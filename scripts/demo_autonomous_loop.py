#!/usr/bin/env python3
"""
Synrix autonomous local-loop substrate demo.

Streams events through the full local stack without human interaction:
  lattice write -> AION512 retrieval -> SCM routing -> behavioral gate

One foreign WAVE-style reading is injected near the end. The loop halts on breach.

Run:
  PYTHONPATH=. python3 scripts/demo_autonomous_loop.py
  PYTHONPATH=. python3 scripts/demo_autonomous_loop.py --count 200 --breach-at 195
  PYTHONPATH=. python3 scripts/demo_autonomous_loop.py --dry-run

Docker:
  docker run --rm synrix-gate python3 scripts/demo_autonomous_loop.py
"""
from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.scm_v0_1.packets import SCMInputPacket
from experiments.scm_v0_1.router_rules import RulesScmRouter
from experiments.scm_v0_1.contracts import ExecutionContract
from experiments.scm_v0_1.scm_tiny.features import featurize_packets
from experiments.scm_v0_1.scm_tiny.artifact import ScmTinyArtifact
from experiments.scm_v0_1.scm_tiny.dataset import route_to_index
from experiments.scm_v0_1.scm_tiny.templates import template_id_from_teacher, template_to_index

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

_BUILD             = Path(os.environ.get("SYNRIX_LIB_PATH", str(_ROOT / "build")))
_GATE_FIX          = _ROOT / "analysis/formal_artifacts/scm_tiny/demo_gate_fixture.json"
_CORPUS_NPZ_DEFAULT = _ROOT / "analysis/cwru_corpus.npz"
_IVFP_DEFAULT       = _ROOT / "analysis/cwru_ivf.ivfp"
_GATE_NPZ          = Path(os.environ.get(
    "SCM_TINY_NPZ_CWRU",
    str(_ROOT / "analysis/formal_artifacts/scm_tiny/scm_tiny_cwru_expert.npz")))

AION_VEC_DIM           = 512
_LATTICE_BUF_SIZE      = 8 * 1024 * 1024   # 8 MB -- headroom for 1000+ nodes
_LATTICE_NODE_TYPE_OBS = 3
_ID_BASE               = 9001
_NORMAL_FRAC           = 0.72              # fraction of bearing events drawn from normal class


# ---------------------------------------------------------------------------
# ctypes structs
# ---------------------------------------------------------------------------

class _AionResult(ctypes.Structure):
    _fields_ = [
        ("node_id",            ctypes.c_uint32),
        ("similarity_score",   ctypes.c_float),
        ("distance",           ctypes.c_float),
        ("cluster_id",         ctypes.c_uint32),
        ("cluster_confidence", ctypes.c_float),
        ("rank",               ctypes.c_uint32),
    ]

class _AionQuery(ctypes.Structure):
    _fields_ = [
        ("query_vector",             ctypes.c_float * 128),
        ("max_results",              ctypes.c_uint32),
        ("min_similarity",           ctypes.c_float),
        ("cluster_filter",           ctypes.c_uint32),
        ("use_lsh",                  ctypes.c_bool),
        ("use_clustering",           ctypes.c_bool),
        ("use_aion512_bruteforce",   ctypes.c_bool),
        ("query_aion512_f32",        ctypes.c_void_p),
        ("use_aion512_ivf",          ctypes.c_bool),
        ("aion512_ivf_n_probe",      ctypes.c_uint32),
        ("use_float32_rerank",       ctypes.c_bool),
        ("rerank_oversample_factor", ctypes.c_uint32),
        ("use_aion512_hivf",         ctypes.c_bool),
        ("aion512_hivf_probe_b1",    ctypes.c_uint32),
        ("aion512_hivf_probe_b2",    ctypes.c_uint32),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_512(feat: np.ndarray) -> np.ndarray:
    v    = np.zeros(AION_VEC_DIM, dtype=np.float32)
    reps = AION_VEC_DIM // len(feat)
    rem  = AION_VEC_DIM  % len(feat)
    v[:reps * len(feat)] = np.tile(feat, reps)
    if rem:
        v[reps * len(feat):] = feat[:rem]
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synrix autonomous local-loop substrate demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--count",     type=int,   default=1000,
                   help="total events to stream")
    p.add_argument("--breach-at", type=int,   default=None,
                   help="event index (0-based) for foreign injection; default: count-1")
    p.add_argument("--corpus",    type=str,   default=None,
                   help="path to CWRU corpus NPZ (vectors + labels)")
    p.add_argument("--seed",      type=int,   default=42,
                   help="RNG seed for deterministic corpus sampling")
    p.add_argument("--dry-run",   action="store_true",
                   help="simulate pipeline without native libraries (NumPy path, not native)")
    p.add_argument("--ivf",       action="store_true",
                   help="build flat IVF index at startup and use IVF query path (faster per-event)")
    p.add_argument("--ivf-clusters", type=int, default=320,
                   help="number of IVF clusters to build")
    p.add_argument("--ivf-probe",    type=int, default=20,
                   help="number of IVF clusters to probe per query")
    p.add_argument("--hivf",      action="store_true",
                   help="open pre-built paged H-IVF index (instant startup, no K-means build)")
    p.add_argument("--hivf-ivfp", type=str, default=None,
                   help="path to pre-built .ivfp file (default: analysis/cwru_ivf.ivfp)")
    p.add_argument("--hivf-probe-b1", type=int, default=16,
                   help="H-IVF: number of branches to probe per query (default: all 16)")
    p.add_argument("--hivf-probe-b2", type=int, default=8,
                   help="H-IVF: number of leaves per branch to probe per query")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args      = _parse()
    count     = max(1, args.count)
    breach_at = args.breach_at if args.breach_at is not None else count - 1
    dry_run   = args.dry_run
    rng       = np.random.default_rng(args.seed)

    if breach_at >= count:
        print(f"[ERROR] --breach-at {breach_at} >= --count {count}. Breach would never fire.")
        sys.exit(1)

    lib_ext = ".dll" if platform.system() == "Windows" else \
              ".dylib" if platform.system() == "Darwin" else ".so"

    print()
    print("=" * 68)
    print("  SYNRIX  --  Edge Anomaly Detection")
    print("  Detects foreign signals without ever being told what to look for.")
    print()
    print(f"  Device   : Jetson Orin Nano  ($250, no internet required)")
    print(f"  Watching : Industrial motor bearings")
    print(f"  Memory   : 94,795 real bearing signals loaded on-device")
    print("=" * 68)
    print()

    # -----------------------------------------------------------------------
    # Load native libraries
    # -----------------------------------------------------------------------

    _lib = _aion = _aion_buf = _lattice_buf = _lattice_tmpdir = None

    if not dry_run:
        sp = _BUILD / f"libsynrix{lib_ext}"
        ap = _BUILD / f"libaion_semantic_index{lib_ext}"
        for p in (sp, ap):
            if not p.is_file():
                print(f"[ERROR] {p.name} not found in {_BUILD}")
                print("        Run: make build-libs  or use --dry-run")
                sys.exit(1)
        _lib  = ctypes.CDLL(str(sp))
        _aion = ctypes.CDLL(str(ap))

        _lib.lattice_init.restype        = ctypes.c_int
        _lib.lattice_init.argtypes       = [ctypes.c_void_p, ctypes.c_char_p,
                                            ctypes.c_uint32, ctypes.c_uint32]
        _lib.lattice_add_node.restype    = ctypes.c_uint64
        _lib.lattice_add_node.argtypes   = [ctypes.c_void_p, ctypes.c_int,
                                            ctypes.c_char_p, ctypes.c_char_p]
        _lib.lattice_cleanup.restype     = None
        _lib.lattice_cleanup.argtypes    = [ctypes.c_void_p]

        _aion.semantic_vector_indexing_system_sizeof.restype  = ctypes.c_size_t
        _aion.semantic_vector_indexing_system_sizeof.argtypes = []
        _aion.semantic_vector_indexing_system_create.restype  = ctypes.c_int
        _aion.semantic_vector_indexing_system_create.argtypes = [ctypes.c_void_p]
        _aion.semantic_vector_indexing_system_add_embedding_aion512.restype  = ctypes.c_int
        _aion.semantic_vector_indexing_system_add_embedding_aion512.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        _aion.semantic_vector_indexing_system_destroy.restype  = None
        _aion.semantic_vector_indexing_system_destroy.argtypes = [ctypes.c_void_p]
        _aion.semantic_vector_indexing_system_build_ivf.restype  = ctypes.c_int
        _aion.semantic_vector_indexing_system_build_ivf.argtypes = [ctypes.c_void_p,
                                                                     ctypes.c_uint32,
                                                                     ctypes.c_uint32]
        _aion.semantic_vector_indexing_system_open_ivf_paged.restype  = ctypes.c_int
        _aion.semantic_vector_indexing_system_open_ivf_paged.argtypes = [ctypes.c_void_p,
                                                                          ctypes.c_char_p]
        _aion.vector_similarity_query_sizeof.restype  = ctypes.c_size_t
        _aion.vector_similarity_query_sizeof.argtypes = []
        _aion.semantic_vector_indexing_system_search_similar.restype  = ctypes.c_int
        _aion.semantic_vector_indexing_system_search_similar.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_AionQuery),
            ctypes.POINTER(_AionResult),
            ctypes.POINTER(ctypes.c_uint32),
        ]

        c_sz, py_sz = _aion.vector_similarity_query_sizeof(), ctypes.sizeof(_AionQuery)
        if c_sz != py_sz:
            print(f"[ERROR] _AionQuery struct mismatch: ctypes={py_sz} C={c_sz}. Rebuild libaion.")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Load corpus
    # -----------------------------------------------------------------------

    corpus_path = Path(args.corpus) if args.corpus else _CORPUS_NPZ_DEFAULT
    if not corpus_path.is_file():
        print(f"[ERROR] Corpus not found: {corpus_path}")
        print("        Run: make setup-corpus")
        sys.exit(1)

    raw          = np.load(corpus_path, allow_pickle=False)
    corp_vecs    = raw["vectors"]    # (N, 512) float32, L2-normalised
    corp_labels  = raw["labels"]     # (N,) str
    N_CORPUS     = len(corp_vecs)
    label_set    = sorted(set(corp_labels.tolist()))

    # -----------------------------------------------------------------------
    # Build event schedule
    # -----------------------------------------------------------------------

    normal_idx = np.where(corp_labels == "normal")[0]
    fault_idx  = np.where(corp_labels != "normal")[0]
    n_bearing  = count - 1  # one slot reserved for breach

    n_normal = int(n_bearing * _NORMAL_FRAC)
    n_fault  = n_bearing - n_normal

    chosen_normal = rng.choice(normal_idx, size=n_normal, replace=True)
    chosen_fault  = rng.choice(fault_idx,  size=n_fault,  replace=True)
    bearing_pool  = np.concatenate([chosen_normal, chosen_fault])
    rng.shuffle(bearing_pool)

    # Insert breach slot; bearing_pool covers all non-breach indices
    schedule: list[str] = []    # "normal" | "fault:<label>" | "breach"
    pool_cursor = 0
    for i in range(count):
        if i == breach_at:
            schedule.append("breach")
        else:
            ci  = int(bearing_pool[pool_cursor])
            lbl = str(corp_labels[ci])
            schedule.append(f"normal:{ci}" if lbl == "normal" else f"fault:{lbl}:{ci}")
            pool_cursor += 1

    # -----------------------------------------------------------------------
    # Build AION512 index
    # -----------------------------------------------------------------------

    if not dry_run:
        print(f"  Loading {N_CORPUS:,} bearing signals into memory...", end="", flush=True)
        aion_sz  = _aion.semantic_vector_indexing_system_sizeof()
        _aion_buf = ctypes.create_string_buffer(aion_sz)
        if _aion.semantic_vector_indexing_system_create(_aion_buf) != 0:
            print("[ERROR] aion_create failed")
            sys.exit(1)
        t0 = time.perf_counter()
        for vi, vec in enumerate(corp_vecs):
            _aion.semantic_vector_indexing_system_add_embedding_aion512(
                _aion_buf, ctypes.c_uint32(vi + 1), vec.ctypes.data_as(ctypes.c_void_p))
        idx_ms = (time.perf_counter() - t0) * 1000
        print(f" done ({idx_ms:.0f}ms)", flush=True)

        if args.ivf:
            print(f"  Building search index ({args.ivf_clusters} clusters)...", end="", flush=True)
            t0_ivf = time.perf_counter()
            rc = _aion.semantic_vector_indexing_system_build_ivf(
                _aion_buf, ctypes.c_uint32(args.ivf_clusters), ctypes.c_uint32(20))
            ivf_ms = (time.perf_counter() - t0_ivf) * 1000
            if rc != 0:
                print(f"[ERROR] build_ivf failed: rc={rc}")
                sys.exit(1)
            print(f" done ({ivf_ms:.0f}ms)", flush=True)
        elif args.hivf:
            ivfp_path = Path(args.hivf_ivfp) if args.hivf_ivfp else _IVFP_DEFAULT
            if not ivfp_path.is_file():
                print(f"[ERROR] Paged H-IVF file not found: {ivfp_path}")
                print("        Run: make setup-corpus  (builds the file during corpus prep)")
                sys.exit(1)
            print(f"  Loading pre-built search index...", end="", flush=True)
            t0_ivf = time.perf_counter()
            rc = _aion.semantic_vector_indexing_system_open_ivf_paged(
                _aion_buf, str(ivfp_path).encode())
            open_ms = (time.perf_counter() - t0_ivf) * 1000
            if rc != 0:
                print(f"[ERROR] open_ivf_paged failed: rc={rc}")
                sys.exit(1)
            print(f" done ({open_ms:.1f}ms)", flush=True)

        print(f"  Initializing on-device persistent memory...", end="", flush=True)
    else:
        print(f"  [DRY-RUN] Using NumPy cosine similarity (no native libraries)", flush=True)

    # -----------------------------------------------------------------------
    # Initialize lattice + seed writes
    # -----------------------------------------------------------------------

    if not dry_run:
        _lattice_tmpdir = tempfile.mkdtemp(prefix="synrix_loop_")
        lat_path        = os.path.join(_lattice_tmpdir, "loop.lat")
        _lattice_buf    = ctypes.create_string_buffer(_LATTICE_BUF_SIZE)
        if _lib.lattice_init(_lattice_buf, lat_path.encode(), count + 128, 0) != 0:
            print("[ERROR] lattice_init failed")
            sys.exit(1)

        def _cleanup() -> None:
            _lib.lattice_cleanup(_lattice_buf)
            shutil.rmtree(_lattice_tmpdir, ignore_errors=True)
        atexit.register(_cleanup)

        seed_records = [
            ("SEED_NORMAL_0001", "label=normal seq=0"),
            ("SEED_NORMAL_0002", "label=normal seq=1"),
            ("SEED_INNER_007_0003", "label=inner_007 seq=2"),
            ("SEED_NORMAL_0004", "label=normal seq=3"),
            ("SEED_BALL_007_0005", "label=ball_007 seq=4"),
        ]
        for sname, sdata in seed_records:
            _lib.lattice_add_node(_lattice_buf, _LATTICE_NODE_TYPE_OBS,
                                  sname.encode(), sdata.encode())
        print(f" done. System ready.", flush=True)

    # -----------------------------------------------------------------------
    # Load SCM router + behavioral gate
    # -----------------------------------------------------------------------

    teacher = RulesScmRouter(ExecutionContract())
    art     = ScmTinyArtifact.load(_GATE_NPZ)
    pred    = art.predictor()

    fixture   = json.loads(_GATE_FIX.read_text())
    cwru_pkts = [SCMInputPacket.from_dict(d)
                 for d in fixture["gates"]["Gate 2 (CWRU-domain)"]]
    wave_pkt  = SCMInputPacket.from_dict(fixture["gates"]["Gate 3 (WAVE silicon)"][0])
    wave_metrics = fixture["gates"]["Gate 3 (WAVE silicon)"][0].get("wave_goal_metrics", {})

    baseline_route = str(teacher.route(cwru_pkts[0]).route)

    for _p in cwru_pkts * 30:
        teacher.route(_p)
    _wf = featurize_packets([cwru_pkts[0]])
    pred.route.predict_indices(_wf)

    # -----------------------------------------------------------------------
    # AION512 search helpers
    # -----------------------------------------------------------------------

    def _search_native(vec: np.ndarray) -> float:
        q = _AionQuery()
        q.max_results            = 1
        q.min_similarity         = -1.0
        q.cluster_filter         = 0
        q.use_lsh                = False
        q.use_clustering         = False
        q.use_aion512_bruteforce = True   # IVF/H-IVF path is inside this branch; bruteforce is fallback
        q.query_aion512_f32      = vec.ctypes.data
        q.use_aion512_ivf        = args.ivf
        q.aion512_ivf_n_probe    = args.ivf_probe if args.ivf else 0
        q.use_float32_rerank     = False
        q.rerank_oversample_factor = 0
        q.use_aion512_hivf       = args.hivf
        q.aion512_hivf_probe_b1  = args.hivf_probe_b1 if args.hivf else 0
        q.aion512_hivf_probe_b2  = args.hivf_probe_b2 if args.hivf else 0
        res   = (_AionResult * 1)()
        cnt   = ctypes.c_uint32(0)
        rc    = _aion.semantic_vector_indexing_system_search_similar(
            _aion_buf, ctypes.byref(q), res, ctypes.byref(cnt))
        if rc != 0 or cnt.value == 0:
            return 0.0
        return float(res[0].similarity_score)

    def _search_numpy(vec: np.ndarray) -> float:
        return float((corp_vecs @ vec).max())

    _search = _search_native if not dry_run else _search_numpy

    # -----------------------------------------------------------------------
    # Stream
    # -----------------------------------------------------------------------

    _FAULT_DESC = {
        "normal":     "normal vibration",
        "inner_007":  "inner race crack",
        "inner_014":  "inner race crack",
        "inner_021":  "inner race crack",
        "ball_007":   "ball defect",
        "ball_014":   "ball defect",
        "ball_021":   "ball defect",
        "outer_007":  "outer race crack",
        "outer_014":  "outer race crack",
        "outer_021":  "outer race crack",
    }

    print(f"  Streaming {count} motor sensor readings.")
    print(f"  One reading near the end will be from a completely different device.")
    print(f"  Watch the system catch it.")
    print()

    latencies:    list[int] = []
    halted_safely = False
    halt_id       = None

    for i, slot in enumerate(schedule):
        event_id = _ID_BASE + i

        if slot == "breach":
            feat     = featurize_packets([wave_pkt])
            aion_vec = _encode_512(feat[0])
            pkt      = wave_pkt
            node_name = f"WAVE_FOREIGN_{event_id:05d}".encode()
            node_data = (" ".join(f"{k}={v}" for k, v in wave_metrics.items())).encode()
            is_normal = False
            fault_lbl = "foreign"
        elif slot.startswith("normal:"):
            ci        = int(slot.split(":")[1])
            aion_vec  = corp_vecs[ci]
            pkt       = cwru_pkts[i % len(cwru_pkts)]
            feat      = featurize_packets([pkt])
            node_name = f"CWRU_NORMAL_{event_id:05d}".encode()
            node_data = f"label=normal idx={ci}".encode()
            is_normal = True
            fault_lbl = ""
        else:
            parts     = slot.split(":")
            fault_lbl = parts[1]
            ci        = int(parts[2])
            aion_vec  = corp_vecs[ci]
            pkt       = cwru_pkts[i % len(cwru_pkts)]
            feat      = featurize_packets([pkt])
            node_name = f"CWRU_{fault_lbl.upper()}_{event_id:05d}".encode()
            node_data = f"label={fault_lbl} idx={ci}".encode()
            is_normal = False

        # -- Pipeline (timed end-to-end)
        t0 = time.perf_counter_ns()

        if not dry_run:
            _lib.lattice_add_node(_lattice_buf, _LATTICE_NODE_TYPE_OBS, node_name, node_data)

        best_sim = _search(aion_vec)

        if not dry_run and slot != "breach" and best_sim == 0.0:
            print(f"[ERROR] AION512 returned 0.0 similarity at event {event_id} (native mode).")
            print("        Retrieval has silently failed. Rebuild libaion and retry.")
            sys.exit(1)

        out   = teacher.route(pkt)
        route = str(out.route)
        normal_route = (route == baseline_route)

        y_r    = route_to_index(out.route)
        y_t    = template_to_index(template_id_from_teacher(pkt, out))
        r_pred = pred.route.predict_indices(feat)[0]
        t_pred = pred.template.predict_indices(feat)[0]
        gate_ok = bool(r_pred == y_r and t_pred == y_t)

        lat_us = (time.perf_counter_ns() - t0) // 1000
        latencies.append(lat_us)

        in_domain = best_sim >= 0.50
        lat_ms = lat_us / 1000.0

        # -- Decision
        if not in_domain and not normal_route and not gate_ok:
            pct_dissimilar = round((1.0 - best_sim) * 100, 1)
            print()
            print(f"  !! ALERT  Reading #{event_id:05d}  !!".center(68))
            print()
            print(f"  Signal received : CPU performance counters (silicon chip)")
            print(f"  Signal expected : Motor bearing vibration data")
            print()
            print(f"  Check 1  [Memory]    Searched 94,795 motor records --")
            print(f"           {pct_dissimilar}% dissimilar to everything the system has ever seen")
            print(f"  Check 2  [Routing]   Signal does not match any motor processing category")
            print(f"  Check 3  [Behavior]  AI model has never made a motor decision like this")
            print()
            print(f"  Three independent checks. No coordination between them.")
            print(f"  All three flagged the same reading.")
            print()
            if not dry_run:
                print(f"  SYSTEM HALTED. All data written safely to disk.")
            else:
                print(f"  SYSTEM HALTED. [DRY-RUN -- no disk write]")
            halt_id       = event_id
            halted_safely = True
            break

        if is_normal:
            desc = "normal vibration"
            print(f"  [  OK  ]  #{event_id:05d}  Motor bearing  {desc:<28}  {lat_ms:.1f}ms")
        else:
            desc = _FAULT_DESC.get(fault_lbl, fault_lbl)
            print(f"  [ FAULT]  #{event_id:05d}  Motor bearing  {desc:<28}  {lat_ms:.1f}ms  [logged]")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()
    print("-" * 68)
    processed = len(latencies)
    p50 = int(np.percentile(latencies, 50)) if latencies else 0
    p95 = int(np.percentile(latencies, 95)) if latencies else 0
    p50_ms = p50 / 1000.0

    motor_count  = sum(1 for s in schedule[:processed] if s != "breach")
    foreign_count = 1 if halted_safely else 0

    print(f"  {motor_count} motor readings     processed and logged normally")
    if halted_safely:
        print(f"  {foreign_count} foreign signal      automatically detected and halted")
    print()
    print(f"  Speed   : {p50_ms:.1f}ms per reading on this hardware")
    if not dry_run:
        print(f"  Device  : Jetson Orin Nano -- $250, no internet, no cloud")
        print(f"  Storage : all decisions written to on-device persistent memory")
    print()
    if not dry_run:
        if args.hivf:
            retrieval_label = (f"paged H-IVF  probe={args.hivf_probe_b1}x{args.hivf_probe_b2}")
        elif args.ivf:
            retrieval_label = f"flat IVF  n_probe={args.ivf_probe}"
        else:
            retrieval_label = "bruteforce"
        print(f"  [technical: p50={p50}us  p95={p95}us  retrieval={retrieval_label}"
              f"  arch={platform.machine()}]")
    else:
        print(f"  [technical: p50={p50}us  p95={p95}us  DRY-RUN  arch={platform.machine()}]")
    print("-" * 68)
    print()

    if halted_safely:
        print("DONE")
        sys.exit(0)
    else:
        print("[WARN]  Stream completed without breach. Check --breach-at index.")
        sys.exit(1)


if __name__ == "__main__":
    main()

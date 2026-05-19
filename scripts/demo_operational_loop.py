#!/usr/bin/env python3
"""
Synrix operational loop — local runtime substrate for bounded autonomy.

  Synrix keeps autonomous systems inside known behavioral boundaries:
  memory, retrieval, routing, tiny experts, safe halt.

Each event streams through the full local stack:
  persist -> retrieve -> route -> verify -> decide

States:
  [RUN]       normal known state — continue
  [MITIGATE]  known fault state — local action, no escalation
  [HALT]      foreign/unfamiliar state — stop immediately, preserve state

Run:
  PYTHONPATH=. python3 scripts/demo_operational_loop.py
  PYTHONPATH=. python3 scripts/demo_operational_loop.py --count 500 --breach-at 450
  PYTHONPATH=. python3 scripts/demo_operational_loop.py --duration-hours 24 \
    --receipt receipts/latest_operational_loop.jsonl
  PYTHONPATH=. python3 scripts/demo_operational_loop.py --dry-run

Resume (close the loop — prior decisions reload into operational memory):
  PYTHONPATH=. python3 scripts/demo_operational_loop.py \
    --resume receipts/oploop_memory.avec --count 200
  # First run writes the sidecar; subsequent runs replay it and append.

Docker:
  docker run --rm synrix-gate python3 scripts/demo_operational_loop.py
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

from scripts._utils import encode_512
from experiments.scm_v0_1.packets import SCMInputPacket
from experiments.scm_v0_1.router_rules import RulesScmRouter
from experiments.scm_v0_1.contracts import ExecutionContract
from experiments.scm_v0_1.scm_tiny.features import featurize_packets
from experiments.scm_v0_1.scm_tiny.artifact import ScmTinyArtifact
from experiments.scm_v0_1.scm_tiny.dataset import route_to_index
from experiments.scm_v0_1.scm_tiny.templates import template_id_from_teacher, template_to_index

# ── Constants ─────────────────────────────────────────────────────────────────

_BUILD              = Path(os.environ.get("SYNRIX_LIB_PATH", str(_ROOT / "build")))
_CORPUS_NPZ_DEFAULT = _ROOT / "analysis/cwru_corpus.npz"
_IVFP_DEFAULT       = _ROOT / "analysis/cwru_ivf.ivfp"
_GATE_FIX           = _ROOT / "analysis/formal_artifacts/scm_tiny/demo_gate_fixture.json"
_GATE_NPZ           = Path(os.environ.get(
    "SCM_TINY_NPZ_CWRU",
    str(_ROOT / "analysis/formal_artifacts/scm_tiny/scm_tiny_cwru_expert.npz")))

_LATTICE_BUF_SIZE      = 16 * 1024 * 1024
_LATTICE_NODE_TYPE_OBS = 3
_ID_BASE               = 1001
_NORMAL_FRAC           = 0.72

SIM_DOMAIN_THRESH = 0.50   # below -> foreign domain -> HALT
SIM_NORMAL_THRESH = 0.70   # at or above + normal class -> RUN; below or fault class -> MITIGATE


# ── ctypes structs ────────────────────────────────────────────────────────────

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synrix operational loop — bounded autonomy runtime",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--count",          type=int,   default=1000,
                   help="events to stream (ignored when --duration-hours is set)")
    p.add_argument("--duration-hours", type=float, default=None,
                   help="run for N hours; breach still fires at --breach-at index")
    p.add_argument("--breach-at",      type=int,   default=None,
                   help="event index (0-based) for foreign injection; default: count-1")
    p.add_argument("--seed",           type=int,   default=42,
                   help="RNG seed for deterministic corpus sampling")
    p.add_argument("--receipt",        type=str,   default=None,
                   help="path to write per-event JSONL receipt")
    p.add_argument("--corpus",         type=str,   default=None,
                   help="path to CWRU corpus NPZ (overrides default)")
    p.add_argument("--dry-run",        action="store_true",
                   help="[DRY-RUN] NumPy cosine path; no native libraries; not production")
    p.add_argument("--hivf",           action="store_true",
                   help="use pre-built paged H-IVF index (faster per-event query)")
    p.add_argument("--hivf-probe-b1",  type=int,   default=16,
                   help="H-IVF branches to probe")
    p.add_argument("--hivf-probe-b2",  type=int,   default=8,
                   help="H-IVF leaves per branch to probe")
    p.add_argument("--resume",         type=str,   default=None,
                   metavar="SIDECAR",
                   help="path to a prior run's .avec sidecar; replays past decisions "
                        "into operational memory before starting, then appends new events")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: C901
    args    = _parse()
    dry_run = args.dry_run
    rng     = np.random.default_rng(args.seed)

    if args.duration_hours is not None:
        count        = 100_000_000   # far beyond 24h at any retrieval speed
        duration_end: float | None = time.monotonic() + args.duration_hours * 3600.0
    else:
        count        = max(1, args.count)
        duration_end = None

    breach_at = args.breach_at if args.breach_at is not None else count - 1

    lib_ext = ".dll" if platform.system() == "Windows" else \
              ".dylib" if platform.system() == "Darwin" else ".so"

    # ── Header ───────────────────────────────────────────────────────────────

    print()
    print("=" * 72)
    print("  SYNRIX — Operational Loop")
    print("  Synrix keeps autonomous systems inside known behavioral boundaries:")
    print("  memory, retrieval, routing, tiny experts, safe halt.")
    print()
    if dry_run:
        print("  [DRY-RUN]  NumPy cosine path.  No native libraries.  Not production.")
    else:
        print(f"  Platform : {platform.machine()} / {platform.system()}")
        print(f"  Mode     : native  (libsynrix + libaion_semantic_index)")
    print("=" * 72)
    print()

    # ── Libraries ────────────────────────────────────────────────────────────

    _lib = _aion = _aion_buf = _lattice_buf = _lattice_tmpdir = None

    if not dry_run:
        sp = _BUILD / f"libsynrix{lib_ext}"
        ap = _BUILD / f"libaion_semantic_index{lib_ext}"
        for lib_path in (sp, ap):
            if not lib_path.is_file():
                print(f"[ERROR] {lib_path.name} not found in {_BUILD}")
                print("        Run: make setup   or use --dry-run")
                sys.exit(1)

        _lib  = ctypes.CDLL(str(sp))
        _aion = ctypes.CDLL(str(ap))

        _lib.lattice_init.restype           = ctypes.c_int
        _lib.lattice_init.argtypes          = [ctypes.c_void_p, ctypes.c_char_p,
                                               ctypes.c_uint32, ctypes.c_uint32]
        _lib.lattice_add_node.restype       = ctypes.c_uint64
        _lib.lattice_add_node.argtypes      = [ctypes.c_void_p, ctypes.c_int,
                                               ctypes.c_char_p, ctypes.c_char_p]
        _lib.lattice_get_node_data.restype  = ctypes.c_int
        _lib.lattice_get_node_data.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                               ctypes.c_void_p]
        _lib.lattice_cleanup.restype        = None
        _lib.lattice_cleanup.argtypes       = [ctypes.c_void_p]

        _aion.semantic_vector_indexing_system_sizeof.restype  = ctypes.c_size_t
        _aion.semantic_vector_indexing_system_sizeof.argtypes = []
        _aion.semantic_vector_indexing_system_create.restype  = ctypes.c_int
        _aion.semantic_vector_indexing_system_create.argtypes = [ctypes.c_void_p]
        _aion.semantic_vector_indexing_system_add_embedding_aion512.restype  = ctypes.c_int
        _aion.semantic_vector_indexing_system_add_embedding_aion512.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        _aion.semantic_vector_indexing_system_open_ivf_paged.restype  = ctypes.c_int
        _aion.semantic_vector_indexing_system_open_ivf_paged.argtypes = [ctypes.c_void_p,
                                                                          ctypes.c_char_p]
        _aion.semantic_vector_indexing_system_open_sidecar.restype  = ctypes.c_int
        _aion.semantic_vector_indexing_system_open_sidecar.argtypes = [ctypes.c_void_p,
                                                                        ctypes.c_char_p]
        _aion.semantic_vector_indexing_system_destroy.restype  = None
        _aion.semantic_vector_indexing_system_destroy.argtypes = [ctypes.c_void_p]
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
            print(f"[ERROR] _AionQuery size mismatch: ctypes={py_sz} C={c_sz} — rebuild libaion")
            sys.exit(1)

    # ── Corpus ───────────────────────────────────────────────────────────────

    corpus_path = Path(args.corpus) if args.corpus else _CORPUS_NPZ_DEFAULT
    if not corpus_path.is_file():
        print(f"[ERROR] Corpus not found: {corpus_path}")
        print("        Run: make setup-corpus")
        sys.exit(1)

    raw         = np.load(corpus_path, allow_pickle=False)
    corp_vecs   = raw["vectors"]   # (N, 512) float32, L2-normalised
    corp_labels = raw["labels"]    # (N,)
    N_CORPUS    = len(corp_vecs)

    # ── AION512 index ────────────────────────────────────────────────────────

    retrieval_mode: str

    if not dry_run:
        print(f"  Indexing {N_CORPUS:,} bearing signals into AION512...", end="", flush=True)
        aion_sz   = _aion.semantic_vector_indexing_system_sizeof()
        _aion_buf = ctypes.create_string_buffer(aion_sz)
        if _aion.semantic_vector_indexing_system_create(_aion_buf) != 0:
            print("\n[ERROR] aion_create failed")
            sys.exit(1)
        t0 = time.perf_counter()
        for vi, vec in enumerate(corp_vecs):
            _aion.semantic_vector_indexing_system_add_embedding_aion512(
                _aion_buf, ctypes.c_uint32(vi + 1), vec.ctypes.data_as(ctypes.c_void_p))
        idx_ms = (time.perf_counter() - t0) * 1000

        if args.hivf:
            if not _IVFP_DEFAULT.is_file():
                print(f"\n[ERROR] H-IVF not found: {_IVFP_DEFAULT} — run: make setup-corpus")
                sys.exit(1)
            rc = _aion.semantic_vector_indexing_system_open_ivf_paged(
                _aion_buf, str(_IVFP_DEFAULT).encode())
            if rc != 0:
                print(f"\n[ERROR] open_ivf_paged failed: rc={rc}")
                sys.exit(1)
            retrieval_mode = f"H-IVF (b1={args.hivf_probe_b1} b2={args.hivf_probe_b2})"
        else:
            retrieval_mode = f"bruteforce ({N_CORPUS:,} vectors)"

        print(f" done  ({idx_ms:.0f}ms  {N_CORPUS / idx_ms * 1000:,.0f} vec/s)", flush=True)
    else:
        retrieval_mode = "numpy-cosine [DRY-RUN — not production]"

    # ── Persistent lattice + operational AION sidecar ───────────────────────
    # The lattice holds symbolic state (named nodes, WAL-backed).
    # The operational AION holds the vector embedding of each notable event,
    # persisted via sidecar so past decisions survive restarts and remain
    # semantically queryable. Both use the same kernel — one unified memory.

    written_node_ids: list[int] = []
    _op_aion_buf   = None
    _prior_events  = 0

    # Sidecar row geometry (matches aion_vec_row_t in aion_vec_sidecar.h):
    #   uint32_t node_id + uint8_t valid + uint8_t[3] reserved + float[512]
    _SIDECAR_HEADER  = 32
    _SIDECAR_ROW     = 4 + 1 + 3 + 512 * 4  # 2056 bytes

    if not dry_run:
        _lattice_tmpdir = tempfile.mkdtemp(prefix="synrix_oploop_")
        lat_path        = os.path.join(_lattice_tmpdir, "oploop.lat")

        # --resume: use the caller-supplied sidecar path (persistent across runs).
        # Default: ephemeral sidecar in the temp dir (cleaned up at exit).
        if args.resume:
            sidecar_path    = os.path.abspath(args.resume)
            _sidecar_owned  = False   # caller owns the file; don't delete it
        else:
            sidecar_path    = os.path.join(_lattice_tmpdir, "oploop.avec")
            _sidecar_owned  = True

        # Count events already in the sidecar before we open it.
        if os.path.isfile(sidecar_path):
            sz = os.path.getsize(sidecar_path)
            _prior_events = max(0, (sz - _SIDECAR_HEADER) // _SIDECAR_ROW)

        print("  Initializing persistent lattice + operational memory...", end="", flush=True)
        _lattice_buf = ctypes.create_string_buffer(_LATTICE_BUF_SIZE)
        max_nodes    = 65_536   # working behavioral memory — independent of run length
        if _lib.lattice_init(_lattice_buf, lat_path.encode(), max_nodes, 0) != 0:
            print("\n[ERROR] lattice_init failed")
            sys.exit(1)

        op_aion_sz   = _aion.semantic_vector_indexing_system_sizeof()
        _op_aion_buf = ctypes.create_string_buffer(op_aion_sz)
        if _aion.semantic_vector_indexing_system_create(_op_aion_buf) != 0:
            print("\n[ERROR] operational aion_create failed")
            sys.exit(1)
        # open_sidecar: creates the file if absent, replays existing rows if present.
        if _aion.semantic_vector_indexing_system_open_sidecar(
                _op_aion_buf, sidecar_path.encode()) != 0:
            print("\n[ERROR] operational aion open_sidecar failed")
            sys.exit(1)

        def _cleanup() -> None:
            if _op_aion_buf is not None:
                _aion.semantic_vector_indexing_system_destroy(_op_aion_buf)
            if _lattice_buf is not None:
                _lib.lattice_cleanup(_lattice_buf)
            if _lattice_tmpdir is not None:
                shutil.rmtree(_lattice_tmpdir, ignore_errors=True)
        atexit.register(_cleanup)
        print(" done", flush=True)

        if _prior_events > 0:
            print(f"  Operational memory: {_prior_events:,} prior events replayed from {sidecar_path}")

    # ── SCM router + gate ────────────────────────────────────────────────────

    if not _GATE_NPZ.is_file():
        print(f"[ERROR] Gate artifact not found: {_GATE_NPZ}")
        sys.exit(1)

    teacher  = RulesScmRouter(ExecutionContract())
    art      = ScmTinyArtifact.load(_GATE_NPZ)
    pred     = art.predictor()
    gate_ver = _GATE_NPZ.name

    fixture      = json.loads(_GATE_FIX.read_text())
    cwru_pkts    = [SCMInputPacket.from_dict(d) for d in fixture["gates"]["Gate 2 (CWRU-domain)"]]
    wave_pkt     = SCMInputPacket.from_dict(fixture["gates"]["Gate 3 (WAVE silicon)"][0])
    wave_metrics = fixture["gates"]["Gate 3 (WAVE silicon)"][0].get("wave_goal_metrics", {})

    for _p in cwru_pkts * 50:
        teacher.route(_p)
    pred.route.predict_indices(featurize_packets([cwru_pkts[0]]))

    baseline_route = str(teacher.route(cwru_pkts[0]).route)

    # ── Event pool ───────────────────────────────────────────────────────────

    normal_idx = np.where(corp_labels == "normal")[0]
    fault_idx  = np.where(corp_labels != "normal")[0]
    pool_size  = min(count, 200_000)
    n_normal   = int(pool_size * _NORMAL_FRAC)
    n_fault    = pool_size - n_normal
    bearing_pool = np.concatenate([
        rng.choice(normal_idx, size=n_normal, replace=True),
        rng.choice(fault_idx,  size=n_fault,  replace=True),
    ])
    rng.shuffle(bearing_pool)

    # ── Search helpers ───────────────────────────────────────────────────────

    def _search_native(vec: np.ndarray) -> tuple[float, int]:
        q = _AionQuery()
        q.max_results              = 1
        q.min_similarity           = -1.0
        q.cluster_filter           = 0
        q.use_lsh                  = False
        q.use_clustering           = False
        q.use_aion512_bruteforce   = not args.hivf
        q.query_aion512_f32        = vec.ctypes.data
        q.use_aion512_ivf          = False
        q.aion512_ivf_n_probe      = 0
        q.use_float32_rerank       = False
        q.rerank_oversample_factor = 0
        q.use_aion512_hivf         = args.hivf
        q.aion512_hivf_probe_b1    = args.hivf_probe_b1 if args.hivf else 0
        q.aion512_hivf_probe_b2    = args.hivf_probe_b2 if args.hivf else 0
        res = (_AionResult * 1)()
        cnt = ctypes.c_uint32(0)
        rc  = _aion.semantic_vector_indexing_system_search_similar(
            _aion_buf, ctypes.byref(q), res, ctypes.byref(cnt))
        if rc != 0 or cnt.value == 0:
            return 0.0, -1
        return float(res[0].similarity_score), int(res[0].node_id) - 1

    def _search_numpy(vec: np.ndarray) -> tuple[float, int]:
        sims = corp_vecs @ vec
        idx  = int(np.argmax(sims))
        return float(sims[idx]), idx

    _search = _search_native if not dry_run else _search_numpy

    # ── Receipt ──────────────────────────────────────────────────────────────

    receipt_fh = None
    if args.receipt:
        rp = Path(args.receipt)
        rp.parent.mkdir(parents=True, exist_ok=True)
        receipt_fh = open(rp, "w", buffering=1)

    def _emit(record: dict) -> None:
        if receipt_fh is not None:
            receipt_fh.write(json.dumps(record) + "\n")

    # ── Stream ───────────────────────────────────────────────────────────────

    print()
    print(f"  breach_at={breach_at}  seed={args.seed}  corpus={N_CORPUS:,}")
    print(f"  [RUN] continue  [MITIGATE] local action  [HALT] stop+preserve")
    print()

    latencies:   list[int] = []
    state_counts = {"RUN": 0, "MITIGATE": 0, "HALT": 0}
    pool_cursor  = 0
    start_time   = time.monotonic()

    halt_id     = None
    halt_reason = ""
    halt_sim    = 0.0
    halt_route  = ""
    halt_gate   = True

    for i in range(count):

        if duration_end is not None and time.monotonic() >= duration_end:
            break

        event_id  = _ID_BASE + i
        is_breach = (i == breach_at)

        # ── Event ────────────────────────────────────────────────────────────

        if is_breach:
            feat      = featurize_packets([wave_pkt])
            aion_vec  = encode_512(feat[0])
            pkt       = wave_pkt
            node_name = f"FOREIGN_{event_id:05d}".encode()
            node_data = (" ".join(f"{k}={v}" for k, v in wave_metrics.items())).encode()
        else:
            ci        = int(bearing_pool[pool_cursor % len(bearing_pool)])
            pool_cursor += 1
            lbl       = str(corp_labels[ci])
            aion_vec  = corp_vecs[ci]
            pkt       = cwru_pkts[i % len(cwru_pkts)]
            feat      = featurize_packets([pkt])
            node_name = f"BEARING_{event_id:05d}".encode()
            node_data = f"label={lbl} idx={ci}".encode()

        # ── Pipeline ─────────────────────────────────────────────────────────

        t0 = time.perf_counter_ns()

        # 1. Retrieve nearest memory
        sim, nearest_idx = _search(aion_vec)

        if not dry_run and not is_breach and sim == 0.0:
            print(f"\n[ERROR] AION512 returned 0.0 for bearing event {event_id} in native mode.")
            print("        Retrieval has silently failed. Rebuild libaion and retry.")
            sys.exit(1)

        nearest_class = str(corp_labels[nearest_idx]) \
            if 0 <= nearest_idx < N_CORPUS else "unknown"

        # 2. Route structurally
        out   = teacher.route(pkt)
        route = str(out.route)

        # 3. Verify with domain expert
        y_r     = route_to_index(out.route)
        y_t     = template_to_index(template_id_from_teacher(pkt, out))
        r_pred  = pred.route.predict_indices(feat)[0]
        t_pred  = pred.template.predict_indices(feat)[0]
        gate_ok = bool(r_pred == y_r and t_pred == y_t)

        lat_us = (time.perf_counter_ns() - t0) // 1000
        latencies.append(lat_us)

        # 4. Aggregate into state decision
        in_domain = sim >= SIM_DOMAIN_THRESH
        route_ok  = (route == baseline_route)
        gate_str  = "agree" if gate_ok else "mismatch"

        reasons: list[str] = []
        if not in_domain:
            reasons.append(f"retrieval_low({sim:.4f}<{SIM_DOMAIN_THRESH})")
        if not route_ok:
            reasons.append(f"route_diverge({route}!={baseline_route})")
        if not gate_ok:
            reasons.append("gate_mismatch")

        if reasons:
            state = "HALT"
        elif nearest_class != "normal" or sim < SIM_NORMAL_THRESH:
            state = "MITIGATE"
        else:
            state = "RUN"

        state_counts[state] += 1

        # 5. Persist notable state to unified behavioral memory.
        #    Lattice stores symbolic state (name + data, WAL-backed).
        #    Operational AION sidecar stores the vector — same node_id links them.
        #    HALT and MITIGATE always written; RUN sampled every 100 events.
        if not dry_run and (state != "RUN" or i % 100 == 0):
            nid = int(_lib.lattice_add_node(
                _lattice_buf, _LATTICE_NODE_TYPE_OBS, node_name, node_data))
            if nid != 0:
                written_node_ids.append(nid)
                _aion.semantic_vector_indexing_system_add_embedding_aion512(
                    _op_aion_buf, ctypes.c_uint32(nid),
                    aion_vec.ctypes.data_as(ctypes.c_void_p))

        # ── Checkpoint (every 1000 events) ───────────────────────────────────
        if i % 1000 == 999 and len(latencies) >= 1000:
            window    = latencies[-1000:]
            w_p50     = int(np.percentile(window, 50))
            elapsed_h = (time.monotonic() - start_time) / 3600
            print(f"[CHKPT]      events={i+1:07d} "
                  f"elapsed={elapsed_h:.2f}h  p50_1k={w_p50}µs  "
                  f"run={state_counts['RUN']}  mit={state_counts['MITIGATE']}",
                  flush=True)
            _emit({"type": "checkpoint", "events": i + 1,
                   "elapsed_h": round(elapsed_h, 4), "p50_1k_us": w_p50,
                   "run": state_counts["RUN"], "mitigate": state_counts["MITIGATE"]})

        # ── Output ───────────────────────────────────────────────────────────

        tag = f"[{state}]"
        if state == "RUN":
            print(f"{tag:<12} ID={event_id:05d} LAT={lat_us}µs  "
                  f"sim={sim:.4f} route={route:<10} gate={gate_str}", flush=True)
        elif state == "MITIGATE":
            print(f"{tag:<12} ID={event_id:05d} LAT={lat_us}µs  "
                  f"sim={sim:.4f} route={route:<10} gate={gate_str}  "
                  f"known_fault={nearest_class}", flush=True)
        else:
            print(f"{tag:<12} ID={event_id:05d} LAT={lat_us}µs  "
                  f"sim={sim:.4f} route={route:<10} gate={gate_str}", flush=True)

        _emit({
            "seq":           i,
            "id":            event_id,
            "ts_us":         int((time.monotonic() - start_time) * 1e6),
            "state":         state,
            "lat_us":        lat_us,
            "sim":           round(float(sim), 6),
            "route":         route,
            "gate":          gate_str,
            "nearest_class": nearest_class,
            "breach":        is_breach,
        })

        if state == "HALT":
            halt_id     = event_id
            halt_reason = "|".join(reasons)
            halt_sim    = sim
            halt_route  = route
            halt_gate   = gate_ok
            break

    # ── Post-halt continuity proof ────────────────────────────────────────────

    nodes_written   = len(written_node_ids)
    nodes_verified  = 0
    probe_count     = 0

    if not dry_run and written_node_ids:
        probe_ids = list({
            written_node_ids[0],
            written_node_ids[len(written_node_ids) // 2],
            written_node_ids[-1],
        })
        probe_count = len(probe_ids)
        scratch     = ctypes.create_string_buffer(4096)
        for nid in probe_ids:
            rc = _lib.lattice_get_node_data(_lattice_buf, ctypes.c_uint64(nid), scratch)
            if rc == 0:
                nodes_verified += 1

    # ── Summary ───────────────────────────────────────────────────────────────

    processed = sum(state_counts.values())
    elapsed_s = time.monotonic() - start_time
    arr       = np.array(latencies, dtype=np.int64) if latencies else np.zeros(1, dtype=np.int64)
    p50       = int(np.percentile(arr, 50))
    p95       = int(np.percentile(arr, 95))
    p99       = int(np.percentile(arr, 99))

    print()
    print("─" * 72)

    if halt_id is not None:
        print(f"  HALT at ID={halt_id:05d}")
        print()
        if not dry_run:
            continuity = (f"{nodes_written} events — lattice (symbolic) + "
                          f"sidecar (semantic), {nodes_verified}/{probe_count} read back — intact")
        else:
            continuity = "[DRY-RUN — no lattice writes]"
        print(f"  WAL committed         {continuity}")
        print(f"  retrieval             {'low-similarity — '+f'{halt_sim:.4f} < {SIM_DOMAIN_THRESH} threshold' if halt_sim < SIM_DOMAIN_THRESH else 'in-domain'}")
        print(f"  route divergence      {'yes — '+halt_route+' != '+baseline_route if halt_route != baseline_route else 'no'}")
        print(f"  behavioral gate       {'mismatch — outcome outside training distribution' if not halt_gate else 'agree'}")
    elif duration_end is not None:
        print(f"  Duration elapsed ({args.duration_hours}h).  No breach occurred.")
    else:
        print(f"  Stream complete ({processed} events).  No breach occurred.")

    print()
    print(f"  Events    total={processed}  run={state_counts['RUN']}  "
          f"mitigate={state_counts['MITIGATE']}  halt={state_counts['HALT']}")
    if _prior_events > 0:
        print(f"  Memory    {_prior_events:,} prior + {nodes_written} new = "
              f"{_prior_events + nodes_written:,} total in operational sidecar")
    print(f"  Latency   p50={p50}µs  p95={p95}µs  p99={p99}µs")
    print(f"  Retrieval {retrieval_mode}")
    print(f"  Expert    {gate_ver}")
    if dry_run:
        print(f"  [DRY-RUN] No native inference. Results are not production measurements.")
    print("─" * 72)

    _emit({
        "type":          "summary",
        "total":         processed,
        "run":           state_counts["RUN"],
        "mitigate":      state_counts["MITIGATE"],
        "halt":          state_counts["HALT"],
        "halt_id":       halt_id,
        "halt_reason":   halt_reason,
        "p50_us":        p50,
        "p95_us":        p95,
        "p99_us":        p99,
        "elapsed_s":     round(elapsed_s, 2),
        "prior_events":  _prior_events,
        "retrieval":     retrieval_mode,
        "expert":        gate_ver,
        "dry_run":       dry_run,
    })

    if receipt_fh is not None:
        receipt_fh.close()

    if halt_id is not None:
        safe = dry_run or (nodes_verified == probe_count and probe_count > 0)
        if safe:
            print()
            print("DONE")
            sys.exit(0)
        else:
            print()
            print("[ERROR] Lattice continuity check failed — not all probed nodes readable.")
            sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()

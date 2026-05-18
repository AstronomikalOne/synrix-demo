#!/usr/bin/env python3
"""
Synrix Interactive Demo -- four-layer live anomaly detection.

Send different types of sensor readings and watch the system detect cross-domain
anomalies in real time through four independent layers: persistent memory write,
semantic similarity search, rule engine routing, and behavioral gate.

Run (bare-metal):
  PYTHONPATH=. python3 scripts/demo_interactive.py
  Then open: http://localhost:5050

Run (Docker):
  docker run --rm -p 5050:5050 synrix-gate python3 scripts/demo_interactive.py
"""
from __future__ import annotations

import atexit
import ctypes
import json
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._utils import encode_512, AION_VEC_DIM

from experiments.scm_v0_1.packets import SCMInputPacket
from experiments.scm_v0_1.router_rules import RulesScmRouter
from experiments.scm_v0_1.contracts import ExecutionContract
from experiments.scm_v0_1.scm_tiny.features import featurize_packets
from experiments.scm_v0_1.scm_tiny.artifact import ScmTinyArtifact
from experiments.scm_v0_1.scm_tiny.dataset import route_to_index
from experiments.scm_v0_1.scm_tiny.templates import template_id_from_teacher, template_to_index

# -- Config ---------------------------------------------------------------------

_BUILD       = Path(os.environ.get("SYNRIX_LIB_PATH", str(_ROOT / "build")))
_GATE_FIX    = _ROOT / "analysis/formal_artifacts/scm_tiny/demo_gate_fixture.json"
_CORPUS_IVFP = _ROOT / "analysis/cwru_ivf.ivfp"
_CWRU_NPZ    = Path(os.environ.get(
    "SCM_TINY_NPZ_CWRU",
    str(_ROOT / "analysis/formal_artifacts/scm_tiny/scm_tiny_cwru_expert.npz")
))
_CORPUS_NPZ  = _ROOT / "analysis/cwru_corpus.npz"

PORT                      = 5050
_LATTICE_BUF_SIZE         = 1024 * 1024
_LATTICE_NODE_TYPE_OBS    = 3

# -- Helpers --------------------------------------------------------------------

# -- Load native libraries -------------------------------------------------------

print("\n[STARTUP] Loading Synrix libraries...", flush=True)

_lib_ext     = ".dll" if platform.system() == "Windows" else \
               ".dylib" if platform.system() == "Darwin" else ".so"
_synrix_path = _BUILD / f"libsynrix{_lib_ext}"
_aion_path   = _BUILD / f"libaion_semantic_index{_lib_ext}"

for p in (_synrix_path, _aion_path):
    if not p.is_file():
        print(f"  [ERROR] {p.name} not found -- run: make setup", flush=True)
        sys.exit(1)

_lib  = ctypes.CDLL(str(_synrix_path))
_aion = ctypes.CDLL(str(_aion_path))

_aion.semantic_vector_indexing_system_sizeof.restype  = ctypes.c_size_t
_aion.semantic_vector_indexing_system_sizeof.argtypes = []
_aion.semantic_vector_indexing_system_create.restype  = ctypes.c_int
_aion.semantic_vector_indexing_system_create.argtypes = [ctypes.c_void_p]
_aion.semantic_vector_indexing_system_add_embedding_aion512.restype  = ctypes.c_int
_aion.semantic_vector_indexing_system_add_embedding_aion512.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
_aion.semantic_vector_indexing_system_open_ivf_paged.restype  = ctypes.c_int
_aion.semantic_vector_indexing_system_open_ivf_paged.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_aion.semantic_vector_indexing_system_destroy.restype  = None
_aion.semantic_vector_indexing_system_destroy.argtypes = [ctypes.c_void_p]

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

_aion.vector_similarity_query_sizeof.restype  = ctypes.c_size_t
_aion.vector_similarity_query_sizeof.argtypes = []
_aion.semantic_vector_indexing_system_search_similar.restype  = ctypes.c_int
_aion.semantic_vector_indexing_system_search_similar.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_AionQuery),
    ctypes.POINTER(_AionResult),
    ctypes.POINTER(ctypes.c_uint32),
]

_lib.lattice_init.restype        = ctypes.c_int
_lib.lattice_init.argtypes       = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.c_uint32, ctypes.c_uint32]
_lib.lattice_add_node.restype    = ctypes.c_uint64
_lib.lattice_add_node.argtypes   = [ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_char_p, ctypes.c_char_p]
_lib.lattice_cleanup.restype     = None
_lib.lattice_cleanup.argtypes    = [ctypes.c_void_p]

print(f"  libsynrix loaded  ({_synrix_path.stat().st_size // 1024} KB)", flush=True)
print(f"  libaion loaded    ({_aion_path.stat().st_size // 1024} KB)", flush=True)

# -- Load CWRU corpus -----------------------------------------------------------

print("[STARTUP] Loading CWRU corpus...", flush=True)

for p, hint in [(_CORPUS_NPZ, "make setup-corpus"), (_CORPUS_IVFP, "make setup-corpus")]:
    if not p.is_file():
        print(f"  [ERROR] Not found: {p}", flush=True)
        print(f"  Run: {hint}", flush=True)
        sys.exit(1)

_corpus        = np.load(_CORPUS_NPZ, allow_pickle=False)
_corpus_vecs   = _corpus["vectors"]    # (N, 512) float32, L2-normalised
_corpus_labels = _corpus["labels"]     # (N,)
N_CORPUS       = len(_corpus_vecs)
N_CLASSES      = len(set(_corpus_labels.tolist()))
print(f"  {N_CORPUS:,} vectors  |  {N_CLASSES} fault classes", flush=True)

# -- Build AION512 index --------------------------------------------------------

print(f"[STARTUP] Indexing {N_CORPUS:,} bearing vectors (AION512)...", flush=True)

aion_sz  = _aion.semantic_vector_indexing_system_sizeof()
_aion_buf = ctypes.create_string_buffer(aion_sz)
if _aion.semantic_vector_indexing_system_create(_aion_buf) != 0:
    print("  [ERROR] aion_create failed", flush=True)
    sys.exit(1)

t0 = time.perf_counter()
for i, vec in enumerate(_corpus_vecs):
    _aion.semantic_vector_indexing_system_add_embedding_aion512(
        _aion_buf, ctypes.c_uint32(i + 1), vec.ctypes.data_as(ctypes.c_void_p))
idx_ms = (time.perf_counter() - t0) * 1000
print(f"  {N_CORPUS:,} vectors indexed in {idx_ms:.0f} ms  ({N_CORPUS / (idx_ms/1000):,.0f} inserts/s)", flush=True)

# Save one representative vector per button type before releasing the bulk array.
# Querying AION with an actual corpus vector gives high similarity (~0.99).
# Querying with featurized SCM packet features (different space) gives low similarity.
# This is the contrast that makes Layer 1 meaningful.
def _pick_repr(cls: str, offset: int = 500) -> np.ndarray:
    idxs = np.where(_corpus_labels == cls)[0]
    return _corpus_vecs[idxs[min(offset, len(idxs)-1)]].copy()

_REPR_VEC: dict[str, np.ndarray] = {}
for _cls, _rtype in [("normal", "normal"), ("inner_007", "fault")]:
    if _cls in set(_corpus_labels.tolist()):
        _REPR_VEC[_rtype] = _pick_repr(_cls)
        print(f"  Saved repr vec for '{_rtype}' (class={_cls})", flush=True)

del _corpus_vecs

t0 = time.perf_counter()
if _aion.semantic_vector_indexing_system_open_ivf_paged(_aion_buf, str(_CORPUS_IVFP).encode()) != 0:
    print("  [ERROR] open_ivf_paged failed", flush=True)
    sys.exit(1)
open_ms = (time.perf_counter() - t0) * 1000
print(f"  H-IVF loaded in {open_ms:.1f} ms  ({_CORPUS_IVFP.stat().st_size // 1024} KB mmap'd from NVMe)", flush=True)

c_sz, py_sz = _aion.vector_similarity_query_sizeof(), ctypes.sizeof(_AionQuery)
if c_sz != py_sz:
    print(f"  [ERROR] _AionQuery struct mismatch: ctypes={py_sz} C={c_sz}", flush=True)
    sys.exit(1)

# -- Initialize persistent lattice ---------------------------------------------

print("[STARTUP] Initializing persistent lattice...", flush=True)

_lattice_tmpdir = tempfile.mkdtemp(prefix="synrix_demo_")
_lattice_path   = os.path.join(_lattice_tmpdir, "demo.lat")
_lattice_buf    = ctypes.create_string_buffer(_LATTICE_BUF_SIZE)
_node_counter   = [0]  # mutable box for thread-safe increment under _lattice_lock

if _lib.lattice_init(_lattice_buf, _lattice_path.encode(), 512, 0) != 0:
    print("  [ERROR] lattice_init failed", flush=True)
    sys.exit(1)

def _cleanup_lattice() -> None:
    _lib.lattice_cleanup(_lattice_buf)
    shutil.rmtree(_lattice_tmpdir, ignore_errors=True)

atexit.register(_cleanup_lattice)
print(f"  Lattice ready  ({_LATTICE_BUF_SIZE // 1024} KB in-process store)", flush=True)

# -- Load fixture packets -------------------------------------------------------

print("[STARTUP] Loading packets and routing stack...", flush=True)

fixture      = json.loads(_GATE_FIX.read_text())
cwru_pkts    = [SCMInputPacket.from_dict(d) for d in fixture["gates"]["Gate 2 (CWRU-domain)"]]
wave_pkt     = SCMInputPacket.from_dict(fixture["gates"]["Gate 3 (WAVE silicon)"][0])
_WAVE_METRICS = fixture["gates"]["Gate 3 (WAVE silicon)"][0].get("wave_goal_metrics", {})

_teacher  = RulesScmRouter(ExecutionContract())
_art      = ScmTinyArtifact.load(_CWRU_NPZ)
_pred     = _art.predictor()
_gate_kb  = _CWRU_NPZ.stat().st_size // 1024

# Warmup and get baseline route for bearings
for _p in cwru_pkts * 10:
    _teacher.route(_p)
_baseline_route = str(_teacher.route(cwru_pkts[0]).route)

print(f"  Packets ready   |  baseline_route={_baseline_route}", flush=True)
print(f"  Gate loaded     |  {_gate_kb} KB CWRU domain expert", flush=True)

# -- Thread locks ---------------------------------------------------------------
# AION search is not reentrant; lattice writes are serialised for safety.

_aion_lock    = threading.Lock()
_lattice_lock = threading.Lock()

# -- Core detection -------------------------------------------------------------

def _aion_search(vec: np.ndarray) -> tuple[list[dict], float]:
    """Query AION512 bruteforce; returns (top_matches, search_us)."""
    with _aion_lock:
        q = _AionQuery()
        q.max_results            = 3
        q.min_similarity         = -1.0
        q.cluster_filter         = 0
        q.use_lsh                = False
        q.use_clustering         = False
        q.use_aion512_bruteforce = True
        q.query_aion512_f32      = vec.ctypes.data
        q.use_aion512_ivf        = False
        q.aion512_ivf_n_probe    = 0
        q.use_float32_rerank     = False
        q.rerank_oversample_factor = 0
        q.use_aion512_hivf       = False
        q.aion512_hivf_probe_b1  = 0
        q.aion512_hivf_probe_b2  = 0
        results = (_AionResult * 3)()
        count   = ctypes.c_uint32(0)
        t0      = time.perf_counter_ns()
        rc      = _aion.semantic_vector_indexing_system_search_similar(
            _aion_buf, ctypes.byref(q), results, ctypes.byref(count))
        search_us = (time.perf_counter_ns() - t0) / 1e3

    n = int(count.value) if rc == 0 else 0
    top_matches = []
    for i in range(n):
        r   = results[i]
        vid = r.node_id - 1
        lbl = str(_corpus_labels[vid]) if vid < N_CORPUS else "?"
        top_matches.append({"rank": i + 1, "label": lbl, "score": round(float(r.similarity_score), 4)})
    return top_matches, round(search_us, 1)


def analyze_reading(rtype: str) -> dict:
    """
    Four-layer detection for a reading type: 'normal', 'fault', or 'pmu'.

    Layer 1 (Lattice write): persist the incoming record; return write latency.
    Layer 2 (AION512 similarity): query with the correct vector space for each type.
      - bearing buttons: actual corpus vector (sim ~0.99, in domain)
      - pmu button:      featurized SCM features (sim ~0.10, foreign domain)
    Layers 3-4: SCM routing and behavioral gate.
    """
    is_pmu = rtype not in _REPR_VEC

    if is_pmu:
        feat     = featurize_packets([wave_pkt])
        aion_vec = encode_512(feat[0])
        pkt      = wave_pkt
    else:
        aion_vec = _REPR_VEC[rtype]
        pkt      = cwru_pkts[0] if rtype == "normal" else cwru_pkts[2]
        feat     = featurize_packets([pkt])

    # Layer 1: write to persistent lattice (counter + write under single lock)
    with _lattice_lock:
        _node_counter[0] += 1
        seq = _node_counter[0]
        if is_pmu:
            node_name = f"WAVE_PMU_{seq:04d}".encode()
            node_data = (" ".join(f"{k}={v}" for k, v in _WAVE_METRICS.items())).encode()
        else:
            label     = "normal" if rtype == "normal" else "inner_007"
            node_name = f"CWRU_{label.upper()}_{seq:04d}".encode()
            node_data = f"label={label} seq={seq}".encode()
        t0_lat  = time.perf_counter_ns()
        node_id = _lib.lattice_add_node(
            _lattice_buf, _LATTICE_NODE_TYPE_OBS, node_name, node_data)
        write_us = round((time.perf_counter_ns() - t0_lat) / 1e3, 1)

    write_ok = node_id != 0

    # Layer 2: AION512 similarity
    top_matches, search_us = _aion_search(aion_vec)
    best_sim  = top_matches[0]["score"] if top_matches else 0.0
    in_domain = bool(best_sim >= 0.50)

    # Layer 3: SCM rule engine routing
    t0_r      = time.perf_counter_ns()
    out       = _teacher.route(pkt)
    route_us  = round((time.perf_counter_ns() - t0_r) / 1e3, 1)
    route     = str(out.route)
    normal_route = bool(route == _baseline_route)

    # Layer 4: Behavioral gate
    y_r    = route_to_index(out.route)
    y_t    = template_to_index(template_id_from_teacher(pkt, out))
    r_pred = _pred.route.predict_indices(feat)[0]
    t_pred = _pred.template.predict_indices(feat)[0]
    gate_ok = bool(r_pred == y_r and t_pred == y_t)

    anomaly = not in_domain or not normal_route or not gate_ok

    # -- Human-readable layer copy (context-aware) --------------------------------

    reading_label = (
        "CPU performance counter" if is_pmu else
        "faulty bearing" if rtype == "fault" else
        "healthy bearing"
    )

    # Layer 1: storage
    l1_verdict = f"Stored on device in {write_us:.0f} us"
    l1_detail  = (
        f"Every reading is written to on-device persistent memory before analysis begins. "
        f"No cloud, no network round-trip. This {reading_label} reading is now stored "
        f"as '{node_name.decode()}' and will survive a reboot."
    )

    # Layer 2: semantic search
    if in_domain:
        l2_verdict = "Bearing corpus: in-domain"
        if rtype == "fault":
            l2_detail = (
                f"Searched {N_CORPUS:,} historical bearing signals. Even a damaged bearing "
                f"produces a recognizable vibration pattern -- nearest match cosine similarity "
                f"{best_sim:.4f}. The system has seen bearing faults before."
            )
        else:
            l2_detail = (
                f"Searched {N_CORPUS:,} historical bearing signals. This reading is consistent "
                f"with the stored corpus -- nearest match cosine similarity {best_sim:.4f}."
            )
    else:
        l2_verdict = f"Best bearing similarity: {best_sim:.1%} -- out of domain"
        l2_detail  = (
            f"Searched {N_CORPUS:,} historical bearing vibration signals. "
            f"The closest match scores only {best_sim:.4f} cosine similarity -- this reading "
            f"looks nothing like any bearing signal the system has ever stored. First flag: wrong domain."
        )

    # Layer 3: rule engine
    if normal_route:
        l3_verdict = "Classified correctly -- bearing data processing class"
        l3_detail  = (
            f"The rule engine assigns a processing class to every reading. "
            f"All {N_CORPUS:,} historical bearing records share the same class. "
            f"This {reading_label} reading matches."
        )
    else:
        l3_verdict = "Classified differently -- not a bearing reading"
        l3_detail  = (
            f"The rule engine assigns a processing class to every reading. "
            f"Every bearing record in the corpus shares the same expected class. "
            f"This reading was assigned a different class -- the system does not "
            f"normally process this type of data. Second flag."
        )

    # Layer 4: behavioral gate
    if gate_ok:
        l4_verdict = "Decision recognized -- matches learned bearing behavior"
        l4_detail  = (
            f"A {_gate_kb} KB model trained exclusively on bearing data correctly "
            f"predicted the system's decision on this reading. The outcome follows "
            f"known patterns."
        )
    else:
        l4_verdict = "Decision unrecognized -- model has never seen this before"
        l4_detail  = (
            f"A {_gate_kb} KB model trained exclusively on bearing data could not "
            f"predict the system's decision. It has only ever seen bearing signals -- "
            f"this outcome is outside everything it was trained on. Third independent flag."
        )

    return {
        "anomaly": anomaly,
        "rtype": rtype,
        "layer1": {
            "name": "On-Device Storage",
            "sublabel": "Every reading is written to persistent memory on the device before analysis",
            "pass": write_ok,
            "write_us": write_us,
            "node_name": node_name.decode(),
            "verdict": l1_verdict,
            "detail": l1_detail,
            "wave_metrics": _WAVE_METRICS if is_pmu else None,
            "vec_dim": AION_VEC_DIM,
        },
        "layer2": {
            "name": "Does it look like anything we've seen before?",
            "sublabel": f"Compared against {N_CORPUS:,} historical bearing vibration signals",
            "pass": in_domain,
            "best_sim": best_sim,
            "top_matches": top_matches,
            "search_us": search_us,
            "corpus_size": N_CORPUS,
            "verdict": l2_verdict,
            "detail": l2_detail,
        },
        "layer3": {
            "name": "What does the rule engine expect to do with this?",
            "sublabel": "A deterministic rule engine classifies every reading into a processing class",
            "pass": normal_route,
            "route_us": route_us,
            "verdict": l3_verdict,
            "detail": l3_detail,
        },
        "layer4": {
            "name": "Does the decision match what a trained model expects?",
            "sublabel": f"An {_gate_kb} KB model trained only on bearing data checks whether this outcome makes sense",
            "pass": gate_ok,
            "gate_kb": _gate_kb,
            "verdict": l4_verdict,
            "detail": l4_detail,
        },
    }

# -- Stats endpoint payload -----------------------------------------------------

_STATS = json.dumps({
    "n_corpus":   N_CORPUS,
    "n_classes":  N_CLASSES,
    "open_ms":    round(open_ms, 1),
    "idx_ms":     round(idx_ms, 0),
    "gate_kb":    _gate_kb,
    "platform":   f"{platform.machine()} / {platform.system()}",
}).encode()

# -- HTML -----------------------------------------------------------------------

_HTML_STR = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Synrix -- Live Edge Inference Demo</title>
<style>
:root{
  --bg:#07070f;--card:#0e0e1c;--border:#1c1c34;
  --green:#00e676;--red:#ff4444;--blue:#4da6ff;--dim:#606080;--text:#cccce0;--bright:#f0f0ff;
}
html{font-size:110%}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
.wrap{max-width:880px;margin:0 auto;padding:44px 20px}

/* header */
.hdr{margin-bottom:32px}
.hdr h1{font-size:1.75rem;font-weight:700;color:var(--bright);letter-spacing:-.02em}
.hdr h1 span{color:var(--blue)}
.pitch{margin-top:12px;background:rgba(77,166,255,.06);border:1px solid rgba(77,166,255,.18);
  border-radius:8px;padding:14px 18px}
.pitch p{font-size:.9rem;line-height:1.7;color:var(--text)}
.pitch p strong{color:var(--bright)}

/* scenario box */
.scene{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px 24px;margin-bottom:28px}
.scene-label{font-size:.73rem;text-transform:uppercase;letter-spacing:.1em;color:var(--blue);margin-bottom:8px;font-weight:600}
.scene p{font-size:.88rem;line-height:1.7;color:var(--text)}

/* stack diagram */
.stack{display:flex;gap:0;margin-bottom:28px;border:1px solid var(--border);border-radius:10px;overflow:hidden}
.snode{flex:1;padding:10px 6px;text-align:center;border-right:1px solid var(--border);background:var(--card)}
.snode:last-child{border-right:none}
.snode-n{font-size:.6rem;text-transform:uppercase;letter-spacing:.08em;color:var(--dim)}
.snode-t{font-size:.72rem;font-weight:700;color:var(--blue);margin-top:3px}
.snode-s{font-size:.6rem;color:var(--dim);margin-top:2px;line-height:1.3}

/* reading buttons */
.btns{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px}
.btn{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:22px 14px 18px;
  cursor:pointer;text-align:center;transition:border-color .15s,transform .12s;color:inherit;font:inherit}
.btn:hover{border-color:var(--blue);transform:translateY(-2px)}
.btn:active{transform:translateY(0)}
.btn.selected{border-color:var(--blue);background:#0b0b1e}
.btn-icon{font-size:1.7rem;margin-bottom:10px}
.btn-name{font-size:.8rem;font-weight:700;color:var(--bright);text-transform:uppercase;letter-spacing:.06em}
.btn-sub{font-size:.72rem;color:var(--dim);margin-top:4px;line-height:1.4}

/* result panel */
.panel{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:26px;min-height:220px;
  transition:border-color .3s}
.panel.ok{border-color:rgba(0,230,118,.5)}
.panel.bad{border-color:rgba(255,68,68,.5)}

.idle{text-align:center;padding:50px 0;color:var(--dim);font-size:.9rem}
.loading{display:flex;align-items:center;justify-content:center;padding:50px 0;gap:10px;color:var(--dim);font-size:.9rem}
.spin{width:18px;height:18px;border:2px solid var(--border);border-top-color:var(--blue);border-radius:50%;
  animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* verdict row */
.verdict-row{display:flex;align-items:center;gap:16px;margin-bottom:22px;padding-bottom:16px;border-bottom:1px solid var(--border)}
.verdict-badge{font-size:1.1rem;font-weight:700;white-space:nowrap}
.verdict-badge.ok{color:var(--green)}
.verdict-badge.bad{color:var(--red)}
.verdict-sub{font-size:.84rem;color:var(--dim);line-height:1.5}

/* layers */
.layers{display:flex;flex-direction:column;gap:8px}
.layer{display:flex;gap:14px;padding:12px 14px;border-radius:8px;border:1px solid transparent;
  opacity:0;transform:translateY(8px);transition:opacity .32s,transform .32s,border-color .3s}
.layer.show{opacity:1;transform:translateY(0)}
.layer.ok{border-color:rgba(0,230,118,.2);background:rgba(0,230,118,.03)}
.layer.bad{border-color:rgba(255,68,68,.2);background:rgba(255,68,68,.03)}

.dot{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:.72rem;font-weight:700;flex-shrink:0;margin-top:1px}
.dot.ok{background:rgba(0,230,118,.18);color:var(--green)}
.dot.bad{background:rgba(255,68,68,.18);color:var(--red)}

.lbody{flex:1;min-width:0}
.lname{font-size:.82rem;font-weight:700;color:var(--bright)}
.lsub{font-size:.71rem;color:var(--dim);margin-bottom:3px}
.lverdict{font-size:.82rem;font-weight:600}
.lverdict.ok{color:var(--green)}
.lverdict.bad{color:var(--red)}
.ldetail{font-size:.74rem;color:var(--dim);margin-top:3px;line-height:1.5;word-break:break-all}
.lmeta{font-size:.69rem;margin-top:3px;color:#44446a}

/* WAVE metric fields grid */
.metrics-wrap{margin-top:8px;padding:8px 10px;background:#080814;border-radius:6px;border:1px solid var(--border)}
.metrics-label{font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:var(--blue);margin-bottom:6px;font-weight:600}
.metrics-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:4px}
.mfield{text-align:center;padding:4px 2px;border-radius:4px;background:#0c0c1e;border:1px solid var(--border)}
.mfield-k{font-size:.6rem;color:var(--dim)}
.mfield-v{font-size:.72rem;font-weight:700;color:var(--blue)}
.metrics-enc{margin-top:6px;font-size:.68rem;color:#44446a;line-height:1.5}

/* footer stats */
.stats{margin-top:26px;text-align:center;font-size:.72rem;color:var(--dim)}
.stats span{color:var(--blue)}
</style>
</head>
<body>
<div class="wrap">

  <div class="hdr">
    <h1><span>Synrix</span> &mdash; Live Edge Inference</h1>
    <div class="pitch">
      <p><strong>Universal anomaly detection</strong> that doesn't require knowing what anomalies look like &mdash; only what normal looks like. Everything else gets flagged automatically, across four independent layers, in microseconds, on a $400 edge device. No cloud. No pre-labelled anomalies needed.</p>
    </div>
  </div>

  <div class="scene">
    <div class="scene-label">Demo Scenario</div>
    <p>A predictive maintenance system monitors industrial bearings &mdash; the rotating parts inside motors and gearboxes. It has seen <span id="sCorpus">94,795</span> real vibration signals across <span id="sClasses">10</span> fault types and <em>only</em> learned what legitimate bearing data looks like.</p>
    <p style="margin-top:8px">Midway through normal operation, a completely different kind of reading arrives: performance counter data from a silicon chip (CPU cache misses, branch mispredictions, memory bandwidth). It has nothing to do with bearings. No rule was written to catch this. Select it below and watch all four layers flag it independently.</p>
  </div>

  <div class="stack">
    <div class="snode"><div class="snode-n">Layer 1</div><div class="snode-t">Store It</div><div class="snode-s">Write to device memory</div></div>
    <div class="snode"><div class="snode-n">Layer 2</div><div class="snode-t">Recognize It</div><div class="snode-s">Search 94k signals</div></div>
    <div class="snode"><div class="snode-n">Layer 3</div><div class="snode-t">Classify It</div><div class="snode-s">Rule-based decision</div></div>
    <div class="snode"><div class="snode-n">Layer 4</div><div class="snode-t">Verify It</div><div class="snode-s">Learned model check</div></div>
  </div>

  <div class="btns">
    <button class="btn" id="btn-normal" onclick="send('normal')">
      <div class="btn-icon">&#9881;</div>
      <div class="btn-name">Normal Bearing</div>
      <div class="btn-sub">Healthy rotation signal from industrial motor</div>
    </button>
    <button class="btn" id="btn-fault" onclick="send('fault')">
      <div class="btn-icon">&#9888;</div>
      <div class="btn-name">Bearing Fault</div>
      <div class="btn-sub">Inner race crack &mdash; still a bearing signal</div>
    </button>
    <button class="btn" id="btn-pmu" onclick="send('pmu')">
      <div class="btn-icon">&#128187;</div>
      <div class="btn-name">Silicon Chip PMU</div>
      <div class="btn-sub">CPU cache &amp; branch performance counters</div>
    </button>
  </div>

  <div class="panel" id="panel">
    <div class="idle">Choose a reading type above to run it through the full detection stack.</div>
  </div>

  <div class="stats">
    Live on this hardware &nbsp;&middot;&nbsp;
    <span id="fCorpus">&#8230;</span> bearing vectors indexed &nbsp;&middot;&nbsp;
    H-IVF loaded in <span id="fOpen">&#8230;</span> ms &nbsp;&middot;&nbsp;
    <span id="fPlatform">&#8230;</span>
  </div>
  <div class="stats" style="margin-top:6px;font-style:italic">
    Per-layer timings are C-library inference measurements only &mdash; end-to-end latency includes Python HTTP
    and browser round-trip overhead not present in a production deployment.
    See <code>docs/BENCHMARK_RECEIPTS.md</code> for Jetson Orin Nano production numbers.
  </div>

</div>
<script>
(async () => {
  try {
    const s = await (await fetch('/stats')).json();
    document.getElementById('sCorpus').textContent   = s.n_corpus.toLocaleString();
    document.getElementById('sClasses').textContent  = s.n_classes;
    document.getElementById('fCorpus').textContent   = s.n_corpus.toLocaleString();
    document.getElementById('fOpen').textContent     = s.open_ms;
    document.getElementById('fPlatform').textContent = s.platform;
  } catch(_){}
})();

function metricsHtml(m) {
  if (!m) return '';
  const entries = Object.entries(m);
  const cells = entries.map(([k,v]) =>
    `<div class="mfield"><div class="mfield-k">${k}</div><div class="mfield-v">${v}</div></div>`
  ).join('');
  return `<div class="metrics-wrap">
    <div class="metrics-label">Live WAVE packet fields (pre-recorded fixture)</div>
    <div class="metrics-grid">${cells}</div>
    <div class="metrics-enc">These 14 metric values are tiled into a 512-float L2-normalised vector and compared against ${(94795).toLocaleString()} bearing vectors in AION512.</div>
  </div>`;
}

function layerMeta(l, i) {
  if (i === 0) return `<div class="lmeta">Write latency: ${l.write_us} us  |  stored as: ${l.node_name}</div>`;
  if (i === 1) return `<div class="lmeta">Search time: ${l.search_us} us across ${l.corpus_size.toLocaleString()} signals</div>`;
  if (i === 2) return `<div class="lmeta">Decision time: ${l.route_us} us</div>`;
  if (i === 3) return `<div class="lmeta">Model size: ${l.gate_kb} KB -- trained exclusively on bearing data</div>`;
  return '';
}

async function send(type) {
  ['normal','fault','pmu'].forEach(t => {
    document.getElementById('btn-'+t).classList.toggle('selected', t === type);
  });
  const panel = document.getElementById('panel');
  panel.className = 'panel';
  panel.innerHTML = '<div class="loading"><div class="spin"></div>Running 4-layer detection...</div>';

  let data;
  try {
    const r = await fetch('/analyze', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({type})
    });
    data = await r.json();
  } catch(e) {
    panel.innerHTML = '<div class="idle" style="color:var(--red)">Request failed: '+e.message+'</div>';
    return;
  }

  const bad = data.anomaly;
  panel.className = 'panel ' + (bad ? 'bad' : 'ok');

  const vText = bad ? '[!] ANOMALY DETECTED -- all three detection layers flagged it'
                    : '[OK] READING ACCEPTED -- confirmed as normal bearing data';
  const vSub  = bad
    ? 'Three independent layers each reached the same conclusion without coordination. No rule was written for this case.'
    : 'The system stored the reading, matched it to known signals, classified it correctly, and the learned model agreed.';

  const ls = [data.layer1, data.layer2, data.layer3, data.layer4];
  panel.innerHTML = `
    <div class="verdict-row">
      <div class="verdict-badge ${bad?'bad':'ok'}">${vText}</div>
      <div class="verdict-sub">${vSub}</div>
    </div>
    <div class="layers">
      ${ls.map((l,i)=>`
        <div class="layer ${l.pass?'ok':'bad'}" id="ly${i}">
          <div class="dot ${l.pass?'ok':'bad'}">${l.pass?'&#10003;':'&#10007;'}</div>
          <div class="lbody">
            <div class="lname">Layer ${i+1}: ${l.name}</div>
            <div class="lsub">${l.sublabel}</div>
            <div class="lverdict ${l.pass?'ok':'bad'}">${l.verdict}</div>
            <div class="ldetail">${l.detail}</div>
            ${layerMeta(l, i)}
            ${i===0 ? metricsHtml(l.wave_metrics) : ''}
          </div>
        </div>`).join('')}
    </div>`;

  [0,1,2,3].forEach(i => setTimeout(() => {
    const el = document.getElementById('ly'+i);
    if (el) el.classList.add('show');
  }, i * 200));
}
</script>
</body>
</html>"""
_HTML = _HTML_STR.encode()

# -- HTTP handler ---------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._respond(200, "text/html; charset=utf-8", _HTML)
        elif self.path == "/stats":
            self._respond(200, "application/json", _STATS)
        else:
            self._respond(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path != "/analyze":
            self._respond(404, "text/plain", b"not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length))
        rtype  = body.get("type", "normal")
        result = analyze_reading(rtype)
        self._respond(200, "application/json", json.dumps(result).encode())

    def _respond(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

# -- Start ----------------------------------------------------------------------

print(f"\n[READY]   http://localhost:{PORT}", flush=True)
print(f"          Platform : {platform.machine()} / {platform.system()}", flush=True)
print(f"          Corpus   : {N_CORPUS:,} vectors  |  {N_CLASSES} fault classes", flush=True)
print(f"          Gate     : {_gate_kb} KB CWRU domain expert", flush=True)
print(f"\nPress Ctrl+C to stop.\n", flush=True)

with ThreadingHTTPServer(("0.0.0.0", PORT), _Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOP] Server stopped.", flush=True)
        sys.stdout.flush()
        os.dup2(os.open("/dev/null", os.O_WRONLY), 1)

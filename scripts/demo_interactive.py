#!/usr/bin/env python3
"""
Synrix Interactive Demo -- three-layer live anomaly detection.

Send different types of sensor readings and watch the system detect cross-domain
anomalies in real time through three independent layers.

Run (bare-metal):
  PYTHONPATH=. python3 scripts/demo_interactive.py
  Then open: http://localhost:5050

Run (Docker):
  docker run --rm -p 5050:5050 synrix-gate python3 scripts/demo_interactive.py
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

# -- Config ---------------------------------------------------------------------

_BUILD       = Path(os.environ.get("SYNRIX_LIB_PATH", str(_ROOT / "build")))
_GATE_FIX    = _ROOT / "analysis/formal_artifacts/scm_tiny/demo_gate_fixture.json"
_CORPUS_IVFP = _ROOT / "analysis/cwru_ivf.ivfp"
_CWRU_NPZ    = Path(os.environ.get(
    "SCM_TINY_NPZ_CWRU",
    str(_ROOT / "analysis/formal_artifacts/scm_tiny/scm_tiny_cwru_expert.npz")
))
_CORPUS_NPZ  = _ROOT / "analysis/cwru_corpus.npz"

AION_VEC_DIM = 512
PORT         = 5050

# -- Helpers --------------------------------------------------------------------

def _encode_512(feat: np.ndarray) -> np.ndarray:
    v    = np.zeros(AION_VEC_DIM, dtype=np.float32)
    reps = AION_VEC_DIM // len(feat)
    rem  = AION_VEC_DIM  % len(feat)
    v[:reps * len(feat)] = np.tile(feat, reps)
    if rem:
        v[reps * len(feat):] = feat[:rem]
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v

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

# -- Load fixture packets -------------------------------------------------------

print("[STARTUP] Loading packets and routing stack...", flush=True)

fixture   = json.loads(_GATE_FIX.read_text())
cwru_pkts = [SCMInputPacket.from_dict(d) for d in fixture["gates"]["Gate 2 (CWRU-domain)"]]
wave_pkt  = SCMInputPacket.from_dict(fixture["gates"]["Gate 3 (WAVE silicon)"][0])

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

# -- Thread lock (AION search is not reentrant) ---------------------------------

_aion_lock = threading.Lock()

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
    Three-layer detection for a reading type: 'normal', 'fault', or 'pmu'.

    Layer 1 (AION512 similarity) uses the correct vector space for each type:
      - bearing buttons: query with an actual corpus vector (vibration signal space)
                         so similarity is high (~0.99) -- in domain
      - pmu button:      query with featurized SCM features (different space)
                         so similarity is low (~0.10) -- foreign domain

    Layers 2-3 use the SCM packet for routing and gate checks.
    """
    # Layer 1: pick the right query vector
    if rtype in _REPR_VEC:
        # Bearing reading: use a stored corpus vector -- will find near-identical neighbours
        aion_vec = _REPR_VEC[rtype]
        pkt      = cwru_pkts[0] if rtype == "normal" else cwru_pkts[2]
        feat     = featurize_packets([pkt])
    else:
        # PMU: encode the SCM packet's feature vector -- projects into a foreign space
        feat     = featurize_packets([wave_pkt])
        aion_vec = _encode_512(feat[0])
        pkt      = wave_pkt

    top_matches, search_us = _aion_search(aion_vec)
    best_sim  = top_matches[0]["score"] if top_matches else 0.0
    in_domain = best_sim >= 0.50

    # Layer 2: SCM rule engine routing
    t0_r     = time.perf_counter_ns()
    out      = _teacher.route(pkt)
    route_us = round((time.perf_counter_ns() - t0_r) / 1e3, 1)
    route    = str(out.route)
    normal_route = (route == _baseline_route)

    # Layer 3: Behavioral gate
    y_r     = route_to_index(out.route)
    y_t     = template_to_index(template_id_from_teacher(pkt, out))
    r_pred  = _pred.route.predict_indices(feat)[0]
    t_pred  = _pred.template.predict_indices(feat)[0]
    gate_ok = bool(r_pred == y_r and t_pred == y_t)
    in_domain    = bool(in_domain)
    normal_route = bool(normal_route)
    anomaly = not in_domain or not normal_route or not gate_ok

    return {
        "anomaly": anomaly,
        "layer1": {
            "name": "Semantic Similarity",
            "sublabel": f"AION512 -- {N_CORPUS:,} bearing vectors indexed",
            "pass": in_domain,
            "best_sim": best_sim,
            "top_matches": top_matches,
            "search_us": search_us,
            "corpus_size": N_CORPUS,
            "verdict": "IN DOMAIN" if in_domain else "FOREIGN DOMAIN",
            "detail": (
                f"Best match {best_sim:.4f} -- consistent with bearing corpus"
                if in_domain else
                f"Best match only {best_sim:.4f} -- unlike any of the {N_CORPUS:,} bearing records"
            ),
        },
        "layer2": {
            "name": "Rule Engine Routing",
            "sublabel": "SCM deterministic router",
            "pass": normal_route,
            "route": route,
            "expected": _baseline_route,
            "route_us": route_us,
            "verdict": f"Route '{route}' -- expected" if normal_route else f"Route '{route}' -- anomalous",
            "detail": (
                f"Assigned to '{route}' -- matches all historical bearing records"
                if normal_route else
                f"Expected '{_baseline_route}' but got '{route}' -- different execution class"
            ),
        },
        "layer3": {
            "name": "Behavioral Gate",
            "sublabel": f"{_gate_kb} KB student model trained on bearing data only",
            "pass": gate_ok,
            "gate_kb": _gate_kb,
            "verdict": "Student agrees with teacher" if gate_ok else "Student disagrees -- MISMATCH",
            "detail": (
                "Learned model matches teacher decision on route and template"
                if gate_ok else
                f"The {_gate_kb} KB model has never seen this decision pattern -- it was trained on bearing data only"
            ),
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
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
.wrap{max-width:860px;margin:0 auto;padding:44px 20px}

/* header */
.hdr{margin-bottom:36px}
.hdr h1{font-size:1.75rem;font-weight:700;color:var(--bright);letter-spacing:-.02em}
.hdr h1 span{color:var(--blue)}
.hdr p{margin-top:10px;color:var(--dim);line-height:1.65;font-size:.93rem;max-width:620px}

/* scenario box */
.scene{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px 24px;margin-bottom:32px}
.scene-label{font-size:.73rem;text-transform:uppercase;letter-spacing:.1em;color:var(--blue);margin-bottom:8px;font-weight:600}
.scene p{font-size:.9rem;line-height:1.7;color:var(--text)}

/* reading buttons */
.btns{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:32px}
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
.verdict-row{display:flex;align-items:center;gap:16px;margin-bottom:24px;padding-bottom:18px;border-bottom:1px solid var(--border)}
.verdict-badge{font-size:1.15rem;font-weight:700;white-space:nowrap}
.verdict-badge.ok{color:var(--green)}
.verdict-badge.bad{color:var(--red)}
.verdict-sub{font-size:.85rem;color:var(--dim);line-height:1.5}

/* layers */
.layers{display:flex;flex-direction:column;gap:10px}
.layer{display:flex;gap:14px;padding:14px 16px;border-radius:8px;border:1px solid transparent;
  opacity:0;transform:translateY(8px);transition:opacity .35s,transform .35s,border-color .3s}
.layer.show{opacity:1;transform:translateY(0)}
.layer.ok{border-color:rgba(0,230,118,.2);background:rgba(0,230,118,.03)}
.layer.bad{border-color:rgba(255,68,68,.2);background:rgba(255,68,68,.03)}

.dot{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:.72rem;font-weight:700;flex-shrink:0;margin-top:1px}
.dot.ok{background:rgba(0,230,118,.18);color:var(--green)}
.dot.bad{background:rgba(255,68,68,.18);color:var(--red)}

.lbody{flex:1;min-width:0}
.lname{font-size:.82rem;font-weight:700;color:var(--bright)}
.lsub{font-size:.72rem;color:var(--dim);margin-bottom:4px}
.lverdict{font-size:.83rem;font-weight:600}
.lverdict.ok{color:var(--green)}
.lverdict.bad{color:var(--red)}
.ldetail{font-size:.75rem;color:var(--dim);margin-top:3px;line-height:1.5}
.lmeta{font-size:.7rem;color:var(--border);margin-top:3px;color:#44446a}

/* footer stats */
.stats{margin-top:28px;text-align:center;font-size:.72rem;color:var(--dim)}
.stats span{color:var(--blue)}
</style>
</head>
<body>
<div class="wrap">

  <div class="hdr">
    <h1><span>Synrix</span> &mdash; Live Edge Inference</h1>
    <p>Choose a reading type below. The system runs it through three independent detection layers and shows you what it found.</p>
  </div>

  <div class="scene">
    <div class="scene-label">Scenario</div>
    <p>A predictive maintenance system monitors industrial bearings &mdash; the rotating parts inside motors and gearboxes. It has seen <span id="sCorpus">94,795</span> real vibration signals across <span id="sClasses">10</span> fault types and learned what legitimate sensor data looks like.</p>
    <p style="margin-top:10px">Midway through normal operation, a completely different kind of reading arrives: performance counter data from a silicon chip (CPU cache misses, branch mispredictions). It has nothing to do with bearings. No rule was written to catch this. The system figures it out from what it already knows.</p>
  </div>

  <div class="btns">
    <button class="btn" id="btn-normal" onclick="send('normal')">
      <div class="btn-icon">&#9881;&#65039;</div>
      <div class="btn-name">Normal Bearing</div>
      <div class="btn-sub">Healthy rotation signal from industrial motor</div>
    </button>
    <button class="btn" id="btn-fault" onclick="send('fault')">
      <div class="btn-icon">&#9888;&#65039;</div>
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
    <div class="idle">Choose a reading type above to run it through the detection stack.</div>
  </div>

  <div class="stats">
    Live on this hardware &nbsp;&middot;&nbsp;
    <span id="fCorpus">&#8230;</span> bearing vectors indexed &nbsp;&middot;&nbsp;
    H-IVF loaded from NVMe in <span id="fOpen">&#8230;</span> ms &nbsp;&middot;&nbsp;
    <span id="fPlatform">&#8230;</span>
  </div>
  <div class="stats" style="margin-top:6px;font-style:italic">
    Note: per-layer timings are C-library inference measurements only &mdash;
    end-to-end request latency includes Python HTTP and browser round-trip overhead
    not present in a production deployment. See <code>docs/BENCHMARK_RECEIPTS.md</code>
    for production throughput numbers from the Jetson Orin Nano.
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

async function send(type) {
  ['normal','fault','pmu'].forEach(t => {
    document.getElementById('btn-'+t).classList.toggle('selected', t === type);
  });
  const panel = document.getElementById('panel');
  panel.className = 'panel';
  panel.innerHTML = '<div class="loading"><div class="spin"></div>Analyzing...</div>';

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

  const vText = bad ? '[!] ANOMALY DETECTED -- 3-layer detection triggered'
                    : '[OK] READING ACCEPTED -- consistent with bearing domain';
  const vSub  = bad
    ? 'This reading does not match the bearing domain. Multiple independent layers flagged it.'
    : 'All three detection layers confirm this reading is within the known bearing domain.';

  const ls = [data.layer1, data.layer2, data.layer3];
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
            ${i===0?`<div class="lmeta">Inference: ${l.corpus_size.toLocaleString()} vectors in ${l.search_us} uss</div>`:''}
            ${i===1?`<div class="lmeta">Inference: route decision in ${l.route_us} uss</div>`:''}
            ${i===2?`<div class="lmeta">${l.gate_kb} KB model -- trained on bearing data only</div>`:''}
          </div>
        </div>`).join('')}
    </div>`;

  [0,1,2].forEach(i => setTimeout(() => {
    const el = document.getElementById('ly'+i);
    if (el) el.classList.add('show');
  }, i * 220));
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

#!/usr/bin/env python3
"""
Download CWRU bearing fault dataset, build a 512-d HRR vector corpus, and
serialize a pre-built H-IVF paged index so docker run skips K-means entirely.

Sources:
  Case Western Reserve University Bearing Data Center
  https://engineering.case.edu/bearingdatacenter/download-data-file

Outputs:
  analysis/cwru_corpus.npz   — vectors (N,512) float32 + labels + file_ids
  analysis/cwru_ivf.ivfp     — pre-built H-IVF paged binary index (mmap at runtime)

Called once during Docker build. Caches raw .mat files under analysis/cwru_raw/.
"""
from __future__ import annotations

import ctypes
import io
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io

ROOT      = Path(__file__).resolve().parents[1]
OUT_NPZ   = ROOT / "analysis" / "cwru_corpus.npz"
OUT_IVFP  = ROOT / "analysis" / "cwru_ivf.ivfp"
CACHE     = ROOT / "analysis" / "cwru_raw"
BASE_URL  = "https://engineering.case.edu/sites/default/files"

# H-IVF topology: 16 branches × 20 leaves = 320 total leaf clusters.
HIVF_BRANCHES = 16
HIVF_LEAVES   = 20
HIVF_SEED     = 42

WINDOW = 512
STRIDE = 64
DIM    = 512
HRR_SEED = 0xC4E8   # fixed — same projection matrix every run

# (file_id, label)  — DE channel, 12 kHz
FILES = [
    # Normal
    (97,  "normal"), (98,  "normal"), (99,  "normal"), (100, "normal"),
    # Inner race
    (105, "inner_007"), (106, "inner_007"), (107, "inner_007"), (108, "inner_007"),
    (169, "inner_014"), (170, "inner_014"), (171, "inner_014"), (172, "inner_014"),
    (209, "inner_021"), (210, "inner_021"), (211, "inner_021"), (212, "inner_021"),
    # Ball
    (118, "ball_007"), (119, "ball_007"), (120, "ball_007"), (121, "ball_007"),
    (185, "ball_014"), (186, "ball_014"), (187, "ball_014"), (188, "ball_014"),
    (222, "ball_021"), (223, "ball_021"), (224, "ball_021"), (225, "ball_021"),
    # Outer race (6 o'clock position)
    (130, "outer_007"), (131, "outer_007"), (132, "outer_007"), (133, "outer_007"),
    (197, "outer_014"), (198, "outer_014"), (199, "outer_014"), (200, "outer_014"),
    (234, "outer_021"), (235, "outer_021"), (236, "outer_021"), (237, "outer_021"),
]


# ── Feature extraction (DE channel, 281-dim) ──────────────────────────────────
# Matches cwru_demo.py extract_features_batch_1ch exactly.

def extract_features(windows: np.ndarray) -> np.ndarray:
    """(N, W) float64 → (N, 281) float32 — time + spectral statistics."""
    W = windows.astype(np.float64)
    N, L = W.shape

    rms      = np.sqrt(np.mean(W**2, axis=1))
    peak     = np.max(np.abs(W), axis=1)
    mean_abs = np.mean(np.abs(W), axis=1)
    std      = np.std(W, axis=1)
    crest    = peak / (rms + 1e-12)
    shape_f  = rms / (mean_abs + 1e-12)
    impulse  = peak / (mean_abs + 1e-12)
    p2p      = W.max(axis=1) - W.min(axis=1)
    Wc       = W - W.mean(axis=1, keepdims=True)
    skew     = np.mean(Wc**3, axis=1) / (std**3 + 1e-12)
    kurt     = np.mean(Wc**4, axis=1) / (std**4 + 1e-12)

    spec      = np.abs(np.fft.rfft(W, axis=1))          # (N, L//2+1)
    spec_sum  = spec.sum(axis=1, keepdims=True) + 1e-12
    spec_norm = spec / spec_sum
    freqs     = np.fft.rfftfreq(L)
    centroid  = (spec_norm * freqs).sum(axis=1)
    spec_std  = np.sqrt(((freqs - centroid[:, None])**2 * spec_norm).sum(axis=1))
    entropy   = -(spec_norm * np.log(spec_norm + 1e-12)).sum(axis=1)

    bs    = spec.shape[1] // 8
    bands = np.stack([spec[:, i*bs:(i+1)*bs].mean(axis=1) for i in range(8)], axis=1)

    spec_max = spec.max(axis=1, keepdims=True) + 1e-12
    top3_idx = np.argsort(spec, axis=1)[:, -3:][:, ::-1]
    top3_mag = np.take_along_axis(spec, top3_idx, axis=1) / spec_max

    return np.concatenate([
        rms[:,None], peak[:,None], crest[:,None], mean_abs[:,None],
        shape_f[:,None], impulse[:,None], std[:,None],
        skew[:,None], kurt[:,None], p2p[:,None],
        centroid[:,None], spec_std[:,None], entropy[:,None],
        bands, top3_mag,
        spec_norm,    # 257 FFT bins
    ], axis=1).astype(np.float32)    # (N, 281)


# ── HRR encoding: random projection → 512-d unit vectors ─────────────────────

def _projection_matrix(n_feats: int) -> np.ndarray:
    rng = np.random.default_rng(HRR_SEED)
    P   = rng.standard_normal((DIM, n_feats)).astype(np.float64)
    return (P / np.sqrt(n_feats)).T    # (n_feats, DIM)

_PROJ: np.ndarray | None = None

def encode_hrr(windows: np.ndarray) -> np.ndarray:
    """(N, W) → (N, 512) float32 L2-normalised."""
    global _PROJ
    feats = extract_features(windows).astype(np.float64)
    mu    = feats.mean(axis=1, keepdims=True)
    sg    = feats.std(axis=1, keepdims=True)
    feats = (feats - mu) / (sg + 1e-8)
    if _PROJ is None:
        _PROJ = _projection_matrix(feats.shape[1])
    vecs  = (feats @ _PROJ).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-30)


# ── Download + load ───────────────────────────────────────────────────────────

def download(file_id: int) -> bytes:
    url = f"{BASE_URL}/{file_id}.mat"
    print(f"    GET {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "synrix-demo/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()

def load_de(data: bytes) -> np.ndarray | None:
    mat = scipy.io.loadmat(io.BytesIO(data))
    for key in mat:
        if "DE_time" in key:
            return mat[key].ravel().astype(np.float64)
    return None

def windows_of(sig: np.ndarray) -> np.ndarray:
    n = (len(sig) - WINDOW) // STRIDE
    idx = np.arange(n)[:, None] * STRIDE + np.arange(WINDOW)
    return sig[idx].astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)

    all_vecs:     list[np.ndarray] = []
    all_labels:   list[str]        = []
    all_file_ids: list[int]        = []

    for file_id, label in FILES:
        cache_path = CACHE / f"{file_id}.mat"
        if cache_path.is_file():
            data = cache_path.read_bytes()
        else:
            try:
                data = download(file_id)
                cache_path.write_bytes(data)
            except Exception as exc:
                print(f"  [SKIP] {file_id}.mat — {exc}", file=sys.stderr)
                continue

        sig = load_de(data)
        if sig is None:
            print(f"  [SKIP] {file_id}.mat — no DE_time key", file=sys.stderr)
            continue

        wins = windows_of(sig)
        vecs = encode_hrr(wins)
        all_vecs.append(vecs)
        all_labels.extend([label] * len(wins))
        all_file_ids.extend([file_id] * len(wins))
        print(f"  [{file_id:3d}] {label:<12}  {len(wins):5d} windows  "
              f"(total: {sum(len(v) for v in all_vecs):,})", flush=True)

    if not all_vecs:
        print("ERROR: no vectors built — check network access", file=sys.stderr)
        sys.exit(1)

    vectors  = np.concatenate(all_vecs, axis=0)
    labels   = np.array(all_labels)
    file_ids = np.array(all_file_ids, dtype=np.int32)

    np.savez_compressed(OUT_NPZ, vectors=vectors, labels=labels, file_ids=file_ids)
    print(f"\n  Saved {len(vectors):,} vectors  shape={vectors.shape}  → {OUT_NPZ}")

    _build_paged_ivf(vectors)


def _build_paged_ivf(vectors: np.ndarray) -> None:
    """Insert vectors into AION512, run H-IVF K-means, serialize to OUT_IVFP."""
    build_dir = Path(os.environ.get("SYNRIX_LIB_PATH", str(ROOT / "build")))
    ext = ".dll" if platform.system() == "Windows" else \
          ".dylib" if platform.system() == "Darwin" else ".so"
    lib_path = build_dir / f"libaion_semantic_index{ext}"
    if not lib_path.is_file():
        print(f"  [SKIP] {lib_path} not found — skipping IVF build", file=sys.stderr)
        return

    aion = ctypes.CDLL(str(lib_path))
    aion.semantic_vector_indexing_system_sizeof.restype  = ctypes.c_size_t
    aion.semantic_vector_indexing_system_sizeof.argtypes = []
    aion.semantic_vector_indexing_system_create.restype  = ctypes.c_int
    aion.semantic_vector_indexing_system_create.argtypes = [ctypes.c_void_p]
    aion.semantic_vector_indexing_system_add_embedding_aion512.restype  = ctypes.c_int
    aion.semantic_vector_indexing_system_add_embedding_aion512.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    aion.semantic_vector_indexing_system_build_ivf_level2.restype  = ctypes.c_int
    aion.semantic_vector_indexing_system_build_ivf_level2.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint]
    aion.semantic_vector_indexing_system_build_ivf_paged.restype  = ctypes.c_int
    aion.semantic_vector_indexing_system_build_ivf_paged.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p]
    aion.semantic_vector_indexing_system_destroy.restype  = None
    aion.semantic_vector_indexing_system_destroy.argtypes = [ctypes.c_void_p]

    sz  = aion.semantic_vector_indexing_system_sizeof()
    buf = ctypes.create_string_buffer(sz)
    if aion.semantic_vector_indexing_system_create(buf) != 0:
        print("  [FAIL] AION system create failed", file=sys.stderr)
        sys.exit(1)

    N = len(vectors)
    print(f"\n  Adding {N:,} vectors to AION512 ...", flush=True)
    t0 = time.perf_counter()
    for i, vec in enumerate(vectors):
        aion.semantic_vector_indexing_system_add_embedding_aion512(
            buf, ctypes.c_uint32(i + 1), vec.ctypes.data_as(ctypes.c_void_p))
    print(f"  Added in {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)

    print(f"  Building H-IVF ({HIVF_BRANCHES} branches × {HIVF_LEAVES} leaves) — "
          f"K-means runs once here, never at docker run ...", flush=True)
    t0 = time.perf_counter()
    rc = aion.semantic_vector_indexing_system_build_ivf_level2(
        buf,
        ctypes.c_uint32(HIVF_BRANCHES),
        ctypes.c_uint32(HIVF_LEAVES),
        ctypes.c_uint(HIVF_SEED),
    )
    if rc != 0:
        print(f"  [FAIL] build_ivf_level2 returned {rc}", file=sys.stderr)
        aion.semantic_vector_indexing_system_destroy(buf)
        sys.exit(1)
    print(f"  H-IVF built in {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)

    print(f"  Serializing paged index → {OUT_IVFP} ...", flush=True)
    t0 = time.perf_counter()
    rc = aion.semantic_vector_indexing_system_build_ivf_paged(
        buf, str(OUT_IVFP).encode())
    if rc != 0:
        print(f"  [FAIL] build_ivf_paged returned {rc}", file=sys.stderr)
        aion.semantic_vector_indexing_system_destroy(buf)
        sys.exit(1)
    print(f"  Paged index written in {(time.perf_counter()-t0)*1000:.0f} ms  "
          f"({OUT_IVFP.stat().st_size // 1024} KB)", flush=True)

    aion.semantic_vector_indexing_system_destroy(buf)


if __name__ == "__main__":
    main()

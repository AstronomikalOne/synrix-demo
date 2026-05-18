"""ctypes wrapper for SCM-Tiny softmax-linear training via liblattice_expert_train.so.

Uses ``lattice_expert_softmax_linear_train_fullbatch_l2`` which applies L2 weight decay
per step — matching Python baseline ``fit`` (l2=1e-4).

Build: ``./scripts/build_lattice_expert_train.sh`` → ``build/liblattice_expert_train.so``
Env override: ``SCM_TINY_TRAIN_LIB`` or ``LATTICE_EXPERT_TRAIN_LIB``
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_lib: ctypes.CDLL | None = None
_lib_loaded: bool = False  # True once we've attempted load (even if failed)


def _load_lib() -> ctypes.CDLL:
    global _lib, _lib_loaded
    if _lib_loaded:
        if _lib is None:
            raise OSError("liblattice_expert_train.so not available")
        return _lib
    _lib_loaded = True

    env = os.environ.get("SCM_TINY_TRAIN_LIB") or os.environ.get("LATTICE_EXPERT_TRAIN_LIB")
    # experiments/scm_v0_1/scm_tiny/ → repo root is parents[3]
    root = Path(__file__).resolve().parents[3]
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(root / "build" / "liblattice_expert_train.so")

    for p in candidates:
        if p.is_file():
            try:
                _lib = ctypes.CDLL(str(p))
                break
            except OSError:
                pass

    if _lib is None:
        raise OSError(
            "liblattice_expert_train.so not found; run: "
            "./scripts/build_lattice_expert_train.sh "
            "(or set SCM_TINY_TRAIN_LIB)"
        )

    fn = _lib.lattice_expert_softmax_linear_train_fullbatch_l2
    fn.restype = ctypes.c_int
    fn.argtypes = (
        ctypes.c_int64,   # N
        ctypes.c_int64,   # D
        ctypes.c_int64,   # K
        ctypes.POINTER(ctypes.c_float),   # X
        ctypes.POINTER(ctypes.c_int64),   # Y
        ctypes.c_int,     # epochs
        ctypes.c_float,   # lr
        ctypes.c_float,   # l2_coeff
        ctypes.POINTER(ctypes.c_float),   # W
        ctypes.POINTER(ctypes.c_float),   # B
    )
    return _lib


def library_available() -> bool:
    try:
        _load_lib()
        return True
    except OSError:
        return False


def train_scm_head_c(
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    steps: int,
    lr: float,
    seed: int,
    l2: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Train one softmax-linear head in C; return (W, b) float32.

    W shape: (D, K); b shape: (K,). Caller initializes with NumPy (same scale as Python baseline).
    Raises ``OSError`` if the .so is not available — caller should fall back to Python.
    """
    lib = _load_lib()
    fn = lib.lattice_expert_softmax_linear_train_fullbatch_l2

    Xf = np.ascontiguousarray(X, dtype=np.float32)
    yi = np.ascontiguousarray(y, dtype=np.int64)
    n, d = Xf.shape
    k = int(num_classes)
    if yi.shape[0] != n:
        raise ValueError("y length mismatch")

    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.05, size=(d, k)).astype(np.float32)
    b = np.zeros(k, dtype=np.float32)

    rc = fn(
        ctypes.c_int64(n),
        ctypes.c_int64(d),
        ctypes.c_int64(k),
        Xf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        yi.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int(int(steps)),
        ctypes.c_float(float(lr)),
        ctypes.c_float(float(l2)),
        W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"lattice_expert_softmax_linear_train_fullbatch_l2 returned {rc}")
    return W, b

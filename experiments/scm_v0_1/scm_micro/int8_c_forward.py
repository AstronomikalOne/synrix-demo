"""
C library forward for `Int8MicroMlpPolicy` (same weight layout as ``scm_micro_int8_forward.c``).

Set ``SCM_MICRO_INT8_LIB`` to ``libscm_micro_int8.so`` (default: ``<repo>/build/libscm_micro_int8.so``).
Build: ``cd NebulOS-Scaffolding && ./scripts/build_scm_micro_int8.sh``
"""
from __future__ import annotations

import os
from ctypes import (
    CDLL,
    POINTER,
    c_float,
    c_int,
    c_int8,
)
from pathlib import Path
from typing import Optional

import numpy as np

from .int8_policy import Int8MicroMlpPolicy
from ..scm_tiny.features import PACKET_FEATURE_DIM


def _default_lib_path() -> Path:
    root = Path(__file__).resolve().parents[3]  # NebulOS-Scaffolding
    return root / "build" / "libscm_micro_int8.so"


def load_scm_micro_int8_lib(path: Optional[os.PathLike[str] | str] = None) -> CDLL:
    p = path or os.environ.get("SCM_MICRO_INT8_LIB")
    if p is None:
        p = _default_lib_path()
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(
            f"libscm_micro_int8 not found: {p} — run ./scripts/build_scm_micro_int8.sh"
        )
    return CDLL(str(p))


def pack_int8_for_c(pol: Int8MicroMlpPolicy) -> dict[str, np.ndarray]:
    """W0, W1, Wr, Wt are (in, out); C expects transposed row-major: (out, in)."""
    d_in = int(pol.W0q.shape[0])
    h0 = int(pol.W0q.shape[1])
    h1 = int(pol.W1q.shape[1])
    return {
        "w0t": np.ascontiguousarray(pol.W0q.T, dtype=np.int8),  # (h0, d_in)
        "w1t": np.ascontiguousarray(pol.W1q.T, dtype=np.int8),  # (h1, h0)
        "wrt": np.ascontiguousarray(pol.Wrq.T, dtype=np.int8),  # (3, h1)
        "wtt": np.ascontiguousarray(pol.Wtq.T, dtype=np.int8),  # (8, h1)
    }


def _bind_forward1(lib: CDLL) -> None:
    if getattr(lib, "_scm_micro_int8_forward1_set", None):
        return
    fn1 = lib.scm_micro_int8_forward1
    fn1.argtypes = [
        POINTER(c_float),
        c_int,
        POINTER(c_int8),
        c_int,
        c_float,
        c_float,
        POINTER(c_float),
        POINTER(c_int8),
        c_int,
        c_float,
        c_float,
        POINTER(c_float),
        POINTER(c_int8),
        POINTER(c_int8),
        c_float,
        c_float,
        c_float,
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
    ]
    fn1.restype = c_int
    lib._scm_micro_int8_forward1_set = True  # type: ignore[attr-defined]


def _bind_forward_n(lib: CDLL) -> None:
    if getattr(lib, "_scm_micro_int8_forward_n_set", None):
        return
    fnn = lib.scm_micro_int8_forward_n
    fnn.argtypes = [
        c_int,
        POINTER(c_float),
        c_int,
        POINTER(c_int8),
        c_int,
        c_float,
        c_float,
        POINTER(c_float),
        POINTER(c_int8),
        c_int,
        c_float,
        c_float,
        POINTER(c_float),
        POINTER(c_int8),
        POINTER(c_int8),
        c_float,
        c_float,
        c_float,
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
    ]
    fnn.restype = c_int
    lib._scm_micro_int8_forward_n_set = True  # type: ignore[attr-defined]


def forward_logits_c_n(
    pol: Int8MicroMlpPolicy,
    X: np.ndarray,
    lib: CDLL,
    packs: Optional[dict[str, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`scm_micro_int8_forward_n` — one native call for the whole batch (N rows)."""
    if X.ndim != 2 or X.shape[1] != PACKET_FEATURE_DIM:
        raise ValueError("X must be (N, PACKET_FEATURE_DIM)")
    N = int(X.shape[0])
    p = pol
    w = packs or pack_int8_for_c(pol)
    h0, d_in = w["w0t"].shape
    h1, _h0b = w["w1t"].shape
    if d_in != PACKET_FEATURE_DIM or _h0b != h0:
        raise ValueError("bad packed weight shapes")

    _bind_forward_n(lib)
    b0 = np.ascontiguousarray(p.b0.astype(np.float32))
    b1 = np.ascontiguousarray(p.b1.astype(np.float32))
    br = np.ascontiguousarray(p.br.astype(np.float32))
    bt = np.ascontiguousarray(p.bt.astype(np.float32))
    w0t = np.ascontiguousarray(w["w0t"])
    w1t = np.ascontiguousarray(w["w1t"])
    wrt = np.ascontiguousarray(w["wrt"])
    wtt = np.ascontiguousarray(w["wtt"])
    Xc = np.ascontiguousarray(X.astype(np.float32))
    out_r = np.empty((N, 3), dtype=np.float32)
    out_t = np.empty((N, 8), dtype=np.float32)
    rc = lib.scm_micro_int8_forward_n(
        c_int(N),
        Xc.ctypes.data_as(POINTER(c_float)),
        c_int(d_in),
        w0t.ctypes.data_as(POINTER(c_int8)),
        c_int(h0),
        c_float(p.s_x),
        c_float(p.s_w0),
        b0.ctypes.data_as(POINTER(c_float)),
        w1t.ctypes.data_as(POINTER(c_int8)),
        c_int(h1),
        c_float(p.s_a0),
        c_float(p.s_w1),
        b1.ctypes.data_as(POINTER(c_float)),
        wrt.ctypes.data_as(POINTER(c_int8)),
        wtt.ctypes.data_as(POINTER(c_int8)),
        c_float(p.s_a1),
        c_float(p.s_wr),
        c_float(p.s_wt),
        br.ctypes.data_as(POINTER(c_float)),
        bt.ctypes.data_as(POINTER(c_float)),
        out_r.ctypes.data_as(POINTER(c_float)),
        out_t.ctypes.data_as(POINTER(c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"scm_micro_int8_forward_n failed: {rc}")
    return out_r, out_t


def forward_logits_c_rowloop(
    pol: Int8MicroMlpPolicy,
    X: np.ndarray,
    lib: CDLL,
    packs: Optional[dict[str, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """N × `scm_micro_int8_forward1` — for benchmarking Python/ctypes per-row overhead."""
    if X.ndim != 2 or X.shape[1] != PACKET_FEATURE_DIM:
        raise ValueError("X must be (N, PACKET_FEATURE_DIM)")
    N = int(X.shape[0])
    p = pol
    w = packs or pack_int8_for_c(pol)
    h0, d_in = w["w0t"].shape
    h1, _h0b = w["w1t"].shape
    if d_in != PACKET_FEATURE_DIM or _h0b != h0:
        raise ValueError("bad packed weight shapes")

    _bind_forward1(lib)
    b0 = np.ascontiguousarray(p.b0.astype(np.float32))
    b1 = np.ascontiguousarray(p.b1.astype(np.float32))
    br = np.ascontiguousarray(p.br.astype(np.float32))
    bt = np.ascontiguousarray(p.bt.astype(np.float32))
    w0t = np.ascontiguousarray(w["w0t"])
    w1t = np.ascontiguousarray(w["w1t"])
    wrt = np.ascontiguousarray(w["wrt"])
    wtt = np.ascontiguousarray(w["wtt"])
    Xc = np.ascontiguousarray(X.astype(np.float32))
    out_r = np.empty((N, 3), dtype=np.float32)
    out_t = np.empty((N, 8), dtype=np.float32)
    d_in_c = c_int(d_in)
    c_h0 = c_int(h0)
    c_h1 = c_int(h1)
    for r in range(N):
        xrow = Xc[r]
        orow = out_r[r]
        otrow = out_t[r]
        rc = lib.scm_micro_int8_forward1(
            xrow.ctypes.data_as(POINTER(c_float)),
            d_in_c,
            w0t.ctypes.data_as(POINTER(c_int8)),
            c_h0,
            c_float(p.s_x),
            c_float(p.s_w0),
            b0.ctypes.data_as(POINTER(c_float)),
            w1t.ctypes.data_as(POINTER(c_int8)),
            c_h1,
            c_float(p.s_a0),
            c_float(p.s_w1),
            b1.ctypes.data_as(POINTER(c_float)),
            wrt.ctypes.data_as(POINTER(c_int8)),
            wtt.ctypes.data_as(POINTER(c_int8)),
            c_float(p.s_a1),
            c_float(p.s_wr),
            c_float(p.s_wt),
            br.ctypes.data_as(POINTER(c_float)),
            bt.ctypes.data_as(POINTER(c_float)),
            orow.ctypes.data_as(POINTER(c_float)),
            otrow.ctypes.data_as(POINTER(c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"scm_micro_int8_forward1 failed: {rc}")
    return out_r, out_t


def forward_logits_c(
    pol: Int8MicroMlpPolicy,
    X: np.ndarray,
    lib: CDLL,
    packs: Optional[dict[str, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Preferred path: one ``scm_micro_int8_forward_n`` call (same as :func:`forward_logits_c_n`)."""
    return forward_logits_c_n(pol, X, lib, packs)

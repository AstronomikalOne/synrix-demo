"""
INT8 / fixed-point *inference* path for SCM-Micro: int8 weights + int32 matmul
accumulation, dequantize with per-tensor scales. Calibrated on a float32 feature
matrix (typical: distillation ``featurize_packets`` batch).

Trained weights stay in ``MicroMlpPolicy`` / ``.npz``; this is deployment-shaped
inference (numpy reference; C/NEON is a follow-on).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mlp_policy import MicroMlpPolicy
from ..scm_tiny.features import PACKET_FEATURE_DIM


def _q_sym(t: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric int8, scale s such that t ≈ s * q."""
    m = float(np.max(np.abs(t)))
    if m < 1e-20:
        return np.zeros(t.shape, dtype=np.int8), 1.0
    s = m / 127.0
    q = np.round(t / s).clip(-128, 127).astype(np.int8)
    return q, s


@dataclass
class Int8MicroMlpPolicy:
    """
    Matmul: int8 × int8 → int32, then * (s_i * s_w) + float bias.
    Scales ``s_x``, ``s_a0``, ``s_a1`` come from calibration data max abs.
    """

    W0q: np.ndarray
    s_w0: float
    W1q: np.ndarray
    s_w1: float
    Wrq: np.ndarray
    s_wr: float
    Wtq: np.ndarray
    s_wt: float
    s_x: float
    s_a0: float
    s_a1: float
    b0: np.ndarray
    b1: np.ndarray
    br: np.ndarray
    bt: np.ndarray

    def forward_logits(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """X float32 (N, PACKET_FEATURE_DIM). Returns route, template logits float32."""
        _n, d = X.shape
        if d != PACKET_FEATURE_DIM:
            raise ValueError(f"bad feature dim {d}")
        xq = np.clip(np.round(X / self.s_x), -128, 127).astype(np.int8)
        z0_i = (xq.astype(np.int32) @ self.W0q.astype(np.int32)) * (self.s_x * self.s_w0)
        z0 = z0_i.astype(np.float32) + self.b0
        a0 = np.maximum(z0, 0.0)
        a0q = np.clip(np.round(a0 / self.s_a0), -128, 127).astype(np.int8)
        z1_i = (a0q.astype(np.int32) @ self.W1q.astype(np.int32)) * (self.s_a0 * self.s_w1)
        z1 = z1_i.astype(np.float32) + self.b1
        a1 = np.maximum(z1, 0.0)
        a1q = np.clip(np.round(a1 / self.s_a1), -128, 127).astype(np.int8)
        lr = (
            a1q.astype(np.int32) @ self.Wrq.astype(np.int32)
        ) * (self.s_a1 * self.s_wr) + self.br.astype(np.float32)
        lt = (
            a1q.astype(np.int32) @ self.Wtq.astype(np.int32)
        ) * (self.s_a1 * self.s_wt) + self.bt.astype(np.float32)
        return lr.astype(np.float32), lt.astype(np.float32)

    @classmethod
    def from_float(
        cls,
        pol: MicroMlpPolicy,
        X_cal: np.ndarray,
    ) -> Int8MicroMlpPolicy:
        if X_cal.ndim != 2 or X_cal.shape[1] != PACKET_FEATURE_DIM:
            raise ValueError("X_cal must be (N, PACKET_FEATURE_DIM)")
        _lr_f, _lt_f, _z0, a0, _z1, a1 = pol.forward(X_cal)
        s_x = max(float(np.max(np.abs(X_cal))), 1e-8) / 127.0
        W0q, s_w0 = _q_sym(pol.W0)
        a0_m = max(float(np.max(a0)), 1e-8) / 127.0
        W1q, s_w1 = _q_sym(pol.W1)
        a1_m = max(float(np.max(np.abs(a1))), 1e-8) / 127.0
        Wrq, s_wr = _q_sym(pol.Wr)
        Wtq, s_wt = _q_sym(pol.Wt)
        return cls(
            W0q=W0q,
            s_w0=s_w0,
            W1q=W1q,
            s_w1=s_w1,
            Wrq=Wrq,
            s_wr=s_wr,
            Wtq=Wtq,
            s_wt=s_wt,
            s_x=s_x,
            s_a0=a0_m,
            s_a1=a1_m,
            b0=pol.b0.astype(np.float32),
            b1=pol.b1.astype(np.float32),
            br=pol.br.astype(np.float32),
            bt=pol.bt.astype(np.float32),
        )

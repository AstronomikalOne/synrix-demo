"""
SCM-Micro: shared 2-layer ReLU MLP trunk + route / template heads (numpy only).

No attention, no tokens — fixed feature dim → fixed logits. Suitable for later INT8
quantization and C/ONNX export. See docs/SYNRIX_CONTROLLER_MODEL_SPEC_V0_1.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from ..packets import RouteKind, SCMInputPacket
from ..scm_tiny.baseline import ROUTE_CLASSES, route_to_index
from ..scm_tiny.features import PACKET_FEATURE_DIM, featurize_packet, featurize_packets
from ..scm_tiny.templates import QUERY_TEMPLATE_IDS, template_to_index

_ArtifactPath = Union[str, Path]
MICRO_VERSION_MAJOR = 0
MICRO_VERSION_MINOR = 1

# Default trunk: 256 → 256 → 128 (matches spec “tiny MLP” ballpark; all dims fixed at construction).
DEFAULT_H0 = 256
DEFAULT_H1 = 128


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(np.clip(x, -50, 50), dtype=np.float64)
    return (e / (e.sum(axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float32)


@dataclass
class MicroMlpPolicy:
    """
    Shared MLP: input (B, d_in) → hidden0 → hidden1 → [route_logits | template_logits].
    Trained with joint cross-entropy (route + template).
    """

    W0: np.ndarray
    b0: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    Wr: np.ndarray
    br: np.ndarray
    Wt: np.ndarray
    bt: np.ndarray

    @property
    def h0(self) -> int:
        return int(self.W0.shape[1])

    @property
    def h1(self) -> int:
        return int(self.W1.shape[1])

    def forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return logits_r, logits_t, z0, a0, z1, a1 (float32)."""
        X = X.astype(np.float64)
        z0 = X @ self.W0.astype(np.float64) + self.b0.astype(np.float64)
        a0 = np.maximum(z0, 0.0)
        z1 = a0 @ self.W1.astype(np.float64) + self.b1.astype(np.float64)
        a1 = np.maximum(z1, 0.0)
        lr = (a1 @ self.Wr.astype(np.float64) + self.br.astype(np.float64)).astype(np.float32)
        lt = (a1 @ self.Wt.astype(np.float64) + self.bt.astype(np.float64)).astype(np.float32)
        return (
            lr,
            lt,
            z0.astype(np.float32),
            a0.astype(np.float32),
            z1.astype(np.float32),
            a1.astype(np.float32),
        )

    def fit(
        self,
        X: np.ndarray,
        y_route: np.ndarray,
        y_templ: np.ndarray,
        *,
        steps: int = 3000,
        lr: float = 0.05,
        l2: float = 1e-5,
    ) -> None:
        """
        X: (N, d_in), y_route / y_templ: (N,) int class indices.
        """
        n, d = X.shape
        if d != PACKET_FEATURE_DIM:
            raise ValueError(f"X second dim {d} != PACKET_FEATURE_DIM {PACKET_FEATURE_DIM}")
        if len(y_route) != n or len(y_templ) != n:
            raise ValueError("label shape mismatch")
        n_r, n_t = int(self.Wr.shape[1]), int(self.Wt.shape[1])
        Yr = np.zeros((n, n_r), dtype=np.float64)
        Yr[np.arange(n), y_route.astype(np.int64)] = 1.0
        Yt = np.zeros((n, n_t), dtype=np.float64)
        Yt[np.arange(n), y_templ.astype(np.int64)] = 1.0

        Xd = X.astype(np.float64)
        for _ in range(int(steps)):
            z0 = Xd @ self.W0.astype(np.float64) + self.b0.astype(np.float64)
            a0 = np.maximum(z0, 0.0)
            z1 = a0 @ self.W1.astype(np.float64) + self.b1.astype(np.float64)
            a1 = np.maximum(z1, 0.0)
            logits_r = a1 @ self.Wr.astype(np.float64) + self.br.astype(np.float64)
            logits_t = a1 @ self.Wt.astype(np.float64) + self.bt.astype(np.float64)
            pr = _softmax(logits_r.astype(np.float32)).astype(np.float64)
            pt = _softmax(logits_t.astype(np.float32)).astype(np.float64)
            dlr = (pr - Yr) / float(n)
            dlt = (pt - Yt) / float(n)
            d_a1 = dlr @ self.Wr.T + dlt @ self.Wt.T
            d_z1 = d_a1 * _relu_grad(z1)
            d_W1 = a0.T @ d_z1 + l2 * self.W1.astype(np.float64)
            d_b1 = d_z1.sum(axis=0)
            d_a0 = d_z1 @ self.W1.T
            d_z0 = d_a0 * _relu_grad(z0)
            d_W0 = Xd.T @ d_z0 + l2 * self.W0.astype(np.float64)
            d_b0 = d_z0.sum(axis=0)
            d_Wr = a1.T @ dlr + l2 * self.Wr.astype(np.float64)
            d_br = dlr.sum(axis=0)
            d_Wt = a1.T @ dlt + l2 * self.Wt.astype(np.float64)
            d_bt = dlt.sum(axis=0)

            self.W0 = (self.W0 - lr * d_W0).astype(np.float32)
            self.b0 = (self.b0 - lr * d_b0).astype(np.float32)
            self.W1 = (self.W1 - lr * d_W1).astype(np.float32)
            self.b1 = (self.b1 - lr * d_b1).astype(np.float32)
            self.Wr = (self.Wr - lr * d_Wr).astype(np.float32)
            self.br = (self.br - lr * d_br).astype(np.float32)
            self.Wt = (self.Wt - lr * d_Wt).astype(np.float32)
            self.bt = (self.bt - lr * d_bt).astype(np.float32)

    @classmethod
    def random(
        cls,
        *,
        d_in: int = PACKET_FEATURE_DIM,
        h0: int = DEFAULT_H0,
        h1: int = DEFAULT_H1,
        n_route: int = 3,
        n_templ: int = 8,
        seed: int = 0,
    ) -> MicroMlpPolicy:
        rng = np.random.default_rng(seed)
        scale0 = np.sqrt(2.0 / max(d_in, 1))
        scale1 = np.sqrt(2.0 / max(h0, 1))
        scale2 = np.sqrt(2.0 / max(h1, 1))
        W0 = rng.normal(0, scale0, (d_in, h0)).astype(np.float32)
        b0 = np.zeros(h0, dtype=np.float32)
        W1 = rng.normal(0, scale1, (h0, h1)).astype(np.float32)
        b1 = np.zeros(h1, dtype=np.float32)
        Wr = rng.normal(0, scale2, (h1, n_route)).astype(np.float32)
        br = np.zeros(n_route, dtype=np.float32)
        Wt = rng.normal(0, scale2, (h1, n_templ)).astype(np.float32)
        bt = np.zeros(n_templ, dtype=np.float32)
        return cls(W0, b0, W1, b1, Wr, br, Wt, bt)

    def save(self, path: _ArtifactPath) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            W0=self.W0,
            b0=self.b0,
            W1=self.W1,
            b1=self.b1,
            Wr=self.Wr,
            br=self.br,
            Wt=self.Wt,
            bt=self.bt,
            version_major=np.int32(MICRO_VERSION_MAJOR),
            version_minor=np.int32(MICRO_VERSION_MINOR),
            feature_dim=np.int32(PACKET_FEATURE_DIM),
        )

    @classmethod
    def load(cls, path: _ArtifactPath) -> MicroMlpPolicy:
        z = np.load(Path(path), allow_pickle=False)
        if int(z["version_major"].item()) != MICRO_VERSION_MAJOR:
            raise ValueError("unsupported micro artifact version")
        if int(z["feature_dim"].item()) != PACKET_FEATURE_DIM:
            raise ValueError("feature_dim mismatch")
        return cls(
            W0=np.asarray(z["W0"], dtype=np.float32),
            b0=np.asarray(z["b0"], dtype=np.float32),
            W1=np.asarray(z["W1"], dtype=np.float32),
            b1=np.asarray(z["b1"], dtype=np.float32),
            Wr=np.asarray(z["Wr"], dtype=np.float32),
            br=np.asarray(z["br"], dtype=np.float32),
            Wt=np.asarray(z["Wt"], dtype=np.float32),
            bt=np.asarray(z["bt"], dtype=np.float32),
        )


@dataclass(frozen=True)
class ScmMicroPredictor:
    """
    Drop-in for ``ShadowScmRouter`` / eval: same ``predict_route`` / ``predict_template_id`` as ``ScmTinyPredictor``.
    """

    policy: MicroMlpPolicy

    def predict_route(self, packet: SCMInputPacket) -> RouteKind:
        x = featurize_packet(packet)[None, :]
        lr, _lt, _z0, _a0, _z1, a1 = self.policy.forward(x)
        i = int(np.argmax(lr[0]))
        return ROUTE_CLASSES[i]

    def predict_template_id(self, packet: SCMInputPacket) -> str:
        x = featurize_packet(packet)[None, :]
        _lr, lt, _z0, _a0, _z1, _a1 = self.policy.forward(x)
        i = int(np.argmax(lt[0]))
        return QUERY_TEMPLATE_IDS[i]

    def predict_route_index_batch(self, X: np.ndarray) -> np.ndarray:
        lr, _t, _, _, _, _ = self.policy.forward(X)
        return np.argmax(lr, axis=1).astype(np.int64)

    def predict_template_index_batch(self, X: np.ndarray) -> np.ndarray:
        _r, lt, _, _, _, _ = self.policy.forward(X)
        return np.argmax(lt, axis=1).astype(np.int64)


def train_micro_policy(
    X: np.ndarray,
    y_route: np.ndarray,
    y_templ: np.ndarray,
    *,
    seed: int = 0,
    steps: int = 3000,
    lr: float = 0.05,
    h0: int = DEFAULT_H0,
    h1: int = DEFAULT_H1,
) -> ScmMicroPredictor:
    pol = MicroMlpPolicy.random(seed=seed, h0=h0, h1=h1)
    pol.fit(X, y_route, y_templ, steps=steps, lr=lr, l2=1e-5)
    return ScmMicroPredictor(pol)

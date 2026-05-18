"""Tiny softmax route classifier (numpy spec; C fast path when liblattice_expert_train.so available)."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from ..packets import SCMInputPacket, RouteKind
from .features import PACKET_FEATURE_DIM, featurize_packet, featurize_packets
from .templates import QUERY_TEMPLATE_IDS, template_to_index
from .train_c import train_scm_head_c as _train_c

ROUTE_CLASSES: Tuple[RouteKind, RouteKind, RouteKind] = ("structured", "semantic", "mixed")

_ROUTE_TO_IDX = {r: i for i, r in enumerate(ROUTE_CLASSES)}


def route_to_index(route: str) -> int:
    try:
        return _ROUTE_TO_IDX[route]  # type: ignore[index]
    except KeyError as e:
        raise ValueError(f"unknown route label: {route!r}") from e


class SoftmaxRouteClassifier:
    """Multiclass logistic regression: predict ``route`` ∈ {structured, semantic, mixed}."""

    def __init__(self, dim: int = PACKET_FEATURE_DIM, seed: int = 0):
        self._seed = seed
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.05, (dim, len(ROUTE_CLASSES))).astype(np.float32)
        self.b = np.zeros(len(ROUTE_CLASSES), dtype=np.float32)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        steps: int = 2500,
        lr: float = 0.15,
        l2: float = 1e-4,
    ) -> None:
        """X: (n, dim), y: (n,) int labels 0..2."""
        if X.ndim != 2 or X.shape[1] != self.W.shape[0]:
            raise ValueError(f"X must be (n, {self.W.shape[0]}), got {X.shape}")
        n = int(X.shape[0])
        if n == 0:
            return
        y = y.astype(np.int64, copy=False)
        if y.shape != (n,):
            raise ValueError("y must be (n,)")

        try:
            W, b = _train_c(
                X,
                y,
                num_classes=len(ROUTE_CLASSES),
                steps=int(steps),
                lr=float(lr),
                seed=self._seed,
                l2=float(l2),
            )
            self.W = W
            self.b = b
            return
        except OSError:
            pass

        Y = np.zeros((n, len(ROUTE_CLASSES)), dtype=np.float32)
        Y[np.arange(n), y] = 1.0

        for _ in range(int(steps)):
            logits = X @ self.W + self.b
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits, dtype=np.float64).astype(np.float32)
            prob = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)
            grad = (prob - Y) / float(n)
            self.W = (self.W - lr * (X.T @ grad) - lr * l2 * self.W).astype(np.float32)
            self.b = (self.b - lr * grad.sum(axis=0)).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W + self.b
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float64).astype(np.float32)
        return exp / (exp.sum(axis=1, keepdims=True) + 1e-12)

    def predict_indices(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def predict_route(self, packet: SCMInputPacket) -> RouteKind:
        x = featurize_packet(packet)[None, :]
        idx = int(self.predict_indices(x)[0])
        return ROUTE_CLASSES[idx]

    def score_accuracy(self, packets: Iterable[SCMInputPacket], routes: Iterable[str]) -> float:
        X = featurize_packets(packets)
        y = np.array([route_to_index(r) for r in routes], dtype=np.int64)
        if X.shape[0] == 0:
            return 1.0
        pred = self.predict_indices(X)
        return float((pred == y).mean())


class SoftmaxTemplateClassifier:
    """Predict discrete query-plan template (``QUERY_TEMPLATE_IDS``) from packet features."""

    def __init__(self, dim: int = PACKET_FEATURE_DIM, seed: int = 0):
        self._seed = seed
        rng = np.random.default_rng(seed)
        n = len(QUERY_TEMPLATE_IDS)
        self.W = rng.normal(0.0, 0.05, (dim, n)).astype(np.float32)
        self.b = np.zeros(n, dtype=np.float32)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        steps: int = 3500,
        lr: float = 0.18,
        l2: float = 1e-4,
        governor_mask: np.ndarray | None = None,
        governor_template_idx: int | None = None,
        governor_lambda: float = 0.0,
    ) -> None:
        if X.ndim != 2 or X.shape[1] != self.W.shape[0]:
            raise ValueError(f"X must be (n, {self.W.shape[0]}), got {X.shape}")
        n = int(X.shape[0])
        if n == 0:
            return
        y = y.astype(np.int64, copy=False)
        if y.shape != (n,):
            raise ValueError("y must be (n,)")
        nc = len(QUERY_TEMPLATE_IDS)
        use_gov = (
            governor_lambda > 0.0
            and governor_mask is not None
            and governor_template_idx is not None
        )
        if use_gov:
            gm = np.asarray(governor_mask, dtype=np.float32).reshape(n)
            if gm.shape != (n,):
                raise ValueError(
                    f"governor_mask must be (n,), got {gm.shape} for n={n}"
                )
            k_gov = int(governor_template_idx)
            if k_gov < 0 or k_gov >= nc:
                raise ValueError(
                    f"governor_template_idx must be in [0,{nc}), got {k_gov}"
                )

        # Governor penalty has no C equivalent — use Python loop only when active.
        if not use_gov:
            try:
                W, b = _train_c(
                    X,
                    y,
                    num_classes=nc,
                    steps=int(steps),
                    lr=float(lr),
                    seed=self._seed,
                    l2=float(l2),
                )
                self.W = W
                self.b = b
                return
            except OSError:
                pass

        Y = np.zeros((n, nc), dtype=np.float32)
        Y[np.arange(n), y] = 1.0
        for _ in range(int(steps)):
            logits = X @ self.W + self.b
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits, dtype=np.float64).astype(np.float32)
            prob = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)
            grad = (prob - Y) / float(n)
            if use_gov:
                m = gm.reshape(n, 1)
                one_hot = np.zeros((n, nc), dtype=np.float32)
                one_hot[:, k_gov] = 1.0
                pk = prob[:, k_gov : k_gov + 1]
                # Penalty L += (lambda/n) * sum_i m_i * p_{ik}; analytic grad adds to CE grad.
                grad = grad + (
                    (float(governor_lambda) / float(n))
                    * m
                    * prob
                    * (one_hot - pk)
                ).astype(np.float32)
            self.W = (self.W - lr * (X.T @ grad) - lr * l2 * self.W).astype(np.float32)
            self.b = (self.b - lr * grad.sum(axis=0)).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W + self.b
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float64).astype(np.float32)
        return exp / (exp.sum(axis=1, keepdims=True) + 1e-12)

    def predict_indices(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def predict_template_id(self, packet: SCMInputPacket) -> str:
        x = featurize_packet(packet)[None, :]
        i = int(self.predict_indices(x)[0])
        return QUERY_TEMPLATE_IDS[i]

    def score_accuracy(
        self, packets: Iterable[SCMInputPacket], template_ids: Iterable[str]
    ) -> float:
        X = featurize_packets(packets)
        y = np.array([template_to_index(t) for t in template_ids], dtype=np.int64)
        if X.shape[0] == 0:
            return 1.0
        pred = self.predict_indices(X)
        return float((pred == y).mean())

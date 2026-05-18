"""Save/load SCM-Tiny numpy classifiers to a single compressed artifact (training-ready packaging)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from ..packets import RouteKind, SCMInputPacket
from .baseline import (
    SoftmaxRouteClassifier,
    SoftmaxTemplateClassifier,
)
from .features import PACKET_FEATURE_DIM, featurize_packet
from .templates import QUERY_TEMPLATE_IDS

_ArtifactPath = Union[str, Path]

# Increment minor when the npz key layout is backward compatible; major for breaking change.
ARTIFACT_VERSION_MAJOR = 0
# 2: PACKET_FEATURE_DIM 270 (14 WAVE metric slots); v1 artifacts were 256-dim — retrain required.
ARTIFACT_VERSION_MINOR = 2


@dataclass(frozen=True)
class ScmTinyPredictor:
    """
    Thin wrapper for inference: route + query-template id from ``SCMInputPacket`` features.
    Weights are numpy softmax classifiers; see ``ScmTinyArtifact`` for I/O.
    """

    route: SoftmaxRouteClassifier
    template: SoftmaxTemplateClassifier

    def predict_route(self, packet: SCMInputPacket) -> RouteKind:
        return self.route.predict_route(packet)

    def predict_template_id(self, packet: SCMInputPacket) -> str:
        return self.template.predict_template_id(packet)

    def predict_template_proba(self, packet: SCMInputPacket) -> np.ndarray:
        """Template softmax probabilities, shape ``(len(QUERY_TEMPLATE_IDS),)``."""
        x = featurize_packet(packet)[None, :]
        return np.asarray(self.template.predict_proba(x)[0], dtype=np.float64)

    def predict_template_topk(self, packet: SCMInputPacket, k: int = 5) -> List[Tuple[str, float]]:
        """Top-``k`` ``(template_id, probability)`` pairs, descending by probability."""
        p = self.predict_template_proba(packet)
        k_eff = max(1, min(int(k), len(QUERY_TEMPLATE_IDS)))
        idx = np.argsort(-p, kind="stable")[:k_eff]
        return [(QUERY_TEMPLATE_IDS[int(i)], float(p[int(i)])) for i in idx]

    def agrees_with_teacher(
        self,
        packet: SCMInputPacket,
        teacher_route: str,
        teacher_template: str,
    ) -> bool:
        return self.predict_route(packet) == teacher_route and self.predict_template_id(
            packet
        ) == teacher_template


@dataclass(frozen=True)
class ScmTinyArtifact:
    """
    On-disk: ``.npz`` with route/template weights and version scalars.
    No pickle — portable across Python versions and arch.
    """

    version_major: int
    version_minor: int
    feature_dim: int
    route: SoftmaxRouteClassifier
    template: SoftmaxTemplateClassifier

    def predictor(self) -> ScmTinyPredictor:
        return ScmTinyPredictor(route=self.route, template=self.template)

    @classmethod
    def from_classifiers(
        cls,
        route: SoftmaxRouteClassifier,
        template: SoftmaxTemplateClassifier,
    ) -> ScmTinyArtifact:
        if route.W.shape[0] != PACKET_FEATURE_DIM or template.W.shape[0] != PACKET_FEATURE_DIM:
            raise ValueError(
                f"expected feature dim {PACKET_FEATURE_DIM}, got "
                f"route {route.W.shape[0]} template {template.W.shape[0]}"
            )
        return cls(
            version_major=ARTIFACT_VERSION_MAJOR,
            version_minor=ARTIFACT_VERSION_MINOR,
            feature_dim=PACKET_FEATURE_DIM,
            route=route,
            template=template,
        )

    def save(self, path: _ArtifactPath) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            route_W=self.route.W,
            route_b=self.route.b,
            template_W=self.template.W,
            template_b=self.template.b,
            version_major=np.int32(self.version_major),
            version_minor=np.int32(self.version_minor),
            feature_dim=np.int32(self.feature_dim),
        )

    @classmethod
    def load(cls, path: _ArtifactPath) -> ScmTinyArtifact:
        p = Path(path)
        z = np.load(p, allow_pickle=False)
        major = int(z["version_major"].item())
        minor = int(z["version_minor"].item())
        fd = int(z["feature_dim"].item())
        if fd != PACKET_FEATURE_DIM:
            raise ValueError(f"artifact feature_dim {fd} != {PACKET_FEATURE_DIM}")
        if major != ARTIFACT_VERSION_MAJOR:
            raise ValueError(
                f"unsupported artifact version {major}.{minor} (expected {ARTIFACT_VERSION_MAJOR}.x)"
            )

        rw, rb = z["route_W"], z["route_b"]
        tw, tb = z["template_W"], z["template_b"]

        route = SoftmaxRouteClassifier.__new__(SoftmaxRouteClassifier)  # type: ignore[call-arg]
        route.W = np.asarray(rw, dtype=np.float32)
        route.b = np.asarray(rb, dtype=np.float32)

        template = SoftmaxTemplateClassifier.__new__(SoftmaxTemplateClassifier)  # type: ignore[call-arg]
        template.W = np.asarray(tw, dtype=np.float32)
        template.b = np.asarray(tb, dtype=np.float32)

        art = cls(
            version_major=major,
            version_minor=minor,
            feature_dim=fd,
            route=route,
            template=template,
        )
        return art


def train_classifiers(
    X: np.ndarray,
    y_route: np.ndarray,
    y_template: np.ndarray,
    *,
    route_seed: int = 7,
    template_seed: int = 11,
    route_steps: int = 4000,
    template_steps: int = 5000,
    route_lr: float = 0.2,
    template_lr: float = 0.22,
    template_governor_mask: np.ndarray | None = None,
    template_governor_idx: int | None = None,
    template_governor_lambda: float = 0.0,
) -> ScmTinyArtifact:
    """
    Train route + template softmax heads on aligned label arrays
    (same row order as ``X``).

    Optional **template governor** (training-only): when ``template_governor_lambda > 0``,
    pass a boolean/float mask aligned with rows of ``X`` and the template index to penalize
    (see :meth:`SoftmaxTemplateClassifier.fit`).
    """
    rr = SoftmaxRouteClassifier(seed=route_seed)
    rr.fit(X, y_route, steps=route_steps, lr=route_lr)
    tt = SoftmaxTemplateClassifier(seed=template_seed)
    tt.fit(
        X,
        y_template,
        steps=template_steps,
        lr=template_lr,
        governor_mask=template_governor_mask,
        governor_template_idx=template_governor_idx,
        governor_lambda=template_governor_lambda,
    )
    return ScmTinyArtifact.from_classifiers(rr, tt)

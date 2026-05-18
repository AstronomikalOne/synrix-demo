"""
Optional multi-NPZ SCM-Tiny routing (Expert Library).

Enable with ``SCM_TINY_EXPERT_DISPATCH=1``. Deterministic rules (no learned router):

* **WAVE** — ``packet.wave_goal_metrics`` is a non-empty dict (14-float tail present).
* **CWRU / mechanical** — ``goal`` is ``ingest_machine_state`` (``canonicalize`` chat_task default).
* **UNSW / network** — ``goal`` is ``ingest_system_log``.
* **Fallback** — ``SCM_TINY_NPZ`` when a specialist path is unset or missing on disk.

Specialist env (optional paths; must exist as files to be used):

* ``SCM_TINY_NPZ_WAVE``, ``SCM_TINY_NPZ_CWRU``, ``SCM_TINY_NPZ_UNSW``

Predictors are cached by resolved path string.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from ..packets import SCMInputPacket
from .artifact import ScmTinyArtifact, ScmTinyPredictor

_CACHE: Dict[str, ScmTinyPredictor] = {}


def expert_dispatch_enabled() -> bool:
    v = (os.environ.get("SCM_TINY_EXPERT_DISPATCH") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _exists(npz: str) -> bool:
    return bool(npz) and Path(npz).is_file()


def classify_expert_kind(packet: SCMInputPacket) -> str:
    """Return ``wave`` | ``cwru`` | ``unsw`` | ``default``."""
    wgm = packet.wave_goal_metrics
    if isinstance(wgm, dict) and len(wgm) > 0:
        return "wave"
    g = (packet.goal or "").strip().lower()
    if g == "ingest_machine_state":
        return "cwru"
    if g == "ingest_system_log":
        return "unsw"
    return "default"


def resolve_expert_npz_path(packet: SCMInputPacket) -> Optional[str]:
    """Pick ``.npz`` path; fall back to ``SCM_TINY_NPZ``."""
    kind = classify_expert_kind(packet)
    fallback = (os.environ.get("SCM_TINY_NPZ") or "").strip()
    if kind == "wave":
        p = (os.environ.get("SCM_TINY_NPZ_WAVE") or "").strip()
        return p if _exists(p) else (fallback if _exists(fallback) else None)
    if kind == "cwru":
        p = (os.environ.get("SCM_TINY_NPZ_CWRU") or "").strip()
        return p if _exists(p) else (fallback if _exists(fallback) else None)
    if kind == "unsw":
        p = (os.environ.get("SCM_TINY_NPZ_UNSW") or "").strip()
        return p if _exists(p) else (fallback if _exists(fallback) else None)
    return fallback if _exists(fallback) else None


def predictor_for_packet(packet: SCMInputPacket) -> Optional[ScmTinyPredictor]:
    """Load/cache Tiny predictor for this packet (dispatch mode)."""
    if not expert_dispatch_enabled():
        return None
    opt = os.environ.get("SCM_TINY_SHADOW", "1").strip().lower()
    if opt in ("0", "false", "no", "off"):
        return None
    path = resolve_expert_npz_path(packet)
    if not path:
        return None
    if path not in _CACHE:
        _CACHE[path] = ScmTinyArtifact.load(Path(path)).predictor()
    return _CACHE[path]


def clear_predictor_cache() -> None:
    """Test hook."""
    _CACHE.clear()

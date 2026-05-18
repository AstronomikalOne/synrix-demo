"""Fixed-size numpy features for SCMInputPacket (hashing trick + light structure + WAVE metrics)."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

import numpy as np

from ..packets import SCMInputPacket

# 240 token hash bins + 16 manual slots (see _manual_slice) + 14 WAVE floats (ingest order).
_HASH_SLOTS = 240
_MANUAL_SLOTS = 16
# Same key order as python-sdk/synrix/wave_metrics_aion512.METRIC_KEYS (tile14 / lattice ingest).
_WAVE_METRIC_KEYS: tuple[str, ...] = (
    "iso",
    "dep",
    "indep",
    "tput",
    "load",
    "cross",
    "branch",
    "cache",
    "tlb",
    "decode",
    "mispred",
    "ltr",
    "cs",
    "mp",
)
_WAVE_METRIC_SLOTS = len(_WAVE_METRIC_KEYS)
PACKET_FEATURE_DIM = _HASH_SLOTS + _MANUAL_SLOTS + _WAVE_METRIC_SLOTS

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def _tokens(*parts: str) -> List[str]:
    out: List[str] = []
    for p in parts:
        out.extend(_TOKEN_RE.findall((p or "").lower()))
    return out


def _hash_bucket(token: str, modulo: int) -> int:
    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % modulo


def _wave_metric_slice(packet: SCMInputPacket) -> np.ndarray:
    """L2-normalized WAVE profile (14,) or zeros when absent / degenerate."""
    wgm = packet.wave_goal_metrics
    out = np.zeros(_WAVE_METRIC_SLOTS, dtype=np.float32)
    if not wgm:
        return out
    for i, k in enumerate(_WAVE_METRIC_KEYS):
        try:
            out[i] = float(wgm.get(k, 0.0))
        except (TypeError, ValueError):
            out[i] = 0.0
    n = float(np.linalg.norm(out))
    if n < 1e-12:
        return np.zeros(_WAVE_METRIC_SLOTS, dtype=np.float32)
    out /= np.float32(n)
    return out


def _event_levels(packet: SCMInputPacket) -> tuple[int, int, int]:
    err = warn = other = 0
    for ev in packet.recent_events:
        if not isinstance(ev, dict):
            other += 1
            continue
        lvl = str(ev.get("level", "")).upper()
        if lvl in ("ERROR", "CRITICAL", "FATAL"):
            err += 1
        elif lvl == "WARN":
            warn += 1
        else:
            other += 1
    return err, warn, other


def featurize_packet(packet: SCMInputPacket) -> np.ndarray:
    """Return float32 vector of shape ``(PACKET_FEATURE_DIM,)``."""
    vec = np.zeros(PACKET_FEATURE_DIM, dtype=np.float32)
    toks = _tokens(packet.goal, packet.query)
    for t in toks:
        vec[_hash_bucket(t, _HASH_SLOTS)] += 1.0

    err, warn, _other = _event_levels(packet)
    base = _HASH_SLOTS
    vec[base + 0] = min(1.0, err / 5.0)
    vec[base + 1] = min(1.0, warn / 5.0)
    vec[base + 2] = 1.0 if packet.recent_events else 0.0
    vec[base + 3] = min(1.0, len(packet.available_memory_types) / 10.0)
    vec[base + 4] = min(1.0, len(packet.active_state) / 10.0)
    # L2 normalize hash slice for scale stability
    hnorm = float(np.linalg.norm(vec[:_HASH_SLOTS])) + 1e-6
    vec[:_HASH_SLOTS] /= hnorm
    base_w = _HASH_SLOTS + _MANUAL_SLOTS
    vec[base_w : base_w + _WAVE_METRIC_SLOTS] = _wave_metric_slice(packet)
    return vec


def featurize_packets(packets: Iterable[SCMInputPacket]) -> np.ndarray:
    rows = [featurize_packet(p) for p in packets]
    if not rows:
        return np.zeros((0, PACKET_FEATURE_DIM), dtype=np.float32)
    return np.stack(rows, axis=0)

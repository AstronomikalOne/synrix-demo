"""Discrete query-plan templates (tokenless enum) derived from the rules teacher."""

from __future__ import annotations

from typing import Tuple

from ..packets import SCMInputPacket, SCMRouterOutput

# Ordered for stable classifier indices (see ``template_to_index``).
QUERY_TEMPLATE_IDS: Tuple[str, ...] = (
    "T_FAILURE_MIXED",
    "T_SEMANTIC_ONLY",
    "T_SEMANTIC_WITH_PREFIX",
    "T_CONSTRAINT",
    "T_HISTORY",
    "T_LOG_INGEST",
    "T_DEFAULT_PREFIX",
)

_TEMPLATE_TO_IDX = {t: i for i, t in enumerate(QUERY_TEMPLATE_IDS)}


def template_to_index(tid: str) -> int:
    try:
        return _TEMPLATE_TO_IDX[tid]
    except KeyError as e:
        raise ValueError(f"unknown template id: {tid!r}") from e


def template_id_from_teacher(packet: SCMInputPacket, out: SCMRouterOutput) -> str:
    """Map teacher ``(packet, router_output)`` to a single template label (hybrid symbolic C1)."""
    goal_l = (packet.goal or "").lower()
    keys = [q.key or "" for q in out.queries]
    types = [q.type for q in out.queries]

    if out.route == "mixed":
        return "T_FAILURE_MIXED"

    if out.route == "semantic":
        n_prefix = sum(1 for t in types if t == "prefix")
        if n_prefix == 0:
            return "T_SEMANTIC_ONLY"
        return "T_SEMANTIC_WITH_PREFIX"

    # structured
    if any((k.startswith("CONSTRAINT:") or k.startswith("CONSTRAINT")) for k in keys):
        return "T_CONSTRAINT"

    if "ingest_system_log" in goal_l or "ingest_log" in goal_l:
        if any((k.startswith("EVENT:") or k.startswith("EVENT")) for k in keys):
            return "T_LOG_INGEST"

    if any(k.startswith("EVENT:") or k.startswith("EVENT") for k in keys):
        return "T_HISTORY"

    return "T_DEFAULT_PREFIX"

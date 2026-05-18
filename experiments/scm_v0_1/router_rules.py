"""Rules-first SCM-Router-0.1 — no ML; bounded query plans only."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .contracts import ExecutionContract
from .packets import (
    MemoryQuerySpec,
    SCMInputPacket,
    SCMRouterOutput,
    WritebackSpec,
)


_FAILURE_RE = re.compile(
    r"\b(fail|failure|fault|error|crash|alarm|e-stop|estop|timeout|abort)\b",
    re.I,
)
_SEMANTIC_RE = re.compile(
    r"\b(similar|like|pattern|anomaly|resembl|closest|nearest)\b",
    re.I,
)
_CONSTRAINT_RE = re.compile(
    r"\b(constraints?|rules?|limit|must not|forbidden|safety|interlocks?)\b",
    re.I,
)
_HISTORY_RE = re.compile(
    r"\b(history|prior|last time|previous|what happened)\b",
    re.I,
)
# CNC / 3D-print / shop-floor vocabulary — boosts mixed or semantic routing with failures
_MACHINERY_RE = re.compile(
    r"\b(spindle|g-?code|gcode|filament|layer|nozzle|bed|toolpath|printer|cnc|mill|"
    r"lathe|warp|clog|jam|runout|overheat|collision|backlash|extruder|stepper|"
    r"chatter|deflection|workpiece|stock|vise|chuck)\b",
    re.I,
)
# ``available_memory_types`` may use ``WAVE_0x``, ``wave_0x``, or ``WAVE_0x:`` — lattice node names
# are ``WAVE_0x`` + hex with no colon after ``0x``, so this token must not become ``WAVE_0x:``.
_WAVE_LATTICE_FAMILY_RE = re.compile(r"(?i)wave_0x")


def _context_signals_failure(packet: SCMInputPacket) -> bool:
    """True if recent log events or sensor readings imply fault (without relying on goal text)."""
    for ev in packet.recent_events[-5:]:
        if not isinstance(ev, dict):
            continue
        lvl = str(ev.get("level", "")).upper()
        if lvl in ("ERROR", "CRITICAL", "FATAL", "WARN"):
            if lvl in ("ERROR", "CRITICAL", "FATAL"):
                return True
        msg = str(ev.get("message", ev.get("msg", "")))
        if _FAILURE_RE.search(msg):
            return True
    for st in packet.active_state[-5:]:
        if not isinstance(st, dict):
            continue
        if str(st.get("kind", "")).lower() != "sensor":
            continue
        sev = str(st.get("severity", "")).lower()
        if sev in ("alarm", "critical", "error", "fault"):
            return True
        if _FAILURE_RE.search(str(st.get("metric", "")) + " " + str(st.get("value", ""))):
            return True
    return False


def _is_wave_goal_metrics(m: Optional[Dict[str, Any]]) -> bool:
    if not m or not isinstance(m, dict):
        return False
    return len(m) >= 1


def _dedupe_queries(queries: List[MemoryQuerySpec]) -> List[MemoryQuerySpec]:
    seen: set = set()
    out: List[MemoryQuerySpec] = []
    for q in queries:
        sig = (q.type, q.key, q.limit, q.k, q.vector_ref)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(q)
    return out


def _pick_prefix_from_types(types: List[str], default: str) -> str:
    """Lattice WAVE rows are named ``WAVE_0x`` + hex (no colon); TASK-like keys use ``NAME:``."""
    for t in types:
        s = str(t).strip()
        if not s:
            continue
        sl = s.rstrip(":").strip()
        if _WAVE_LATTICE_FAMILY_RE.fullmatch(sl):
            return "WAVE_0x"
        if not s.endswith(":"):
            return f"{s}:"
        return s
    return default


class RulesScmRouter:
    def __init__(self, contract: ExecutionContract | None = None):
        self.contract = contract or ExecutionContract()

    def route(self, packet: SCMInputPacket) -> SCMRouterOutput:
        text = packet.canonical_text()
        lim = self.contract.clamp_prefix_limit(self.contract.max_prefix_limit)
        max_q = self.contract.max_queries_per_turn
        k = self.contract.clamp_k(8)

        queries: List[MemoryQuerySpec] = []
        reasoning: List[str] = []
        route: str = "structured"
        intent = "retrieve_context"
        writeback: List[WritebackSpec] = []

        base_prefix = _pick_prefix_from_types(
            packet.available_memory_types, "TASK:"
        )

        # WAVE / Level 4.5: packet carries opcode-relevant metrics → similarity on WAVE_0x corpus.
        if _is_wave_goal_metrics(packet.wave_goal_metrics):
            route = "semantic"
            reasoning.append(
                "wave_goal_metrics present: similarity on WAVE opcode corpus (vector_ref=wave_goal_metrics)"
            )
            k_wave = self.contract.clamp_k(self.contract.max_similarity_k)
            queries.append(
                MemoryQuerySpec(
                    type="similarity",
                    vector_ref="wave_goal_metrics",
                    k=k_wave,
                )
            )
            if max_q > 1:
                queries.append(
                    MemoryQuerySpec(
                        type="prefix",
                        key="WAVE_0x",
                        limit=min(lim, self.contract.clamp_prefix_limit(100)),
                    )
                )
            queries = _dedupe_queries(queries)
            if len(queries) > max_q:
                queries = queries[:max_q]
            writeback.append(
                WritebackSpec(
                    op="WRITE_EVENT",
                    args={
                        "kind": "scm_router_v0_1",
                        "route": route,
                        "intent": intent,
                        "query_count": len(queries),
                        "wave_goal_metrics": True,
                    },
                )
            )
            return SCMRouterOutput(
                route=route,  # type: ignore[arg-type]
                queries=queries,
                writeback=writeback,
                intent=intent,
                reasoning_steps=reasoning,
                confidence=0.88,
                fallback=None,
            )

        ctx_failure = _context_signals_failure(packet)
        text_failure = bool(_FAILURE_RE.search(text))
        machinery = bool(_MACHINERY_RE.search(text))

        if ctx_failure and not text_failure:
            reasoning.append("context signal: ERROR log or alarm-class sensor — treat as failure class")

        # Failure / fault class (natural language or structured ERROR / alarm sensor context)
        if text_failure or ctx_failure:
            route = "mixed"
            reasoning.append("failure-related: structured failure keys + semantic hook")
            queries.append(
                MemoryQuerySpec(
                    type="prefix",
                    key="TASK:FAILURE:",
                    limit=lim,
                )
            )
            queries.append(
                MemoryQuerySpec(
                    type="prefix",
                    key=f"{base_prefix}FAILURE:",
                    limit=lim,
                )
            )
            queries.append(
                MemoryQuerySpec(
                    type="similarity",
                    vector_ref="HRR(goal+query)",
                    k=k,
                )
            )
        elif machinery and _SEMANTIC_RE.search(text):
            route = "semantic"
            reasoning.append("machinery domain + semantic cue: similarity-first")
            queries.append(
                MemoryQuerySpec(
                    type="similarity",
                    vector_ref="HRR(goal+query)",
                    k=k,
                )
            )
            if max_q > 1:
                queries.append(
                    MemoryQuerySpec(type="prefix", key=base_prefix, limit=min(50, lim))
                )
        elif _SEMANTIC_RE.search(text):
            route = "semantic"
            reasoning.append("semantic cue: similarity-first")
            queries.append(
                MemoryQuerySpec(
                    type="similarity",
                    vector_ref="HRR(goal+query)",
                    k=k,
                )
            )
            if max_q > 1:
                queries.append(
                    MemoryQuerySpec(type="prefix", key=base_prefix, limit=min(50, lim))
                )
        elif _CONSTRAINT_RE.search(text):
            route = "structured"
            reasoning.append("constraint cue: prefix CONSTRAINT and active rules")
            queries.append(
                MemoryQuerySpec(type="prefix", key="CONSTRAINT:", limit=lim)
            )
            if max_q > 1:
                queries.append(
                    MemoryQuerySpec(type="prefix", key=base_prefix, limit=lim)
                )
        elif _HISTORY_RE.search(text):
            route = "structured"
            reasoning.append("history cue: recent events prefix")
            queries.append(
                MemoryQuerySpec(type="prefix", key="EVENT:", limit=lim)
            )
            if max_q > 1:
                queries.append(
                    MemoryQuerySpec(type="prefix", key=base_prefix, limit=lim)
                )
        else:
            reasoning.append("default: bounded prefix on primary memory type")
            queries.append(
                MemoryQuerySpec(type="prefix", key=base_prefix, limit=lim)
            )
            g = (packet.goal or "").lower()
            if (
                ("ingest_log" in g or "ingest_system_log" in g)
                and packet.recent_events
                and max_q > len(queries)
            ):
                reasoning.append("system log ingress: add EVENT prefix for timeline correlation")
                queries.append(
                    MemoryQuerySpec(type="prefix", key="EVENT:", limit=min(lim, 50))
                )
            if (
                ("ingest_sensor" in g or "ingest_machine_state" in g)
                and max_q > len(queries)
            ):
                reasoning.append("machine state ingress: add primary prefix for last-known-good context")
                queries.append(
                    MemoryQuerySpec(type="prefix", key=base_prefix, limit=min(50, lim))
                )

        queries = _dedupe_queries(queries)
        # Enforce max_queries_per_turn and strip extras
        if len(queries) > max_q:
            queries = queries[:max_q]

        if not queries:
            return SCMRouterOutput(
                route="structured",
                queries=[
                    MemoryQuerySpec(type="prefix", key=base_prefix, limit=lim)
                ],
                writeback=[],
                intent="fallback_default",
                reasoning_steps=["empty plan; injected safe default prefix"],
                confidence=0.5,
                fallback="NO_OP",
            )

        # Optional writeback: log routing decision as event (caller persists)
        writeback.append(
            WritebackSpec(
                op="WRITE_EVENT",
                args={
                    "kind": "scm_router_v0_1",
                    "route": route,
                    "intent": intent,
                    "query_count": len(queries),
                },
            )
        )

        return SCMRouterOutput(
            route=route,  # type: ignore[arg-type]
            queries=queries,
            writeback=writeback,
            intent=intent,
            reasoning_steps=reasoning,
            confidence=0.85 if route != "structured" else 0.9,
            fallback=None,
        )

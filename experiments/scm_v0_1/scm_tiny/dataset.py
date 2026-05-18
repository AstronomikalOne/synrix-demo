"""Teacher-labeled examples for SCM-Tiny distillation (RulesScmRouter = teacher)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..contracts import ExecutionContract
from ..packets import SCMInputPacket
from ..router_rules import RulesScmRouter
from .baseline import route_to_index
from .jsonl_inputs import load_router_input_jsonl
from .templates import template_id_from_teacher, template_to_index

StructuredExample = Tuple[SCMInputPacket, str]


def _router() -> RulesScmRouter:
    return RulesScmRouter(ExecutionContract(max_queries_per_turn=4, max_prefix_limit=100))


def distillation_examples() -> List[Dict[str, Any]]:
    """Return labeled rows: ``packet`` (SCMInputPacket), ``route`` (teacher), ``tag`` (debug).

    Diverse paraphrases so a tiny model can generalize beyond ``tracegen.synthetic_trace``.
    """
    r = _router()
    rows: List[Dict[str, Any]] = []

    def add(pkt: SCMInputPacket, tag: str) -> None:
        out = r.route(pkt)
        rows.append(
            {
                "tag": tag,
                "packet": pkt,
                "route": out.route,
                "template_id": template_id_from_teacher(pkt, out),
            }
        )

    # --- mixed: failure language ---
    failure_goals = [
        "Diagnose recent faults",
        "Why did the job stop",
        "Investigate alarm condition",
        "Trace spindle crash",
    ]
    failure_queries = [
        "motor error and timeout",
        "e-stop during cut",
        "alarm on tool 3",
        "timeout waiting for home",
        "fault after warmup",
    ]
    for g in failure_goals:
        for q in failure_queries:
            add(
                SCMInputPacket(
                    goal=g,
                    query=q,
                    available_memory_types=["TASK", "EVENT"],
                ),
                "failure_text",
            )

    # --- mixed: context-only (ERROR in recent_events, quiet goal text) ---
    for msg in ("spindle vibration alarm", "servo fault", "lubrication pressure low"):
        add(
            SCMInputPacket(
                goal="ingest_system_log",
                query="",
                available_memory_types=["TASK", "EVENT"],
                recent_events=[
                    {
                        "channel": "system_log",
                        "level": "ERROR",
                        "message": msg,
                        "logger": "cnc",
                        "ts": 1.0,
                    }
                ],
            ),
            "failure_context_log",
        )

    # --- semantic: similarity / pattern cues (no failure words) ---
    semantic_goals = [
        "Find jobs similar to last week",
        "Locate nearest anomaly cases",
        "Match pattern in production",
        "Warping issue pattern search",
    ]
    semantic_queries = [
        "pattern like before",
        "closest to reference run",
        "similar to baseline lot",
        "resembles prior bad lot",
    ]
    for g in semantic_goals:
        for q in semantic_queries:
            add(
                SCMInputPacket(
                    goal=g,
                    query=q,
                    available_memory_types=["TASK"],
                ),
                "semantic_text",
            )

    # --- semantic: machinery + semantic cue ---
    for g, q in (
        ("PLA print quality", "layer pattern like last batch"),
        ("CNC finish", "chatter similar to prior job"),
        ("Nozzle clog", "closest prior incident"),
    ):
        add(
            SCMInputPacket(goal=g, query=q, available_memory_types=["TASK", "EVENT"]),
            "machinery_semantic",
        )

    # --- structured: constraints / safety ---
    for g, q in (
        ("Safety constraints for CNC", ""),
        ("List interlocks on lathe", "rules and limits"),
        ("Forbidden moves", "must not violate envelope"),
        ("Cell safety rules", "interlocks and limits"),
        ("Operator guardrails", "what is not allowed"),
        ("Program boundaries", "envelope must hold"),
    ):
        add(
            SCMInputPacket(goal=g, query=q, available_memory_types=["CONSTRAINT", "TASK"]),
            "constraint",
        )

    # --- structured: history ---
    for g, q in (
        ("Timeline before pause", "what happened prior"),
        ("Prior errors on line 2", "history of faults"),
        ("Earlier shifts", "what happened last time"),
        ("Previous cycle context", "prior events on spindle"),
    ):
        add(
            SCMInputPacket(goal=g, query=q, available_memory_types=["EVENT"]),
            "history",
        )

    # --- structured: default prefix (open work, summaries) ---
    for g, q in (
        ("Summarize open work orders", ""),
        ("List active jobs", "current queue"),
        ("Shop status", "running programs"),
        ("Backlog snapshot", "open tasks only"),
        ("Workcell overview", "jobs in progress"),
    ):
        add(
            SCMInputPacket(goal=g, query=q, available_memory_types=["JOB", "TASK"]),
            "default_prefix",
        )

    for mem in (
        ["TASK"],
        ["JOB"],
        ["EVENT"],
        ["TASK", "JOB"],
    ):
        add(
            SCMInputPacket(
                goal="Inventory current ops",
                query="",
                available_memory_types=list(mem),
            ),
            "default_mem",
        )

    # --- semantic with max_queries_per_turn=1 → similarity-only plan ---
    r_one = RulesScmRouter(ExecutionContract(max_queries_per_turn=1, max_prefix_limit=100))
    for g, q in (
        ("Find similar jobs", "pattern like last week"),
        ("Nearest anomaly cases", "closest to reference"),
    ):
        pkt = SCMInputPacket(goal=g, query=q, available_memory_types=["TASK"])
        out = r_one.route(pkt)
        rows.append(
            {
                "tag": "semantic_single_query_contract",
                "packet": pkt,
                "route": out.route,
                "template_id": template_id_from_teacher(pkt, out),
            }
        )

    # --- INFO system log ingest (structured + EVENT correlation; not mixed) ---
    for msg in ("cycle start acknowledged", "operator acknowledged prompt"):
        pkt = SCMInputPacket(
            goal="ingest_system_log",
            query="",
            available_memory_types=["TASK", "EVENT"],
            recent_events=[
                {
                    "channel": "system_log",
                    "level": "INFO",
                    "message": msg,
                    "logger": "cnc",
                    "ts": 1.0,
                }
            ],
        )
        out = r.route(pkt)
        rows.append(
            {
                "tag": "log_ingest_info",
                "packet": pkt,
                "route": out.route,
                "template_id": template_id_from_teacher(pkt, out),
            }
        )

    return rows


def packets_and_labels_for_scm_tiny(
    *,
    jsonl_path: Path | None = None,
    max_jsonl_rows: int = 1_000_000,
    merge_distillation: bool = False,
) -> tuple[list[SCMInputPacket], np.ndarray, np.ndarray]:
    """
    Build supervised rows for SCM-Tiny: packets plus integer route/template labels.

    * **Default** (``jsonl_path is None``): built-in ``distillation_examples()`` labels
      (preserves per-row teacher semantics, including narrow contracts encoded in the set).
    * **JSONL only**: each line is a packet; labels come from
      ``RulesScmRouter(ExecutionContract())`` + ``template_id_from_teacher`` (same as
      ``train_scm_micro.py``).
    * **Merge**: prepend distillation rows (stored labels), then append teacher-labeled
      JSONL rows — use for volume without losing the small curated slice.
    """
    packets: list[SCMInputPacket] = []
    routes_i: list[int] = []
    tmpl_i: list[int] = []

    if jsonl_path is None:
        rows = distillation_examples()
        for r in rows:
            packets.append(r["packet"])
            routes_i.append(route_to_index(r["route"]))
            tmpl_i.append(template_to_index(r["template_id"]))
        return packets, np.array(routes_i, dtype=np.int64), np.array(tmpl_i, dtype=np.int64)

    jl = load_router_input_jsonl(jsonl_path, max(0, max_jsonl_rows))
    if not jl:
        raise ValueError(f"no JSONL rows loaded from {jsonl_path}")

    if merge_distillation:
        for r in distillation_examples():
            packets.append(r["packet"])
            routes_i.append(route_to_index(r["route"]))
            tmpl_i.append(template_to_index(r["template_id"]))

    router = RulesScmRouter(ExecutionContract())
    for row in jl:
        pkt = row["packet"]
        out = router.route(pkt)
        packets.append(pkt)
        routes_i.append(route_to_index(str(out.route)))
        tmpl_i.append(template_to_index(template_id_from_teacher(pkt, out)))

    y_r = np.array(routes_i, dtype=np.int64)
    y_t = np.array(tmpl_i, dtype=np.int64)
    return packets, y_r, y_t

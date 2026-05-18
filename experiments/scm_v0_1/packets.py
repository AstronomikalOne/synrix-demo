"""Structured packets for SCM-Router-0.1 (spec input/output shapes)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


RouteKind = Literal["structured", "semantic", "mixed"]
QueryType = Literal["prefix", "exact", "similarity"]


@dataclass
class MemoryQuerySpec:
    type: QueryType
    key: Optional[str] = None
    limit: Optional[int] = None
    vector_ref: Optional[str] = None  # e.g. "HRR(goal)" — symbolic, not raw bytes
    k: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.key is not None:
            d["key"] = self.key
        if self.limit is not None:
            d["limit"] = self.limit
        if self.vector_ref is not None:
            d["vector"] = self.vector_ref
        if self.k is not None:
            d["k"] = self.k
        return d

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MemoryQuerySpec":
        return cls(
            type=raw["type"],
            key=raw.get("key"),
            limit=raw.get("limit"),
            vector_ref=raw.get("vector"),
            k=raw.get("k"),
        )


@dataclass
class WritebackSpec:
    op: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "args": dict(self.args)}


@dataclass
class SCMInputPacket:
    goal: str
    query: str = ""
    active_state: List[Any] = field(default_factory=list)
    retrieved_nodes: List[Any] = field(default_factory=list)
    graph_edges: List[Any] = field(default_factory=list)
    recent_events: List[Any] = field(default_factory=list)
    constraints: List[Any] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    available_memory_types: List[str] = field(default_factory=list)
    # Optional WAVE profile (14 float metrics, same keys as wave_tagged ingest). Drives
    # RulesScmRouter similarity with vector_ref ``wave_goal_metrics`` — no lattice gold read.
    wave_goal_metrics: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SCMInputPacket":
        wgm = raw.get("wave_goal_metrics")
        if wgm is not None and not isinstance(wgm, dict):
            wgm = None
        return cls(
            goal=str(raw.get("goal", "")),
            query=str(raw.get("query", "")),
            active_state=list(raw.get("active_state", [])),
            retrieved_nodes=list(raw.get("retrieved_nodes", [])),
            graph_edges=list(raw.get("graph_edges", [])),
            recent_events=list(raw.get("recent_events", [])),
            constraints=list(raw.get("constraints", [])),
            available_tools=list(raw.get("available_tools", [])),
            available_memory_types=list(raw.get("available_memory_types", [])),
            wave_goal_metrics=dict(wgm) if isinstance(wgm, dict) else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "goal": self.goal,
            "query": self.query,
            "active_state": self.active_state,
            "retrieved_nodes": self.retrieved_nodes,
            "graph_edges": self.graph_edges,
            "recent_events": self.recent_events,
            "constraints": self.constraints,
            "available_tools": self.available_tools,
            "available_memory_types": self.available_memory_types,
        }
        if self.wave_goal_metrics is not None:
            d["wave_goal_metrics"] = dict(self.wave_goal_metrics)
        return d

    def canonical_text(self) -> str:
        """Lowercased blend for rule matching (not for embedding)."""
        parts = [self.goal, self.query]
        return " ".join(p for p in parts if p).lower()


@dataclass
class SCMRouterOutput:
    route: RouteKind
    queries: List[MemoryQuerySpec]
    writeback: List[WritebackSpec]
    intent: str = "route_only"
    reasoning_steps: List[str] = field(default_factory=list)
    confidence: float = 1.0
    fallback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "route": self.route,
            "memory_queries": [q.to_dict() for q in self.queries],
            "reasoning_steps": list(self.reasoning_steps),
            "writeback": [w.to_dict() for w in self.writeback],
            "confidence": self.confidence,
            "fallback": self.fallback,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SCMRouterOutput":
        """Parse router JSON (e.g. after ``validate_router_output_dict``)."""
        queries = [MemoryQuerySpec.from_dict(q) for q in raw.get("memory_queries", [])]
        writeback = [
            WritebackSpec(op=str(w["op"]), args=dict(w.get("args") or {}))
            for w in raw.get("writeback", [])
        ]
        return cls(
            route=raw["route"],  # type: ignore[arg-type]
            queries=queries,
            writeback=writeback,
            intent=str(raw.get("intent", "retrieve_context")),
            reasoning_steps=list(raw.get("reasoning_steps") or []),
            confidence=float(raw["confidence"]) if raw.get("confidence") is not None else 1.0,
            fallback=raw.get("fallback"),
        )

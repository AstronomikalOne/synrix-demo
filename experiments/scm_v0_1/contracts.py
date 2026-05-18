"""Execution contract defaults (SCM v0.1 spec § Execution Contract)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionContract:
    max_ops_per_turn: int = 8
    max_prefix_limit: int = 100
    max_similarity_k: int = 16
    # 0 = disabled (default). When set, adapter uses a Python thread boundary; FFI may still run past timeout.
    query_timeout_ms: int = 0
    max_queries_per_turn: int = 4
    control_safe_mode: bool = True

    def clamp_prefix_limit(self, n: int) -> int:
        return max(1, min(int(n), self.max_prefix_limit))

    def clamp_k(self, k: int) -> int:
        return max(1, min(int(k), self.max_similarity_k))

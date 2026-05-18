"""
C1b — template query policy: validate a query plan against the shapes allowed for a template id.

Derived from ``router_rules.py`` / ``template_id_from_teacher`` semantics. Used to gate teacher
plans against the student's predicted template (observability first; execution unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .packets import MemoryQuerySpec
from .scm_tiny.templates import QUERY_TEMPLATE_IDS

# Templates where every query must be non-prefix (similarity-only arms).
TEMPLATE_PREFIX_BANNED: frozenset = frozenset({"T_SEMANTIC_ONLY"})


@dataclass(frozen=True)
class QueryShapeRule:
    """One structural constraint on ``MemoryQuerySpec`` rows in a plan."""

    type: str  # "prefix" | "similarity"
    key_prefix: Optional[str] = None
    vector_ref: Optional[str] = None
    required: bool = False


# Declarative policy for templates that decompose cleanly into AND-of rules.
# See ``validate_queries_for_template`` for composite templates (failure mixed, log ingest).
TEMPLATE_QUERY_POLICY: Dict[str, List[QueryShapeRule]] = {
    "T_SEMANTIC_ONLY": [
        # Prefix banned via ``TEMPLATE_PREFIX_BANNED`` below.
        QueryShapeRule("similarity", required=True),
    ],
    "T_SEMANTIC_WITH_PREFIX": [
        QueryShapeRule("similarity", required=True),
        QueryShapeRule("prefix", required=True),
    ],
    "T_CONSTRAINT": [
        QueryShapeRule("prefix", key_prefix="CONSTRAINT:", required=True),
    ],
    "T_HISTORY": [
        QueryShapeRule("prefix", key_prefix="EVENT:", required=True),
    ],
    "T_LOG_INGEST": [
        QueryShapeRule("prefix", key_prefix="EVENT:", required=True),
    ],
    "T_DEFAULT_PREFIX": [
        QueryShapeRule("prefix", required=True),
    ],
}


@dataclass(frozen=True)
class TemplatePolicyResult:
    ok: bool
    template_id: str
    failures: List[str]


def _query_matches_rule(q: MemoryQuerySpec, rule: QueryShapeRule) -> bool:
    if q.type != rule.type:
        return False
    if rule.type == "prefix":
        key = q.key or ""
        if rule.key_prefix is not None and not key.startswith(rule.key_prefix):
            return False
        return True
    if rule.type == "similarity":
        ref = q.vector_ref or ""
        if rule.vector_ref is not None and ref != rule.vector_ref:
            return False
        return True
    return False


def _failure_mixed_ok(queries: List[MemoryQuerySpec]) -> bool:
    """``TASK:FAILURE:`` / ``*FAILURE:`` prefix plus ``HRR(goal+query)`` similarity (rules teacher)."""
    has_hrr = any(
        q.type == "similarity" and (q.vector_ref or "") == "HRR(goal+query)" for q in queries
    )
    has_fail_pfx = any(
        q.type == "prefix" and q.key and "FAILURE" in q.key for q in queries
    )
    return bool(has_hrr and has_fail_pfx)


def _log_ingest_ok(queries: List[MemoryQuerySpec]) -> bool:
    """Log ingest adds ``EVENT:`` in addition to the default primary prefix (two prefix arms)."""
    pref = [q for q in queries if q.type == "prefix"]
    if len(pref) < 2:
        return False
    return any((q.key or "").startswith("EVENT:") for q in pref)


def validate_queries_for_template(
    queries: List[MemoryQuerySpec],
    template_id: str,
) -> TemplatePolicyResult:
    """
    Check that ``queries`` satisfy the student's template policy.

    Unknown ``template_id`` → ``ok=False``, ``unknown_template``.
    """
    failures: List[str] = []
    if template_id not in QUERY_TEMPLATE_IDS:
        return TemplatePolicyResult(False, template_id, ["unknown_template"])

    if template_id == "T_FAILURE_MIXED":
        if not _failure_mixed_ok(queries):
            if not any(
                q.type == "similarity" and (q.vector_ref or "") == "HRR(goal+query)"
                for q in queries
            ):
                failures.append("failure_mixed_requires_similarity_HRR_goal_query")
            if not any(
                q.type == "prefix" and q.key and "FAILURE" in q.key for q in queries
            ):
                failures.append("failure_mixed_requires_failure_class_prefix")
        return TemplatePolicyResult(len(failures) == 0, template_id, failures)

    if template_id in TEMPLATE_PREFIX_BANNED and any(q.type == "prefix" for q in queries):
        failures.append("semantic_only_bans_prefix")

    rules = TEMPLATE_QUERY_POLICY.get(template_id, [])
    for rule in rules:
        if not rule.required:
            continue
        if not any(_query_matches_rule(q, rule) for q in queries):
            failures.append(
                f"missing_required_rule:{rule.type}:"
                f"key_prefix={rule.key_prefix!r}:vector_ref={rule.vector_ref!r}"
            )

    if template_id == "T_LOG_INGEST" and not _log_ingest_ok(queries):
        failures.append("log_ingest_requires_event_prefix_and_primary_prefix")

    return TemplatePolicyResult(len(failures) == 0, template_id, failures)

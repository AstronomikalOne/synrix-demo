"""
Single entry for SCM v0.1 routing in processes that should pick up **optional** INT8
Micro shadow hints without a second code path.

Use :func:`get_scm_router` everywhere you would have called ``RulesScmRouter()``.

**Rule authority (env-gated):**

* ``SCM_ROUTER_MODE=gated`` — C1b ``TemplateGatedScmRouter``: teacher output unchanged; append
  ``template_gated:`` line with ``policy_ok`` vs SCM-Tiny's predicted template (needs ``SCM_TINY_NPZ``
  or expert-dispatch NPZs). Does **not** load SCM-Micro INT8 shadow.
* ``SCM_ROUTER_MODE=rules`` (or ``rules_only`` / ``strict``) — return **only**
  :class:`RulesScmRouter`: no shadow class, no Micro line, same behavior as
  v0.1 before shadow integration.
* Default / ``shadow`` — return :class:`ShadowScmRouterMicroInt8`, which **always**
  delegates to rules for execution. Optional shadows append ``reasoning_steps`` lines only:

  * **SCM-Tiny** (numpy student): ``SCM_TINY_NPZ`` → ``.npz`` path; opt-out with
    ``SCM_TINY_SHADOW=0``. Line prefix ``shadow_student:``.
    **Expert Library:** ``SCM_TINY_EXPERT_DISPATCH=1`` with optional
    ``SCM_TINY_NPZ_WAVE`` / ``SCM_TINY_NPZ_CWRU`` / ``SCM_TINY_NPZ_UNSW`` (each must
    exist on disk to be used; else ``SCM_TINY_NPZ`` fallback). Dispatch adds
    ``expert=<kind>`` to the shadow line.
  * **SCM-Micro INT8**: ``SCM_MICRO_NPZ`` + ``SCM_MICRO_INT8_LIB`` (and
    ``SCM_MICRO_INT8_SHADOW`` not opt-out). Line prefix ``shadow_scm_micro_int8:``.

  Both can be active; teacher output is unchanged.

**Other env:** ``SCM_MICRO_NPZ``, ``SCM_MICRO_INT8_LIB``, ``SCM_MICRO_INT8_SHADOW=0``,
``SCM_TINY_NPZ``, ``SCM_TINY_SHADOW=0``, ``SCM_TINY_EXPERT_DISPATCH``,
``SCM_TINY_NPZ_WAVE``, ``SCM_TINY_NPZ_CWRU``, ``SCM_TINY_NPZ_UNSW``.
Optional INT8 calibration (domain-aware scales): ``SCM_MICRO_INT8_CALIB_JSONL``,
``SCM_MICRO_INT8_CALIB_MAX_ROWS`` (see :class:`ShadowScmRouterMicroInt8`).
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Union

from .contracts import ExecutionContract
from .packets import SCMInputPacket, SCMRouterOutput
from .router_rules import RulesScmRouter
from .shadow_micro_int8 import ShadowScmRouterMicroInt8, _tiny_predictor_from_env
from .shadow_router import TemplateGatedScmRouter
from .scm_tiny.expert_dispatch import expert_dispatch_enabled
from .scm_tiny.templates import template_id_from_teacher

ScmRouter = Union[RulesScmRouter, ShadowScmRouterMicroInt8, TemplateGatedScmRouter]


def _router_mode_is_gated() -> bool:
    return (os.environ.get("SCM_ROUTER_MODE") or "").strip().lower() == "gated"


def _router_mode_allows_shadow() -> bool:
    v = (os.environ.get("SCM_ROUTER_MODE") or "shadow").strip().lower()
    if v in ("rules", "rules_only", "strict"):
        return False
    if v in ("gated",):
        return False
    if v in ("shadow", "shadow_micro", "shadow_micro_int8", "default", ""):
        return True
    return True


def get_scm_router(contract: Optional[ExecutionContract] = None) -> ScmRouter:
    """
    Router boundary for SCM v0.1: **strict rules** or **rules + optional shadow**.

    * ``SCM_ROUTER_MODE=gated`` — ``TemplateGatedScmRouter`` (C1b): teacher plan validated
      against SCM-Tiny's predicted template shape; needs ``SCM_TINY_NPZ`` or expert dispatch NPZs.
    * ``SCM_ROUTER_MODE`` forces ``RulesScmRouter`` when set to rules — Micro is
      never loaded; authority is unambiguous.
    * Otherwise the shadow wrapper is used (inert if artifacts / env disable it).
    """
    if _router_mode_is_gated():
        st = None if expert_dispatch_enabled() else _tiny_predictor_from_env()
        if st is None and not expert_dispatch_enabled():
            return RulesScmRouter(contract)
        return TemplateGatedScmRouter(st, contract)
    if not _router_mode_allows_shadow():
        return RulesScmRouter(contract)
    return ShadowScmRouterMicroInt8(contract)


_RE_AGREE = re.compile(r"agree_with_teacher=(True|False)")
_RE_EXPERT = re.compile(r"expert=([a-zA-Z0-9_]+)")
_RE_ROUTE = re.compile(r"route=([a-zA-Z0-9_-]+)")
_RE_TMPL = re.compile(r"template=([A-Za-z0-9_]+)")
_RE_POLICY_OK = re.compile(r"policy_ok=(True|False)")
_RE_TEMPLATE_AGREE = re.compile(r"template_agree=(True|False)")


def parse_template_gated_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse ``template_gated:`` line (C1b policy observation)."""
    if "template_gated:" not in line:
        return None
    raw = line.strip()
    if "error=" in raw and "no_student_predictor" in raw:
        return {
            "present": True,
            "ok": False,
            "kind": "template_gated",
            "error": "no_student_predictor",
            "execution_mode": "policy_fallback",
            "raw": raw,
        }
    policy_ok: Optional[bool] = None
    m = _RE_POLICY_OK.search(raw)
    if m:
        policy_ok = m.group(1) == "True"
    agree: Optional[bool] = None
    m_a = _RE_TEMPLATE_AGREE.search(raw)
    if m_a:
        agree = m_a.group(1) == "True"
    m_st = re.search(r"student_template=([A-Za-z0-9_]+)", raw)
    m_tt = re.search(r"teacher_template=([A-Za-z0-9_]+)", raw)
    m_em = re.search(r"execution_mode=([A-Za-z0-9_]+)", raw)
    out: Dict[str, Any] = {
        "present": True,
        "ok": True,
        "kind": "template_gated",
        "policy_ok": policy_ok,
        "template_agree": agree,
        "student_template": m_st.group(1) if m_st else None,
        "teacher_template": m_tt.group(1) if m_tt else None,
        "execution_mode": m_em.group(1) if m_em else None,
        "raw": raw,
    }
    if "policy_failures=" in raw:
        out["policy_failures"] = raw.split("policy_failures=", 1)[-1].strip()
    return out


def parse_scm_tiny_shadow_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse ``shadow_student:`` line (SCM-Tiny); shape aligned with Micro parser."""
    if "shadow_student:" not in line:
        return None
    raw = line.strip()
    if "error=" in raw:
        err_part = raw.split("error=", 1)[-1].strip()
        return {
            "present": True,
            "ok": False,
            "error": err_part[:500],
            "raw": raw,
            "kind": "tiny",
        }
    agree: Optional[bool] = None
    m = _RE_AGREE.search(raw)
    if m:
        agree = m.group(1) == "True"
    m_r = _RE_ROUTE.search(raw)
    m_t = _RE_TMPL.search(raw)
    m_e = _RE_EXPERT.search(raw)
    out: Dict[str, Any] = {
        "present": True,
        "ok": True,
        "agree_with_teacher": agree,
        "suggested_route": m_r.group(1) if m_r else None,
        "suggested_template": m_t.group(1) if m_t else None,
        "raw": raw,
        "kind": "tiny",
    }
    if m_e:
        out["expert_kind"] = m_e.group(1)
    return out


def parse_scm_micro_shadow_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single ``reasoning_steps`` line ``shadow_scm_micro_int8: ...`` or
    ``shadow_scm_micro_fp32: ...``.
    Returns a small dict for audit / metrics, or None if not a shadow line.
    """
    if "shadow_scm_micro_int8:" not in line and "shadow_scm_micro_fp32:" not in line:
        return None
    raw = line.strip()
    if "error=" in raw:
        err_part = raw.split("error=", 1)[-1].strip()
        return {
            "present": True,
            "ok": False,
            "error": err_part[:500],
            "raw": raw,
        }
    agree: Optional[bool] = None
    m = _RE_AGREE.search(raw)
    if m:
        agree = m.group(1) == "True"
    m_r = _RE_ROUTE.search(raw)
    m_t = _RE_TMPL.search(raw)
    return {
        "present": True,
        "ok": True,
        "agree_with_teacher": agree,
        "suggested_route": m_r.group(1) if m_r else None,
        "suggested_template": m_t.group(1) if m_t else None,
        "raw": raw,
    }


def shadow_audit_from_router_out(
    router_out: SCMRouterOutput,
) -> Optional[Dict[str, Any]]:
    """Extract shadow summary from ``router_out.reasoning_steps`` (Micro preferred, else Tiny)."""
    steps: List[str] = list(router_out.reasoning_steps or [])
    for s in steps:
        p = parse_scm_micro_shadow_line(s)
        if p is not None:
            return p
    for s in steps:
        p = parse_template_gated_line(s)
        if p is not None:
            return p
    for s in steps:
        p = parse_scm_tiny_shadow_line(s)
        if p is not None:
            return p
    return None


def build_scm_shadow_log(
    router_out: SCMRouterOutput,
    packet: SCMInputPacket,
) -> Dict[str, Any]:
    """
    Structured log for **authoritative rules execution** + optional **Micro agreement**.

    * ``execution`` — always from ``router_out`` (what will run; rules are canonical).
    * ``agreement`` — teacher = rules output; ``suggested_*`` from the shadow line when
      present; otherwise nulls.
    * ``packet`` — ``goal_fingerprint`` (sha256 prefix) and length; not full text.
    """
    g = str(packet.goal or "")
    fp = hashlib.sha256(g.encode("utf-8")).hexdigest()[:16]
    allows = _router_mode_allows_shadow()
    line = shadow_audit_from_router_out(router_out) if allows else None
    if not allows:
        mode = "rules"
    elif line and line.get("kind") == "template_gated":
        mode = "template_gated"
    elif line and line.get("kind") == "tiny":
        mode = "shadow_tiny"
    else:
        mode = "shadow_micro"
    execution = {
        "route": router_out.route,
        "memory_query_count": len(router_out.queries or []),
        "writeback_count": len(router_out.writeback or []),
        "intent": router_out.intent,
        "confidence": router_out.confidence,
    }
    t_route = str(router_out.route)
    t_templ = template_id_from_teacher(packet, router_out)

    base: Dict[str, Any] = {
        "router_mode": mode,
        "rule_authority": (
            "rules_only" if not allows else "rules_plus_optional_shadow_hint"
        ),
        "execution": execution,
        "packet": {
            "goal_len": len(g),
            "goal_fingerprint": fp,
        },
    }
    if not allows:
        base["shadow_injected"] = False
        base["agreement"] = None
        base["line_parse"] = None
        return base

    base["shadow_injected"] = line is not None
    if line and line.get("ok"):
        base["agreement"] = {
            "authoritative_route": t_route,
            "authoritative_template": t_templ,
            "agree_with_teacher": line.get("agree_with_teacher"),
            "suggested_route": line.get("suggested_route"),
            "suggested_template": line.get("suggested_template"),
        }
    elif line and not line.get("ok"):
        base["agreement"] = {
            "authoritative_route": t_route,
            "authoritative_template": t_templ,
            "agree_with_teacher": None,
            "suggested_route": None,
            "suggested_template": None,
            "shadow_error": line.get("error"),
        }
    else:
        base["agreement"] = {
            "authoritative_route": t_route,
            "authoritative_template": t_templ,
            "agree_with_teacher": None,
            "suggested_route": None,
            "suggested_template": None,
        }
    base["line_parse"] = line
    return base

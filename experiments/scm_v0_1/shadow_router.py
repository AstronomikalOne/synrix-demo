"""
Run the rules teacher for executable query plans; attach student (SCM-Tiny) agreement
metadata for training/eval and safe shadow deploys. Execution always remains teacher-valid.

For INT8 Micro + ``forward_n``, see ``shadow_micro_int8.ShadowScmRouterMicroInt8`` (same contract).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .contracts import ExecutionContract
from .packets import SCMInputPacket, SCMRouterOutput
from .router_rules import RulesScmRouter
from .scm_tiny.artifact import ScmTinyPredictor
from .scm_tiny.expert_dispatch import expert_dispatch_enabled, predictor_for_packet
from .scm_tiny.templates import template_id_from_teacher
from .template_query_policy import validate_queries_for_template


class ShadowScmRouter:
    """
    ``route()`` returns the same ``SCMRouterOutput`` as ``RulesScmRouter`` (queries +
    writebacks) but appends a ``reasoning_steps`` line with student route/template and
    whether they match the teacher. Does not change ``route``/``queries`` fields — the
    student is observed, not in control, until you promote a stricter mode elsewhere.
    """

    def __init__(
        self,
        student: ScmTinyPredictor,
        contract: Optional[ExecutionContract] = None,
    ):
        self._student = student
        self._teacher = RulesScmRouter(contract)

    def route(self, packet: SCMInputPacket) -> SCMRouterOutput:
        out = self._teacher.route(packet)
        t_template = template_id_from_teacher(packet, out)
        s_route = self._student.predict_route(packet)
        s_tmpl = self._student.predict_template_id(packet)
        agree = (s_route == out.route) and (s_tmpl == t_template)
        extra = (
            f"shadow_student: route={s_route} template={s_tmpl} "
            f"agree_with_teacher={agree}"
        )
        new_steps = list(out.reasoning_steps or []) + [extra]
        return replace(out, reasoning_steps=new_steps)


class TemplateGatedScmRouter:
    """
    Teacher runs; student predicts template; teacher plan is validated against the student's
    allowed query shapes (C1b). Mismatch is recorded on ``reasoning_steps`` only — execution
    remains teacher-valid (no plan mutation).
    """

    def __init__(
        self,
        student: Optional[ScmTinyPredictor],
        contract: Optional[ExecutionContract] = None,
    ) -> None:
        self._student = student
        self._teacher = RulesScmRouter(contract)
        self._dispatch = expert_dispatch_enabled()

    def _predictor(self, packet: SCMInputPacket) -> Optional[ScmTinyPredictor]:
        if self._dispatch:
            p = predictor_for_packet(packet)
            if p is not None:
                return p
        return self._student

    def route(self, packet: SCMInputPacket) -> SCMRouterOutput:
        out = self._teacher.route(packet)
        t_template = template_id_from_teacher(packet, out)
        tiny = self._predictor(packet)
        tag = "template_gated"
        if tiny is None:
            extra = (
                f"{tag}: error=no_student_predictor teacher_template={t_template} "
                "policy_ok=False execution_mode=policy_fallback"
            )
            new_steps = list(out.reasoning_steps or []) + [extra]
            return replace(
                out,
                reasoning_steps=new_steps,
                fallback="no_student_predictor",
            )

        s_tmpl = tiny.predict_template_id(packet)
        result = validate_queries_for_template(out.queries, s_tmpl)
        agree = s_tmpl == t_template
        if result.ok:
            execution_mode = "policy_admitted"
            fallback: Optional[str] = None
        else:
            execution_mode = "policy_fallback"
            fallback = "template_policy_mismatch:" + ":".join(result.failures)
        extra = (
            f"{tag}: student_template={s_tmpl} teacher_template={t_template} "
            f"template_agree={agree} policy_ok={result.ok} "
            f"execution_mode={execution_mode}"
        )
        if not result.ok:
            extra += f" policy_failures={result.failures}"
        new_steps = list(out.reasoning_steps or []) + [extra]
        return replace(out, reasoning_steps=new_steps, fallback=fallback)

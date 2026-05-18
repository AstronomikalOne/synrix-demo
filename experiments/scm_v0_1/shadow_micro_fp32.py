"""
FP32 SCM-Micro shadow (numpy ``MicroMlpPolicy``) for **diagnostics** — same contract as
:class:`ShadowScmRouterMicroInt8` but no quantization. Rules remain authoritative.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .contracts import ExecutionContract
from .packets import SCMInputPacket, SCMRouterOutput
from .router_rules import RulesScmRouter
from .scm_micro.mlp_policy import MicroMlpPolicy
from .scm_tiny.baseline import ROUTE_CLASSES
from .scm_tiny.features import featurize_packet
from .scm_tiny.templates import QUERY_TEMPLATE_IDS, template_id_from_teacher

_ArtifactPath = Union[str, Path, os.PathLike[str]]


def _resolve(p: Optional[_ArtifactPath], env_key: str) -> Optional[Path]:
    if p is not None:
        return Path(p)
    v = os.environ.get(env_key)
    return Path(v) if v else None


class ShadowScmRouterMicroFp32:
    """Teacher + optional FP32 Micro line in ``reasoning_steps`` (eval / ablations)."""

    def __init__(
        self,
        contract: Optional[ExecutionContract] = None,
        *,
        micro_npz: Optional[_ArtifactPath] = None,
    ) -> None:
        self._teacher = RulesScmRouter(contract)
        self._mlp: Optional[MicroMlpPolicy] = None

        opt = os.environ.get("SCM_MICRO_FP32_SHADOW", "1").strip().lower()
        if opt in ("0", "false", "no", "off"):
            return

        p = _resolve(micro_npz, "SCM_MICRO_NPZ")
        if p is None or not p.is_file():
            return
        self._mlp = MicroMlpPolicy.load(p)

    @property
    def enabled(self) -> bool:
        return self._mlp is not None

    def route(self, packet: SCMInputPacket) -> SCMRouterOutput:
        out = self._teacher.route(packet)
        if not self.enabled:
            return out

        t_template = template_id_from_teacher(packet, out)
        mlp = self._mlp
        if mlp is None:
            return out
        try:
            x = featurize_packet(packet)[None, :]
            lr, lt, *_ = mlp.forward(x)
            ir = int(np.argmax(lr[0]))
            it = int(np.argmax(lt[0]))
            s_route = ROUTE_CLASSES[ir]
            s_tmpl = QUERY_TEMPLATE_IDS[it]
            agree = (s_route == out.route) and (s_tmpl == t_template)
            extra = (
                f"shadow_scm_micro_fp32: route={s_route} template={s_tmpl} "
                f"agree_with_teacher={agree} path=numpy_forward"
            )
        except Exception as e:  # noqa: BLE001
            extra = f"shadow_scm_micro_fp32: error={type(e).__name__}:{e}"
        new_steps = list(out.reasoning_steps or []) + [extra]
        return replace(out, reasoning_steps=new_steps)

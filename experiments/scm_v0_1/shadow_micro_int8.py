"""
SCM-Micro INT8 (``forward_n``) as a *shadow* hint: ``RulesScmRouter`` always owns
execution; one ``reasoning_steps`` line records suggested route/template vs teacher.

**Spine remains rules + contracts.** Micro suggests; rules decide.

When ``SCM_MICRO_NPZ`` and ``SCM_MICRO_INT8_LIB`` are not both set to existing files
(or constructor paths are omitted / invalid), the router behaves as ``RulesScmRouter``
only (no shadow line). Optional env ``SCM_MICRO_INT8_SHADOW=0`` forces that off
even if paths exist.

Optional **domain calibration** for INT8 scales: ``SCM_MICRO_INT8_CALIB_JSONL`` (JSONL
of packets) and ``SCM_MICRO_INT8_CALIB_MAX_ROWS`` (default loader cap). If unset,
calibration features come from the fixed distillation corpus only.

Optional **SCM-Tiny** (numpy distilled student): set ``SCM_TINY_NPZ`` to a ``.npz``
artifact; a ``shadow_student:`` line is appended before any Micro line. Disable with
``SCM_TINY_SHADOW=0``. Teacher routing is unchanged.

**Expert Library:** ``SCM_TINY_EXPERT_DISPATCH=1`` with ``SCM_TINY_NPZ[_WAVE|_CWRU|_UNSW]``
selects the student per packet (see ``scm_tiny.expert_dispatch``). Shadow line includes
``expert=<kind>`` when dispatch is on.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .contracts import ExecutionContract
from .packets import SCMInputPacket, SCMRouterOutput
from .router_rules import RulesScmRouter
from .scm_micro.int8_c_forward import forward_logits_c_n, load_scm_micro_int8_lib
from .scm_micro.int8_policy import Int8MicroMlpPolicy
from .scm_micro.mlp_policy import MicroMlpPolicy
from .scm_tiny.artifact import ScmTinyArtifact, ScmTinyPredictor
from .scm_tiny.baseline import ROUTE_CLASSES
from .scm_tiny.dataset import distillation_examples
from .scm_tiny.features import featurize_packet, featurize_packets
from .scm_tiny.jsonl_inputs import load_router_input_jsonl
from .scm_tiny.expert_dispatch import classify_expert_kind, expert_dispatch_enabled, predictor_for_packet
from .scm_tiny.templates import QUERY_TEMPLATE_IDS, template_id_from_teacher

_ArtifactPath = Union[str, Path, os.PathLike[str]]


def _default_calibration_X() -> np.ndarray:
    rows = distillation_examples()
    return featurize_packets([r["packet"] for r in rows])


def _calibration_X_from_jsonl(path: Path, max_rows: int) -> Optional[np.ndarray]:
    """
    Optional domain-aware INT8 calibration: scales ``s_x`` / activations are derived
    from ``X_cal`` max-abs. Using distillation-only features while evaluating on UCI /
    UNSW captures can mis-scale quantized activations (often route survives, template
    collapses).
    """
    if not path.is_file():
        return None
    cap = int(max_rows)
    if cap <= 0:
        cap = 2048
    rows = load_router_input_jsonl(path, cap)
    if not rows:
        return None
    return featurize_packets([r["packet"] for r in rows])


def _resolve(p: Optional[_ArtifactPath], env_key: str) -> Optional[Path]:
    if p is not None:
        return Path(p)
    v = os.environ.get(env_key)
    return Path(v) if v else None


def _tiny_predictor_from_env() -> Optional[ScmTinyPredictor]:
    """Load SCM-Tiny from ``SCM_TINY_NPZ`` when present; opt-out via ``SCM_TINY_SHADOW=0``."""
    opt = os.environ.get("SCM_TINY_SHADOW", "1").strip().lower()
    if opt in ("0", "false", "no", "off"):
        return None
    v = (os.environ.get("SCM_TINY_NPZ") or "").strip()
    if not v:
        return None
    p = Path(v)
    if not p.is_file():
        return None
    return ScmTinyArtifact.load(p).predictor()


class ShadowScmRouterMicroInt8:
    """
    Like ``ShadowScmRouter`` but the student is **INT8 Micro** via ``forward_logits_c_n``
    (native ``scm_micro_int8_forward_n`` in the host ``.so``). Teacher output is
    unchanged; optional **Tiny** then optional **Micro** metadata lines are appended.
    """

    def __init__(
        self,
        contract: Optional[ExecutionContract] = None,
        *,
        micro_npz: Optional[_ArtifactPath] = None,
        int8_lib: Optional[_ArtifactPath] = None,
        calibration_X: Optional[np.ndarray] = None,
    ) -> None:
        self._teacher = RulesScmRouter(contract)
        self._tiny_legacy: Optional[ScmTinyPredictor] = (
            None if expert_dispatch_enabled() else _tiny_predictor_from_env()
        )
        self._tiny_dispatch: bool = expert_dispatch_enabled()
        self._int8: Optional[Int8MicroMlpPolicy] = None
        self._lib: Optional[object] = None

        opt = os.environ.get("SCM_MICRO_INT8_SHADOW", "1").strip().lower()
        if opt in ("0", "false", "no", "off"):
            return

        p = _resolve(micro_npz, "SCM_MICRO_NPZ")
        l = _resolve(int8_lib, "SCM_MICRO_INT8_LIB")
        if p is None or l is None or not p.is_file() or not l.is_file():
            return

        try:
            mlp = MicroMlpPolicy.load(p)
        except ValueError:
            # Stale ``SCM_MICRO_NPZ`` (e.g. feature_dim mismatch vs current featurizer): skip Micro shadow.
            return
        if calibration_X is not None:
            x_cal = np.asarray(calibration_X, dtype=np.float32)
        else:
            cal_path = os.environ.get("SCM_MICRO_INT8_CALIB_JSONL")
            if cal_path:
                max_r_raw = os.environ.get("SCM_MICRO_INT8_CALIB_MAX_ROWS", "2048")
                try:
                    max_r = int(max_r_raw)
                except ValueError:
                    max_r = 2048
                xc = _calibration_X_from_jsonl(Path(cal_path), max_r)
                if xc is not None and xc.size:
                    x_cal = xc.astype(np.float32)
                else:
                    x_cal = _default_calibration_X()
            else:
                x_cal = _default_calibration_X()
        self._int8 = Int8MicroMlpPolicy.from_float(mlp, x_cal)
        self._lib = load_scm_micro_int8_lib(l)

    @property
    def enabled(self) -> bool:
        return self._int8 is not None and self._lib is not None

    @property
    def tiny_shadow_enabled(self) -> bool:
        if self._tiny_dispatch:
            # Dispatch mode: shadow runs if SCM_TINY_SHADOW allows and any NPZ may resolve.
            opt = os.environ.get("SCM_TINY_SHADOW", "1").strip().lower()
            if opt in ("0", "false", "no", "off"):
                return False
            fb = (os.environ.get("SCM_TINY_NPZ") or "").strip()
            if fb and Path(fb).is_file():
                return True
            for key in ("SCM_TINY_NPZ_WAVE", "SCM_TINY_NPZ_CWRU", "SCM_TINY_NPZ_UNSW"):
                p = (os.environ.get(key) or "").strip()
                if p and Path(p).is_file():
                    return True
            return False
        return self._tiny_legacy is not None

    def route(self, packet: SCMInputPacket) -> SCMRouterOutput:
        out = self._teacher.route(packet)
        steps: List[str] = list(out.reasoning_steps or [])
        t_template = template_id_from_teacher(packet, out)

        tiny: Optional[ScmTinyPredictor] = None
        if self._tiny_dispatch:
            tiny = predictor_for_packet(packet)
        else:
            tiny = self._tiny_legacy
        if tiny is not None:
            try:
                s_route_t = tiny.predict_route(packet)
                s_tmpl_t = tiny.predict_template_id(packet)
                agree_t = (s_route_t == out.route) and (s_tmpl_t == t_template)
                kind = classify_expert_kind(packet)
                if self._tiny_dispatch:
                    steps.append(
                        f"shadow_student: expert={kind} route={s_route_t} template={s_tmpl_t} "
                        f"agree_with_teacher={agree_t}"
                    )
                else:
                    steps.append(
                        f"shadow_student: route={s_route_t} template={s_tmpl_t} "
                        f"agree_with_teacher={agree_t}"
                    )
            except Exception as e:  # noqa: BLE001
                steps.append(f"shadow_student: error={type(e).__name__}:{e}")

        if not self.enabled:
            return replace(out, reasoning_steps=steps)

        int8, lib = self._int8, self._lib
        if int8 is None or lib is None:
            return replace(out, reasoning_steps=steps)
        try:
            x = featurize_packet(packet)[None, :]
            lr, lt = forward_logits_c_n(int8, x, lib)
            ir = int(np.argmax(lr[0]))
            it = int(np.argmax(lt[0]))
            s_route = ROUTE_CLASSES[ir]
            s_tmpl = QUERY_TEMPLATE_IDS[it]
            agree = (s_route == out.route) and (s_tmpl == t_template)
            extra = (
                f"shadow_scm_micro_int8: route={s_route} template={s_tmpl} "
                f"agree_with_teacher={agree} path=c_forward_n"
            )
        except Exception as e:  # noqa: BLE001 - shadow must never break teacher path
            extra = f"shadow_scm_micro_int8: error={type(e).__name__}:{e}"
        steps.append(extra)
        return replace(out, reasoning_steps=steps)

#!/usr/bin/env python3
"""
Exercise ``get_scm_router()`` with ``SCM_TINY_EXPERT_DISPATCH=1`` on synthetic packets.

If ``SCM_TINY_NPZ`` / specialist paths are unset or missing, trains a minimal Tiny
``.npz`` from distillation rows into a temp dir (same artifact copied per slot).

Usage::

  cd NebulOS-Scaffolding
  PYTHONPATH=. python3 scripts/smoke_scm_tiny_expert_dispatch.py

Or point at real experts::

  SCM_TINY_EXPERT_DISPATCH=1 \\
  SCM_TINY_NPZ_WAVE=... SCM_TINY_NPZ_CWRU=... SCM_TINY_NPZ_UNSW=... SCM_TINY_NPZ=... \\
  PYTHONPATH=. python3 scripts/smoke_scm_tiny_expert_dispatch.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from experiments.scm_v0_1.contracts import ExecutionContract
from experiments.scm_v0_1.packets import SCMInputPacket
from experiments.scm_v0_1.router_entry import get_scm_router, parse_scm_tiny_shadow_line
from experiments.scm_v0_1.scm_tiny import (
    distillation_examples,
    featurize_packets,
    route_to_index,
    template_to_index,
    train_classifiers,
)
from experiments.scm_v0_1.scm_tiny.expert_dispatch import clear_predictor_cache


def _exists(p: str) -> bool:
    return bool(p) and Path(p).is_file()


def _ensure_npz_paths() -> tuple[str, str, str, str]:
    """Return (wave, cwru, unsw, fallback) paths that exist on disk."""
    w = (os.environ.get("SCM_TINY_NPZ_WAVE") or "").strip()
    c = (os.environ.get("SCM_TINY_NPZ_CWRU") or "").strip()
    u = (os.environ.get("SCM_TINY_NPZ_UNSW") or "").strip()
    fb = (os.environ.get("SCM_TINY_NPZ") or "").strip()
    if _exists(w) and _exists(c) and _exists(u) and _exists(fb):
        return w, c, u, fb

    rows = distillation_examples()
    packets = [r["packet"] for r in rows]
    X = featurize_packets(packets)
    y_r = np.array([route_to_index(r["route"]) for r in rows], dtype=np.int64)
    y_t = np.array([template_to_index(r["template_id"]) for r in rows], dtype=np.int64)
    art = train_classifiers(X, y_r, y_t, route_steps=400, template_steps=400)
    td = tempfile.mkdtemp(prefix="scm_tiny_dispatch_smoke_")
    base = Path(td) / "tiny.npz"
    art.save(str(base))
    wave = str(Path(td) / "wave.npz")
    cwru = str(Path(td) / "cwru.npz")
    unsw = str(Path(td) / "unsw.npz")
    fb = str(Path(td) / "fallback.npz")
    for dest in (wave, cwru, unsw, fb):
        shutil.copy2(base, dest)
    print(f"(trained minimal Tiny artifacts in {td})", file=sys.stderr)
    return wave, cwru, unsw, fb


def main() -> int:
    clear_predictor_cache()

    wave_p, cwru_p, unsw_p, fb_p = _ensure_npz_paths()

    keys = (
        "SCM_ROUTER_MODE",
        "SCM_MICRO_NPZ",
        "SCM_MICRO_INT8_LIB",
        "SCM_MICRO_INT8_SHADOW",
        "SCM_TINY_SHADOW",
        "SCM_TINY_EXPERT_DISPATCH",
        "SCM_TINY_NPZ_WAVE",
        "SCM_TINY_NPZ_CWRU",
        "SCM_TINY_NPZ_UNSW",
        "SCM_TINY_NPZ",
    )
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["SCM_ROUTER_MODE"] = "shadow"
        os.environ["SCM_MICRO_NPZ"] = ""
        os.environ["SCM_MICRO_INT8_LIB"] = ""
        os.environ["SCM_MICRO_INT8_SHADOW"] = "0"
        os.environ["SCM_TINY_SHADOW"] = "1"
        os.environ["SCM_TINY_EXPERT_DISPATCH"] = "1"
        os.environ["SCM_TINY_NPZ_WAVE"] = wave_p
        os.environ["SCM_TINY_NPZ_CWRU"] = cwru_p
        os.environ["SCM_TINY_NPZ_UNSW"] = unsw_p
        os.environ["SCM_TINY_NPZ"] = fb_p

        router = get_scm_router(ExecutionContract())
        tiny_ok = getattr(router, "tiny_shadow_enabled", False)
        print(f"tiny_shadow_enabled={tiny_ok}", file=sys.stderr)

        scenarios: list[tuple[str, SCMInputPacket]] = [
            (
                "wave",
                SCMInputPacket(
                    goal="ingest_system_log",
                    query="q",
                    wave_goal_metrics={"silicon_truth": 0.1},
                ),
            ),
            ("cwru", SCMInputPacket(goal="ingest_machine_state", query="probe")),
            ("unsw", SCMInputPacket(goal="ingest_system_log", query="net")),
            ("default", SCMInputPacket(goal="custom_routing_goal", query="x")),
        ]

        for label, pkt in scenarios:
            out = router.route(pkt)
            lines = [s for s in (out.reasoning_steps or []) if "shadow_student:" in (s or "")]
            if not lines:
                print(f"{label}: NO shadow_student line")
                continue
            raw = lines[0]
            parsed = parse_scm_tiny_shadow_line(raw)
            ek = parsed.get("expert_kind") if parsed else None
            print(f"{label}: {raw}")
            if ek:
                print(f"       parsed expert_kind={ek}")
    finally:
        clear_predictor_cache()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

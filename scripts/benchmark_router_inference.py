#!/usr/bin/env python3
"""Router inference throughput benchmark — rules / shadow / gated modes.

Measures per-packet wall time (ns) and throughput (packets/sec) for each
SCM router mode on a fixed synthetic packet corpus.

Usage:
  cd NebulOS-Scaffolding
  PYTHONPATH=. python3 scripts/benchmark_router_inference.py
  PYTHONPATH=. python3 scripts/benchmark_router_inference.py --packets 2000 --json-out /tmp/router_bench.json

Modes timed:
  rules   — RulesScmRouter only; no student shadow, no Tiny load
  shadow  — ShadowScmRouterMicroInt8 + Tiny shadow (SCM_TINY_NPZ); Micro INT8 disabled
  gated   — TemplateGatedScmRouter (C1b policy); Tiny shadow for template check
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.scm_v0_1.packets import SCMInputPacket  # noqa: E402
from experiments.scm_v0_1.router_entry import get_scm_router  # noqa: E402


_DEFAULT_NPZ = str(
    _ROOT / "analysis/formal_artifacts/scm_tiny/scm_tiny_mixed_unsw_wave_cwru.npz"
)

_GOALS = [
    "ingest_system_log",
    "ingest_machine_state",
    "custom_goal_42",
    "ingest_system_log",
]
_QUERIES = ["wave_q", "cwru_q", "default_q", "unsw_q"]


def _synthetic_packets(n: int, seed: int) -> List[SCMInputPacket]:
    rng = np.random.default_rng(seed)
    out: List[SCMInputPacket] = []
    for i in range(n):
        goal = _GOALS[i % len(_GOALS)]
        query = f"{_QUERIES[i % len(_QUERIES)]}_{rng.integers(0, 1 << 20)}"
        wm = {"silicon_truth": float(rng.random())} if i % 4 == 0 else {}
        out.append(SCMInputPacket(goal=goal, query=query, wave_goal_metrics=wm))
    return out


def _summ(name: str, arr: List[float]) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "name": name,
        "count": int(len(arr)),
        "min_ns": float(np.min(a)),
        "mean_ns": float(np.mean(a)),
        "median_ns": float(np.median(a)),
        "p99_ns": float(np.percentile(a, 99)),
        "max_ns": float(np.max(a)),
        "stdev_ns": float(statistics.stdev(arr)) if len(arr) > 1 else 0.0,
        "throughput_pps": float(1e9 / np.mean(a)) if np.mean(a) > 0 else 0.0,
    }


def _bench_mode(
    mode_name: str,
    packets: List[SCMInputPacket],
    warmup: int,
    npz: str,
) -> Dict:
    env_patch = {
        "SCM_MICRO_INT8_SHADOW": "0",
        "SCM_TINY_NPZ": npz,
        "SCM_TINY_SHADOW": "1",
        "SCM_ROUTER_MODE": mode_name if mode_name != "shadow" else "shadow",
    }
    saved = {k: os.environ.get(k) for k in env_patch}
    try:
        for k, v in env_patch.items():
            os.environ[k] = v

        router = get_scm_router()

        # Warmup
        for pkt in packets[:max(1, warmup)]:
            router.route(pkt)

        ns: List[float] = []
        for pkt in packets:
            t0 = time.perf_counter_ns()
            router.route(pkt)
            ns.append(float(time.perf_counter_ns() - t0))

    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return _summ(mode_name, ns)


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark SCM router inference throughput.")
    ap.add_argument("--packets", type=int, default=3000, metavar="N")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--npz",
        type=str,
        default=_DEFAULT_NPZ,
        help="SCM-Tiny .npz for shadow/gated modes (default: canonical mixed artifact)",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    npz = args.npz
    if not Path(npz).is_file():
        print(f"SCM-Tiny .npz not found: {npz}", file=sys.stderr)
        print("Run: make train-scm-tiny-mixed-unsw-wave-cwru", file=sys.stderr)
        return 1

    packets = _synthetic_packets(args.packets, args.seed)
    print(
        f"Benchmarking {args.packets} packets × 3 modes "
        f"(warmup={args.warmup}, seed={args.seed})",
        file=sys.stderr,
    )

    results = {}
    for mode in ("rules", "shadow", "gated"):
        print(f"  timing mode={mode!r} ...", file=sys.stderr, end="", flush=True)
        r = _bench_mode(mode, packets, args.warmup, npz)
        results[mode] = r
        pps = r["throughput_pps"]
        p50_us = r["median_ns"] / 1e3
        p99_us = r["p99_ns"] / 1e3
        print(f"  {pps:,.0f} pps  p50={p50_us:.1f}µs  p99={p99_us:.1f}µs", file=sys.stderr)

    summary = {
        "packets_per_mode": args.packets,
        "warmup": args.warmup,
        "seed": args.seed,
        "npz": npz,
        "modes": results,
        "note": (
            "shadow mode: Tiny shadow active, Micro INT8 disabled. "
            "gated mode: TemplateGatedScmRouter (C1b policy). "
            "Timings are Python-process wall time on this host."
        ),
    }

    print(json.dumps(summary, indent=2))

    # Human-readable table to stderr
    print("\nRouter inference benchmark — Jetson Orin Nano", file=sys.stderr)
    print(f"{'Mode':<10} {'pps':>10} {'p50 µs':>10} {'p99 µs':>10} {'mean µs':>10}", file=sys.stderr)
    print("-" * 52, file=sys.stderr)
    for mode, r in results.items():
        print(
            f"{mode:<10} {r['throughput_pps']:>10,.0f}"
            f" {r['median_ns']/1e3:>10.1f}"
            f" {r['p99_ns']/1e3:>10.1f}"
            f" {r['mean_ns']/1e3:>10.1f}",
            file=sys.stderr,
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

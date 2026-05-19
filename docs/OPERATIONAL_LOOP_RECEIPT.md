# Operational Loop Receipt

Verification record for `scripts/demo_operational_loop.py`.

## What this demonstrates

A continuously running local runtime that enforces behavioral boundaries on an event stream without external coordination, cloud dependency, or probabilistic models.

Three states:

| State | Condition | Action |
|-------|-----------|--------|
| `[RUN]` | In-domain, normal class, gate agrees | Continue |
| `[MITIGATE]` | In-domain, fault class detected, gate agrees | Log and continue |
| `[HALT]` | Any layer flags foreign/unfamiliar state | Stop immediately, preserve state |

HALT fires when **any** of:
- Retrieval similarity < 0.50 (foreign domain)
- Route diverges from baseline (structural mismatch)
- Behavioral gate disagrees (outcome outside training distribution)

No coordination between layers. Each reaches its conclusion independently.

## Run

```bash
# Default: 1000 events, breach at event 999
PYTHONPATH=. python3 scripts/demo_operational_loop.py

# Demo recording — suppress [RUN] lines, sample every 10th [MITIGATE]
PYTHONPATH=. python3 scripts/demo_operational_loop.py --count 1000 --quiet

# Short smoke run
PYTHONPATH=. python3 scripts/demo_operational_loop.py --count 50 --breach-at 40

# With JSONL receipt
PYTHONPATH=. python3 scripts/demo_operational_loop.py \
  --receipt receipts/latest_operational_loop.jsonl

# Resume: prior decisions reload into operational memory before the next run starts
PYTHONPATH=. python3 scripts/demo_operational_loop.py \
  --resume receipts/oploop_memory.avec --count 200
# First run creates the sidecar; subsequent runs replay it and append.

# H-IVF retrieval (faster per-event query)
PYTHONPATH=. python3 scripts/demo_operational_loop.py --hivf

# Dry-run (no native libs required; labeled simulation)
PYTHONPATH=. python3 scripts/demo_operational_loop.py --dry-run
```

Or via make:

```bash
make demo-operational-loop
```

## Expected output format

```
[RUN]        ID=01002 LAT=7174µs  sim=1.0010 route=mixed      gate=agree
[MITIGATE]   ID=01008 LAT=7313µs  sim=1.0007 route=mixed      gate=agree  known_fault=outer_014
[HALT]       ID=01026 LAT=6870µs  sim=0.0981 route=semantic   gate=mismatch
```

After HALT:

```
  WAL committed         26 notable events committed, 3/3 sampled read back — lattice intact
  retrieval             low-similarity — 0.0981 < 0.5 threshold
  route divergence      yes — semantic != mixed
  behavioral gate       mismatch — outcome outside training distribution
```

Ends with `DONE` only if halt was safe and lattice continuity verified.

## Reference numbers (Jetson Orin Nano, aarch64 / NEON, bruteforce)

| Metric | Value |
|--------|-------|
| p50 latency | ~7 ms (Python + ctypes overhead; C library inference is ~176 µs) |
| p95 latency | ~7.4 ms |
| p99 latency | ~7.6 ms |
| AION insert rate | ~80,000 vec/s |
| Lattice writes | 3 µs p50 |
| WAL continuity | 3/3 nodes read back after halt |

Latency is dominated by Python interpreter and ctypes call boundaries around the C library. The C inference path (`libsynrix.so` + `libaion_semantic_index.so`) runs at ~176 µs. See `docs/BENCHMARK_RECEIPTS.md`.

## Claim tiers

**Tier A (proven in this repo):**
- Three-layer independent breach detection on foreign domain input
- WAL/lattice continuity verified by node read-back after halt
- MITIGATE state fires on known fault classes without triggering halt
- Deterministic replay with `--seed`

**Tier B (shown on device video, source proprietary):**
- 41,841 WAVE silicon PMU signals as the live corpus
- Phi encoder + Sentry + harness hotswap path

**Tier C (roadmap):**
- 24h continuous run receipt (see D4 in GO_LIVE_SPEC.md)
- Fleet-level promotion lifecycle

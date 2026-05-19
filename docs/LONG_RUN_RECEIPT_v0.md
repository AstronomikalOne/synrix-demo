# Long-Run Receipt v0

500,000-event stability run. Jetson Orin Nano, aarch64 / NEON, 2026-05-18.

## Run command

```bash
nohup env PYTHONPATH=. SYNRIX_LIB_PATH=build python3 -u scripts/demo_operational_loop.py \
  --count 500_000 \
  --seed 42 \
  --resume receipts/oploop_memory.avec \
  --receipt receipts/latest_operational_loop.jsonl \
  > receipts/oploop_stab.log 2>&1 &
```

## Final summary

```
  WAL committed         143655 events — lattice (symbolic) + sidecar (semantic), 3/3 read back — intact
  retrieval             low-similarity — 0.0981 < 0.5 threshold
  route divergence      yes — semantic != mixed
  behavioral gate       mismatch — outcome outside training distribution

  Events    total=500000  run=359864  mitigate=140135  halt=1
  Latency   p50=7990µs  p95=8256µs  p99=8561µs
  Retrieval bruteforce (94,795 vectors)
  Expert    scm_tiny_cwru_expert.npz
```

## State distribution

| State | Count | Fraction |
|-------|-------|----------|
| RUN | 359,864 | 71.97% |
| MITIGATE | 140,135 | 28.03% |
| HALT | 1 | breach at event 501,000 (by design) |

Design target: 72% RUN / 28% MITIGATE (`_NORMAL_FRAC=0.72`). Actual: **71.97% / 28.03%**.

## Latency curve (p50 per 1k-event window, sampled every 50k events)

| Events | Elapsed | p50 | RSS |
|--------|---------|-----|-----|
| 1,000 | 0.00h | 6,818 µs | 310 MB |
| 51,000 | 0.10h | 6,746 µs | 324 MB |
| 101,000 | 0.20h | 6,814 µs | 341 MB |
| 151,000 | 0.30h | 6,814 µs | 358 MB |
| 201,000 | 0.42h | 7,208 µs | 374 MB |
| 251,000 | 0.52h | 7,467 µs | 391 MB |
| 301,000 | 0.63h | 7,970 µs | 408 MB |
| 351,000 | 0.77h | 7,962 µs | 425 MB |
| 401,000 | 0.88h | 7,118 µs | 441 MB |
| 451,000 | 0.99h | 7,103 µs | 458 MB |
| 500,000 | 1.09h | 7,990 µs | 475 MB |

p50 oscillates between 6.7–8.1 ms across the full run with no monotonic drift. Variation is periodic — lattice auto-grow events (65k→131k→262k nodes) cause brief page-fault bursts, then settle. RSS grows linearly (~330 KB per 1k events) as expected for lattice node allocation on a run writing every 100th event plus all MITIGATE events.

## WAL and sidecar continuity

- **143,655 behavioral events** written to `receipts/oploop_memory.avec` (sidecar, semantic layer)
- **3/3 sampled node IDs** (first, mid, last) read back from lattice after halt — symbolic layer intact
- Sidecar format: 32-byte header + 2,056 bytes/row (`uint32 node_id + uint8 valid + uint8[3] pad + float[512]`)
- Same `node_id` links symbolic lattice entry (name + data) to semantic vector on disk

## Breach behavior

HALT fires at event 501,000 (injected breach, `--breach-at` default). All three layers agree independently:

| Layer | Signal |
|-------|--------|
| Retrieval | sim=0.0981 < 0.50 threshold (foreign domain) |
| Router | semantic != mixed (route divergence) |
| Gate | mismatch — outcome outside training distribution |

## Claim tier

**Tier A (proven in this repo, reproducible from fresh clone):**
- 500,000 in-domain events processed without false positive halt
- State distribution matches design within 0.03%
- p50 latency stable — no monotonic drift over 1.09h
- Three-layer independent breach detection fires correctly on injected foreign input
- WAL/lattice continuity verified by node read-back after halt
- Behavioral memory (sidecar) persists to disk; `--resume` replays prior events into operational index on next run
- MITIGATE fires on known fault classes without triggering halt

**Tier B (shown on device video, source proprietary):**
- 41,841 WAVE silicon PMU signals as live corpus
- Phi encoder + Sentry + harness hotswap path

**Tier C (roadmap):**
- Fleet-level promotion lifecycle

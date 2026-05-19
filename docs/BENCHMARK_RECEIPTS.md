# Benchmark Receipts

All measurements from Jetson Orin Nano (8-core Cortex-A78AE, 8 GB LPDDR5, aarch64).

---

## Triple Behavioral Equivalence Gate

**Setup:** Canonical softmax-linear student (C-trained, `liblattice_expert_train.so`, seeds 7/11) vs deterministic rule-based teacher. Criterion: student must agree with teacher on route AND template for every packet.

| Gate | Domain | n | Route | Template |
|---|---|---|---|---|
| Gate 1 | UNSW-NB15 network traffic | 4 (demo) / full holdout | 1.000 | 1.000 |
| Gate 2 | CWRU bearing fault signals | 4 (demo) / full holdout | 1.000 | 1.000 |
| Gate 3 | WAVE silicon PMU data | 4 (demo) / full holdout | 1.000 | 1.000 |

Demo gate: 12 packets from actual training-distribution JSONL (not synthetic).  
Full gate: thousands of disjoint holdout rows per domain.

---

## Router Inference Throughput

**Setup:** 2000 packets per mode, 100-packet warmup, seed=42 synthetic packets.

| Mode | pps | p50 | p99 |
|---|---|---|---|
| rules | 52,964 | 19 µs | 26 µs |
| gated | 5,189 | 177 µs | 207 µs |
| shadow | 3,421 | 271 µs | 316 µs |

**Rules:** Deterministic rule-based routing only.  
**Gated:** Rules + softmax-linear template policy check (C-trained head).  
**Shadow:** Rules + softmax-linear student in observer mode (no execution change).

Live demo runs will vary ±5% from the above.

---

## Expert Library Holdout Gates

Per-domain specialists trained on 8,000 disjoint rows, gated on 2,000 disjoint rows (no overlap with training).

| Domain | Train n | Holdout n | Route | Template |
|---|---|---|---|---|
| CWRU | 8,000 | 2,000 | 1.000 | 1.000 |
| WAVE | 8,000 | 2,000 | 1.000 | 1.000 |
| UNSW | holdout-random corpus | 2,000 | 1.000 | 1.000 |

---

## Lattice Write Throughput

**Setup:** 500,000-node persistent lattice, `MAP_SHARED` + seqlock, Jetson Orin Nano NVMe.

| Metric | Value |
|---|---|
| Write p50 | 73 µs |
| Write p99 | ~200 µs |
| SQLite equivalent | ~62 ms (850× slower at p50) |

---

## AION512 Semantic Index

**Setup:** 94,795 CWRU bearing vectors (512-float L2-normalized embeddings, NEON SDOT kernel). The private WAVE corpus (41,841 vectors) produces similar QPS at this scale.

| Metric | Value |
|---|---|
| QPS (aarch64 NEON) | 165,000 |
| QPS (x86 scalar fallback) | ~20,000–40,000 (hardware-dependent) |

---

## Operational Loop — 500k-event stability run

**Setup:** `demo_operational_loop.py --count 500_000 --seed 42`, Jetson Orin Nano, aarch64/NEON, bruteforce retrieval over 94,795 CWRU vectors. See `docs/LONG_RUN_RECEIPT_v0.md` for full receipt.

| Metric | Value |
|---|---|
| Total events | 500,000 |
| Runtime | 1.09 h |
| p50 latency | 7,990 µs |
| p95 latency | 8,256 µs |
| p99 latency | 8,561 µs |
| RUN (in-domain, normal) | 359,864 (71.97%) |
| MITIGATE (in-domain, fault) | 140,135 (28.03%) |
| HALT (false positive) | 0 |
| HALT (injected breach) | 1 — fires correctly |
| WAL continuity | 3/3 nodes read back after halt |
| Behavioral memory | 143,655 events on disk (sidecar) |

p50 was flat across the full 1.09h run with no monotonic drift. State distribution matches design target (72%/28%) within 0.03%.

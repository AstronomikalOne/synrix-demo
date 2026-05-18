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

**Setup:** 41,841 WAVE vectors, 512-float L2-normalized embeddings, NEON SDOT kernel.

| Metric | Value |
|---|---|
| QPS (aarch64 NEON) | 165,000 |
| QPS (x86 scalar fallback) | ~20,000–40,000 (hardware-dependent) |

# Synrix — Edge Anomaly Routing Demo

A reproducibility artifact for domain-bound anomaly detection on embedded hardware: public bearing data corpus, AION512 vector retrieval, deterministic routing, and an 11 KB behavioral gate. Runs on a $250 Jetson Orin Nano with no cloud dependency.

Includes a behavioral equivalence gate, router throughput benchmark, and interactive web demo.

## Demo video (two parts)

| Part | Where to record | In this repo? |
|------|-----------------|---------------|
| **A — Edge stack** (gate, e2e, cross-domain catch) | **This repo** — `make demo-screen-record` | Yes — fully reproducible |
| **B — IP stack** (WAVE 41k, phi encoders, Sentry/harness hotswap) | Private `aion-omega` only | Video + log in [`media/`](media/README.md); **no C source** |

See [`docs/DEMO_SPLIT.md`](docs/DEMO_SPLIT.md) and [`media/README.md`](media/README.md) for file names and upload notes.

## Quick start

### Docker (recommended)

**Step 1 — open a terminal and navigate into the repo folder:**

```bash
cd synrix-demo        # or wherever you cloned/unzipped it
```

**Step 2 — build the image (the `.` at the end is required — it tells Docker where the code is):**

```bash
# Linux / macOS
docker build -t synrix-gate .

# Windows (Command Prompt or PowerShell) — no sudo
docker build -t synrix-gate .
```

> **Linux note:** If you get a permission error, prefix with `sudo`, or add your user to the `docker` group.

**Step 3 — run:**

```bash
docker run --rm synrix-gate
```

**Common mistake:** Running `docker build -t synrix-gate` without the `.` gives `requires 1 argument` — the dot is the build path, not punctuation.

```bash
# End-to-end edge inference pipeline: lattice → AION512 → routing → behavioral gate
docker run --rm synrix-gate python3 scripts/demo_e2e_pipeline.py
```

### Bare-metal (no Docker)

```bash
# Copy pre-built native libs for your architecture into build/
make setup

# Gate benchmark
PYTHONPATH=. python3 scripts/demo_synrix_gate.py

# E2E pipeline — downloads CWRU dataset (~134 MB) and builds vector index first
make setup-corpus
PYTHONPATH=. python3 scripts/demo_e2e_pipeline.py
```

### Gate benchmark expected output

```
[PASS] Gate 1 (UNSW-domain)   n=4  route=1.000  tmpl=1.000
[PASS] Gate 2 (CWRU-domain)   n=4  route=1.000  tmpl=1.000
[PASS] Gate 3 (WAVE silicon)  n=4  route=1.000  tmpl=1.000

Triple gate result: 3/3  ✓

  Mode        pps     p50 µs   p99 µs
  rules    <live>    <live>    <live>
  gated    <live>    <live>    <live>
  shadow   <live>    <live>    <live>

Canonical gate path: VERIFIED on this hardware.
```

Throughput numbers are measured live on your hardware and will vary. See `docs/BENCHMARK_RECEIPTS.md` for reference numbers from the Jetson Orin Nano.

### Interactive web demo (live anomaly detection)

A browser-based demo where you send different types of sensor readings and watch the
system process each one through four independent layers in real time.

```bash
# Docker (recommended — handles all dependencies)
docker run --rm -p 5050:5050 synrix-gate python3 scripts/demo_interactive.py
# Then open: http://localhost:5050

# Bare-metal (requires make setup + make setup-corpus first)
make run-interactive
# Then open: http://localhost:5050
```

Three buttons let you send a reading:
- **Normal Bearing** — healthy vibration signal from an industrial motor
- **Bearing Fault** — inner race crack (still a bearing signal, correctly handled)
- **Silicon Chip PMU** — CPU performance counters (wrong domain entirely)

Each reading passes through four layers:
1. **Store it** — written to on-device persistent memory in microseconds
2. **Recognize it** — searched against 94,795 historical bearing signals
3. **Classify it** — rule engine assigns a processing class
4. **Verify it** — an 11 KB model trained only on bearing data checks the decision

Bearing readings (both normal and fault) show all green across all four layers.
The PMU reading triggers layers 2, 3, and 4: similarity drops to ~9.8%, the
rule engine assigns it a different class, and the learned model has never seen
this outcome in bearing data — three independent flags, no coordination between them.

Startup takes ~15 seconds while 94,795 vectors are loaded into AION512.

> **Note on latency:** The per-layer timings shown in the UI are C-library inference measurements only. End-to-end request latency includes Python HTTP handling and browser round-trip overhead not present in a production deployment. See `docs/BENCHMARK_RECEIPTS.md` for production throughput numbers from the Jetson Orin Nano.

> **Python vs the actual system:** Every demo in this repo is orchestrated from Python for readability. The system itself is the C libraries (`libsynrix.so`, `libaion_semantic_index.so`, `liblattice_expert_train.so`). Python is the wrapper. In production, those libraries are called directly from C — the ~176us figure in the benchmark receipts is the real system latency. The millisecond-range numbers you see in demo output are Python interpreter overhead, ctypes call boundaries, and NumPy operations layered on top of sub-millisecond C inference.

---

### Operational loop — continuous behavioral enforcement

The flagship demo. Runs 1000 events through the full stack at 7ms each, enforcing behavioral boundaries in real time. MITIGATE fires on known faults without stopping. HALT fires when all three independent layers agree the input is foreign. After halt, WAL continuity is proven by node read-back.

```bash
# Bare-metal (Jetson — full NEON path, ~7ms p50)
PYTHONPATH=. SYNRIX_LIB_PATH=build python3 scripts/demo_operational_loop.py --count 1000 --quiet

# Docker (x86_64 scalar path, ~15ms p50)
docker run --rm synrix-gate python3 scripts/demo_operational_loop.py --count 1000 --quiet

# Dry-run (no native libs required)
docker run --rm synrix-gate python3 scripts/demo_operational_loop.py --dry-run
```

`--quiet` suppresses `[RUN]` lines and samples every 10th `[MITIGATE]` — recommended for demo recording. Remove it to see every event.

Expected output (quiet mode):

```
[MITIGATE]   ID=01002 LAT=7166µs  sim=1.0004 route=mixed      gate=agree  known_fault=outer_021
[MITIGATE]   ID=01046 LAT=7446µs  sim=1.0006 route=mixed      gate=agree  known_fault=inner_014
...
[CHKPT]      events=0001000 elapsed=0.00h  p50_1k=6956µs  rss=306MB  run=719  mit=280
[HALT]       ID=02000 LAT=6608µs  sim=0.0981 route=semantic   gate=mismatch

  WAL committed         288 events — lattice (symbolic) + sidecar (semantic), 3/3 read back — intact
  retrieval             low-similarity — 0.0981 < 0.5 threshold
  route divergence      yes — semantic != mixed
  behavioral gate       mismatch — outcome outside training distribution

  Events    total=1000  run=719  mitigate=280  halt=1
  Latency   p50=6956µs  p95=7336µs  p99=7446µs
  Retrieval bruteforce (94,795 vectors)
```

500k-event stability receipt: see `docs/LONG_RUN_RECEIPT_v0.md` and `docs/BENCHMARK_RECEIPTS.md`.

---

### Autonomous agent loop (terminal stream)

A terminal demo showing Synrix as an agent substrate — streams 1000 events through the full stack (lattice write → AION512 search → SCM routing → behavioral gate), then halts and freezes state when a foreign reading breaches the domain manifold.

```bash
# Docker — bruteforce retrieval (~7ms/query, no IVF build overhead)
docker run --rm -v $(pwd)/analysis:/app/analysis synrix-gate \
  python3 scripts/demo_autonomous_loop.py --count 1000

# Docker — paged H-IVF retrieval (~1.4ms/query, instant startup from pre-built index)
docker run --rm -v $(pwd)/analysis:/app/analysis synrix-gate \
  python3 scripts/demo_autonomous_loop.py --count 1000 --hivf

# Docker — flat IVF retrieval (~1.5ms/query after ~2min IVF build)
docker run --rm -v $(pwd)/analysis:/app/analysis synrix-gate \
  python3 scripts/demo_autonomous_loop.py --count 1000 --ivf

# Bare-metal
PYTHONPATH=. python3 scripts/demo_autonomous_loop.py --count 1000
PYTHONPATH=. python3 scripts/demo_autonomous_loop.py --count 1000 --hivf
PYTHONPATH=. python3 scripts/demo_autonomous_loop.py --count 1000 --ivf
```

Expected output (bruteforce):

```
[OK]    ID: 09001 | LAT: 6798us | L1 lattice match
[OK]    ID: 09002 | LAT: 6903us | L1 lattice match
[WARN]  ID: 09008 | LAT: 6667us | L2 manifold check -> fault:outer_014
...
[CRITICAL] ID: 09999 | Manifold breach -- 3 independent layers flagged
           Layer 1: cosine displacement 0.9019 -- out of learned space
           Layer 2: execution class mismatch
           Layer 3: behavioral policy divergence

[HALT]  Loop frozen. WAL committed. fdatasync verified. State intact.

  p50 latency : 6798us   (AION512 bruteforce over 94,795 vectors)
  p95 latency : 7129us
```

With `--hivf`: p50 drops to ~1.4ms with **instant startup** — opens the pre-built paged H-IVF index
(`cwru_ivf.ivfp`, included in the Docker image) in <1ms; no K-means build required. Probes all
16 branches, top 8 leaves per branch (128 of 320 total leaf buckets per query).

With `--ivf`: p50 drops to ~1.5ms after a ~2-minute K-means build (320 clusters, 20 probes).

---

### End-to-end pipeline expected output

```
  Step 1 — Persistent lattice: write 100 bearing sensor records (10 × 10 fault classes)
  [OK]   100 nodes written across 10 fault classes  |  Write p50: <live> µs

  Step 2 — AION512: index <N> bearing vectors, load pre-built H-IVF
  [OK]   <N> vectors added in <live> ms  |  H-IVF loaded in <live> ms

  Step 3 — New reading arrives: silicon PMU data (wrong domain)
  [WARN] Best match: <score> — PMU reading has low similarity to all <N> bearing records

  Step 4 — SCM routing: WAVE_PMU_READING routes to 'semantic'  ← different!

  Step 5 — Behavioral gate: WAVE_PMU_READING  ✗ MISMATCH  ← gate catches it!
```

Throughput numbers are from a Jetson Orin Nano (aarch64 / NEON). x86 will be lower — the dispatch banner tells you which path is active.

---

## What this is

A self-contained reproducibility artifact for a paper on edge inference routing. It demonstrates:

- **Behavioral equivalence gate** — a C-trained softmax-linear student matches a deterministic rule-based teacher on 1.000/1.000 across three real-world domains (UNSW-NB15 network traffic, CWRU bearing fault signals, WAVE silicon PMU data)
- **Router throughput** — three routing modes (rules / gated / shadow) measured on real hardware
- **Cache-pressure resilience** — batch inference over a 50k-vector synthetic stress corpus (~52 MB, exceeds L3; separate from the 94,795-vector CWRU bearing corpus used in the interactive demo)

## What this is not

- The full Synrix research platform
- A general-purpose database or vector store
- A product release

**What is in this repo:**

| Component | Status |
|---|---|
| `libsynrix.so` — persistent knowledge lattice | Pre-built binary; C source is proprietary |
| `libaion_semantic_index.so` — AION512 vector index | Pre-built binary; C source is proprietary |
| `liblattice_expert_train.so` — C softmax-linear trainer | Pre-built binary; C source is proprietary |
| SCM routing module (`experiments/scm_v0_1/`) | Python source, included |
| Pre-trained model weights (12 KB `.npz` files) | Included |
| CWRU public bearing corpus (94,795 vectors) | Downloaded on first run via `make setup-corpus` |

**What is NOT in this repo:**

- φ/PSS probe subsystem — proprietary, not included
- Live WAVE PMU collection pipeline — proprietary, not included
- C source for any native library — proprietary, not included
- Full training corpus — only the pre-trained weights are shipped

---

## Hardware dispatch

The image builds C libraries from source — it will produce the correct binary for whatever architecture you build on.

```
# aarch64 (Jetson Orin Nano, Raspberry Pi 5, etc.)
[DISPATCH] arch=aarch64    kernel=NEON_SDOT [ACTIVE]

# x86_64 (any Intel/AMD desktop or CI runner)
[DISPATCH] arch=x86_64     kernel=SCALAR_CPP [WARNING: paper claims require aarch64 / NEON]
```

Paper throughput claims are from aarch64 with NEON SDOT. The x86 scalar fallback is correct but slower.

---

## Optional modes

```bash
# Cache-pressure stress — 50k synthetic vectors, ~52 MB matrix, exceeds L3
docker run --rm synrix-gate python3 scripts/demo_synrix_gate.py --stress-scale 50000

# Show C training pipeline (trains fresh from 65 built-in examples, then gates with canonical)
docker run --rm synrix-gate python3 scripts/demo_synrix_gate.py --train-fresh

# Expert Library dispatch — routes each packet to its domain specialist
docker run --rm -e SCM_TINY_EXPERT_DISPATCH=1 synrix-gate \
  python3 scripts/smoke_scm_tiny_expert_dispatch.py
```

Or via make:

```bash
make build
make run
make run-stress
make run-expert
make run-fresh
```

---

## Artifacts

| File | Description |
|---|---|
| `analysis/formal_artifacts/scm_tiny/scm_tiny_mixed_unsw_wave_cwru_cpath.npz` | Canonical C-trained artifact (seeds 7/11, mixed corpus) |
| `analysis/formal_artifacts/scm_tiny/scm_tiny_cwru_expert.npz` | CWRU domain expert (1.000/1.000 on 2k holdout) |
| `analysis/formal_artifacts/scm_tiny/scm_tiny_wave_expert.npz` | WAVE domain expert (1.000/1.000 on 2k holdout) |
| `analysis/formal_artifacts/scm_tiny/scm_tiny_unsw_heldout_random_train.npz` | UNSW domain expert (1.000/1.000 on 2k holdout) |
| `analysis/formal_artifacts/scm_tiny/demo_gate_fixture.json` | 12 gate packets (4 per domain, from real training-distribution JSONL) |

Artifacts are pure weights (12 KB each). No corpus data is included.

---

## Benchmarks

All numbers are measured live from your hardware when you run the demo. Reference numbers from the Jetson Orin Nano (aarch64 / NEON) are in [`docs/BENCHMARK_RECEIPTS.md`](docs/BENCHMARK_RECEIPTS.md).

---

## Layout

```
scripts/                Demo and benchmark scripts
experiments/scm_v0_1/  SCM routing module (Python)
analysis/               Pre-trained artifacts, gate fixture, and CWRU bearing corpus
lib/                    Pre-built native libraries (linux-aarch64, linux-x86_64)
docs/                   Architecture notes and benchmark receipts
```

The C source for all native libraries (`libsynrix.so`, `libaion_semantic_index.so`, `liblattice_expert_train.so`) is proprietary and not included in this repository. Pre-built binaries are provided in `lib/`.

---

## Limitations

- Gate fixture is 12 packets (4 per domain). The paper's full gate uses thousands of real holdout rows per domain. This demo verifies the claim on a sampled subset.
- Throughput numbers are single-run, 2000-packet measurements. See `docs/BENCHMARK_RECEIPTS.md` for methodology.
- The stress QPS number (`--stress-scale`) is a single sample and varies with thermal load — it demonstrates the path executes under memory pressure, not a stable throughput figure.
- The x86 scalar fallback is correct but will not reproduce the NEON throughput numbers.

---

## Platform support

| Platform | Gate demo | E2E pipeline |
|---|---|---|
| `linux-aarch64` (Jetson, Raspberry Pi 5) | ✓ | ✓ |
| `linux-x86_64` | ✓ | ✓ (scalar path; see `docs/BUILDING.md`) |
| `darwin-arm64`, `win32-x86_64` | contact for access | contact for access |

See [`docs/BUILDING.md`](docs/BUILDING.md) for details.

---

## License

Non-commercial research and evaluation use only. The native libraries are
provided as pre-built binaries — their source is proprietary. See [LICENSE](LICENSE).

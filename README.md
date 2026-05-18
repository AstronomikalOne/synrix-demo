# Synrix — Public Reproducibility Artifact

Behavioral equivalence gate + router throughput demo for the Synrix edge inference stack.

## Quick start

### Docker (recommended)

```bash
sudo docker build -t synrix-gate .

# Behavioral equivalence gate + router benchmark (default)
sudo docker run --rm synrix-gate

# End-to-end edge inference pipeline: lattice → AION512 → routing → behavioral gate
sudo docker run --rm synrix-gate python3 scripts/demo_e2e_pipeline.py
```

> **Note:** `sudo` is required on most Linux systems unless your user is in the `docker` group. On Mac/Windows with Docker Desktop, omit `sudo`.

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
system analyze each one through three independent detection layers in real time.

```bash
# Bare-metal (requires make setup + make setup-corpus first)
make run-interactive
# Then open: http://localhost:5050

# Docker
docker run --rm -p 5050:5050 synrix-gate python3 scripts/demo_interactive.py
# Then open: http://localhost:5050
```

Three buttons let you send:
- **Normal Bearing** — healthy vibration signal from an industrial motor
- **Bearing Fault** — inner race crack (still a bearing signal, correctly handled)
- **Silicon PMU** — CPU performance counters (wrong domain entirely)

The bearing readings show all green — high similarity to the 94,795-vector corpus,
expected routing class, gate agrees. The PMU reading triggers all three layers:
similarity collapses to ~0.098, route changes to a different execution class, and
the 11 KB learned model — trained only on bearing data — disagrees with the teacher.

Startup takes ~15 seconds while 94,795 vectors are loaded into AION512.

> **Note on latency:** The per-layer timings shown in the UI are C-library inference measurements only. End-to-end request latency includes Python HTTP handling and browser round-trip overhead not present in a production deployment. See `docs/BENCHMARK_RECEIPTS.md` for production throughput numbers from the Jetson Orin Nano.

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

Throughput numbers are from a Jetson Orin Nano (aarch64 / NEON). x86 will be lower — the dispatch banner tells you which path is active. The e2e pipeline requires `libaion_semantic_index.so` — currently included for `linux-aarch64` only (see Platform support).

---

## What this is

A self-contained reproducibility artifact for a paper on edge inference routing. It demonstrates:

- **Behavioral equivalence gate** — a C-trained softmax-linear student matches a deterministic rule-based teacher on 1.000/1.000 across three real-world domains (UNSW-NB15 network traffic, CWRU bearing fault signals, WAVE silicon PMU data)
- **Router throughput** — three routing modes (rules / gated / shadow) measured on real hardware
- **Cache-pressure resilience** — batch inference over a 50k-vector, ~52 MB synthetic corpus (exceeds L3)

## What this is not

- The full Synrix research platform
- A general-purpose database or vector store
- A product release

The full platform includes a persistent knowledge lattice, AION512 semantic index, and φ/PSS probe subsystem. This repo contains only what is needed to verify the paper's routing and equivalence claims.

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

The C source for `libsynrix.so` and `liblattice_expert_train.so` is proprietary and not included in this repository. Pre-built binaries are provided in `lib/`.

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
| `linux-x86_64` | ✓ | — (`libaion_semantic_index.so` not yet bundled) |
| `darwin-arm64`, `win32-x86_64` | contact for access | contact for access |

See [`docs/BUILDING.md`](docs/BUILDING.md) for details.

---

## License

Non-commercial research and evaluation use only. The native libraries are
provided as pre-built binaries — their source is proprietary. See [LICENSE](LICENSE).

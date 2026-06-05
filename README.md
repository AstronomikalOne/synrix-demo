# Synrix — Runtime Trust Infrastructure for Embedded Hardware

On-device behavioral enforcement for edge AI. No cloud. No network dependency.
Ships as a 12 KB model + two native libraries that run on a $250 Jetson Orin Nano.

Three independent enforcement layers fire simultaneously when an input is foreign to the device's learned domain — retrieval similarity collapse, routing class divergence, and behavioral gate mismatch. When all three agree, execution halts and a structured CRA Article 14 incident report is generated automatically.

Stable to 500,000 events with flat p50 latency. Demonstrated on real silicon.

---

## What it does

```
Input arrives
    │
    ├─ Layer 1: AION512 retrieval — cosine similarity vs 94,795 stored vectors
    │           sim=0.098 → LOW  (threshold 0.50)
    │
    ├─ Layer 2: SCM routing — deterministic rule teacher + learned student shadow
    │           teacher=semantic, student=mixed → DIVERGENCE
    │
    └─ Layer 3: Behavioral gate — 11 KB model trained only on in-domain outputs
                gate=mismatch → OUTSIDE TRAINING DISTRIBUTION

    → Three independent flags. No coordination. HALT + incident report generated.
```

Normal bearing signals: all three layers agree, execution continues.
Foreign signals (wrong domain, fault class, or adversarial input): all three diverge.

---

## Quick start

### Docker (any platform)

```bash
docker build -t synrix-gate .
docker run --rm synrix-gate
```

### Bare-metal (Jetson / Linux x86_64)

```bash
make setup          # copies pre-built native libs for your arch
make setup-corpus   # downloads CWRU bearing dataset (~134 MB, one time)
PYTHONPATH=. SYNRIX_LIB_PATH=build python3 scripts/demo_synrix_gate.py
```

### Python wheel (requires pre-built libs)

```bash
pip install synrix-kernel
python -m synrix doctor --runtime
```

---

## Demos

### Behavioral gate (triple domain verification)

Verifies the learned student model matches the deterministic rule teacher at 1.000/1.000 across three independent real-world domains:

```bash
docker run --rm synrix-gate python3 scripts/demo_synrix_gate.py
```

Expected:
```
[PASS] Gate 1 (UNSW-domain)   n=4  route=1.000  tmpl=1.000
[PASS] Gate 2 (CWRU-domain)   n=4  route=1.000  tmpl=1.000
[PASS] Gate 3 (WAVE silicon)  n=4  route=1.000  tmpl=1.000

Triple gate result: 3/3  ✓
```

### Operational loop — continuous enforcement

1,000 events through the full stack at ~7ms each on Jetson (NEON path). MITIGATE fires on known bearing faults. HALT fires when all three layers flag a foreign input simultaneously.

```bash
# Jetson bare-metal (~7ms p50, NEON path)
PYTHONPATH=. SYNRIX_LIB_PATH=build python3 scripts/demo_operational_loop.py --count 1000 --quiet

# Docker (~15ms p50, scalar path)
docker run --rm synrix-gate python3 scripts/demo_operational_loop.py --count 1000 --quiet
```

Expected output (quiet mode):
```
[MITIGATE]   ID=01002 LAT=7166µs  sim=1.0004 route=mixed  gate=agree  known_fault=outer_021
[MITIGATE]   ID=01046 LAT=7446µs  sim=1.0006 route=mixed  gate=agree  known_fault=inner_014
...
[CHKPT]      events=0001000 elapsed=0.00h  p50_1k=6956µs  rss=306MB
[HALT]       ID=02000 LAT=6608µs  sim=0.0981 route=semantic  gate=mismatch

  retrieval      sim=0.0981 < 0.50 threshold
  route          divergence: teacher=semantic student=mixed
  gate           mismatch — outside training distribution

  Events    total=1000  run=719  mitigate=280  halt=1
  Latency   p50=6956µs  p95=7336µs  p99=7446µs
```

500,000-event stability receipt (1.09h flat latency): [`docs/LONG_RUN_RECEIPT_v0.md`](docs/LONG_RUN_RECEIPT_v0.md)

### Interactive web demo

```bash
docker run --rm -p 5050:5050 synrix-gate python3 scripts/demo_interactive.py
# open http://localhost:5050
```

Send normal bearing, bearing fault, and silicon PMU readings. Watch all four
layers process each in real time. The PMU reading triggers all three enforcement
layers — three independent flags, no coordination.

### End-to-end pipeline

```bash
docker run --rm synrix-gate python3 scripts/demo_e2e_pipeline.py
```

Writes 100 bearing records to the persistent lattice, builds a vector index,
then processes a foreign PMU reading through all enforcement layers.

### PHI optimization transfer — optimizer that recognizes

Three-phase demo of PHI's optimization memory system. Works from any clone with no binary or lattice required.

```bash
python3 scripts/demo_phi_transfer.py
```

Expected output:

```
Act 1 — Memory: what PHI stored
  PHIFP:ggml_vec_dot_q8_0_q8_0  dead@30   1.53×  certified
  PHIFP:ggml_vec_dot_q4_0_q8_0  swap@1,2  1.33×  sound
  PHIFP:ggml_vec_dot_q4_K_q8_K  dead@0    1.27×  certified
  PHIFP:ggml_vec_dot_q5_K_q8_K  dead@29   1.44×  certified

  ✓  Lattice stores the mutation descriptor. Retrieval is sub-millisecond.

Act 2 — Transfer gate: does the mutation apply to a new binary?
  expected  c31ca64e  →  actual  c21ca64e
  △  MISS (exact) — byte differs; trying register-normalized comparison…

  Mask bits[4:0] (Rd — output register field):
    expected masked  0x4ea61cc0
    actual   masked  0x4ea61cc0  ← MATCH

  ✓  TRANSFER_CANDIDATE — Rd-rename only (L1)
  ✓  2/4 correct promotions. 2/4 correct rejections.

Act 3 — Recognition: identify by behavior profile, not by name
  func_offset  0xca68  (102 instructions, no symbol provided)

  ANN search results:
    [1]  PHIFP:ggml_vec_dot_q8_0_q8_0    0.9877    1.53×
    [2]  PHIFP:ggml_vec_dot_q4_0_q8_0    0.9877    1.33×
    [3]  PHIFP:ggml_vec_dot_q4_K_q8_K    0.9426    1.27×
    [4]  PHIFP:ggml_vec_dot_q5_K_q8_K    0.9311    1.44×

  ✓  PHI recognized the function family — no symbol name used.
  △  Gate correctly refused: opcode class changed in this variant.
     Warm-start returned: try dead@30 in targeted re-search.
```

**Claim:** given an unnamed binary region, PHI retrieves prior optimization candidates from behavioral similarity rather than symbol identity.

The MISS after retrieval is correct behavior. PHI recognized the function family and refused to apply a mutation across an opcode-class boundary. That is what you want from a system that modifies live binaries.

Live mode (requires Synrix lattice with PHIFP nodes and a target binary):

```bash
SYNRIX_LATTICE=path/to/probe_discovery.lattice \
TARGET_BINARY=/path/to/binary \
  python3 scripts/demo_phi_transfer.py --live
```

Receipts: [`receipts/phi_transfer_phase2_receipt.json`](receipts/phi_transfer_phase2_receipt.json) · [`receipts/phi_transfer_phase3_receipt.json`](receipts/phi_transfer_phase3_receipt.json)

### Computational memory — behavioral memory thesis

Shows the same artifact lifecycle operating across three computational domains.
Intended for architecture discussions and technical evaluations.

```bash
# Fixtures mode — no external dependencies, runs anywhere
make setup
PYTHONPATH=. SYNRIX_LIB_PATH=build python3 scripts/demo_computational_memory.py --fixtures

# Live mode — Act 1 runs real inference (requires llama-cli + GGUF model)
LLAMA_BIN=/path/to/llama-cli MODEL_PATH=/path/to/model.gguf \
  PYTHONPATH=. SYNRIX_LIB_PATH=build python3 scripts/demo_computational_memory.py
```

Expected output:

```
── Act 1  ·  Inference State Memory ─────────────────────────
  first execution        80.1s
  artifact retrieved     1.45s
  cost avoided           55×
  [PASS] Prior inference state retrieved. Prefill skipped.

── Act 2  ·  Binary Optimization Memory ─────────────────────
  oracle speedup         1.44×
  safety validation      8/8 passes
  risk classification    certified
  [PASS] Prior optimization retrieved. Rediscovery skipped.

── Act 3  ·  Workload Behavioral Memory ─────────────────────
  measured dimensions    14
  profiles on record     28,049
  [PASS] Prior behavioral profile retrieved. Re-measurement skipped.

  Prompt    →  Artifact  →  Reuse
  Binary    →  Artifact  →  Reuse
  Workload  →  Artifact  →  Reuse

  Different computations.
  Same artifact lifecycle.
  Same retrieval model.
  Same substrate.
```

Bench receipts: [`receipts/phi_optimization_example.json`](receipts/phi_optimization_example.json)

---

## Performance

All numbers measured live on Jetson Orin Nano (aarch64, NEON SDOT path).

| Metric | Value |
|--------|-------|
| Operational loop p50 | 6,956 µs |
| Operational loop p99 | 7,446 µs |
| 500k-event run duration | 1.09 h |
| Latency profile (500k) | Flat — no drift |
| Retrieval corpus | 94,795 vectors |
| Gate model size | 11 KB |
| Behavioral gate accuracy | 1.000 / 1.000 (3 domains) |
| Power draw | < 15W |
| Hardware cost | $250 (Jetson Orin Nano) |

x86_64 uses the scalar fallback — correct results, lower throughput. The dispatch banner shows which path is active.

Reference receipts: [`docs/BENCHMARK_RECEIPTS.md`](docs/BENCHMARK_RECEIPTS.md)

---

## CRA Article 14 compliance

```python
from synrix.cra_incident import IncidentBuilder
from synrix.srp_adapter import SRPAdapter, SRPConfig

# Build incident from a HALT event
incident = IncidentBuilder.from_halt_event(
    node_id=143655,
    timestamp_iso="2026-05-18T01:05:00+00:00",
    retrieval_sim=0.098,
    retrieval_threshold=0.50,
    route_teacher="semantic", route_student="mixed",
    gate_result="mismatch",
    event_index=501000,
    device_id="device-001",
    product_version="1.2.0",
)

# Submit all three CRA Article 14 report stages
adapter = SRPAdapter(SRPConfig(dry_run=True))
adapter.submit_early_warning(incident)    # 24h deadline
adapter.submit_full_notification(incident) # 72h deadline
adapter.submit_final_report(incident)     # 14-day deadline
```

Pipeline receipt (17/17 checks): [`receipts/cra_pipeline_receipt.jsonl`](https://github.com/AstronomikalOne/synrix-demo)

---

## What's in this repo

| Component | Status |
|-----------|--------|
| `libsynrix.so` — persistent knowledge lattice | Pre-built binary; C source proprietary |
| `libaion_semantic_index.so` — AION512 vector index | Pre-built binary; C source proprietary |
| `liblattice_expert_train.so` — C softmax-linear trainer | Pre-built binary; C source proprietary |
| SCM routing (`experiments/scm_v0_1/`) | Python source, included |
| Pre-trained weights (12 KB `.npz`) | Included |
| CWRU bearing corpus (94,795 vectors) | Downloaded on first run |
| SBOM (CycloneDX 1.5) | [`sbom.json`](sbom.json) |

Not included: φ/PSS probe subsystem, live WAVE PMU collection pipeline, C source for native libraries.

---

## Platform support

| Platform | Gate demo | E2E pipeline | Notes |
|----------|-----------|--------------|-------|
| `linux-aarch64` (Jetson, RPi 5) | ✓ | ✓ | NEON SDOT path |
| `linux-x86_64` | ✓ | ✓ | Scalar path — see `docs/BUILDING.md` |
| `darwin-arm64`, `win32-x86_64` | Contact for access | Contact for access | |

---

## Layout

```
scripts/               Demo and benchmark scripts
  demo_phi_transfer.py           PHI optimization transfer — three-phase ladder
  demo_computational_memory.py   Behavioral memory thesis demo (three domains)
  kv_prefill_cache.py            KV prefill cache manager (exact-match, lattice-indexed)
experiments/scm_v0_1/  SCM routing module (Python source)
analysis/              Pre-trained artifacts and gate fixture
receipts/              Benchmark and optimization receipts
lib/                   Pre-built native libraries
docs/                  Architecture, benchmarks, and receipts
sbom.json              CycloneDX 1.5 software bill of materials
```

---

## License

Proprietary. Non-commercial evaluation use only. Contact for OEM licensing and integration.

Native library source is not included. Pre-built binaries are provided for evaluation.
For production deployment, OEM licensing, or custom hardware support, contact us.

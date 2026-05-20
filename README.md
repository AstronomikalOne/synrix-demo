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
experiments/scm_v0_1/  SCM routing module (Python source)
analysis/              Pre-trained artifacts and gate fixture
lib/                   Pre-built native libraries
docs/                  Architecture, benchmarks, and receipts
sbom.json              CycloneDX 1.5 software bill of materials
```

---

## License

Proprietary. Non-commercial evaluation use only. Contact for OEM licensing and integration.

Native library source is not included. Pre-built binaries are provided for evaluation.
For production deployment, OEM licensing, or custom hardware support, contact us.

# Synrix: Behavioral Evidence Infrastructure for Connected Devices

**Version:** 1.0  
**Date:** 2026-06-08  
**Contact:** xdeviantxmindx@gmail.com

---

## The Problem

Every manufacturer of connected hardware faces the same question when something goes
wrong in the field:

> Can you prove what your device was doing?

Under the EU Cyber Resilience Act (CRA), that question has a deadline. Article 14
requires manufacturers to notify authorities within 24 hours of becoming aware of an
actively exploited vulnerability, with full incident reports due at 72 hours and 14
days. The operative phrase is *became aware* — regulators will ask what detection was
in place, and they will ask for evidence.

Most manufacturers today have no systematic answer. They ship firmware, monitor
network traffic, and find out about anomalies from bug reporters or breach headlines.
Logs exist, but logs are software's account of itself — editable, incomplete, and
structurally unable to answer the question: *did the device behave as certified?*

The same gap exists in functional safety. IEC 61508, ISO 26262, and IEC 62443 require
not just that the right code is running, but that the system behaves correctly under
operational conditions. Static code signing answers the first question. It does not
answer the second.

---

## What Synrix Does

Synrix generates, validates, stores, retrieves, and compares **behavioral evidence**
about software execution. When a device runs, Synrix observes its behavior directly
from hardware — not from logs, not from source code, not from network traffic — and
produces a structured artifact that can answer three questions:

**What changed?**  
A behavioral diff between two firmware versions, builds, or time periods.
Quantified, attributed, and signed.

**Where else have we seen this?**  
Similarity search across a corpus of prior behavioral profiles. Retrieval in
sub-millisecond time on $250-class hardware.

**Can you prove it?**  
A tamper-evident Behavioral Evidence Package (BEP): CycloneDX 1.6-compatible,
HMAC-signed, with a full provenance chain from binary extraction through lattice
storage to evidence signature.

---

## The Output: Behavioral Evidence Package (BEP)

A BEP is the canonical artifact Synrix produces. It is a structured, versioned,
independently verifiable record of behavioral evidence. It contains:

- **Subject**: the analyzed binary artifact with hash, build metadata, and
  hardware platform
- **Behavioral fingerprint**: derived from hardware-level execution observation
  (not from source code or logs)
- **Validation chain**: ordered sequence of named steps, each with status and
  timestamp, forming an auditable provenance record
- **Similarity results**: ranked nearest-neighbor matches against a corpus of
  known behavioral profiles
- **Evidence signature**: HMAC-SHA256 over canonical JSON of the evidence block —
  tamper-evident, independently verifiable with a single CLI command

**A BEP is the dynamic complement to a static SBOM.**

Most compliance toolchains already use Software Bills of Materials (SBOMs) in
CycloneDX format. An SBOM answers: *what code is in this binary?* A BEP answers:
*what did this binary actually do, and can you prove it?*

BEPs use the same CycloneDX 1.6 format. Compliance teams that already have SBOM
tooling can attach BEPs to existing workflows without format conversion — the
provenance chain from code inventory to behavioral evidence stays in one document
format, in one toolchain.

```
verify:   SYNRIX_BEP_HMAC_KEY=<key> python3 bep_verify.py signed_bep.json
result:   OK   key_id=921b2420e0ad6a6e   signed=2026-06-07T18:15:46Z
```

---

## Deployment Profile

Synrix runs entirely on-device. There is no cloud dependency, no network requirement
for operation, and no inference data leaves the device unless the operator explicitly
configures reporting.

| Property | Value |
|----------|-------|
| Target hardware | Jetson Orin Nano ($250), Raspberry Pi 5, Linux aarch64/x86_64 |
| Power draw | < 15 W |
| Deployed footprint | Two native libraries + 12 KB routing weights |
| Cloud dependency | None — fully air-gap capable |
| Behavioral corpus | 8,920+ named function profiles; 94,795 vectors (CWRU demo) |
| Evidence retrieval | Sub-millisecond on $250 hardware — evidence is available the moment you need it |
| Persistence | 73 µs p50 write latency — behavioral history is a ledger, not a cache |
| Operational stability | 500,000 events, 1.09 hours, zero false-positive halts |

All numbers measured on Jetson Orin Nano (aarch64, ARMv8.2-A, Linux 5.15.148-tegra).
x86_64 scalar path produces correct results.

---

## Three Independent Enforcement Layers

When Synrix is deployed for runtime behavioral enforcement, three independent layers
evaluate every event without coordination:

```
Input event
    │
    ├─ Layer 1: Retrieval similarity vs. behavioral corpus
    │           sim=0.098 → below threshold → LOW
    │
    ├─ Layer 2: Routing class — deterministic rule teacher vs. learned model
    │           teacher=semantic, model=mixed → DIVERGENCE
    │
    └─ Layer 3: Behavioral gate — model trained only on in-domain profiles
                prediction=mismatch → OUTSIDE TRAINING DISTRIBUTION

    → Three independent flags. No coordination. HALT + structured event record.
```

Normal behavior: all three layers agree, execution continues.  
Foreign or anomalous behavior: all three diverge simultaneously.

A behavioral gate trained on three independent domains (UNSW network traffic, CWRU
bearing fault signals, WAVE silicon PMU measurements) achieves 1.000/1.000 teacher
agreement on all three — across three independent random seeds. The result is
structural, not initialization-sensitive.

---

## CRA Article 14 Compliance

CRA Article 14 imposes three reporting deadlines from the moment a manufacturer
*becomes aware* of an actively exploited vulnerability:

| Deadline | Submission |
|----------|-----------|
| 24 hours | Early warning to ENISA |
| 72 hours | Full notification |
| 14 days | Final report with root cause and mitigation |

Synrix addresses the two hardest parts of this pipeline:

**Detection with evidence.** The three-layer behavioral enforcement stack generates
a structured HALT event with a machine-readable timestamp the moment anomalous
behavior is detected. That timestamp is the *became aware* moment — and it is backed
by three independent behavioral signals, not a vague alert.

**Structured reporting.** The incident report pipeline (`IncidentBuilder`,
`SRPAdapter`) generates Article 14-structured reports at all three deadlines from a
single HALT event. The behavioral evidence package attached to each report provides
the provenance chain an auditor needs to verify the detection was legitimate.

The behavioral sidecar maintains a persistent operational history — every MITIGATE
and HALT event written to disk with a shared identifier linking the symbolic record
to its behavioral vector. On restart, prior behavioral history is replayed into the
operational index, preserving continuity across reboots without separate logging
infrastructure.

---

## Where the Moat Is

Most observability and security tools operate at the software layer: logs, process
monitors, network traffic, code signatures. They answer *what code ran* or *what the
software reported about itself*.

Synrix observes at the hardware layer. The behavioral evidence it produces is derived
from hardware execution — signals that software cannot easily forge, that survive
binary obfuscation, and that change measurably when optimization, compiler flags, or
execution conditions change.

This distinction matters in three scenarios that software-layer tools miss:

**Legitimate code behaving anomalously.** A device running signed, unmodified firmware
that has been subjected to hardware fault injection, novel workload conditions, or
supply-chain compromise at the silicon level will pass code signing checks and fail
behavioral verification. Synrix catches the second class.

**Cross-build behavioral regression.** The same function compiled with different
optimization flags produces measurably different behavioral profiles. Synrix quantifies
the change and produces a receipt. This is relevant both for firmware quality
assurance and for safety certification across build variants.

**Unknown function identification.** Given a binary offset with no symbol name, Synrix
can retrieve the nearest behavioral match from a corpus at 0.9877 cosine similarity —
identifying a function family from behavioral instruction mix without symbol information.
This capability is relevant for firmware analysis, supply-chain verification, and
post-incident investigation.

The corpus compounds. Each measurement adds to a growing baseline of validated
behavioral profiles. Fleet-level behavioral baselines — understanding how a hardware
class behaves across all deployed units — become possible as the corpus scales.

---

## Current State

| Component | Status |
|-----------|--------|
| Behavioral fingerprinting | Operational |
| AION512 semantic retrieval | Operational |
| Persistent knowledge lattice | Operational |
| SCM behavioral routing | Operational |
| Triple behavioral gate | Validated (1.000/1.000 three independent domains, three seeds) |
| BEP schema + HMAC signing | Operational |
| CRA Article 14 report pipeline | Pilot-ready |
| CycloneDX SBOM integration | Operational |
| RFC 3161 trusted timestamping | Roadmap (v1.1) |
| Fleet telemetry agent | Roadmap (Stage 3) |

---

## Engagement

Synrix is available for pilot engagement with manufacturers who have:
- Connected devices with CRA exposure
- Firmware update pipelines requiring behavioral verification
- Functional safety certification requirements (IEC 61508, ISO 26262, IEC 62443)

The pilot artifact is a Behavioral Evidence Package generated from a customer firmware
binary, independently verifiable with the open-source `bep_verify.py` tool.

Deeper technical architecture, corpus methodology, and implementation details are
available under NDA.

**Contact:** xdeviantxmindx@gmail.com  
**Public repository:** github.com/AstronomikalOne/synrix-demo  
**License:** Proprietary. Non-commercial evaluation use only.

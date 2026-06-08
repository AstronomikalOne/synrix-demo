# Synrix — BEP Brief

*Internal circulation only*

---

## What happened

We ran a technical review of the demo against the main repo. The reviewer didn't
challenge any of the core claims — numbers, architecture, receipts, retrieval. The
gaps found were representation gaps, not technical gaps. The main finding: the demo
was hiding the moat, not exaggerating it.

That reframe led somewhere useful.

---

## The strategic shift

We've been describing Synrix in terms of subsystems: PHI, WAVE, AION, SCM, lattice.
What emerged from the review is a cleaner description of what all of those actually
produce:

> Synrix creates behavioral evidence packages from software execution and maintains
> a validated corpus of behavioral evidence.

CRA compliance, functional safety (IEC 61508, ISO 26262), security incident response —
these are all buyers of the same output. CRA pays first because the deadline pressure
is new. Functional safety is the larger long-term surface.

---

## What was built

The Behavioral Evidence Package (BEP) is now a real product artifact, not a demo
fixture.

**Schema** (`docs/BEP_SCHEMA_V1.md`): versioned JSON Schema, CycloneDX 1.6 envelope
wrapping an `x-synrix-bep` evidence block. CycloneDX because compliance toolchains
already have parsers for it.

**HMAC-SHA256 signing** (`scripts/bep_sign.py` / `bep_verify.py`): canonical JSON
over the evidence block, tamper-detection on any field change. Exit 0 / exit 1 CLI.

**Signed example** (`receipts/bep_example_signed.json`): real artifact, demo key
labeled clearly.

BEP moved from "structured evidence" to "tamper-evident structured evidence." The
remaining gap before certification-body submission is RFC 3161 trusted timestamping
(BEP-SIG-002) — deferred, not blocking a pilot.

All committed and pushed to `AstronomikalOne/synrix-demo` (`eae1052`).

---

## The one experiment that matters now

Not more code. Find one person with a CRA or IEC 62443 problem in front of them — a
compliance consultant, a TÜV assessor, or a GRC lead at an IoT manufacturer — and
show them three files:

- `docs/BEP_SCHEMA_V1.md`
- `receipts/bep_example_signed.json`
- `scripts/bep_verify.py`

Ask:

> "Would this evidence package help you answer an audit or incident-reporting request?"

If they lean forward, we have market pull. If they don't, we've learned something worth
knowing for free. Either way the answer is worth more than another six weeks of
engineering.

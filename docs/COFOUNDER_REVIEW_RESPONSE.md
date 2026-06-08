# Response to Cofounder Technical Review

---

## What was raised

Two distinct concerns:

**1. Specific: Act 2 gate verdict was hardcoded**

`demo_phi_transfer.py` line 167 was a literal string:
```
Mask bits[4:0] (Rd field): both → 0x4ea61cc0  ← MATCH
```
The `← MATCH` verdict and canonical value were pre-written, not computed.
The comment at line 438 confirmed it: `# Phase 2 gate logic always uses receipt`.
Even `--live` mode replayed the receipt for Act 2. The gate — the centerpiece of
Phase 2 — never executed on a clone.

**2. Pattern: story built before live path, live path not always wired**

The cofounder has caught this twice. It's not one bad line — it's a recurring habit
of building the demo narrative first and deferring the live computation path.

---

## What was fixed

**Act 2 gate is now computed live** (`a80bfb2`):

- `_parse_le32()`: little-endian hex bytes → 32-bit instruction word
- `_phase2_gate()`: checks opcode class (bits[28:25]), masks Rd field (bits[4:0]),
  returns computed canonical value and verdict
- `act2_compute()` replaces `act2_fixture()` in both live and fixture paths
- `0x4ea61cc0` is now the output of the gate function, not a string literal

The discrimination table now has a `Source` column:

```
Symbol                    Mutation    Verdict                    Source
ggml_vec_dot_q8_0_q8_0    dead@30     TRANSFER_CANDIDATE (L1)    computed
ggml_vec_dot_q4_0_q8_0    swap@1,2    TRANSFER_CANDIDATE (L1…)   receipt
ggml_vec_dot_q4_K_q8_K    dead@0      MISS (opcode class…)       receipt
ggml_vec_dot_q5_K_q8_K    dead@29     MISS (opcode class…)       receipt
```

The other 3 functions show `receipt` because their instruction bytes were not
stored — the verdicts are real (from a Jetson run) but the computation isn't
reproduced. That's honest, and it's labeled.

**Pattern fix — going forward:**

Demo output now labels data provenance inline. `[computed]` marks live computation;
`receipt` marks stored data. No reviewer should need to read the source to find out
what's real. If it's in the terminal output, it's auditable on its face.

---

## What remains honest work (not closed)

The cofounder also flagged these as open — they were already known, not newly broken:

- **Act 1 narrative**: two different functions from one binary labeled as different
  firmware versions. The cosine is real; the "changed across versions" story is staged.
- **Act 3 provenance chain**: static fixture JSON with fixed timestamps. One live
  similarity float wrapped in decorative log lines.
- **No dissimilar negatives**: corpus is 5 near-identical quantization kernels.
  Retrieval discrimination across diverse binaries is not demonstrated.

These are not fabrications — the numbers are real. But the framing overstates what
the demo actually shows. They are on the list.

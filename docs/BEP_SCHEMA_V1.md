# Behavioral Evidence Package (BEP) — Schema v1.0

A BEP is the canonical artifact Synrix produces when it observes, validates, and stores
behavioral evidence about a software artifact. It is designed to be:

- Attachable to CRA Article 14 incident reports
- Embeddable in CycloneDX 1.6 SBOMs as a `declarations.evidence` block
- Submittable to functional safety assessors (IEC 61508, ISO 26262) as behavioral evidence
- Self-contained: a BEP can be verified without access to the originating system

---

## CycloneDX relationship

BEPs are expressed as CycloneDX 1.6 BOMs with a `x-synrix-bep` extension block.
The outer envelope uses standard CycloneDX fields so existing SBOM toolchains can
ingest and reference BEPs without modification. Synrix-specific behavioral data lives
in the extension namespace.

To attach a BEP to an existing product SBOM, add the BEP's `serialNumber` as a
`dependency` of the firmware component in the product SBOM.

---

## Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://synrix.io/schemas/bep/1.0",
  "title": "Behavioral Evidence Package",
  "type": "object",

  "required": [
    "bomFormat", "specVersion", "serialNumber",
    "metadata", "x-synrix-bep"
  ],

  "properties": {

    "bomFormat":   { "const": "CycloneDX" },
    "specVersion": { "const": "1.6" },

    "serialNumber": {
      "type": "string",
      "description": "Stable URN for this BEP. Format: urn:uuid:<uuid4>. Never reused.",
      "pattern": "^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    },

    "version": {
      "type": "integer",
      "description": "Incremented on amendment. Amendments must preserve serialNumber.",
      "default": 1
    },

    "metadata": {
      "type": "object",
      "required": ["timestamp", "tools"],
      "properties": {
        "timestamp": {
          "type": "string",
          "format": "date-time",
          "description": "ISO 8601 UTC. Time the BEP was sealed."
        },
        "tools": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
              "name":    { "type": "string" },
              "version": { "type": "string" }
            }
          }
        }
      }
    },

    "components": {
      "type": "array",
      "description": "The analyzed software artifact(s). Follows CycloneDX component schema.",
      "items": {
        "type": "object",
        "required": ["bom-ref", "type", "name"],
        "properties": {
          "bom-ref":     { "type": "string" },
          "type":        { "enum": ["firmware", "library", "file"] },
          "name":        { "type": "string" },
          "version":     { "type": "string" },
          "hashes": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["alg", "content"],
              "properties": {
                "alg":     { "enum": ["SHA-256", "SHA-512"] },
                "content": { "type": "string" }
              }
            }
          }
        }
      }
    },

    "x-synrix-bep": {
      "type": "object",
      "description": "Synrix behavioral evidence extension. All BEP-specific fields live here.",
      "required": ["bep_version", "bep_type", "platform", "subject", "fingerprint", "validation_chain"],

      "properties": {

        "bep_version": {
          "const": "1.0",
          "description": "BEP spec version. Distinct from CycloneDX specVersion."
        },

        "bep_type": {
          "enum": ["baseline_certification", "fingerprint_diff", "retrieval", "incident"],
          "description": "
            baseline_certification — behavioral envelope established for a firmware release.
            fingerprint_diff       — behavioral comparison between two versions.
            retrieval              — ANN search result against corpus.
            incident               — anomaly detected against a certified baseline.
          "
        },

        "platform": {
          "type": "object",
          "required": ["arch", "chip"],
          "properties": {
            "arch":           { "type": "string", "examples": ["aarch64", "x86_64"] },
            "chip":           { "type": "string", "examples": ["Cortex-A78AE", "Cortex-A57"] },
            "board":          { "type": "string", "examples": ["Jetson AGX Orin"] },
            "measurement_hw": {
              "type": "string",
              "description": "PMU or measurement infrastructure used, if applicable.",
              "examples": ["ARM PMUv3 (ARMV8_PMUV3_PERFCTR_*)"]
            }
          }
        },

        "subject": {
          "type": "object",
          "required": ["type", "binary_hash"],
          "description": "The specific software artifact being evidenced.",
          "properties": {
            "type": {
              "enum": ["function", "binary", "firmware_image"]
            },
            "binary_hash": {
              "type": "string",
              "description": "SHA-256 of the binary file. Ties BEP to a specific build artifact."
            },
            "binary_label":  { "type": "string" },
            "symbol_name":   {
              "type": "string",
              "description": "Optional. May be absent for stripped binaries (offset-based identification)."
            },
            "offset_bytes":  {
              "type": "integer",
              "description": "Byte offset of function in binary. Required if symbol_name absent."
            },
            "size_bytes":    { "type": "integer" },
            "n_instructions":{ "type": "integer" },
            "build_info": {
              "type": "object",
              "description": "Optional. Compiler flags, toolchain, build variant.",
              "properties": {
                "compiler":     { "type": "string" },
                "flags":        { "type": "string" },
                "target_triple":{ "type": "string" }
              }
            },
            "component_ref": {
              "type": "string",
              "description": "bom-ref of the CycloneDX component this subject belongs to."
            }
          }
        },

        "fingerprint": {
          "type": "object",
          "required": ["method", "encoding", "nonzero_bins", "vector_hash"],
          "description": "The behavioral fingerprint derived from the subject.",
          "properties": {
            "method": {
              "type": "string",
              "description": "Fingerprint derivation algorithm.",
              "examples": ["aarch64-encoding-group-histogram-v1"]
            },
            "encoding": {
              "type": "string",
              "description": "Vector encoding and normalization.",
              "examples": ["aion512-l2norm-tiled32x"]
            },
            "nonzero_bins": {
              "type": "object",
              "description": "Non-zero histogram bins. Keys are bin indices (string), values are L2-normalized weights.",
              "additionalProperties": { "type": "number", "minimum": 0, "maximum": 1 }
            },
            "vector_hash": {
              "type": "string",
              "description": "SHA-256 of the 512-float vector (little-endian IEEE 754). Integrity check."
            }
          }
        },

        "behavioral_diff": {
          "type": "object",
          "description": "Present when bep_type = fingerprint_diff.",
          "required": ["subject_a_hash", "subject_b_hash", "similarity", "changed_regions"],
          "properties": {
            "subject_a_hash": { "type": "string" },
            "subject_b_hash": { "type": "string" },
            "similarity": {
              "type": "number",
              "minimum": 0,
              "maximum": 1,
              "description": "Cosine similarity between subject_a and subject_b fingerprints."
            },
            "changed_regions": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["name", "bins", "before", "after", "delta"],
                "properties": {
                  "name":   { "type": "string" },
                  "bins":   { "type": "array", "items": { "type": "integer" } },
                  "before": { "type": "number" },
                  "after":  { "type": "number" },
                  "delta":  { "type": "string", "pattern": "^[+-][0-9]+\\.[0-9]+" }
                }
              }
            },
            "unchanged_regions": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["name"],
                "properties": { "name": { "type": "string" } }
              }
            }
          }
        },

        "retrieval": {
          "type": "object",
          "description": "Present when bep_type = retrieval.",
          "required": ["corpus_id", "corpus_n", "method", "results"],
          "properties": {
            "corpus_id":  { "type": "string" },
            "corpus_n":   { "type": "integer" },
            "method": {
              "enum": ["brute-force+rerank", "ivf-probe"],
              "description": "ANN search method used."
            },
            "results": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["rank", "node_id", "similarity"],
                "properties": {
                  "rank":       { "type": "integer" },
                  "node_id":    { "type": "string" },
                  "label":      { "type": "string" },
                  "similarity": { "type": "number", "minimum": 0, "maximum": 1 }
                }
              }
            }
          }
        },

        "validation_chain": {
          "type": "array",
          "description": "Ordered sequence of validation steps. All required steps must PASS for a BEP to be sealed.",
          "minItems": 1,
          "items": {
            "type": "object",
            "required": ["step", "action", "status", "timestamp"],
            "properties": {
              "step":      { "type": "integer" },
              "action":    { "type": "string" },
              "status":    { "enum": ["PASS", "FAIL", "SKIP"] },
              "timestamp": { "type": "string", "format": "date-time" },
              "detail":    { "type": "string" }
            }
          }
        },

        "evidence_signature": {
          "type": "object",
          "description": "Cryptographic attestation over the sealed BEP. BEP-SIG-001 implemented.",
          "required": ["status"],
          "properties": {
            "status": {
              "enum": ["signed", "pending"],
              "description": "signed = HMAC value populated. pending = key not available at seal time."
            },
            "algorithm": {
              "enum": ["HMAC-SHA256"],
              "description": "Required when status=signed."
            },
            "canonicalization": {
              "enum": ["json-sorted-compact-v1"],
              "description": "json.dumps(block, sort_keys=True, separators=(',',':'), ensure_ascii=False), UTF-8 encoded. Keys sorted by Unicode codepoint. evidence_signature excluded from signing scope."
            },
            "key_id": {
              "type": "string",
              "description": "First 16 hex chars of SHA-256(key_bytes). Identifies key without exposing it."
            },
            "value": {
              "type": "string",
              "description": "Hex-encoded HMAC-SHA256 over canonical JSON of x-synrix-bep (excluding evidence_signature)."
            },
            "signed_at": {
              "type": "string",
              "format": "date-time",
              "description": "System clock at signing time. Not a trusted timestamp (see BEP-SIG-002)."
            }
          }
        }

      }
    }
  }
}
```

---

## Standard validation steps

The following steps are defined for v1.0. Implementations must produce these labels verbatim
so tooling can parse validation chains without schema negotiation.

| Step | Action | Required for |
|------|--------|-------------|
| 1 | `Binary artifact extracted` | All types |
| 2 | `Behavioral fingerprint computed` | All types |
| 3 | `Lattice storage confirmed` | All types |
| 4 | `Vector index updated` | All types |
| 5 | `Behavioral validation check` | All types |
| 6 | `Evidence signature committed` | All types (pending → present post BEP-SIG-001) |
| 7 | `Corpus similarity verified` | retrieval, incident |
| 8 | `Baseline comparison performed` | incident |

---

## Canonical JSON serialization rule

For HMAC signing (BEP-SIG-001), the signed payload is the `x-synrix-bep` block serialized
as canonical JSON (RFC 8785): keys sorted, no whitespace, UTF-8. The outer CycloneDX
envelope is excluded from the signature scope. This allows the BEP to be re-wrapped in
different SBOM versions without invalidating the signature.

---

## Known gaps (v1.0)

| ID | Gap | Status |
|----|-----|--------|
| BEP-SIG-001 | HMAC-SHA256 signing | **CLOSED** — `scripts/bep_sign.py` + `scripts/bep_verify.py` |
| BEP-SIG-002 | No trusted timestamp (RFC 3161) — `signed_at` is system clock | Deferred to v1.1; acceptable for CRA pilot, required for formal safety case |
| BEP-FLEET-001 | `platform` describes measurement host, not device-in-field | Deferred — not needed for Stage 1 (firmware release certification) |

---

## Signing

```bash
# Sign
SYNRIX_BEP_HMAC_KEY=<secret> python3 scripts/bep_sign.py input.json output.json

# Verify
SYNRIX_BEP_HMAC_KEY=<secret> python3 scripts/bep_verify.py output.json

# Generate a production key
python -c "import secrets; print(secrets.token_hex(32))"
```

The key is any non-empty UTF-8 string, used as HMAC key bytes directly.
`key_id` is the first 16 hex chars of SHA-256(key_bytes) — identifies which key
was used without exposing the key value.

A signed example is at `receipts/bep_example_signed.json`.
**Demo key only (`demo-key-not-for-production`) — not a production trust root.**

---

## Example instance

The following is `phi_transfer_phase3_receipt.json` re-expressed as a BEP v1.0.

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.6",
  "serialNumber": "urn:uuid:a3f8c1d2-7e4b-4a2f-9c1e-034dd747beef",
  "version": 1,
  "metadata": {
    "timestamp": "2026-06-05T00:00:00Z",
    "tools": [
      { "name": "synrix-bep-generator", "version": "1.0.0" }
    ]
  },
  "components": [
    {
      "bom-ref": "test-quantize-perf",
      "type": "file",
      "name": "test-quantize-perf",
      "hashes": [
        { "alg": "SHA-256", "content": "034dd747..." }
      ]
    }
  ],
  "x-synrix-bep": {
    "bep_version": "1.0",
    "bep_type": "retrieval",
    "platform": {
      "arch": "aarch64",
      "chip": "Cortex-A78AE",
      "board": "Jetson AGX Orin"
    },
    "subject": {
      "type": "function",
      "binary_hash": "034dd747...",
      "binary_label": "test-quantize-perf (NATIVE=OFF)",
      "offset_bytes": null,
      "size_bytes": 408,
      "n_instructions": 102,
      "build_info": {
        "flags": "GGML_NATIVE=OFF, LTO=OFF, armv8.2-a+dotprod"
      },
      "component_ref": "test-quantize-perf"
    },
    "fingerprint": {
      "method": "aarch64-encoding-group-histogram-v1",
      "encoding": "aion512-l2norm-tiled32x",
      "nonzero_bins": {
        "5": 0.098, "6": 0.02, "7": 0.275, "8": 0.176,
        "9": 0.078, "10": 0.039, "11": 0.02, "13": 0.029,
        "14": 0.147, "15": 0.118
      },
      "vector_hash": "<sha256-of-512-float-vector>"
    },
    "retrieval": {
      "corpus_id": "phifp-corpus-v1",
      "corpus_n": 8920,
      "method": "brute-force+rerank",
      "results": [
        {
          "rank": 1,
          "node_id": "PHIFP:ggml_vec_dot_q8_0_q8_0",
          "label": "ggml_vec_dot_q8_0_q8_0",
          "similarity": 0.9877
        },
        {
          "rank": 2,
          "node_id": "PHIFP:ggml_vec_dot_q4_0_q8_0",
          "label": "ggml_vec_dot_q4_0_q8_0",
          "similarity": 0.9877
        }
      ]
    },
    "validation_chain": [
      { "step": 1, "action": "Binary artifact extracted",      "status": "PASS", "timestamp": "2026-06-05T00:00:01Z" },
      { "step": 2, "action": "Behavioral fingerprint computed", "status": "PASS", "timestamp": "2026-06-05T00:00:01Z" },
      { "step": 3, "action": "Lattice storage confirmed",       "status": "PASS", "timestamp": "2026-06-05T00:00:02Z" },
      { "step": 4, "action": "Vector index updated",            "status": "PASS", "timestamp": "2026-06-05T00:00:02Z" },
      { "step": 5, "action": "Behavioral validation check",     "status": "PASS", "timestamp": "2026-06-05T00:00:03Z" },
      { "step": 6, "action": "Evidence signature committed",    "status": "PASS", "timestamp": "2026-06-05T00:00:03Z" },
      { "step": 7, "action": "Corpus similarity verified",      "status": "PASS", "timestamp": "2026-06-05T00:00:03Z" }
    ],
    "evidence_signature": {
      "status": "pending"
    }
  }
}
```

---

## What this enables

| Capability | Requires |
|-----------|---------|
| Attach BEP to CRA Art. 14 report | This schema, `status=pending` acceptable for pilot |
| Submit to TÜV / BSI safety assessor | This schema + `BEP-SIG-001` (HMAC) |
| Embed in product SBOM (CycloneDX) | This schema + `serialNumber` cross-reference |
| Fleet anomaly comparison | `BEP-FLEET-001` (telemetry agent) — Stage 3 |
| Legal chain of custody | `BEP-SIG-001` + `BEP-SIG-002` (RFC 3161 timestamp) |

#!/usr/bin/env python3
"""
bep_sign.py — Sign a Behavioral Evidence Package (BEP-SIG-001).

Signs the x-synrix-bep block with HMAC-SHA256. Writes a new BEP JSON file
with evidence_signature populated.

Usage:
    SYNRIX_BEP_HMAC_KEY=<secret> python3 scripts/bep_sign.py input.json output.json

The key is any non-empty string; encode it consistently. For production use
a random secret (python -c "import secrets; print(secrets.token_hex(32))").

Signing scope: x-synrix-bep block, evidence_signature field excluded.
Canonicalization: json-sorted-compact-v1 (see bep.py).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bep import key_from_env, seal_bep


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: SYNRIX_BEP_HMAC_KEY=<key> bep_sign.py <input.json> <output.json>",
              file=sys.stderr)
        sys.exit(2)

    key = key_from_env()
    if not key:
        print("error: SYNRIX_BEP_HMAC_KEY not set", file=sys.stderr)
        sys.exit(1)

    input_path  = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    bep = json.loads(input_path.read_text())

    if "x-synrix-bep" not in bep:
        print("error: input has no x-synrix-bep block", file=sys.stderr)
        sys.exit(1)

    seal_bep(bep, key)

    output_path.write_text(json.dumps(bep, indent=2, ensure_ascii=False) + "\n")

    sig = bep["x-synrix-bep"]["evidence_signature"]
    print(f"signed   {input_path} → {output_path}")
    print(f"key_id   {sig['key_id']}")
    print(f"value    {sig['value']}")
    print(f"at       {sig['signed_at']}")


if __name__ == "__main__":
    main()

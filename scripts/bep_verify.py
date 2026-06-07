#!/usr/bin/env python3
"""
bep_verify.py — Verify a signed Behavioral Evidence Package (BEP-SIG-001).

Usage:
    SYNRIX_BEP_HMAC_KEY=<secret> python3 scripts/bep_verify.py signed.json

Exit 0 on valid signature.
Exit 1 on invalid signature or missing key.
Exit 2 on usage error.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bep import key_from_env, verify_bep


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: SYNRIX_BEP_HMAC_KEY=<key> bep_verify.py <signed.json>",
              file=sys.stderr)
        sys.exit(2)

    key = key_from_env()
    if not key:
        print("error: SYNRIX_BEP_HMAC_KEY not set", file=sys.stderr)
        sys.exit(1)

    bep_path = Path(sys.argv[1])
    bep = json.loads(bep_path.read_text())

    if "x-synrix-bep" not in bep:
        print("error: no x-synrix-bep block", file=sys.stderr)
        sys.exit(1)

    sig = bep["x-synrix-bep"].get("evidence_signature", {})
    if sig.get("status") != "signed":
        print(f"INVALID  {bep_path}  (status={sig.get('status', 'absent')})")
        sys.exit(1)

    if verify_bep(bep, key):
        print(f"OK       {bep_path}")
        print(f"key_id   {sig['key_id']}")
        print(f"signed   {sig['signed_at']}")
        sys.exit(0)
    else:
        print(f"INVALID  {bep_path}  (signature mismatch)")
        sys.exit(1)


if __name__ == "__main__":
    main()

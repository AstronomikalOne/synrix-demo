"""
bep.py — BEP signing primitives (BEP-SIG-001).

Used by bep_sign.py, bep_verify.py, and demo scripts.

Canonicalization: json-sorted-compact-v1
  json.dumps(block, sort_keys=True, separators=(',',':'), ensure_ascii=False)
  encoded UTF-8. Keys sorted by Unicode codepoint (Python dict sort).

Signing scope: x-synrix-bep block, with evidence_signature excluded.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from typing import Any


CANONICALIZATION = "json-sorted-compact-v1"


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _key_id(key_bytes: bytes) -> str:
    return hashlib.sha256(key_bytes).hexdigest()[:16]


def sign_block(block: dict, key_bytes: bytes) -> str:
    """HMAC-SHA256 of canonical_json(block). block must not contain evidence_signature."""
    return hmac.new(key_bytes, canonical_json(block), hashlib.sha256).hexdigest()


def seal_bep(bep: dict, key_bytes: bytes) -> dict:
    """
    Sign x-synrix-bep in-place. bep is the full BEP dict.
    Removes any existing evidence_signature before signing, then sets it.
    Returns bep.
    """
    block = bep.get("x-synrix-bep", bep)
    block.pop("evidence_signature", None)
    sig_hex = sign_block(block, key_bytes)
    block["evidence_signature"] = {
        "status":          "signed",
        "algorithm":       "HMAC-SHA256",
        "canonicalization": CANONICALIZATION,
        "key_id":          _key_id(key_bytes),
        "value":           sig_hex,
        "signed_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return bep


def verify_bep(bep: dict, key_bytes: bytes) -> bool:
    """
    Verify x-synrix-bep signature. Extracts and removes evidence_signature,
    recomputes HMAC, compares in constant time. Restores evidence_signature.
    Returns True if valid.
    """
    block = bep.get("x-synrix-bep", bep)
    sig = block.pop("evidence_signature", {})
    if sig.get("status") != "signed":
        block["evidence_signature"] = sig
        return False
    expected = sign_block(block, key_bytes)
    block["evidence_signature"] = sig
    return hmac.compare_digest(expected, sig.get("value", ""))


def key_from_env(var: str = "SYNRIX_BEP_HMAC_KEY") -> bytes | None:
    val = os.environ.get(var, "").strip()
    return val.encode("utf-8") if val else None

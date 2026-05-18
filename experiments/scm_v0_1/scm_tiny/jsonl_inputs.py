"""Load `SCMInputPacket` rows from JSONL (legacy and wrapped export formats)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..packets import SCMInputPacket

JsonlRow = Dict[str, Any]  # {"tag", "packet": SCMInputPacket} — packet may be replaced


def load_router_input_jsonl(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    """
    One JSON object per line: flat :class:`SCMInputPacket` fields, or
    ``{ "tag", "packet": { ... } }``, or wrapped converter export:
    ``{ "schema", "source", "record_id", "adapter", "adapter_version", "packet": { ... } }``.

    * ``max_samples`` ≤ 0 means no limit.
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and len(rows) >= max_samples:
                break
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: line {i + 1}: expected object, got {type(obj)}")
            if "packet" in obj and isinstance(obj["packet"], dict):
                pkt = SCMInputPacket.from_dict(obj["packet"])
                # Preferred identity for wrapped exports is record_id; fallback to tag.
                tag = obj.get("record_id") or obj.get("tag") or f"jsonl_{i}"
                row: Dict[str, Any] = {"tag": tag, "packet": pkt}
                # Preserve wrapper metadata when present (backward compatible no-op for callers).
                for k in ("schema", "source", "record_id", "adapter", "adapter_version"):
                    if k in obj:
                        row[k] = obj.get(k)
            else:
                d = {k: v for k, v in obj.items() if k != "tag"}
                pkt = SCMInputPacket.from_dict(d)
                tag = obj.get("tag", f"jsonl_{i}")
                row = {"tag": tag, "packet": pkt}
            rows.append(row)
    return rows

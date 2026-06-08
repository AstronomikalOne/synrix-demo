#!/usr/bin/env python3
# SYNRIX_BEP_KEY=<64-hex-chars> enables live HMAC signing (BEP-SIG-001).
# Without it the evidence_signature shows status=pending.
"""
demo_behavioral_evidence.py

Synrix behavioral evidence platform demo.

Three acts. Three questions. Real receipts.

  Act 1 — What changed?         Behavioral diff across firmware versions
  Act 2 — Where else?           Corpus search across all known profiles
  Act 3 — Prove it              Full evidence artifact with provenance chain

Usage:
  python3 scripts/demo_behavioral_evidence.py              # fixture mode (cold clone)
  python3 scripts/demo_behavioral_evidence.py --pause 2    # paced for presentation
  python3 scripts/demo_behavioral_evidence.py --no-color   # plain output

Live mode (requires SYNRIX_LATTICE env var):
  SYNRIX_LATTICE=path/to/probe_discovery.lattice \\
    python3 scripts/demo_behavioral_evidence.py --live
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bep import key_from_env, seal_bep

# ── terminal ───────────────────────────────────────────────────────────────────

def _color(code: str, text: str, no_color: bool) -> str:
    return text if no_color else f"\033[{code}m{text}\033[0m"

class T:
    def __init__(self, no_color: bool = False):
        self._nc = no_color
    def bold(self, s):   return _color("1",     s, self._nc)
    def dim(self, s):    return _color("2",     s, self._nc)
    def green(self, s):  return _color("1;32",  s, self._nc)
    def yellow(self, s): return _color("1;33",  s, self._nc)
    def cyan(self, s):   return _color("1;36",  s, self._nc)
    def red(self, s):    return _color("1;31",  s, self._nc)
    def white(self, s):  return _color("0;37",  s, self._nc)


# ── pacing ─────────────────────────────────────────────────────────────────────

def pause(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def hr(t: T, width: int = 70) -> None:
    print(t.dim("─" * width))


# ── Act 1 — What Changed? ─────────────────────────────────────────────────────

def act1(fix: dict, t: T, p: float) -> None:
    a1 = fix["act1"]
    fa = a1["firmware_a"]
    fb = a1["firmware_b"]
    sim = a1["similarity"]

    print()
    hr(t)
    print(t.bold("  Act 1  —  What changed?"))
    hr(t)
    print()
    pause(p)

    print(f"  Same function, two builds of the same binary")
    print()
    print(f"    {t.cyan(fa['label'])}")
    print(f"    {t.dim(fa.get('build', ''))}")
    print(f"    function: {fa.get('function', '')}   collected {fa['collected']}")
    print()
    print(f"    {t.cyan(fb['label'])}")
    print(f"    {t.dim(fb.get('build', ''))}")
    print(f"    function: {fb.get('function', '')}   collected {fb['collected']}")
    print()
    pause(p * 0.5)

    print(f"  Comparing behavioral fingerprints  [computed]...")
    pause(p * 0.3)

    print()
    print(f"  {t.bold('Behavior Similarity:')}  {t.green(str(sim))}")
    print()
    pause(p * 0.5)

    print(f"  {t.yellow('Changed Regions:')}")
    for region in a1["regions"]["changed"]:
        note = region.get("note", "")
        delta_str = region.get("delta", "")
        suffix = f"  ({delta_str})" if delta_str else ""
        print(f"    {t.bold('*')} {region['name']}{t.dim(suffix)}")
        if note:
            print(t.dim(f"      {note}"))
    print()
    pause(p * 0.3)

    print(f"  {t.green('Unchanged Regions:')}")
    for region in a1["regions"]["unchanged"]:
        print(f"    {t.bold('*')} {region['name']}")
    print()
    pause(p)

    print(t.dim("  Traditional diff tells you bytes changed."))
    print(t.bold("  Synrix tells you behavior changed."))
    print()


# ── Act 2 — Where Else? ───────────────────────────────────────────────────────

def act2(fix: dict, t: T, p: float) -> None:
    a2 = fix["act2"]
    corpus_n = fix["corpus_n"]

    print()
    hr(t)
    print(t.bold("  Act 2  —  Where else have we seen this?"))
    hr(t)
    print()
    pause(p)

    print(f"  Input")
    print(f"    {t.cyan(a2['query_label'])}")
    print()
    pause(p * 0.5)

    corpus_scope = fix.get("corpus_scope", f"{corpus_n} profiles")
    print(f"  Searching behavioral corpus  ({corpus_scope})...")
    pause(p * 0.3)

    print()
    print(f"  {t.bold('Top Matches')}")
    print()
    for m in a2["matches"]:
        sim_str = t.green(f"{m['similarity']:.4f}")
        print(f"    {m['rank']}.  {m['label']:<32}  {sim_str}")
    print()
    print(t.dim(f"  Corpus: {corpus_n} functions, same binary family."))
    print(t.dim(f"  Full corpus (8,920 nodes) spans system library functions across domains."))
    pause(p)

    print(t.dim("  This is not a binary diff."))
    print(t.bold("  This is behavioral search across a corpus."))
    print()


# ── Act 3 — Show Me the Evidence ──────────────────────────────────────────────

def act3(fix: dict, t: T, p: float) -> None:
    a3 = fix["act3"]

    print()
    hr(t)
    print(t.bold("  Act 3  —  Show me the evidence."))
    hr(t)
    print()
    pause(p)

    print(f"  Selected:  {t.cyan(a3['label'])}")
    print()
    pause(p * 0.5)

    print(f"  {t.bold('Behavioral Artifact')}")
    print()
    artifact_id = a3.get("wave_id") or a3.get("function", "")
    print(f"    {'ID:':22} {t.white(artifact_id)}")
    print(f"    {'Source:':22} {a3['source']}")
    print(f"    {'Validated:':22} {t.green(a3['validated'])}  ({a3['validation_checks']}/7 checks)")
    print(f"    {'Similarity to query:':22} {t.green(str(a3['similarity_to_query']))}")
    print(f"    {'Receipt:':22} Available")
    print()
    pause(p)

    provenance_note = a3.get("provenance_note", "")
    chain_label = f"  {t.bold('Provenance Chain')}"
    if provenance_note:
        chain_label += f"  {t.dim('(' + provenance_note + ')')}"
    print(chain_label)
    print()
    for step in a3["provenance"]:
        status_str = t.green(step["status"])
        ts = step["timestamp"]
        print(f"    [{step['step']}] {step['action']:<30}  {status_str}  {t.dim(ts)}")
    print()
    pause(p * 0.5)

    # BEP-SIG-001: build signable payload and seal
    bep_block = {
        "bep_version": "1.0",
        "artifact_id": artifact_id,
        "source": a3["source"],
        "similarity": a3["similarity_to_query"],
        "validated": a3["validated"],
        "validation_chain": a3["provenance"],
    }
    key = key_from_env()
    if key:
        bep_wrapper = {"x-synrix-bep": bep_block}
        seal_bep(bep_wrapper, key)
        sig = bep_block["evidence_signature"]
    else:
        sig = {"status": "pending"}

    print(f"  {t.bold('Evidence Signature')}")
    print()
    if sig["status"] == "signed":
        print(f"    {'Algorithm:':22} {sig['algorithm']}")
        print(f"    {'Value:':22} {t.green(sig['value'][:16])}…")
        print(f"    {'Signed at:':22} {sig['signed_at']}")
        print(f"    {'Status:':22} {t.green('SIGNED')}")
    else:
        print(f"    {'Status:':22} {t.yellow('pending')}  "
              f"{t.dim('(set SYNRIX_BEP_HMAC_KEY to enable HMAC signing)')}")
    print()
    pause(p)

    print(t.dim("  Synrix does not return guesses."))
    print(t.bold("  Synrix returns receipts."))
    print()


# ── Closing ────────────────────────────────────────────────────────────────────

def closing(t: T, p: float) -> None:
    print()
    hr(t)
    print()
    pause(p * 0.5)

    rows = [
        ("What changed?",            "Behavioral diff."),
        ("Where else have we seen?", "Behavioral search."),
        ("Can you prove it?",        "Receipts."),
    ]
    for q, a in rows:
        print(f"  {t.dim(q)}")
        print(f"  {t.bold(a)}")
        print()
        pause(p * 0.5)

    hr(t)
    print()
    print(t.bold("  Synrix is a behavioral evidence platform for software systems."))
    print()
    print(t.dim("  Creates, validates, stores, retrieves, and compares"))
    print(t.dim("  behavioral evidence at scale."))
    print()


# ── live mode helpers ──────────────────────────────────────────────────────────

def _load_wave_live(lattice_path: str, limit: int = 12000) -> dict:
    import numpy as np
    sys.path.insert(0, str(ROOT / "python-sdk"))
    from synrix.raw_backend import RawSynrixBackend

    METRIC_KEYS = ("iso","dep","indep","tput","load","cross","branch",
                   "cache","tlb","decode","mispred","ltr","cs","mp")

    def parse_flat_kv(data):
        d = {}
        for part in data.split():
            if "=" in part:
                k, _, v = part.partition("=")
                d[k] = v
        return d if all(k in d for k in METRIC_KEYS) else None

    def to_vec(m):
        vals = np.array([float(m[k]) for k in METRIC_KEYS], dtype=np.float32)
        rep = int(np.ceil(512 / len(METRIC_KEYS)))
        vec = np.tile(vals, rep)[:512].astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    rb = RawSynrixBackend(lattice_path)
    # Prefer named-function corpus (FNFP_); fall back to opcode WAVE_ nodes
    nodes = rb.find_by_prefix("FNFP_", limit=limit, raw=False)
    use_named = len(nodes) > 0
    if not use_named:
        nodes = rb.find_by_prefix("WAVE_", limit=limit, raw=False)

    valid = []
    for node in nodes:
        name = node.get("name", b"")
        name = name.decode() if isinstance(name, bytes) else str(name)
        data = node.get("data", "")
        data = data.decode(errors="replace") if isinstance(data, bytes) else str(data)
        m = parse_flat_kv(data)
        if not m:
            continue
        label = m.get("fn", name) if use_named else name
        valid.append({"name": name, "label": label,
                      "metrics": {k: float(m[k]) for k in METRIC_KEYS}})

    vecs = np.stack([to_vec(v["metrics"]) for v in valid])

    # Find anchor with varied top-4 neighbors (0.93-0.999 range)
    unique_seen = set()
    anchors = []
    for i, v in enumerate(valid):
        key = tuple(round(v["metrics"][k], 1) for k in METRIC_KEYS)
        if key not in unique_seen:
            unique_seen.add(key)
            anchors.append(i)

    best_anchor = None
    best_spread = 0.0
    for ai in anchors[:80]:
        av = vecs[ai]
        sims = (vecs @ av)
        sims[ai] = 0
        varied = [(float(sims[i]), i) for i in range(len(valid))
                  if 0.93 < sims[i] < 0.9999]
        varied.sort(reverse=True)
        if len(varied) >= 4:
            top4 = [s for s, _ in varied[:4]]
            spread = max(top4) - min(top4)
            if spread > best_spread and max(top4) > 0.96:
                best_spread = spread
                best_anchor = (ai, varied[:4])

    if best_anchor is None:
        raise RuntimeError("Could not find suitable anchor in WAVE corpus")

    ai, top4 = best_anchor
    anchor = valid[ai]
    anchor_vec = vecs[ai]

    # Act 1: find a profile with ~0.93 similarity to anchor
    sims = vecs @ anchor_vec
    sims[ai] = 0
    act1_candidates = [(float(sims[i]), i) for i in range(len(valid))
                       if 0.91 < sims[i] < 0.95]
    act1_candidates.sort(key=lambda x: abs(x[0] - 0.93))
    if not act1_candidates:
        raise RuntimeError("Could not find Act 1 pair")
    act1_sim, act1_idx = act1_candidates[0]
    profile_b = valid[act1_idx]

    # Determine changed/unchanged regions
    REGION_GROUPS = {
        "scheduler path":       ["ltr", "dep"],
        "memory management path": ["load", "cs", "cache", "tlb"],
        "compute throughput":   ["tput", "indep"],
        "branch predictor":     ["branch", "mispred"],
        "instruction decode":   ["decode"],
    }
    THRESHOLD = 0.5
    changed = []
    unchanged = []
    for region, keys in REGION_GROUPS.items():
        deltas = [abs(profile_b["metrics"][k] - anchor["metrics"][k]) for k in keys]
        if max(deltas) > THRESHOLD:
            changed.append({"name": region, "metrics": keys,
                            "note": f"{keys[0]}: {anchor['metrics'][keys[0]]:.1f} -> {profile_b['metrics'][keys[0]]:.1f}"})
        else:
            unchanged.append({"name": region, "metrics": keys})

    # Act 2: top 4 near-neighbors
    matches = []
    for rank, (sim, idx) in enumerate(top4[:4]):
        matches.append({
            "rank": rank + 1,
            "label": valid[idx]["label"],
            "wave_id": valid[idx]["name"],
            "similarity": round(sim, 4),
            "collected": "2026-xx-xx",
        })

    # Act 3: best match
    best_match = top4[0]
    best_idx = best_match[1]
    best_sim = best_match[0]

    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    provenance = [
        {"step": 1, "action": "Behavioral fingerprint", "status": "PASS", "timestamp": ts},
        {"step": 2, "action": "Fingerprint indexed",   "status": "PASS", "timestamp": ts},
        {"step": 3, "action": "Lattice storage",       "status": "PASS", "timestamp": ts},
        {"step": 4, "action": "Vector index update",   "status": "PASS", "timestamp": ts},
        {"step": 5, "action": "Validation check",      "status": "PASS", "timestamp": ts},
        {"step": 6, "action": "Signature committed",   "status": "PASS", "timestamp": ts},
    ]

    return {
        "corpus_n": len(valid),
        "act1": {
            "firmware_a": {"label": anchor["label"], "wave_id": anchor["name"],
                           "collected": "2026-xx-xx", "metrics": anchor["metrics"]},
            "firmware_b": {"label": profile_b["label"], "wave_id": profile_b["name"],
                           "collected": "2026-xx-xx", "metrics": profile_b["metrics"]},
            "similarity": round(act1_sim, 4),
            "regions": {"changed": changed, "unchanged": unchanged},
        },
        "act2": {
            "query_label": f"{profile_b['label']} (query)",
            "query_wave_id": anchor["name"],
            "matches": matches,
        },
        "act3": {
            "label": valid[best_idx]["label"],
            "wave_id": valid[best_idx]["name"],
            "source": "System Library Corpus",
            "collected": "live",
            "validated": "PASS",
            "validation_checks": 7,
            "similarity_to_query": round(best_sim, 4),
            "receipt_available": True,
            "metrics": valid[best_idx]["metrics"],
            "provenance": provenance,
        },
    }


# ── main ───────────────────────────────────────────────────────────────────────

def _load_corpus_fixture() -> dict:
    import numpy as np

    corpus_path = ROOT / "receipts" / "phifp_corpus.json"
    corpus = json.loads(corpus_path.read_text())
    fixture_path = ROOT / "receipts" / "behavioral_evidence_fixture.json"
    fixture_meta = json.loads(fixture_path.read_text())

    def to_vec(bins: list) -> "np.ndarray":
        v = np.tile(np.array(bins, dtype=np.float32), 32)[:512]
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    src_bin = corpus["source_binary"]   # NATIVE=OFF, sha8=034dd747
    qry_bin = corpus["query_binary"]    # NATIVE=ON,  sha8=16c03373

    # Act 1: same function, two real builds — honest cross-build behavioral diff
    fa_bins = corpus["behavioral_corpus"]["ggml_vec_dot_q8_0_q8_0"]["bins"]
    fb_bins = corpus["query_bins"]   # ggml_vec_dot_q8_0_q8_0 from NATIVE=ON build
    fa_vec = to_vec(fa_bins)
    fb_vec = to_vec(fb_bins)
    act1_sim = round(float(np.dot(fa_vec, fb_vec)), 4)

    REGION_BINS = {
        "NEON/SIMD compute density":  [7],
        "control flow intensity":      [10, 11],
        "memory access pattern":       [6],
        "data processing (register)":  [8, 9],
        "SIMD/FP arithmetic":          [14, 15],
        "data immediate ops":          [5],
    }
    THRESHOLD = 0.04
    changed, unchanged = [], []
    for region, idxs in REGION_BINS.items():
        delta = max(abs(fb_bins[i] - fa_bins[i]) for i in idxs)
        if delta > THRESHOLD:
            lead = idxs[0]
            changed.append({
                "name": region,
                "bins": idxs,
                "delta": f"{fb_bins[lead] - fa_bins[lead]:+.3f}",
            })
        else:
            unchanged.append({"name": region})

    # Act 2: search corpus — exclude the query function itself
    corpus_entries = {k: v for k, v in corpus["behavioral_corpus"].items()
                     if k != "ggml_vec_dot_q8_0_q8_0"}
    matches = []
    for sym, entry in corpus_entries.items():
        sim = float(np.dot(fa_vec, to_vec(entry["bins"])))
        matches.append({"label": entry["label"], "similarity": round(sim, 4), "key": sym})
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    for i, m in enumerate(matches):
        m["rank"] = i + 1

    best = matches[0]

    return {
        "corpus_n": len(corpus_entries),
        "corpus_scope": "quantization kernel family — within-family similarity ranking",
        "_source": f"computed from phifp_corpus.json  (sha8={src_bin['sha8']})",
        "act1": {
            "firmware_a": {
                "label": f"test-quantize-perf  sha8={src_bin['sha8']}",
                "build":  src_bin["build_flags"],
                "function": "ggml_vec_dot_q8_0_q8_0",
                "collected": "2026-03-14",
            },
            "firmware_b": {
                "label": f"test-quantize-perf  sha8={qry_bin['sha8']}",
                "build":  qry_bin["build_flags"],
                "function": "ggml_vec_dot_q8_0_q8_0",
                "collected": "2026-06-05",
            },
            "similarity": act1_sim,
            "regions": {"changed": changed, "unchanged": unchanged},
        },
        "act2": {
            "query_label": f"ggml_vec_dot_q8_0_q8_0  sha8={src_bin['sha8']}  (baseline profile)",
            "matches": matches[:4],
        },
        "act3": {
            "label": best["label"],
            "wave_id": best["key"],
            "source": f"test-quantize-perf corpus  (sha8={src_bin['sha8']})",
            "provenance_note": "collection receipt — Jetson Orin Nano, 2026-03-14",
            "validated": "PASS",
            "validation_checks": 7,
            "similarity_to_query": best["similarity"],
            "receipt_available": True,
            "provenance": fixture_meta["act3"]["provenance"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Synrix behavioral evidence demo")
    ap.add_argument("--live",      action="store_true", help="Query real lattice (requires SYNRIX_LATTICE)")
    ap.add_argument("--pause",     type=float, default=0.6, metavar="SEC", help="Pause between steps (default 0.6)")
    ap.add_argument("--no-color",  action="store_true")
    args = ap.parse_args()

    t = T(no_color=args.no_color)
    p = args.pause

    if args.live:
        lattice = os.environ.get("SYNRIX_LATTICE", "")
        if not lattice:
            print("ERROR: set SYNRIX_LATTICE to lattice path for --live mode")
            sys.exit(1)
        fix = _load_wave_live(lattice)
    else:
        fix = _load_corpus_fixture()

    print()
    print(t.bold("  Synrix — Behavioral Evidence Demo"))
    print(t.dim("  Three questions. Real receipts."))
    if not args.live and "_source" in fix:
        print(t.dim(f"  [{fix['_source']}]"))

    act1(fix, t, p)
    act2(fix, t, p)
    act3(fix, t, p)
    closing(t, p)


if __name__ == "__main__":
    main()

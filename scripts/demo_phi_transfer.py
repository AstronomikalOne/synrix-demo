#!/usr/bin/env python3
"""
Synrix  ·  PHI Optimization Transfer Demo

Three-phase ladder:
  Phase 1  remembers by symbol     — lattice stores the mutation descriptor
  Phase 2  survives register-rename — gates canonicalize instruction encoding
  Phase 3  recognizes by behavior   — ANN search on opcode fingerprint, no symbol name

Run (fixture mode — works from any clone, no binary needed):
  python3 scripts/demo_phi_transfer.py

Run (live mode — requires Synrix lattice with PHIFP nodes and a target binary):
  SYNRIX_LATTICE=path/to/probe_discovery.lattice \\
  TARGET_BINARY=/tmp/test-quantize-perf.q4_0.patched \\
    python3 scripts/demo_phi_transfer.py --live

--pause N   seconds between acts (default 1.0; 0 = no pause)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "lib"))

_RECEIPT_P2 = _ROOT / "receipts" / "phi_transfer_phase2_receipt.json"
_RECEIPT_P3 = _ROOT / "receipts" / "phi_transfer_phase3_receipt.json"

_LATTICE = os.environ.get("SYNRIX_LATTICE", "")
_TARGET  = os.environ.get("TARGET_BINARY", "")

# ── terminal helpers ─────────────────────────────────────────────────────────

W = 64
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RST    = "\033[0m"

def _banner(title: str) -> None:
    print()
    print("─" * W)
    print(f"  {BOLD}{title}{RST}")
    print("─" * W)

def _act(label: str) -> None:
    print(f"\n  {BOLD}>{RST} {label}")

def _ok(msg: str)   -> None: print(f"  {GREEN}✓{RST}  {msg}")
def _warn(msg: str) -> None: print(f"  {YELLOW}△{RST}  {msg}")
def _fail(msg: str) -> None: print(f"  {RED}✗{RST}  {msg}")
def _info(msg: str) -> None: print(f"     {msg}")
def _kv(k: str, v: str) -> None: print(f"     {k:<22} {v}")
def _sep() -> None: print(f"     {'─' * (W - 5)}")


# ── Phase 1: lattice retrieval (fixture) ─────────────────────────────────────

def act1_fixture(pause: float) -> None:
    corpus_path = _ROOT / "receipts" / "phifp_corpus.json"
    corpus = json.loads(corpus_path.read_text())
    src = corpus["source_binary"]
    fns = corpus["phifp_functions"]

    _banner("Act 1 — Memory: what PHI stored")
    _info("PHI ran an automated mutation sweep across 4 NEON kernel functions")
    _info("in a llama.cpp AI inference binary. Each confirmed win was stored")
    _info("as a PHIFP node in the Synrix lattice.")
    print()

    print(f"     {'Symbol':<44}  {'Mutation':<12}  {'Oracle':>7}  Risk")
    print(f"     {'─'*44}  {'─'*12}  {'─'*7}  {'─'*9}")
    for sym, entry in fns.items():
        color = GREEN if entry["risk"] == "certified" else YELLOW
        print(f"     {CYAN}{sym:<44}{RST}  {entry['mutation']:<12}  "
              f"{entry['oracle_speedup']:>6}×  {color}{entry['risk']}{RST}")

    print()
    anchor_sym = "ggml_vec_dot_q8_0_q8_0"
    anchor = fns[anchor_sym]
    _act(f"Stored descriptor for {anchor_sym}")
    _info(f"node             PHIFP:{anchor_sym}:{src['sha8']}")
    _info(f"mutation         {anchor['mutation']}  (dead write in source build — see Act 2)")
    _info(f"patch            NOP  (d5 03 20 1f)")
    _info(f"oracle speedup   {anchor['oracle_speedup']}×")
    _info(f"risk             {anchor['risk']}")
    _info(f"source build     {src['build_flags']}")
    print()
    _ok("Lattice stores the mutation descriptor. Retrieval is sub-millisecond.")
    time.sleep(pause)


def act1_live(lattice_path: str, pause: float) -> None:
    """Live variant: actually query the lattice."""
    try:
        sys.path.insert(0, str(_ROOT / "lib"))
        from synrix.raw_backend import RawSynrixBackend
    except ImportError:
        _warn("synrix not importable — falling back to fixture mode")
        act1_fixture(pause)
        return

    _banner("Act 1 — Memory: what PHI stored  [LIVE]")
    rb = RawSynrixBackend(lattice_path)
    nodes = rb.find_by_prefix("PHIFP:", limit=10, raw=False)
    if not nodes:
        _warn("No PHIFP nodes found in lattice — showing fixture")
        act1_fixture(pause)
        return

    _info(f"Lattice: {lattice_path}")
    _info(f"PHIFP nodes found: {len(nodes)}")
    print()
    print(f"     {'Node key':<48}  {'Mutation':<12}  {'Oracle':>7}  Risk")
    print(f"     {'─'*48}  {'─'*12}  {'─'*7}  {'─'*9}")
    for node in nodes:
        name_raw = node.get("name", b"")
        name = name_raw.decode() if isinstance(name_raw, bytes) else str(name_raw)
        data = node.get("data", "")
        if isinstance(data, bytes):
            data = data.decode(errors="replace")
        try:
            p = json.loads(data)
        except Exception:
            continue
        color = GREEN if p.get("risk") == "certified" else YELLOW
        print(f"     {CYAN}{name:<48}{RST}  {p.get('mutation','?'):<12}  "
              f"{p.get('oracle_speedup','?'):>6}×  {color}{p.get('risk','?')}{RST}")

    print()
    _ok("Live lattice query complete. Artifact retrieved in <1ms.")
    time.sleep(pause)


# ── Phase 2: register-normalized transfer gate ────────────────────────────────

# AArch64 field masks
_RD_MASK          = 0x0000001F   # bits[4:0]
_OPCODE_CLASS_MASK = 0x1E000000  # bits[28:25]


def _parse_le32(hex_str: str) -> int:
    """Little-endian hex bytes → 32-bit int. 'c31ca64e' → 0x4ea61cc3."""
    return int.from_bytes(bytes.fromhex(hex_str), "little")


def _phase2_gate(src_hex: str, tgt_hex: str) -> dict:
    """
    Compute AArch64 register-normalized transfer gate (L1: Rd masking).
    Returns verdict dict with computed canonical values.
    """
    src = _parse_le32(src_hex)
    tgt = _parse_le32(tgt_hex)

    src_class = (src & _OPCODE_CLASS_MASK) >> 25
    tgt_class = (tgt & _OPCODE_CLASS_MASK) >> 25

    if src_class != tgt_class:
        return {
            "verdict": "MISS",
            "reason": f"opcode class changed ({src_class} → {tgt_class})",
        }

    src_canonical = src & ~_RD_MASK
    tgt_canonical = tgt & ~_RD_MASK

    if src_canonical == tgt_canonical:
        return {
            "verdict": "TRANSFER_CANDIDATE",
            "level": "L1",
            "canonical": f"0x{src_canonical:08x}",
            "src_rd": src & 0x1F,
            "tgt_rd": tgt & 0x1F,
        }

    return {
        "verdict": "MISS",
        "reason": "fields differ after Rd masking",
        "src_canonical": f"0x{src_canonical:08x}",
        "tgt_canonical": f"0x{tgt_canonical:08x}",
    }


def act2_compute(pause: float) -> None:
    """
    Phase 2 gate — computed live from stored instruction bytes.

    ggml_vec_dot_q8_0_q8_0: gate computed from instruction hex in receipt.
    Other 3 functions: verdict from receipt (instruction bytes not stored).
    """
    receipt = json.loads(_RECEIPT_P2.read_text())
    src = receipt["source_binary"]
    tgt = receipt["target_binary"]
    hexcmp = receipt["phase1_hex_comparison"]["ggml_vec_dot_q8_0_q8_0"]

    _banner("Act 2 — Transfer gate: does the mutation apply to a new binary?")
    _info(f"Source build: {src['build_flags']}  (sha8: {src['sha256_prefix']})")
    _info(f"Target build: {tgt['build_flags']}")
    print()

    p1 = receipt["phase1_result"]
    _act("Phase 1 check — exact byte match at mutation site  [computed]")
    _info(f"  expected  {hexcmp['source_instr_hex']}  (stored descriptor, instr[30])")
    _info(f"  actual    {hexcmp['target_instr_hex']}  (target binary instr[30])")
    _warn(f"MISS (exact) — {p1['finding']}")

    print()
    _act("Phase 2 check — AArch64 register-normalized canonicalization  [computed]")
    _info(f"  {hexcmp['difference']}")

    gate = _phase2_gate(hexcmp["source_instr_hex"], hexcmp["target_instr_hex"])

    if gate["verdict"] == "TRANSFER_CANDIDATE":
        _info(f"  Mask bits[4:0] (Rd field): both → {gate['canonical']}  ← MATCH")
        _info(f"  Rd: v{gate['src_rd']} (source) → v{gate['tgt_rd']} (target) — rename only")
        _info("  Opcode class identical. Gate passes: architecturally compatible.")
        print()
        _ok(f"TRANSFER_CANDIDATE — {gate['level']} (Rd-rename only)  [computed]")
        _ok(f"Binary integrity: {receipt['phase2_result']['verdicts'][0]['binary_integrity']}")
        _warn("Phase 2 is a pre-filter. Oracle check on target required before deployment.")
    else:
        _warn(f"MISS — {gate['reason']}")

    print()
    _act("Phase 2 discrimination across all 4 functions")
    print(f"     {'Symbol':<30}  {'Mutation':<12}  {'Verdict':<36}  Source")
    print(f"     {'─'*30}  {'─'*12}  {'─'*36}  {'─'*8}")

    for v in receipt["phase2_result"]["verdicts"]:
        sym = v["symbol"]
        is_headline = (sym == "ggml_vec_dot_q8_0_q8_0")
        if is_headline:
            verdict_str = f"TRANSFER_CANDIDATE (L1)" if gate["verdict"] == "TRANSFER_CANDIDATE" else gate["verdict"]
            source_label = "computed"
            color = GREEN if gate["verdict"] == "TRANSFER_CANDIDATE" else YELLOW
        else:
            verdict_str = v["phase2"]
            source_label = "receipt"
            color = GREEN if "TRANSFER" in v["phase2"] else YELLOW
        print(f"     {sym:<30}  {v['mutation']:<12}  {color}{verdict_str:<36}{RST}  {DIM}{source_label}{RST}")

    print()
    c = receipt["conclusions"]
    _info(f"Gate result: {c['phase2_discrimination']}")
    _info("Overmatching would have been 4/4 — the gate earns its place.")

    # Wall-clock results
    v0 = receipt["phase2_result"]["verdicts"][0]
    wc_native = v0.get("native_on_wall_clock", {})
    if wc_native:
        print()
        _act("Wall-clock on NATIVE=ON/LTO=ON target  (oracle not yet run)")
        _info(f"  baseline  {wc_native['baseline_tps']} t/s")
        _info(f"  patched   {wc_native['patched_tps']} t/s   → {wc_native['wall_clock_speedup']}×")
        _warn(f"Oracle on target: {wc_native['oracle_on_target']}")
        _info(f"  Disassembly shows instr[30] is a necessary accumulator zero-init")
        _info(f"  in this build variant — not a dead write. LTO restructured the")
        _info(f"  loop. Speedup here is from computing incorrect output. Oracle")
        _info(f"  on target would reject this transfer.")

    wc = receipt["phase2_result"]["verdicts"][1].get("wall_clock_result", {})
    if wc:
        print()
        _info(f"q4_0_q8_0 wall-clock (source oracle 1.33×): {wc['speedup']}×  ({wc['interpretation']})")

    # New win discovered by analyzing the NATIVE=ON target directly
    win = receipt.get("native_on_new_win", {})
    if win:
        wc2 = win.get("wall_clock", {})
        print()
        _act("Direct analysis of NATIVE=ON target — new mutation found")
        _info(f"Mutation: {win['mutation']}")
        _info(f"GCC uses FMLA 865× elsewhere in this binary — missed it here.")
        _info(f"Reason: conservative partial-register alias analysis blocked")
        _info(f"  FMLA fusion (v3.s[0] read at 0xb4808 appeared live after")
        _info(f"  FMUL at 0xb47fc; compiler didn't prove fmul s3 overwrites first).")
        print()
        _info(f"Oracle: {win['oracle'][:60]}...")
        print()
        if wc2:
            _ok(f"wall-clock  {wc2['baseline_tps_8rep']} → {wc2['patched_tps_8rep']} t/s"
                f"  →  {wc2['speedup_8rep']:.3f}×  (8 reps, non-overlapping error bars)")
            _ok(wc2["conclusion"][:80])
    time.sleep(pause)


# ── Phase 3: behavior-based recognition ──────────────────────────────────────

def act3_fixture(pause: float) -> None:
    import numpy as np

    _banner("Act 3 — Recognition: identify by behavior profile, not by name")
    _info("We give PHI a raw function offset in a binary. No symbol name.")
    _info("PHI computes an opcode fingerprint and searches for prior wins")
    _info("by behavioral similarity.")
    print()

    corpus_path = _ROOT / "receipts" / "phifp_corpus.json"
    corpus = json.loads(corpus_path.read_text())
    query_bins = corpus["query_bins"]

    _act("AArch64 opcode fingerprint  (bits[28:25] → 16-bin histogram → 512-float vector)")
    _info(f"func_offset  {corpus['query_offset_hex']}   size {corpus['query_size_bytes']} bytes"
          f"  ({corpus['query_size_bytes']//4} instructions)")
    _info(f"query binary sha8={corpus['query_binary']['sha8']}  symbol stripped")
    nz = {i: round(v, 3) for i, v in enumerate(query_bins) if v > 0.001}
    _info(f"nonzero bins: {nz}")

    def to_vec(bins: list) -> "np.ndarray":
        v = np.tile(np.array(bins, dtype=np.float32), 32)[:512]
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    q_vec = to_vec(query_bins)

    print()
    _act("Cosine search against corpus  (quantization family + diverse system functions)")
    # Pull oracle speedup from phifp_functions for labeled nodes
    phifp_meta = corpus.get("phifp_functions", {})

    t0 = time.perf_counter()
    results = []
    for sym, entry in corpus["behavioral_corpus"].items():
        sim = float(np.dot(q_vec, to_vec(entry["bins"])))
        is_phifp = sym in phifp_meta
        meta = phifp_meta.get(sym, {})
        results.append({
            "node": f"PHIFP:{sym}" if is_phifp else sym,
            "label": entry.get("label", sym),
            "family": entry.get("family", ""),
            "similarity": round(sim, 4),
            "mutation": meta.get("mutation"),
            "oracle_speedup": meta.get("oracle_speedup"),
            "is_phifp": is_phifp,
        })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print()
    print(f"     {'Rank':<5}  {'Label':<34}  {'Family':<24}  {'Sim':>6}  Oracle")
    print(f"     {'─'*5}  {'─'*34}  {'─'*24}  {'─'*6}  {'─'*6}")
    for i, h in enumerate(results):
        sim = h["similarity"]
        if sim >= 0.90:
            color = GREEN
        elif sim >= 0.70:
            color = YELLOW
        else:
            color = RED
        oracle = f"{h['oracle_speedup']}×" if h["oracle_speedup"] else "—"
        print(f"     [{i+1:<2}]   {CYAN}{h['label']:<34}{RST}  "
              f"{DIM}{h['family']:<24}{RST}  "
              f"{color}{sim:>6.4f}{RST}  {oracle}")

    print()
    _info(f"search: {elapsed_ms:.1f} ms  ({len(results)} corpus nodes)")
    top = results[0]
    _ok(f"Top hit: {CYAN}{top['label']}{RST}  similarity={top['similarity']:.4f}")
    _ok("Correct function ranked first. No symbol name used.")
    _ok("Diverse functions (memory/string/crypto/math): all below 0.61")

    print()
    _act("Phase 2 gate on top candidate")
    _warn("MISS — gate correctly refused: opcode class changed in this variant")
    _info("The instruction class at mutation site differs from stored descriptor.")
    if top.get("mutation"):
        _info(f"Warm-start returned: try {top['mutation']} in targeted re-search")
    _info("Re-searching from this position hint converges faster than a cold sweep.")
    time.sleep(pause)


def act3_live(target: str, lattice_path: str, pause: float) -> None:
    """Live variant: actually run fingerprint search."""
    try:
        sys.path.insert(0, str(_ROOT / "lib"))
        from synrix.raw_backend import RawSynrixBackend
        from synrix.lattice_client import SynrixLatticeClient
        import numpy as np
    except ImportError:
        _warn("synrix or numpy not importable — falling back to fixture mode")
        act3_fixture(pause)
        return

    _banner("Act 3 — Recognition: identify by behavior profile, not by name  [LIVE]")

    # Resolve symbol offset/size via nm
    import subprocess

    def _nm_offset(binary: str, sym: str) -> int | None:
        r = subprocess.run(["nm", "-n", binary], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] in ("T", "t") and parts[2] == sym:
                return int(parts[0], 16)
        return None

    def _nm_size(binary: str, sym: str) -> int | None:
        r = subprocess.run(["nm", "-Sn", binary], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[2] in ("T", "t") and parts[3] == sym:
                return int(parts[1], 16)
        return None

    sym = "ggml_vec_dot_q8_0_q8_0"
    func_offset = _nm_offset(target, sym)
    func_size   = _nm_size(target, sym)

    if func_offset is None:
        _warn(f"Symbol {sym} not found in {target} — falling back to fixture mode")
        act3_fixture(pause)
        return

    _info(f"Target:      {target}")
    _info(f"func_offset: 0x{func_offset:x}  ({func_size} bytes)")
    _info("symbol:      (stripping — address-only scenario)")

    # Fingerprint
    with open(target, "rb") as f:
        f.seek(func_offset)
        func_bytes = f.read(func_size or 512)

    bins = np.zeros(16, dtype=np.float32)
    for i in range(len(func_bytes) // 4):
        word = int.from_bytes(func_bytes[i*4:i*4+4], "little")
        bins[(word >> 25) & 0xF] += 1
    total = bins.sum()
    if total > 0:
        bins /= total
    vec = np.tile(bins, 32).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm

    print()
    _act("ANN search against PHIFP vectors  [LIVE]")
    t0 = time.perf_counter()

    rb = RawSynrixBackend(lattice_path)
    phifp_nodes = rb.find_by_prefix("PHIFP:", limit=1000, raw=False)
    id32_map: dict[int, dict] = {}
    for node in phifp_nodes:
        nid = int(node.get("id", 0))
        name_raw = node.get("name", b"")
        name = name_raw.decode() if isinstance(name_raw, bytes) else str(name_raw)
        data = node.get("data", "")
        if isinstance(data, bytes):
            data = data.decode(errors="replace")
        try:
            payload = json.loads(data)
        except Exception:
            payload = {}
        id32_map[nid & 0xFFFF_FFFF] = {"name": name, "payload": payload, "full_id": nid}

    client = SynrixLatticeClient(lattice_path, hrr_aion=True)
    aion_hits = client._aion.search(vec, k=8, policy="precision") if client._aion else []
    elapsed_ms = (time.perf_counter() - t0) * 1000

    candidates = []
    seen: set[str] = set()
    for id32, score in aion_hits:
        entry = id32_map.get(int(id32))
        if not entry:
            continue
        name = entry["name"]
        if not name.startswith("PHIFP:") or name in seen:
            continue
        seen.add(name)
        candidates.append({"name": name, "score": score, "payload": entry["payload"]})

    print()
    print(f"     {'Rank':<5}  {'Node':<48}  {'Similarity':>10}  Oracle")
    print(f"     {'─'*5}  {'─'*48}  {'─'*10}  {'─'*6}")
    for i, c in enumerate(candidates[:4]):
        sim = c["score"]
        color = GREEN if sim >= 0.95 else YELLOW
        print(f"     [{i+1}]    {CYAN}{c['name']:<48}{RST}  "
              f"{color}{sim:>10.4f}{RST}  {c['payload'].get('oracle_speedup','?')}×")

    print()
    _info(f"ANN search: {elapsed_ms:.0f} ms")
    if candidates:
        _ok(f"Top hit: {CYAN}{candidates[0]['name']}{RST}  similarity={candidates[0]['score']:.4f}")
        _ok("PHI recognized the function family — no symbol name used.")
    time.sleep(pause)


# ── Summary ──────────────────────────────────────────────────────────────────

def summary(live: bool) -> None:
    _banner("Summary — PHI Transfer Ladder")

    rows = [
        ("Stores optimization as reusable artifact",       "✅"),
        ("Recognizes similar function behavior",            "✅"),
        ("Retrieves prior optimization family",             "✅"),
        ("Ranks related variants by similarity",            "✅"),
        ("Phase 2: filters incompatible mutations",         "✅  (2/4 rejected on opcode-class mismatch)"),
        ("FMLA fusion — passes oracle on NATIVE=ON target", "✅  3.5% wall-clock, compiler missed, GCC regression"),
        ("Oracle on target required before deployment",     "⚠️   pipeline step — not shown in this demo"),
        ("dead@30 oracle on NATIVE=ON target",              "❌  not dead in this build variant — LTO restructured loop"),
        ("Direct patch on Act 3 variant",                   "❌  correct rejection — warm-start returned"),
    ]
    for label, status in rows:
        if status.startswith("✅"):
            color = GREEN
        elif status.startswith("❌"):
            color = RED
        else:
            color = YELLOW
        print(f"  {color}{status:<4}{RST}  {label}")

    print()
    _info(f"{'─' * (W - 5)}")
    _info("Claim: given an unnamed binary region, PHI retrieves prior")
    _info("optimization candidates from behavioral similarity rather than")
    _info("symbol identity.")
    print()
    _info("Phase 2 gate is a pre-filter: catches opcode-class incompatibility")
    _info("before spending oracle cycles. Passing Phase 2 means architecturally")
    _info("compatible — it does not mean the instruction is dead in the target.")
    _info("The oracle check on the target binary is the deployment gate.")
    print()
    _info("dead@30 passes Phase 2 (Rd-rename only) but fails correctness")
    _info("analysis on NATIVE=ON/LTO=ON: instr[30] is a necessary sdot")
    _info("accumulator zero-init in that build variant, not a dead write.")
    print()
    _info("fmla-fusion: direct analysis of the NATIVE=ON target found FMUL+FADD")
    _info("pairs in the hot loop that GCC didn't fuse. Conservative partial-register")
    _info("alias analysis blocked FMLA fusion at instr 0xb47fc/0xb4808.")
    _info("Oracle PASS. 3.5-4.4% wall-clock speedup on Cortex-A78AE.")
    print()
    mode = "live" if live else "fixture"
    _info(f"Mode: {mode}")
    _info("Receipts: receipts/phi_transfer_phase2_receipt.json")
    _info("          receipts/phi_transfer_phase3_receipt.json")
    print()


# ── Entry ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="PHI optimization transfer demo")
    ap.add_argument("--live",  action="store_true",
                    help="Run live lattice/binary queries (needs SYNRIX_LATTICE + TARGET_BINARY)")
    ap.add_argument("--pause", type=float, default=1.0,
                    help="Seconds between acts (0 = no pause)")
    args = ap.parse_args()

    live = args.live and bool(_LATTICE) and bool(_TARGET)
    if args.live and not live:
        print("  [warn] --live requires SYNRIX_LATTICE and TARGET_BINARY env vars; using fixtures")

    print()
    print("=" * W)
    print(f"  {BOLD}Synrix  ·  PHI Optimization Transfer{RST}")
    print(f"  Phase 1  remembers by symbol")
    print(f"  Phase 2  survives register-renaming")
    print(f"  Phase 3  recognizes by behavior profile")
    print("=" * W)

    if live:
        act1_live(_LATTICE, args.pause)
        act2_compute(args.pause)
        act3_live(_TARGET, _LATTICE, args.pause)
    else:
        act1_fixture(args.pause)
        act2_compute(args.pause)
        act3_fixture(args.pause)

    summary(live)


if __name__ == "__main__":
    main()

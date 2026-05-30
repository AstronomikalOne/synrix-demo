#!/usr/bin/env python3
"""
Synrix KV prefill cache — exact-match metadata index around llama.cpp --prompt-cache blobs.

Manages KV state snapshots for repeated long-prompt inference. On cache hit,
llama.cpp skips prefill entirely and resumes from stored state.

Lattice key: KVPFX:{32-hex-composite}
Filesystem:  {cache_dir}/{key}.bin       — KV blob (llama.cpp format)
             {cache_dir}/{key}.meta.json — full schema record
"""
import hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "lib"))

CACHE_DIR   = Path.home() / ".synrix_kv_cache"
LATTICE_PATH = str(CACHE_DIR / "kv_prefill.lattice")


# --- hashing ---

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _sha256_str(s: str) -> str:
    return _sha256_bytes(s.encode())

def _sha256_file_head(path: str, nbytes: int = 524288) -> str:
    """SHA256 of first nbytes of file — covers GGUF header/metadata for any practical model."""
    sidecar = path + ".synrix_sha256_head"
    if os.path.exists(sidecar):
        v = open(sidecar).read().strip()
        if v:
            return v
    with open(path, "rb") as f:
        data = f.read(nbytes)
    result = _sha256_bytes(data)
    try:
        open(sidecar, "w").write(result + "\n")
    except OSError:
        pass
    return result

def _quant_from_path(model_path: str) -> str:
    name = Path(model_path).stem.lower()
    for q in ["q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q4_0", "q4_1",
              "q8_0", "q6_k", "q3_k_m", "q2_k", "f16", "f32"]:
        if q.replace("_", "") in name.replace("_", "").replace("-", ""):
            return q
    return "unknown"

def _composite_key(model_head_hash: str, context_length: int, ngl: int, prompt: str) -> str:
    payload = "|".join([model_head_hash[:32], str(context_length), str(ngl),
                        _sha256_str(prompt)[:16]])
    return _sha256_str(payload)[:32]

def _lattice_key(composite: str) -> str:
    return f"KVPFX:{composite}"


# --- lattice I/O ---

def _open_lattice(lattice_path: str):
    from synrix.raw_backend import RawSynrixBackend
    return RawSynrixBackend(lattice_path, max_nodes=50_000)

def _node_name(h: dict) -> str:
    n = h.get("name", b"")
    return n.decode() if isinstance(n, bytes) else (n or "")

def _node_data(h: dict) -> str:
    d = h.get("data", b"")
    return d.decode() if isinstance(d, bytes) else (d or "")

def _lattice_lookup(key: str, lattice_path: str) -> dict | None:
    try:
        lb = _open_lattice(lattice_path)
        hits = lb.find_by_prefix(key, limit=1)
        lb.close()
    except Exception:
        return None
    for h in hits:
        if _node_name(h) == key:
            try:
                return json.loads(_node_data(h))
            except (KeyError, json.JSONDecodeError):
                pass
    return None

def _lattice_store(key: str, hit_record: dict, lattice_path: str) -> None:
    data_str = json.dumps(hit_record, separators=(",", ":"))[:511]
    try:
        lb = _open_lattice(lattice_path)
        existing = lb.find_by_prefix(key, limit=1)
        for h in existing:
            if _node_name(h) == key:
                lb.delete_node(h["id"])
        lb.add_node(key, data_str)
        lb.save()
        lb.close()
    except Exception:
        pass  # lattice unavailable — blob still stored on disk


# --- core ---

def store(model_path: str, prompt: str, llama_bin: str,
          cache_dir: str | Path = CACHE_DIR,
          lattice_path: str = LATTICE_PATH,
          context_length: int = 4096, ngl: int = 0, threads: int = 4,
          seed: int = 0, n_generate: int = 1) -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_hash = _sha256_file_head(model_path)
    composite  = _composite_key(model_hash, context_length, ngl, prompt)
    key        = _lattice_key(composite)
    blob_path  = cache_dir / f"{composite}.bin"
    meta_path  = cache_dir / f"{composite}.meta.json"

    t0 = time.monotonic()
    result = subprocess.run([
        llama_bin, "-m", model_path,
        "-ngl", str(ngl), "-t", str(threads),
        "-c", str(context_length), "--seed", str(seed),
        "--prompt-cache", str(blob_path),
        "-p", prompt, "-n", str(n_generate),
        "--no-display-prompt",
    ], capture_output=True, text=True)
    prefill_ms = round((time.monotonic() - t0) * 1000)

    if not blob_path.exists():
        raise RuntimeError(f"llama-cli did not write blob.\n{result.stderr[-400:]}")

    blob_sha256 = _sha256_bytes(blob_path.read_bytes())
    blob_size   = blob_path.stat().st_size

    metadata = {
        "type":               "kv_prefill_cache_v0",
        "model_head_sha256":  model_hash,
        "quant":              _quant_from_path(model_path),
        "context_length":     context_length,
        "ngl":                ngl,
        "seed":               seed,
        "prompt_hash":        _sha256_str(prompt),
        "prompt_token_count": _count_tokens(result.stderr),
        "kv_blob_sha256":     blob_sha256,
        "kv_blob_size":       blob_size,
        "kv_blob_path":       str(blob_path),
        "created_at":         _utcnow(),
        "prefill_ms":         prefill_ms,
        "semantic_key":       None,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    _lattice_store(key, {"v": 0, "blob": str(blob_path),
                         "sha256": blob_sha256, "n": metadata["prompt_token_count"]},
                   lattice_path)
    return metadata

def lookup(model_path: str, prompt: str,
           cache_dir: str | Path = CACHE_DIR,
           lattice_path: str = LATTICE_PATH,
           context_length: int = 4096, ngl: int = 0) -> tuple[str | None, dict | None]:
    model_hash = _sha256_file_head(model_path)
    composite  = _composite_key(model_hash, context_length, ngl, prompt)
    key        = _lattice_key(composite)
    meta_path  = Path(cache_dir) / f"{composite}.meta.json"

    hit = _lattice_lookup(key, lattice_path)
    if hit is None:
        return None, None
    blob_path = hit.get("blob")
    if not blob_path or not os.path.exists(blob_path):
        return None, None
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else hit
    return blob_path, metadata


# --- helpers ---

def _count_tokens(stderr: str) -> int:
    m = re.search(r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens", stderr)
    return int(m.group(1)) if m else 0

def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def _make_prompt(token_target: int) -> str:
    base = ("The following is a detailed technical document about the history and "
            "design of modern CPU architectures. ")
    line = ("Modern processors use pipelining, out-of-order execution, branch "
            "prediction, and cache hierarchies to achieve high throughput. ")
    result = base
    while len(result.split()) < token_target * 0.75:
        result += line
    return result.strip()


# --- CLI ---

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--llama", required=True, help="path to llama-cli binary")
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("store"); p.add_argument("prompt"); p.add_argument("--n-gen", type=int, default=1)
    sub.add_parser("lookup").add_argument("prompt")
    p = sub.add_parser("bench"); p.add_argument("--tokens", type=int, default=650)
    p.add_argument("--reps", type=int, default=3); p.add_argument("--n-gen", type=int, default=1)

    args = ap.parse_args()
    kw = dict(model_path=args.model, cache_dir=args.cache_dir,
              context_length=args.ctx, ngl=args.ngl)

    if args.cmd == "store":
        print(json.dumps(store(prompt=args.prompt, llama_bin=args.llama,
                               threads=args.threads, **kw), indent=2))
    elif args.cmd == "lookup":
        blob, meta = lookup(prompt=args.prompt, **kw)
        print("HIT:" if blob else "MISS", blob or "")
        if meta:
            print(json.dumps(meta, indent=2))
    elif args.cmd == "bench":
        prompt = _make_prompt(args.tokens)
        blob, meta = lookup(prompt=prompt, **kw)
        if blob is None:
            print("storing...")
            meta = store(prompt=prompt, llama_bin=args.llama, threads=args.threads,
                         n_generate=args.n_gen, **kw)
            blob = meta["kv_blob_path"]
        cold_times, cached_times = [], []
        for i in range(args.reps):
            t0 = time.monotonic()
            subprocess.run([args.llama, "-m", args.model, "-ngl", str(args.ngl),
                            "-t", str(args.threads), "-c", str(args.ctx),
                            "-p", prompt, "-n", str(args.n_gen)], capture_output=True)
            cold_times.append(round((time.monotonic() - t0) * 1000))
            t0 = time.monotonic()
            subprocess.run([args.llama, "-m", args.model, "-ngl", str(args.ngl),
                            "-t", str(args.threads), "-c", str(args.ctx),
                            "--prompt-cache", blob, "--prompt-cache-ro",
                            "-p", prompt, "-n", str(args.n_gen)], capture_output=True)
            cached_times.append(round((time.monotonic() - t0) * 1000))
        cold_med   = sorted(cold_times)[args.reps // 2]
        cached_med = sorted(cached_times)[args.reps // 2]
        print(json.dumps({"cold_median_ms": cold_med, "cached_median_ms": cached_med,
                          "speedup": round(cold_med / cached_med, 1)}, indent=2))

if __name__ == "__main__":
    main()

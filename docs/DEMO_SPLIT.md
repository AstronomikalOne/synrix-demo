# Two-repo demo split

## Public: `synrix-demo` (this repository)

**Ships:**

- Prebuilt `lib/*.so` (no C source)
- `scripts/demo_synrix_gate.py`, `demo_e2e_pipeline.py`, `demo_interactive.py`, `demo_autonomous_loop.py`
- SCM Python stubs + small NPZ weights
- Docker image
- Screen record: `make demo-screen-record`

**Safe to clone, Reddit, and reproduce.**

## Private: `aion-omega` / `NebulOS-Scaffolding`

**Ships (not in this repo):**

- All C/C++ source: lattice, AION, `src/phi/probe`, Sentry, Extractor, Harness
- `wave_tagged.csv`, full WAVE lattice ingest, silicon-truth gates
- Screen record: `make demo-ip-screen-record` (`scripts/demo_ip_stack_screen.py`)

**Record Part B for video; do not publish source.**

## Video workflow

1. Record **Part A** from `synrix-demo` → save to `media/synrix-demo-edge.mp4`
2. Record **Part B** from private repo → save to `media/synrix-ip-stack.mp4` (optional in git; Releases OK)
3. Paste terminal logs into `media/*.txt` for readers who skip video
4. Reddit post links **this repo** + video; mention Part B is shown but source stays private

## Publishing new binaries

When private `main` changes native code:

1. `make -C NebulOS-Scaffolding build-libs` (private)
2. Copy `build/*.so` → `synrix-demo/lib/linux-aarch64/`
3. Tag and push `synrix-demo` only

Never copy `src/` into `synrix-demo`.

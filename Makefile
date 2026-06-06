.PHONY: build run run-stress run-expert run-fresh run-interactive setup setup-corpus \
  demo-screen-record demo-operational-loop demo-computational-memory demo-behavioral-evidence

# ── Docker path (recommended) ─────────────────────────────────────────────────
build:
	docker build -t synrix-gate .

# ── Bare-metal path ───────────────────────────────────────────────────────────
# Copy pre-built native libraries for the current architecture into build/.
setup:
	@ARCH=$$(uname -m); \
	LIBDIR=lib/linux-$$ARCH; \
	if [ ! -d "$$LIBDIR" ]; then \
	  echo "[ERROR] No pre-built libraries for $$ARCH (looked in $$LIBDIR)"; exit 1; fi; \
	mkdir -p build; \
	cp $$LIBDIR/*.so build/; \
	echo "[OK] Copied $$(ls $$LIBDIR/*.so | wc -l) libraries from $$LIBDIR to build/"

# Download CWRU bearing corpus and build H-IVF index (needed for demo_e2e_pipeline.py).
# Requires internet access (~134 MB download). Cached under analysis/cwru_raw/.
setup-corpus:
	PYTHONPATH=. SYNRIX_LIB_PATH=build python3 scripts/prepare_cwru_corpus.py

run:
	docker run --rm synrix-gate

run-stress:
	docker run --rm synrix-gate python3 scripts/demo_synrix_gate.py --stress-scale 50000

run-expert:
	docker run --rm -e SCM_TINY_EXPERT_DISPATCH=1 synrix-gate \
	  python3 scripts/smoke_scm_tiny_expert_dispatch.py

run-fresh:
	docker run --rm synrix-gate python3 scripts/demo_synrix_gate.py --train-fresh

# Interactive web demo — three-layer live detection (bare-metal only; requires make setup + make setup-corpus)
# Open http://localhost:5050 after running.
run-interactive:
	PYTHONPATH=. SYNRIX_LIB_PATH=build python3 scripts/demo_interactive.py

# Paced terminal output for screen recording (gate + e2e). See docs/DEMO_SPLIT.md
demo-screen-record:
	bash scripts/demo_screen_record.sh

# PHI optimization transfer demo — three-phase ladder: remember → transfer → recognize.
# Works from any clone (fixture mode). Live mode: SYNRIX_LATTICE=... TARGET_BINARY=... make demo-phi-transfer --live
demo-phi-transfer:
	python3 scripts/demo_phi_transfer.py $(ARGS)

# Behavioral memory thesis demo — three computational domains, same artifact lifecycle.
# Requires: make setup. For live Act 1: LLAMA_BIN=... MODEL_PATH=... make demo-computational-memory
# Without those env vars, falls back to --fixtures automatically.
demo-computational-memory:
	PYTHONPATH=. SYNRIX_LIB_PATH=build python3 scripts/demo_computational_memory.py

# Behavioral evidence demo — three acts: diff, search, prove.
# Works from any clone (fixture mode). Live mode: SYNRIX_LATTICE=... make demo-behavioral-evidence ARGS="--live"
demo-behavioral-evidence:
	python3 scripts/demo_behavioral_evidence.py $(ARGS)

# Operational loop: streams events through full stack, halts on foreign domain.
# Writes receipt to receipts/latest_operational_loop.jsonl.
demo-operational-loop:
	PYTHONPATH=. SYNRIX_LIB_PATH=build python3 scripts/demo_operational_loop.py \
	  --receipt receipts/latest_operational_loop.jsonl

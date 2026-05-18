#!/usr/bin/env bash
# PUBLIC — screen record: gate + E2E (no proprietary phi/WAVE source required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=".:${PYTHONPATH:+:$PYTHONPATH}"
export SYNRIX_LIB_PATH="${SYNRIX_LIB_PATH:-$ROOT/build}"

if [[ ! -f "$SYNRIX_LIB_PATH/libsynrix.so" ]]; then
  echo "Running make setup ..." >&2
  make setup
fi

if [[ ! -f analysis/cwru_corpus.npz ]]; then
  echo "CWRU corpus missing — run: make setup-corpus (or use Docker build)" >&2
  echo "Continuing with gate-only if e2e fails ..." >&2
fi

clear
echo "  Synrix public demo — record Part 1 (edge stack)"
echo "  IP stack (WAVE 41k + phi hotswap): private repo, Part 2 video"
echo ""
sleep 2
clear

echo "══════════════════════════════════════════════════════════"
echo "  Part A — Behavioral equivalence gate"
echo "══════════════════════════════════════════════════════════"
PYTHONPATH=. SYNRIX_LIB_PATH="$SYNRIX_LIB_PATH" python3 scripts/demo_synrix_gate.py
echo ""
sleep 2

if [[ -f analysis/cwru_corpus.npz && -f analysis/cwru_ivf.ivfp ]]; then
  echo "  The same behavioral equivalence machinery used to validate the canonical"
  echo "  student is now applied live during cross-domain routing."
  echo ""
  sleep 2
  echo "══════════════════════════════════════════════════════════"
  echo "  Part B — End-to-end pipeline (lattice → AION → SCM)"
  echo "══════════════════════════════════════════════════════════"
  PYTHONPATH=. SYNRIX_LIB_PATH="$SYNRIX_LIB_PATH" python3 scripts/demo_e2e_pipeline.py
else
  echo "[SKIP] Part B — run: make setup-corpus"
fi

echo ""
echo "Done. Drop your .mp4 in media/ — see media/README.md"

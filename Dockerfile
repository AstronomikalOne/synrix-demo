FROM python:3.12-slim

LABEL description="Synrix edge inference — behavioral equivalence gate + router benchmark"
LABEL version="1.0"

WORKDIR /app

COPY scripts/demo_synrix_gate.py        scripts/demo_synrix_gate.py
COPY scripts/demo_e2e_pipeline.py       scripts/demo_e2e_pipeline.py
COPY scripts/demo_interactive.py        scripts/demo_interactive.py
COPY scripts/benchmark_router_inference.py  scripts/benchmark_router_inference.py
COPY scripts/smoke_scm_tiny_expert_dispatch.py  scripts/smoke_scm_tiny_expert_dispatch.py
COPY scripts/prepare_cwru_corpus.py     scripts/prepare_cwru_corpus.py
COPY experiments/                       experiments/
COPY analysis/formal_artifacts/scm_tiny/scm_tiny_mixed_unsw_wave_cwru_cpath.npz \
                                        analysis/formal_artifacts/scm_tiny/scm_tiny_mixed_unsw_wave_cwru_cpath.npz
COPY analysis/formal_artifacts/scm_tiny/scm_tiny_cwru_expert.npz \
                                        analysis/formal_artifacts/scm_tiny/scm_tiny_cwru_expert.npz
COPY analysis/formal_artifacts/scm_tiny/scm_tiny_wave_expert.npz \
                                        analysis/formal_artifacts/scm_tiny/scm_tiny_wave_expert.npz
COPY analysis/formal_artifacts/scm_tiny/scm_tiny_unsw_heldout_random_train.npz \
                                        analysis/formal_artifacts/scm_tiny/scm_tiny_unsw_heldout_random_train.npz
COPY analysis/formal_artifacts/scm_tiny/demo_gate_fixture.json \
                                        analysis/formal_artifacts/scm_tiny/demo_gate_fixture.json
COPY lib/                               lib/

RUN pip install --no-cache-dir numpy scipy

# Install pre-built native libraries for the target architecture.
# Docker always runs Linux containers, so only linux-* paths apply here.
# Native Mac/Windows binaries (dylib/dll) are in lib/darwin-*/lib/win32-x86_64/.
RUN mkdir -p build && \
    ARCH=$(uname -m) && \
    if   [ "$ARCH" = "aarch64" ]; then LIBDIR=linux-aarch64; \
    elif [ "$ARCH" = "x86_64"  ]; then LIBDIR=linux-x86_64; \
    else echo "Unsupported arch: $ARCH" && exit 1; fi && \
    if [ -f "lib/${LIBDIR}/libsynrix.so" ]; then \
        cp "lib/${LIBDIR}/libsynrix.so"                build/libsynrix.so && \
        cp "lib/${LIBDIR}/liblattice_expert_train.so"  build/liblattice_expert_train.so && \
        cp "lib/${LIBDIR}/libaion_semantic_index.so"   build/libaion_semantic_index.so; \
    else \
        echo ""; \
        echo "  Pre-built binaries not available for ${ARCH} (${LIBDIR})."; \
        echo "  See docs/BUILDING.md for options."; \
        echo ""; \
        exit 1; \
    fi

RUN test -f build/libsynrix.so && \
    test -f build/liblattice_expert_train.so && \
    test -f build/libaion_semantic_index.so && \
    echo "Native libraries ready ($(uname -m))"

# CWRU corpus: copy pre-built artifacts from the local build context when available
# (avoids network downloads that can fail mid-build and produce a partial corpus).
# If neither file is present in the build context, prepare_cwru_corpus.py downloads
# the raw .mat files from the CWRU site and builds from scratch.
COPY analysis/cwru_corpus.npz  analysis/cwru_corpus.npz
COPY analysis/cwru_ivf.ivfp    analysis/cwru_ivf.ivfp

RUN if [ -f analysis/cwru_corpus.npz ] && [ -f analysis/cwru_ivf.ivfp ]; then \
        echo "  CWRU corpus present — skipping download"; \
    else \
        PYTHONPATH=/app SYNRIX_LIB_PATH=/app/build python3 scripts/prepare_cwru_corpus.py; \
    fi

ENV PYTHONPATH=/app
ENV SYNRIX_LIB_PATH=/app/build
# AION_SEMANTIC_INDEX_SCALAR is intentionally NOT set here — on aarch64 the
# hardware dispatcher selects the native NEON INT8 kernel automatically.
# Set it explicitly when running on x86_64: docker run -e AION_SEMANTIC_INDEX_SCALAR=1 ...

# Expert Library paths — activate with: docker run -e SCM_TINY_EXPERT_DISPATCH=1
ENV SCM_TINY_NPZ=/app/analysis/formal_artifacts/scm_tiny/scm_tiny_mixed_unsw_wave_cwru_cpath.npz
ENV SCM_TINY_NPZ_WAVE=/app/analysis/formal_artifacts/scm_tiny/scm_tiny_wave_expert.npz
ENV SCM_TINY_NPZ_CWRU=/app/analysis/formal_artifacts/scm_tiny/scm_tiny_cwru_expert.npz
ENV SCM_TINY_NPZ_UNSW=/app/analysis/formal_artifacts/scm_tiny/scm_tiny_unsw_heldout_random_train.npz

CMD ["python3", "scripts/demo_synrix_gate.py"]

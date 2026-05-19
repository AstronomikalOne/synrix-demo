# Building native libraries

The demo requires three native libraries:

```
build/libsynrix.so                  # Persistent lattice engine
build/liblattice_expert_train.so    # Softmax-linear C trainer
build/libaion_semantic_index.so     # AION512 semantic vector index (e2e pipeline only)
```

## Pre-built binaries (included)

| Platform | Gate demo | E2E pipeline | Location |
|---|---|---|---|
| `linux-aarch64` (Jetson Orin Nano, Raspberry Pi 5, etc.) | ✓ | ✓ | `lib/linux-aarch64/` |
| `linux-x86_64` | ✓ | ✓ (via CI build) | `lib/linux-x86_64/` |
| `darwin-arm64` (Apple Silicon) | contact for access | contact for access | — |
| `win32-x86_64` | contact for access | contact for access | — |

## x86_64 Linux

`lib/linux-x86_64/` includes all three libraries. `libaion_semantic_index.so` is built for x86_64 via the `.github/workflows/build-x86_64-libs.yml` CI workflow (scalar path, no NEON). If you cloned before the workflow ran, rebuild the Docker image after the workflow commits the binary.

## Providing your own binaries

If you have obtained the native libraries through other means, place them at:

```
lib/linux-x86_64/libsynrix.so
lib/linux-x86_64/liblattice_expert_train.so
lib/linux-x86_64/libaion_semantic_index.so   # optional — enables e2e pipeline
```

Then rebuild the Docker image:

```bash
docker build -t synrix-gate .
```

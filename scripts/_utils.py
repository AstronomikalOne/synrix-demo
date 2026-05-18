from __future__ import annotations
import numpy as np

AION_VEC_DIM = 512


def encode_512(feat: np.ndarray) -> np.ndarray:
    """Tile a feature vector to 512 floats and L2-normalise."""
    v    = np.zeros(AION_VEC_DIM, dtype=np.float32)
    reps = AION_VEC_DIM // len(feat)
    rem  = AION_VEC_DIM  % len(feat)
    v[:reps * len(feat)] = np.tile(feat, reps)
    if rem:
        v[reps * len(feat):] = feat[:rem]
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v

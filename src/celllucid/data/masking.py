"""Masks depend on cell identity/view/seed, never on hidden expression values."""

import hashlib
import numpy as np


def hidden_mask(cell_ids, available, fraction=0.25, seed=42, view=0):
    available = np.asarray(available, dtype=bool)
    if available.ndim != 1 or not 0 < fraction < 1:
        raise ValueError("Expected a one-dimensional availability mask and 0 < fraction < 1")
    candidates = np.flatnonzero(available)
    if len(candidates) < 3:
        raise ValueError("Masked prediction requires at least three measured genes")
    n_hidden = min(len(candidates) - 1, max(2, round(len(candidates) * fraction)))
    result = np.zeros((len(cell_ids), len(available)), dtype=bool)
    for row, cell in enumerate(cell_ids):
        key = f"{seed}:{view}:{cell}".encode()
        local_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        rng = np.random.default_rng(local_seed)
        result[row, rng.choice(candidates, n_hidden, replace=False)] = True
    return result


def masked_views(counts, cell_ids, available, **kwargs):
    mask = hidden_mask(cell_ids, available, **kwargs)
    visible = np.array(counts, dtype=np.float32, copy=True)
    visible[mask] = 0
    return visible, mask

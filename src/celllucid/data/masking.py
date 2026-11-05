"""Masks depend on cell identity/view/seed, never on hidden expression values.

View numbers keep every use apart: 0 is the fixed probe mask of the evaluation pools, 7 their
stability view, 1 the initialization sample, and the p-th training occurrence of a cell uses
TRAINING_VIEW_OFFSET + p, so a repeated cell never sees the same mask twice.
"""

import hashlib

import numpy as np

PROBE_VIEW = 0
INITIALIZATION_VIEW = 1
STABILITY_VIEW = 7
TRAINING_VIEW_OFFSET = 1000


def hidden_mask(cell_ids, available, fraction=0.25, seed=42, view=0):
    """Hidden genes per cell; view is one number for all cells or one per cell."""
    available = np.asarray(available, dtype=bool)
    if available.ndim != 1 or not 0 < fraction < 1:
        raise ValueError("Expected a one-dimensional availability mask and 0 < fraction < 1")
    candidates = np.flatnonzero(available)
    if len(candidates) < 3:
        raise ValueError("Masked prediction requires at least three measured genes")
    n_hidden = min(len(candidates) - 1, max(2, round(len(candidates) * fraction)))
    views = np.broadcast_to(np.asarray(view), (len(cell_ids),))
    result = np.zeros((len(cell_ids), len(available)), dtype=bool)
    for row, (cell, cell_view) in enumerate(zip(cell_ids, views, strict=True)):
        key = f"{seed}:{int(cell_view)}:{cell}".encode()
        local_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        rng = np.random.default_rng(local_seed)
        result[row, rng.choice(candidates, n_hidden, replace=False)] = True
    return result


def masked_views(counts, cell_ids, available, **kwargs):
    mask = hidden_mask(cell_ids, available, **kwargs)
    visible = np.array(counts, dtype=np.float32, copy=True)
    visible[mask] = 0
    return visible, mask

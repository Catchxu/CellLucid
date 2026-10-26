"""Explicit splits; cell splits are development diagnostics, not cross-study validation."""

import numpy as np


def split_indices(obs, *, validation_fraction=0.2, seed=42, group_key=None, split_key=None):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if split_key:
        if split_key not in obs:
            raise ValueError(f"Missing split column: {split_key}")
        labels = obs[split_key].astype(str).str.lower()
        train = np.flatnonzero(labels.isin(["train", "training"]))
        val = np.flatnonzero(labels.isin(["val", "valid", "validation"]))
    elif group_key:
        if group_key not in obs or obs[group_key].isna().any():
            raise ValueError("Group splitting requires complete group metadata")
        groups = obs[group_key].astype(str).to_numpy()
        unique = np.unique(groups)
        if len(unique) < 2:
            raise ValueError("Need at least two groups for a held-out group split")
        shuffled = np.random.default_rng(seed).permutation(unique)
        n_val = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
        heldout = set(shuffled[:n_val])
        is_val = np.array([g in heldout for g in groups])
        train, val = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    else:
        order = np.random.default_rng(seed).permutation(len(obs))
        n_val = min(len(obs) - 1, max(1, round(len(obs) * validation_fraction)))
        val, train = np.sort(order[:n_val]), np.sort(order[n_val:])
    if not len(train) or not len(val):
        raise ValueError("Both training and validation splits must contain cells")
    if set(train) & set(val):
        raise ValueError("Training and validation overlap")
    return train, val

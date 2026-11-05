import numpy as np
import pytest
from celllucid.data.masking import (
    INITIALIZATION_VIEW, PROBE_VIEW, STABILITY_VIEW, TRAINING_VIEW, hidden_mask, masked_views,
)


def test_mask_is_independent_of_target_values_and_batch_order():
    available = np.array([True] * 8 + [False, False])
    x = np.arange(20).reshape(2, 10).astype(float)
    visible, hidden = masked_views(x, ["a", "b"], available, seed=2)
    changed = x.copy()
    changed[hidden] += 1000
    visible2, hidden2 = masked_views(changed, ["a", "b"], available, seed=2)
    np.testing.assert_array_equal(hidden, hidden2)
    np.testing.assert_array_equal(visible, visible2)
    np.testing.assert_array_equal(hidden[::-1], hidden_mask(["b", "a"], available, seed=2))
    assert not hidden[:, ~available].any()
    assert not np.array_equal(hidden, hidden_mask(["a", "b"], available, seed=2, view=1))


def test_views_give_distinct_masks():
    available = np.ones(40, dtype=bool)
    views = (PROBE_VIEW, INITIALIZATION_VIEW, TRAINING_VIEW, STABILITY_VIEW)
    assert len(set(views)) == 4
    masks = [hidden_mask(["a"], available, seed=2, view=v)[0] for v in views]
    assert len({m.tobytes() for m in masks}) == 4


def test_mask_requires_usable_gene_set():
    with pytest.raises(ValueError):
        hidden_mask(["a"], [True, True])

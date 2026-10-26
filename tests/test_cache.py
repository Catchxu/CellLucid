import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from celllucid.data.cache import build_feature_cache, FeatureCache
from celllucid.data.masking import masked_views
from celllucid.trainer.topic import train, prepare_rows


class VisibleEncoder:
    genes = [f"g{i}" for i in range(6)]
    dimension = 3

    def identity(self):
        return {"fingerprint": "fixed-test-backbone"}

    def gene_vectors(self, genes):
        return torch.tensor([[i + 1, (i % 2) + 1, (i % 3) + 1] for i in range(len(genes))]).float()

    def encode(self, counts, genes, obs=None):
        # This test encoder makes visible-expression leakage directly observable.
        return torch.tensor(
            np.column_stack([counts.sum(1), counts[:, 0] + 1, counts[:, 1] + 1])
        ).float()


def fixture_dataset(path):
    x = np.random.default_rng(9).integers(1, 8, size=(20, 6)).astype("f4")
    ad.AnnData(
        x,
        obs=pd.DataFrame(
            {"donor": [f"d{i // 4}" for i in range(20)]}, index=[f"cell{i}" for i in range(20)]
        ),
        var=pd.DataFrame(index=VisibleEncoder.genes),
    ).write_h5ad(path)
    return x


def test_cached_features_use_masked_inputs_and_views_stay_in_same_split(tmp_path):
    source = tmp_path / "source.h5ad"
    x = fixture_dataset(source)
    out = tmp_path / "masked.h5"
    build_feature_cache(source, out, VisibleEncoder(), batch_size=3, mask_fraction=0.4, n_views=2)
    with FeatureCache(out) as cache:
        h, counts, mask = cache.read(np.arange(20))
        visible, _ = masked_views(
            x, [f"cell{i}" for i in range(20)], np.ones(6, bool), fraction=0.4
        )
        np.testing.assert_array_equal(h[:, 0], visible.sum(1))
        np.testing.assert_array_equal(counts, x)
        assert mask.any()
        tr, va, _ = prepare_rows(
            [cache], {"validation_fraction": 0.2, "seed": 5, "group_key": "donor"}
        )
        train_cells = set(cache.indices[tr[:, 1]])
        val_cells = set(cache.indices[va[:, 1]])
        assert train_cells.isdisjoint(val_cells)
        assert len(train_cells) + len(val_cells) == 20
    a = ad.read_h5ad(source)
    a.X[0, 0] += 1
    a.write_h5ad(source)
    with pytest.raises(ValueError, match="changed"):
        FeatureCache(out)


def test_training_pipeline_and_full_cache_rejection(tmp_path):
    source = tmp_path / "source.h5ad"
    fixture_dataset(source)
    out = tmp_path / "masked.h5"
    build_feature_cache(source, out, VisibleEncoder(), batch_size=4, mask_fraction=0.4, n_views=2)
    config = {
        "output_dir": str(tmp_path / "run"),
        "caches": [str(out)],
        "epochs": 2,
        "batch_size": 8,
        "model": {"n_topics": 3, "ot_iterations": 20},
        "split": {"group_key": "donor", "validation_fraction": 0.2, "seed": 4},
    }
    result = train(config)
    assert result["topic_update_norm"] > 0 and np.isfinite(result["initial_validation_ce"])
    assert (tmp_path / "run/topic_bank/manifest.json").exists()
    full = tmp_path / "full.h5"
    build_feature_cache(source, full, VisibleEncoder(), batch_size=4)
    config.update(output_dir=str(tmp_path / "bad_run"), caches=[str(full)])
    with pytest.raises(ValueError, match="full-input"):
        train(config)


def test_pipeline_resume_matches_continuous_training(tmp_path):
    from celllucid.tasks.topic_bank import TopicBank

    source = tmp_path / "source.h5ad"
    fixture_dataset(source)
    cache = tmp_path / "features.h5"
    build_feature_cache(source, cache, VisibleEncoder(), mask_fraction=0.4, n_views=2)
    config = {
        "output_dir": str(tmp_path / "resumed"),
        "caches": [str(cache)],
        "epochs": 1,
        "batch_size": 8,
        "model": {"n_topics": 3, "ot_iterations": 20},
        "split": {"group_key": "donor", "validation_fraction": 0.2, "seed": 4},
    }
    train(config)
    with pytest.raises(FileExistsError, match="already has"):
        train(config)
    config["epochs"] = 2
    result = train(config, tmp_path / "resumed/checkpoints/latest")
    resumed = TopicBank(result["bank"])
    config["output_dir"] = str(tmp_path / "continuous")
    fresh_result = train(config)
    continuous = TopicBank(fresh_result["bank"])
    torch.testing.assert_close(resumed.topics, continuous.topics, rtol=0, atol=0)
    torch.testing.assert_close(resumed.dictionary, continuous.dictionary, rtol=0, atol=0)

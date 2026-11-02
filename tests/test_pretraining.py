import json
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from data.prepare_topics import prepare
from celllucid.data.atlas import AtlasBatches
from celllucid.trainer.topic import train, forward_batch
from celllucid.tasks.topic_bank import TopicBank


class VisibleEncoder(torch.nn.Module):
    genes = [f"g{i}" for i in range(6)]
    dimension = 3
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.eval()

    def identity(self):
        return {"fingerprint": "fixed-test-backbone"}

    def gene_vectors(self, genes):
        return torch.tensor([[i + 1, i % 2 + 1, i % 3 + 1] for i in range(len(genes))]).float()

    @torch.no_grad()
    def encode(self, counts, genes, obs=None):
        return torch.tensor(
            np.column_stack([counts.sum(1), counts[:, 0] + 1, counts[:, 1] + 1])
        ).float()


def corpus(tmp_path):
    records = []
    for label, study, normalized in [
        ("a", "s1", False),
        ("b", "s2", False),
        ("duplicate", "s1", False),
        ("normalized", "s3", True),
    ]:
        path = tmp_path / (label + ".h5ad")
        x = np.random.default_rng(9).integers(1, 8, size=(20, 6)).astype("f4")
        if normalized:
            x += 0.25
        ad.AnnData(
            (sparse.csc_matrix(x) if label == "a" else sparse.csr_matrix(x)),
            obs=pd.DataFrame(index=[f"cell{i}" for i in range(20)]),
            var=pd.DataFrame(index=VisibleEncoder.genes),
        ).write_h5ad(path)
        if label == "b":
            with h5py.File(path, "r+") as h:
                del h["obs"]
                del h["var"]
                h.create_dataset(
                    "obs",
                    data=np.array(
                        [(f"cell{i}".encode(),) for i in range(20)], dtype=[("index", "S12")]
                    ),
                )
                h.create_dataset(
                    "var",
                    data=np.array(
                        [(g.encode(),) for g in VisibleEncoder.genes], dtype=[("index", "S12")]
                    ),
                )
        records.append(
            {
                "file_id": label,
                "file_version": "v1",
                "size_bytes": path.stat().st_size,
                "format": "h5ad",
                "file_name": path.name,
                "local_path": str(path),
                "project_ids": [study],
                "metadata": {"protocols": []},
            }
        )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"records": records}))
    checkpoint = tmp_path / "backbone/vocabs"
    checkpoint.mkdir(parents=True)
    with h5py.File(checkpoint / "homo_sapiens_gene.h5", "w") as h:
        h.create_dataset("keys", data=np.array(VisibleEncoder.genes, dtype="S2"))
    mapping = tmp_path / "symbols.json"
    mapping.write_text(json.dumps({g: g for g in VisibleEncoder.genes}))
    cfg = {
        "count_sources": ["X"],
        "validation_block": 1000,
        "exclude_methods": ["spatial", "atac"],
        "batch_size": 4,
        "min_genes": 1,
        "shard_cells": 7,
        "split_seed": 42,
        "validation_fraction": 0.2,
    }
    output = tmp_path / "atlas"
    manifest = prepare(inventory, output, cfg, checkpoint.parent, mapping)
    return output / "manifest.json", manifest


def settings():
    return {
        "backbone": {"device": "cpu"},
        "model": {"n_topics": 3, "ot_iterations": 20},
        "training": {
            "seed": 42,
            "batch_size": 4,
            "encoder_batch_size": 2,
            "workers": 0,
            "mask_fraction": 0.4,
            "optimizer": {"lr": 0.005},
            "grad_clip": 1.0,
            "save_every_steps": 2,
            "log_every_steps": 100,
        },
    }


def test_inventory_coverage_shards_and_distributed_partition(tmp_path):
    _, manifest = corpus(tmp_path)
    assert manifest["source_files"] == 4 and manifest["included_files"] == 2
    assert manifest["excluded_files"] == 1 and manifest["cells"] == 40
    train = {s["study_id"] for s in manifest["shards"] if s["split"] == "train"}
    val = {s["study_id"] for s in manifest["shards"] if s["split"] == "validation"}
    assert train.isdisjoint(val)
    for shard in manifest["shards"]:
        assert shard["cells"] <= 7
    rank0 = list(AtlasBatches(manifest, "train", 4, 42, rank=0, world_size=2))
    rank1 = list(AtlasBatches(manifest, "train", 4, 42, rank=1, world_size=2))
    assert len(rank0) == len(rank1)
    cells = {v for batch in rank0 + rank1 for v in batch["cell_ids"]}
    assert len(cells) == 20


def test_online_masking_and_topic_only_training(tmp_path):
    path, manifest = corpus(tmp_path)
    config = settings()
    encoder = VisibleEncoder()
    from torch.utils.data import DataLoader

    batch = next(iter(DataLoader(AtlasBatches(manifest, "train", 4, 42), batch_size=None)))
    h, counts, hidden = forward_batch(encoder, batch, encoder.genes, config["training"], 0)
    torch.testing.assert_close(h[:, 0], (counts * ~hidden).sum(1))
    _, _, other = forward_batch(encoder, batch, encoder.genes, config["training"], 1)
    assert not torch.equal(hidden, other)
    result = train(config, path, tmp_path / "run", 1, backbone=encoder)
    assert result["topic_update_norm"] > 0 and encoder.anchor.grad is None
    assert not encoder.anchor.requires_grad
    assert not list((tmp_path / "run").glob("*.h5"))


def test_online_resume_matches_continuous(tmp_path):
    path, _ = corpus(tmp_path)
    config = settings()
    train(config, path, tmp_path / "resumed", 2, max_steps=1, backbone=VisibleEncoder())
    resumed = train(
        config,
        path,
        tmp_path / "resumed",
        2,
        resume=tmp_path / "resumed/checkpoints/latest",
        backbone=VisibleEncoder(),
    )
    continuous = train(config, path, tmp_path / "continuous", 2, backbone=VisibleEncoder())
    a = TopicBank(Path(resumed["output_dir"]) / f"topic_bank_step_{resumed['step']:09d}")
    b = TopicBank(Path(continuous["output_dir"]) / f"topic_bank_step_{continuous['step']:09d}")
    torch.testing.assert_close(a.topics, b.topics, rtol=0, atol=0)

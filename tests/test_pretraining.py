import json
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from data.prepare_topics import gene_projection, prepare
from celllucid.checkpoint import load_checkpoint
from celllucid.data.cache import TrainingCache, build_cache, cache_initialization
from celllucid.data.masking import TRAINING_VIEW, hidden_mask, masked_views
from celllucid.models.initialization import empty_model
from celllucid.trainer.topic import learning_rate, train
from celllucid.tasks.topic_bank import TopicBank, export_topic_bank, training_gene_scale


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
    def encode(self, counts, genes, obs=None, pad_to=None):
        return torch.tensor(
            np.column_stack([counts.sum(1), counts[:, 0] + 1, counts[:, 1] + 1])
        ).float()


BARCODES = ["".join(np.random.default_rng(i).choice(list("ACGT"), 16)) for i in range(20)]


def corpus(tmp_path):
    records = []
    # Every file holds the same counts, so only shared names or barcodes mark copies:
    # "duplicate" is a log1p copy of "a" listed first, "renamed" a renamed copy of "b", and
    # "colliding" reuses the barcodes of "b" for cells expressing half of its genes.
    # "logged" holds log1p counts of its own cells, "panel" measures half of the genes, and
    # "listed" is excluded by name.
    for label, study, shift, methods, names in [
        ("duplicate", "s1", None, [], [f"cell{i}" for i in range(20)]),
        ("a", "s1", 0, ["10x 3' v3", "scATAC-seq"], [f"cell{i}" for i in range(20)]),
        ("b", "s2", 0, ["Visium Spatial Gene Expression"], [f"b_{c}-1" for c in BARCODES]),
        ("colliding", "s2", 0, [], [f"{c}-2" for c in BARCODES]),
        ("renamed", "s2", 0, [], [f"{c}-1-0" for c in BARCODES]),
        ("normalized", "s3", 0.25, [], [f"normalized{i}" for i in range(20)]),
        ("negative", "s4", -10, [], [f"negative{i}" for i in range(20)]),
        ("peaks", "s5", 0, ["scATAC-seq"], [f"peak{i}" for i in range(20)]),
        ("logged", "s6", None, [], [f"logged{i}" for i in range(20)]),
        ("panel", "s7", 0, [], [f"panel{i}" for i in range(20)]),
        ("listed", "s8", 0, [], [f"listed{i}" for i in range(20)]),
    ]:
        path = tmp_path / (label + ".h5ad")
        x = np.random.default_rng(9).integers(1, 8, size=(20, 6)).astype("f4")
        x = np.log1p(x) if shift is None else x + shift
        if label == "colliding":
            x[:, :3] = 0
        if label == "panel":
            x = x[:, :3]
        ad.AnnData(
            (sparse.csc_matrix(x) if label == "a" else sparse.csr_matrix(x)),
            obs=pd.DataFrame(index=names),
            var=pd.DataFrame(index=VisibleEncoder.genes[: x.shape[1]]),
        ).write_h5ad(path)
        if label == "b":
            with h5py.File(path, "r+") as h:
                del h["obs"]
                del h["var"]
                h.create_dataset(
                    "obs",
                    data=np.array([(n.encode(),) for n in names], dtype=[("index", "S24")]),
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
                "metadata": {"protocols": [{"libraryConstructionApproach": methods}]},
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
        "exclude_methods": ["atac"],
        "min_measured_genes": 4,
        "exclude_files": {"listed.h5ad": "a copy of a"},
        "duplicate_jaccard": 0.9,
        "batch_size": 4,
        "min_genes": 1,
        "shard_cells": 7,
        "validation_studies": ["s3"],
    }
    output = tmp_path / "atlas"
    manifest = prepare(inventory, output, cfg, checkpoint.parent, mapping)
    return output / "manifest.json", manifest


def settings(tmp_path, manifest):
    return {
        "backbone": {"device": "cpu"},
        "cache": {"seed": 42, "mask_fraction": 0.4, "encoder_batch_size": 4,
                  "chunk_samples": 16, "initialization_cells_per_study": 16,
                  "evaluation_pools": None},
        "model": {"n_topics": 3, "temperature": 0.1, "ot_epsilon": 0.1, "ot_iterations": 50,
                  "gene_prior_uniform_share": 0.01, "excluded_genes": None, "kmeans_iterations": 10},
        "training": {"seed": 42, "device": "cpu", "batch_size": 8,
                     "compile_sinkhorn": False, "optimizer": {"lr": 0.01, "weight_decay": 0.0}, "warmup_steps": 0, "min_lr": 0.001,
                     "grad_clip": 1.0, "log_every_steps": 1, "eval_every_steps": 1, "save_every_steps": 1},
        "evaluation": {"pools": None},
    }


def test_inventory_coverage_and_study_split(tmp_path):
    _, manifest = corpus(tmp_path)
    assert manifest["source_files"] == 11 and manifest["included_files"] == 5
    assert manifest["excluded_files"] == 4 and manifest["cells"] == 100
    reports = {r["file_id"]: r for r in json.loads((tmp_path / "atlas/coverage.json").read_text())["reports"]}
    assert reports["a"]["integer_counts"] and not reports["normalized"]["integer_counts"]
    assert reports["negative"]["status"] == "excluded"
    # log1p counts are restored to the integer counts.
    assert reports["logged"]["log1p_counts"] and reports["logged"]["integer_counts"]
    restored = []
    for shard in manifest["shards"]:
        if shard["file_id"] == "logged":
            with h5py.File(shard["path"], "r") as handle:
                restored.append(ad.io.sparse_dataset(handle["X"])[:].toarray())
    counts = np.random.default_rng(9).integers(1, 8, size=(20, 6))
    np.testing.assert_array_equal(np.concatenate(restored), counts)
    # Visium and the RNA files of a multiome project are prepared; ATAC-only files are not.
    assert reports["peaks"]["reason"].startswith("Excluded assay")
    assert reports["panel"]["reason"] == "3 backbone genes measured, fewer than 4"
    assert reports["listed"]["reason"] == "Excluded file: a copy of a"
    assert reports["a"]["retained_cells"] == reports["b"]["retained_cells"] == 20
    # The stored counts are kept over the log1p copy listed first, a renamed copy is found
    # by its barcodes, and different cells that share barcodes are kept.
    matches = {
        label: {(m["file_id"], m["key"]): (m["matched"], m["removed"]) for m in reports[label]["duplicate_matches"]}
        for label in ("duplicate", "colliding", "renamed")
    }
    assert matches["duplicate"] == {("a", "name"): (20, 20)}
    assert matches["colliding"] == {("b", "barcode"): (20, 0)}
    assert matches["renamed"] == {("b", "barcode"): (20, 20), ("colliding", "barcode"): (20, 0)}
    assert [reports[label]["status"] for label in matches] == ["empty", "included", "empty"]
    train_studies = {s["study_id"] for s in manifest["shards"] if s["split"] == "train"}
    validation = {s["study_id"] for s in manifest["shards"] if s["split"] == "validation"}
    assert validation == {"s3"} and train_studies == {"s1", "s2", "s6"}
    for shard in manifest["shards"]:
        assert shard["cells"] <= 7


def test_cache_holds_every_eligible_cell_once_with_its_mask(tmp_path):
    path, manifest = corpus(tmp_path)
    config = settings(tmp_path, manifest)
    result = build_cache(config, path, tmp_path / "cache", backbone=VisibleEncoder())
    cache = TrainingCache(tmp_path / "cache")
    train_cells = []
    for shard in manifest["shards"]:
        if shard["split"] == "train":
            with h5py.File(shard["path"], "r") as handle:
                train_cells += list(handle["cell_ids"].asstr()[:])
    records = []
    for part in sorted((tmp_path / "cache").glob("part_*.h5")):
        with h5py.File(part, "r") as handle:
            records += list(zip(handle["cell_id"].asstr()[:], handle["hidden_bits"][:]))
    assert len(cache) == result["samples"] == len(records) == len(train_cells)
    assert sorted(cell for cell, _ in records) == sorted(train_cells)
    first = next(s["path"] for s in manifest["shards"] if s["split"] == "train")
    with h5py.File(first, "r") as handle:
        cell = handle["cell_ids"].asstr()[0]
        counts = ad.io.sparse_dataset(handle["X"])[np.array([0]), :].toarray().astype("f4")
        available = handle["available"][:]
    index = next(i for i, (c, _) in enumerate(records) if c == cell)
    visible, hidden = masked_views(counts, [cell], available, fraction=0.4, seed=42, view=TRAINING_VIEW)
    h, x, hidden_batch = cache.batch(np.array([index]), "cpu")
    torch.testing.assert_close(h, VisibleEncoder().encode(visible, VisibleEncoder.genes).half().float())
    torch.testing.assert_close(hidden_batch, torch.from_numpy(hidden))
    torch.testing.assert_close(x, torch.from_numpy(counts * hidden))
    np.testing.assert_array_equal(hidden, hidden_mask([cell], available, fraction=0.4, seed=42, view=TRAINING_VIEW))
    studies = {s["study_id"] for s in manifest["shards"] if s["split"] == "train"}
    initialization = cache_initialization(tmp_path / "cache")
    assert initialization["masked"].shape == (16 * len(studies), 3)
    assert np.isclose(initialization["proportion_mean"].sum(), 1)
    # The gene scale for nnls inference covers exactly the cached training cells.
    counts = []
    for shard in manifest["shards"]:
        if shard["split"] == "train":
            with h5py.File(shard["path"], "r") as handle:
                counts.append(ad.io.sparse_dataset(handle["X"])[:].toarray())
    counts = np.concatenate(counts).astype(np.float64)
    scale, source = training_gene_scale(config, path, tmp_path / "cache", workers=2)
    np.testing.assert_allclose(scale, (counts / counts.sum(1, keepdims=True) * 1e6).std(0), rtol=1e-9, atol=1e-6)
    assert source["cells"] == len(cache)


def test_gene_identifiers_align_with_versions_and_symbol_prefixes():
    var = pd.DataFrame(index=["TNMD_ENSG00000000005", "ENSG00000000419.12", "ENSA", "unknown"])
    genes = ["ENSG00000000005", "ENSG00000000419", "ENSG00000143420"]
    projection, available, _ = gene_projection(var, genes, {"ENSA": "ENSG00000143420"})
    assert available.all() and projection.nnz == 3


def test_learning_rate_warmup_and_cosine_decay():
    rates = [learning_rate(step, 10, 1.0, 4, 0.1) for step in range(10)]
    assert rates[:4] == [0.25, 0.5, 0.75, 1.0]
    assert rates[4] == 1.0 and abs(rates[-1] - 0.1) < 1e-12
    assert all(a > b for a, b in zip(rates[4:], rates[5:]))
    assert {learning_rate(step, 10, 0.5, 0, 0.5) for step in range(10)} == {0.5}


def test_training_resume_matches_continuous_and_exports(tmp_path):
    path, manifest = corpus(tmp_path)
    config = settings(tmp_path, manifest)
    build_cache(config, path, tmp_path / "cache", backbone=VisibleEncoder())
    continuous = train(config, tmp_path / "cache", tmp_path / "continuous")
    train(config, tmp_path / "cache", tmp_path / "resumed", max_steps=1)
    resumed = train(config, tmp_path / "cache", tmp_path / "resumed", resume=True)
    steps = len(TrainingCache(tmp_path / "cache")) // 8
    assert steps >= 2 and continuous["step"] == resumed["step"] == steps
    gene_vectors = cache_initialization(tmp_path / "cache")["gene_vectors"]
    models = []
    for run in ("continuous", "resumed"):
        model = empty_model(gene_vectors, VisibleEncoder.genes, config["model"], "cpu", False)
        load_checkpoint(tmp_path / run / "checkpoints/latest", model)
        models.append(model)
    for name, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[name], rtol=0, atol=0)
    log = [json.loads(line) for line in (tmp_path / "continuous/metrics.jsonl").open()]
    assert {"ce_gain_over_background", "cell_effective_topics", "cell_topic_cosine", "max_topic_mass"} <= set(log[0])
    bank = TopicBank(export_topic_bank(models[0], VisibleEncoder.genes, VisibleEncoder.genes, tmp_path / "bank",
                                       backbone={"fingerprint": "fixed-test-backbone"}))
    h = torch.randn(5, 3) + 10
    torch.testing.assert_close(bank.transform(h), models[0].theta(h, inference=True))

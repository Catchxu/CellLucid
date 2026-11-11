"""Portable topic definitions and cell assignment without frozen backbone weights.

A bank folder holds topics.safetensors, genes.json (Ensembl IDs in backbone input order),
symbols.json (gene symbols aligned with genes.json) and manifest.json (settings and checksums).

Cells get topic usage in one of three ways: from their TranscriptFormer embedding (transform, the
default), or in gene space from their counts with the dictionary fixed: bag-of-words maximum
likelihood under the training loss (fold_in), or starCAT-style non-negative least squares
(project). project needs the bank's gene_scale, each gene's standard deviation of counts per
million over training cells; banks without it (an optional tensor) serve the other two, and
celllucid-add-gene-scale adds it to an exported bank.
"""

from multiprocessing import Pool
from pathlib import Path
import argparse
import json
import math
import re
import anndata as ad
import h5py
import numpy as np
import torch
from scipy import sparse
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from ..config import file_digest, write_json, fingerprint

FORMAT_VERSION = 5
FILES = ("topics.safetensors", "genes.json", "symbols.json", "manifest.json")


def _write_bank(tensors, genes, symbols, output_dir, payload):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    save_file(tensors, str(output / "topics.safetensors"))
    write_json(output / "genes.json", list(genes))
    write_json(output / "symbols.json", list(symbols))
    payload = {**payload, "files": {name: file_digest(output / name) for name in FILES[:-1]}}
    payload["bank_id"] = fingerprint(payload)
    write_json(output / "manifest.json", payload)
    return output


def export_topic_bank(model, genes, symbols, output_dir, *, backbone, training=None, gene_scale=None,
                      gene_scale_source=None):
    if len(genes) != len(model.gene_vectors) or len(set(genes)) != len(genes) or len(symbols) != len(genes):
        raise ValueError("Invalid ordered gene vocabulary")
    with torch.no_grad():
        tensors = {
            "topics": F.normalize(model.topics, dim=-1).detach().cpu().contiguous(),
            "dictionary": model.dictionary().detach().cpu().contiguous(),
            "gene_prior": model.gene_prior.detach().cpu().contiguous(),
            "inference_mean": model.inference_mean.detach().cpu().contiguous(),
            "train_mean": model.train_mean.detach().cpu().contiguous(),
            "gene_map": model.gene_map.weight.detach().cpu().contiguous(),
        }
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.hyperparameters(),
        "backbone": backbone,
        "training": training or {},
        "gene_representation": "projected_input",
        "cell_input": "unmasked counts, centered by inference_mean",
    }
    if gene_scale is not None:
        tensors["gene_scale"] = torch.as_tensor(gene_scale, dtype=torch.float32).contiguous()
        payload["gene_scale"] = gene_scale_source
    return _write_bank(tensors, genes, symbols, output_dir, payload)


def add_gene_scale(source, output_dir, gene_scale, gene_scale_source):
    """Copy of the bank at `source` with a gene scale (one value per bank gene) and its provenance."""
    bank = TopicBank(source)
    if len(gene_scale) != len(bank.genes):
        raise ValueError("The gene scale needs one value per bank gene")
    tensors = load_file(str(bank.path / "topics.safetensors"))
    tensors["gene_scale"] = torch.as_tensor(gene_scale, dtype=torch.float32).contiguous()
    payload = {k: v for k, v in bank.manifest.items() if k not in ("files", "bank_id")}
    payload["gene_scale"] = gene_scale_source
    return _write_bank(tensors, bank.genes, bank.symbols, output_dir, payload)


class TopicBank:
    def __init__(self, path="assets/topic_banks/default", device="cpu"):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        unsigned = {k: v for k, v in self.manifest.items() if k != "bank_id"}
        if fingerprint(unsigned) != self.manifest.get("bank_id"):
            raise ValueError("Topic-bank manifest fingerprint mismatch")
        if self.manifest["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported topic bank")
        for name, expected in self.manifest["files"].items():
            if file_digest(self.path / name) != expected:
                raise ValueError(f"Topic bank checksum mismatch: {name}")
        self.genes = json.loads((self.path / "genes.json").read_text())
        self.symbols = json.loads((self.path / "symbols.json").read_text())
        tensors = load_file(str(self.path / "topics.safetensors"), device=str(device))
        self.topics = tensors["topics"]
        self.dictionary = tensors["dictionary"]
        self.gene_prior = tensors["gene_prior"]
        self.inference_mean = tensors["inference_mean"]
        self.gene_scale = tensors.get("gene_scale")  # optional: only project needs it
        self.temperature = self.manifest["model"]["temperature"]
        if self.dictionary.shape != (len(self.topics), len(self.genes)):
            raise ValueError("Dictionary shape does not match ordered genes/topics")
        if (
            not torch.isfinite(self.dictionary).all()
            or (self.dictionary < 0).any()
            or not torch.allclose(
                self.dictionary.sum(-1),
                torch.ones(len(self.topics), device=self.dictionary.device),
                atol=1e-5,
            )
        ):
            raise ValueError("Invalid dictionary probabilities")
        # Genes the training loss scored: all but those whose symbol matches model.excluded_genes.
        training = self.manifest["training"]
        excluded = training["model"]["excluded_genes"] if "model" in training else None
        self.targets = torch.tensor(
            [excluded is None or re.match(excluded, symbol) is None for symbol in self.symbols],
            device=self.dictionary.device,
        )

    @torch.no_grad()
    def transform(self, embeddings):
        """Topic usage of cells encoded from unmasked counts."""
        h = torch.as_tensor(embeddings, device=self.topics.device, dtype=torch.float32)
        if h.ndim != 2 or h.shape[1] != self.topics.shape[1] or not torch.isfinite(h).all():
            raise ValueError("Cell embeddings have invalid shape or values")
        scores = F.normalize(h - self.inference_mean, dim=-1) @ self.topics.T
        return torch.softmax(scores / self.temperature, dim=-1)

    def _counts(self, counts):
        x = torch.as_tensor(counts, device=self.dictionary.device, dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != len(self.genes) or not torch.isfinite(x).all() or (x < 0).any():
            raise ValueError("Counts must be finite, non-negative and cells x bank genes")
        if (x[:, self.targets].sum(1) <= 0).any():
            raise ValueError("A cell has no counts on the bank's target genes")
        return x

    @torch.no_grad()
    def fold_in(self, counts, iterations=500, tolerance=1e-6):
        """Bag-of-words topic usage of cells from their counts (cells x bank genes), dictionary fixed.

        The maximum-likelihood usage under the training loss, which scores usage @ dictionary
        renormalized over the target genes: EM runs on each topic renormalized over those genes,
        and the result maps back to the model's usage (each topic divided by its mass on the
        target genes, then renormalized). EM stops after `iterations` or once no usage changes by
        more than `tolerance` in one step.
        """
        x = self._counts(counts)[:, self.targets]
        d = self.dictionary[:, self.targets]
        mass = d.sum(1)
        d = d / mass[:, None]
        total = x.sum(1, keepdim=True)
        theta = torch.full((len(x), len(d)), 1 / len(d), device=x.device)
        for _ in range(iterations):
            new = theta * ((x / (theta @ d).clamp_min(torch.finfo(torch.float32).tiny)) @ d.T) / total
            converged = (new - theta).abs().max() <= tolerance
            theta = new
            if converged:
                break
        theta = theta / mass
        return theta / theta.sum(1, keepdim=True)

    @torch.no_grad()
    def project(self, counts, max_value=10.0, iterations=5000, tolerance=1e-7):
        """starCAT-style topic usage of cells from their counts (cells x bank genes), dictionary fixed.

        Each cell's counts per million, every gene divided by the bank's gene_scale (its standard
        deviation of counts per million over training cells), are fitted by non-negative least
        squares to the dictionary in the same units, on the target genes with a positive scale;
        usages are normalized to sum to one. Scaled values of cells and dictionary are clipped at
        `max_value` (as scanpy's and Seurat's scaling do), so genes that are very rare in training,
        whose tiny scale would inflate them, cannot dominate the fit. FISTA on the Gram matrix stops
        after `iterations` or once a step changes no usage by more than `tolerance` times the largest.
        """
        if self.gene_scale is None:
            raise ValueError("This topic bank has no gene_scale; add one with celllucid-add-gene-scale")
        x = self._counts(counts)
        scale = self.gene_scale.float()
        genes = self.targets & (scale > 0)
        x = ((x / x.sum(1, keepdim=True) * 1e6)[:, genes] / scale[genes]).clamp_max(max_value)
        spectra = (self.dictionary[:, genes] * 1e6 / scale[genes]).clamp_max(max_value)
        gram = spectra @ spectra.T
        b = x @ spectra.T
        step = 1 / torch.linalg.eigvalsh(gram).max()
        theta = torch.zeros_like(b)
        y, t = theta.clone(), 1.0
        for _ in range(iterations):
            new = (y - step * (y @ gram - b)).clamp_min(0)
            t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
            y = new + (t - 1) / t_next * (new - theta)
            converged = (new - theta).abs().max() <= tolerance * new.abs().max()
            theta, t = new, t_next
            if converged:
                break
        return theta / theta.sum(1, keepdim=True)

    def top_genes(self, n=20):
        values, indices = self.dictionary.topk(min(n, len(self.genes)), dim=-1)
        return [
            [
                {"gene": self.symbols[j], "gene_id": self.genes[j], "weight": float(v)}
                for j, v in zip(ii.tolist(), vv.tolist(), strict=True)
            ]
            for ii, vv in zip(indices, values, strict=True)
        ]


def _shard_moments(path, excluded):
    """Cells, and per-gene sums and sums of squares of counts per million, of a corpus shard's
    cells outside `excluded`."""
    with h5py.File(path, "r") as handle:
        keep = ~np.isin(handle["cell_ids"].asstr()[:], list(excluded))
        counts = ad.io.sparse_dataset(handle["X"])[:][keep].astype(np.float64)
    cpm = sparse.diags(1e6 / np.asarray(counts.sum(1)).ravel()) @ counts
    return cpm.shape[0], np.asarray(cpm.sum(0)).ravel(), np.asarray(cpm.multiply(cpm).sum(0)).ravel()


def training_gene_scale(config, manifest_path, cache, workers=16):
    """Gene scale of TopicBank.project: each gene's standard deviation of counts per million over
    every training cell of `cache`, i.e. the train-split cells of the corpus manifest it was built
    from minus the evaluation-pool cells, as in build_cache. Raw counts are streamed from the
    corpus shards, `workers` shards at a time. Returns the scale and its provenance."""
    from ..data.cache import cache_identity

    identity = cache_identity(cache)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["identity"] != identity["manifest_identity"]:
        raise ValueError("The cache was built from a different corpus manifest")
    pools = config["cache"]["evaluation_pools"]
    evaluation = (set() if pools is None else
                  {c["cell_id"] for c in json.loads((Path(pools) / "cells.json").read_text())["cells"]})
    shards = [shard["path"] for shard in manifest["shards"] if shard["split"] == "train"]
    with Pool(workers) as pool:
        moments = pool.starmap(_shard_moments, [(path, evaluation) for path in shards])
    cells = sum(m[0] for m in moments)
    if cells != identity["samples"]:
        raise ValueError("The training cells do not match the cache")
    mean = sum(m[1] for m in moments) / cells
    variance = sum(m[2] for m in moments) / cells - mean**2
    scale = np.sqrt(np.clip(variance, 0, None))  # clip float rounding below zero
    source = {"statistic": "standard deviation of counts per million over every training cell", "cells": cells,
              "manifest_identity": identity["manifest_identity"]}
    return scale, source


def add_gene_scale_main():
    parser = argparse.ArgumentParser(
        description="Add the training gene scale (for nnls inference) to an exported topic bank")
    parser.add_argument("--bank", required=True, help="Exported topic-bank folder")
    parser.add_argument("--config", required=True, help="Configuration the bank's training cache was built with")
    parser.add_argument("--manifest", required=True, help="Corpus manifest the training cache was built from")
    parser.add_argument("--cache", required=True, help="Training cache the bank was trained on")
    parser.add_argument("--output", required=True, help="New topic-bank folder")
    parser.add_argument("--workers", type=int, default=16, help="Corpus shards read in parallel")
    args = parser.parse_args()
    from ..config import load_config
    from ..data.cache import cache_identity

    bank = TopicBank(args.bank)
    if bank.manifest["training"]["cache"]["manifest_identity"] != cache_identity(args.cache)["manifest_identity"]:
        raise ValueError("The bank was trained on a different cache")
    scale, source = training_gene_scale(load_config(args.config), args.manifest, args.cache, args.workers)
    print(add_gene_scale(args.bank, args.output, scale, source))


def main():
    parser = argparse.ArgumentParser(description="Export a pretrained topic checkpoint as a topic bank")
    parser.add_argument("--checkpoint", default="runs/hca_pretraining/checkpoints/latest")
    parser.add_argument("--cache", required=True, help="Training cache the checkpoint was trained on")
    parser.add_argument("--output", default="runs/hca_pretraining/topic_bank")
    args = parser.parse_args()
    from ..checkpoint import load_checkpoint
    from ..data.cache import cache_identity, cache_initialization
    from ..models.initialization import empty_model, gene_symbols

    info = json.loads((Path(args.checkpoint) / "checkpoint.json").read_text())
    identity = cache_identity(args.cache)
    if info["metadata"]["cache"]["manifest_identity"] != identity["manifest_identity"]:
        raise ValueError("Checkpoint was trained on a different cache")
    model = empty_model(cache_initialization(args.cache)["gene_vectors"], identity["genes"], info["metadata"]["model"],
                        "cpu", False)
    load_checkpoint(args.checkpoint, model)
    symbols = gene_symbols(identity["genes"], info["metadata"]["model"]["gene_symbols"])
    print(export_topic_bank(model, identity["genes"], symbols, args.output,
                            backbone={"fingerprint": identity["backbone_fingerprint"]},
                            training=info["metadata"]))


if __name__ == "__main__":
    main()

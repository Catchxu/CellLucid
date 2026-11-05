"""Study-balanced training cache of frozen embeddings with a fresh mask for every occurrence.

Plan. Every training study contributes the same number of samples. Its eligible cells (train
split, not dropped duplicates, not evaluation-pool cells or their duplicate copies) are
shuffled in passes; pass p gives each cell its p-th occurrence, masked with view
TRAINING_VIEW_OFFSET + p. Large studies use part of one pass and small studies repeat in
later passes, never with the same mask. A study-balanced initialization sample is encoded
with INITIALIZATION_VIEW and without a mask; it provides the k-means topics, the training and
inference cell means and the empirical gene marginal.

Build. Ranks encode disjoint chunks and write each chunk atomically as one HDF5 part, so an
interrupted build resumes by skipping finished parts. A part stores FP16 embeddings, the
hidden-gene set as packed bits, and the nonzero hidden counts in sparse form.
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from ..config import file_digest, fingerprint, load_config, write_json
from .masking import INITIALIZATION_VIEW, TRAINING_VIEW_OFFSET, masked_views

INITIALIZATION_STREAM = 2**31 - 1  # seed stream of the initialization sample, apart from passes
VIEW_RULE = (f"occurrence p of a cell uses mask view {TRAINING_VIEW_OFFSET} + p; "
             f"the initialization sample uses view {INITIALIZATION_VIEW} and an unmasked encoding")


def excluded_cells(duplicate_of, evaluation_ids):
    """Dropped duplicates, evaluation cells, and every duplicate copy of an evaluation cell."""
    evaluation_ids = set(evaluation_ids)
    twins = {kept for dropped, kept in duplicate_of.items() if dropped in evaluation_ids}
    return set(duplicate_of) | evaluation_ids | twins


def eligible_cells(manifest, excluded):
    studies = defaultdict(list)
    for shard in manifest["shards"]:
        if shard["split"] != "train":
            continue
        with h5py.File(shard["path"], "r") as handle:
            ids = handle["cell_ids"].asstr()[:]
        studies[shard["study_id"]].extend(
            (shard["path"], row, cell_id) for row, cell_id in enumerate(ids) if cell_id not in excluded)
    return dict(sorted(studies.items()))


def training_plan(eligible, samples_per_study, seed):
    """(path, row, cell_id, study, view) for each sample, study-balanced by upsampling."""
    samples = []
    for position, (study, cells) in enumerate(eligible.items()):
        taken, occurrence = 0, 0
        while taken < samples_per_study:
            order = np.random.default_rng(np.random.SeedSequence([seed, position, occurrence])).permutation(len(cells))
            for index in order[: samples_per_study - taken]:
                samples.append((*cells[index], study, TRAINING_VIEW_OFFSET + occurrence))
            taken += min(len(cells), samples_per_study - taken)
            occurrence += 1
    return samples


def initialization_plan(eligible, cells_per_study, seed):
    picks = []
    for position, (study, cells) in enumerate(eligible.items()):
        rng = np.random.default_rng(np.random.SeedSequence([seed, position, INITIALIZATION_STREAM]))
        picks.extend((*cells[index], study, INITIALIZATION_VIEW) for index in rng.permutation(len(cells))[:cells_per_study])
    return picks


def read_samples(samples):
    """Counts, cell IDs, assays and availability for samples, grouped by shard path."""
    by_path = defaultdict(list)
    for position, sample in enumerate(samples):
        by_path[sample[0]].append(position)
    for path, positions in by_path.items():
        rows = np.array([samples[p][1] for p in positions])
        unique, inverse = np.unique(rows, return_inverse=True)
        with h5py.File(path, "r") as handle:
            counts = ad.io.sparse_dataset(handle["X"])[unique, :].toarray().astype("f4")[inverse]
            ids = handle["cell_ids"].asstr()[unique][inverse]
            assay = handle["assay"].asstr()[unique][inverse]
            available = handle["available"][:]
        assert list(ids) == [samples[p][2] for p in positions]
        yield positions, counts, ids, assay, available


def masked_inputs(samples, n_genes, settings):
    """Counts, masked inputs, hidden masks and assays of the samples, in sample order.

    Masks are drawn per shard (its availability) and per sample (its view); encoding then
    runs over all samples together so microbatches stay full across shards.
    """
    n = len(samples)
    counts = np.empty((n, n_genes), dtype="f4")
    visible = np.empty_like(counts)
    hidden = np.empty((n, n_genes), dtype=bool)
    assay = np.empty(n, dtype=object)
    views = np.array([s[4] for s in samples])
    for positions, rows, ids, assays, available in read_samples(samples):
        visible[positions], hidden[positions] = masked_views(
            rows, ids, available, fraction=settings["mask_fraction"], seed=settings["seed"], view=views[positions])
        counts[positions], assay[positions] = rows, assays
    return counts, visible, hidden, assay


def encode(backbone, genes, inputs, assay, microbatch):
    return torch.cat([
        backbone.encode(inputs[s:s + microbatch], genes, pd.DataFrame({"assay": assay[s:s + microbatch]}), pad_to=microbatch)
        for s in range(0, len(inputs), microbatch)
    ]).numpy()


def encode_training_chunk(backbone, genes, samples, settings):
    counts, visible, hidden, assay = masked_inputs(samples, len(genes), settings)
    embeddings = encode(backbone, genes, visible, assay, settings["encoder_batch_size"]).astype("f2")
    targets = counts * hidden
    nonzero = []
    for row in targets:
        index = np.flatnonzero(row)
        nonzero.append((index.astype("u2"), row[index]))
    return embeddings, np.packbits(hidden, axis=1), nonzero


def write_atomic(path, datasets):
    temporary = path.with_name(path.name + ".tmp")
    with h5py.File(temporary, "w") as handle:
        for name, value in datasets.items():
            handle.create_dataset(name, data=value)
    os.replace(temporary, path)


def write_training_part(path, samples, embeddings, bits, nonzero):
    lengths = np.array([len(index) for index, _ in nonzero])
    strings = h5py.string_dtype("utf-8")
    write_atomic(path, {
        "embeddings": embeddings,
        "hidden_bits": bits,
        "nonzero_ptr": np.concatenate([[0], np.cumsum(lengths)]).astype("i8"),
        "nonzero_index": np.concatenate([index for index, _ in nonzero]),
        "nonzero_count": np.concatenate([count for _, count in nonzero]).astype("f4"),
        "cell_id": np.array([s[2] for s in samples], dtype=strings),
        "study": np.array([s[3] for s in samples], dtype=strings),
        "view": np.array([s[4] for s in samples], dtype="i4"),
    })


def build_initialization(path, backbone, genes, samples, settings):
    counts, visible, _, assay = masked_inputs(samples, len(genes), settings)
    strings = h5py.string_dtype("utf-8")
    write_atomic(path, {
        "masked": encode(backbone, genes, visible, assay, settings["encoder_batch_size"]),
        "unmasked": encode(backbone, genes, counts, assay, settings["encoder_batch_size"]),
        "proportion_mean": (counts / counts.sum(1, keepdims=True)).astype("f8").mean(0),
        "gene_vectors": backbone.gene_vectors(genes).numpy(),
        "cell_id": np.array([s[2] for s in samples], dtype=strings),
        "study": np.array([s[3] for s in samples], dtype=strings),
    })


def build_cache(config, manifest_path, output, *, backbone=None):
    distributed = "RANK" in os.environ
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if distributed else (0, 1)
    device = config["backbone"]["device"]
    if distributed:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
        # Other ranks wait at the first barrier while rank 0 encodes the initialization sample.
        dist.init_process_group("nccl", device_id=torch.device(device), timeout=timedelta(hours=6))
    settings = config["cache"]
    output = Path(output).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    duplicates = json.loads(Path(settings["duplicates"]).read_text())
    if duplicates["manifest_identity"] != manifest["identity"]:
        raise ValueError("Duplicate list was computed for a different corpus")
    evaluation = ([] if settings["evaluation_pools"] is None else
                  [c["cell_id"] for c in json.loads((Path(settings["evaluation_pools"]) / "cells.json").read_text())["cells"]])
    eligible = eligible_cells(manifest, excluded_cells(duplicates["duplicate_of"], evaluation))
    samples = sorted(training_plan(eligible, settings["samples_per_study"], settings["seed"]),
                     key=lambda s: (s[0], s[1], s[4]))
    chunks = [samples[i:i + settings["chunk_samples"]] for i in range(0, len(samples), settings["chunk_samples"])]
    genes = json.loads(Path(manifest["genes_path"]).read_text())
    backbone = backbone if backbone is not None else _backbone(config, device)
    if backbone.genes != genes:
        raise ValueError("Corpus gene order differs from the frozen backbone")
    identity = {
        "format_version": 1,
        "manifest_identity": manifest["identity"],
        "backbone_fingerprint": backbone.identity()["fingerprint"],
        "genes": genes,
        "duplicates_digest": file_digest(settings["duplicates"]),
        "evaluation_cells_digest": fingerprint(sorted(evaluation)),
        "settings": {k: settings[k] for k in ("seed", "samples_per_study", "mask_fraction", "chunk_samples",
                                               "initialization_cells_per_study")},
        "views": VIEW_RULE,
        "eligible_cells": {study: len(cells) for study, cells in eligible.items()},
        "samples": len(samples),
        "parts": len(chunks),
    }
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "identity.json").exists():
            if json.loads((output / "identity.json").read_text()) != json.loads(json.dumps(identity)):
                raise ValueError("Existing cache was built with a different identity")
        else:
            write_json(output / "identity.json", identity)
        if not (output / "initialization.h5").exists():
            build_initialization(output / "initialization.h5", backbone, genes,
                                 initialization_plan(eligible, settings["initialization_cells_per_study"], settings["seed"]),
                                 settings)
    if distributed:
        dist.barrier()
    for index in range(rank, len(chunks), world):
        part = output / f"part_{index:05d}.h5"
        if part.exists():
            continue
        write_training_part(part, chunks[index], *encode_training_chunk(backbone, genes, chunks[index], settings))
        print(f"rank {rank}: part {index + 1}/{len(chunks)}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        write_json(output / "complete.json", {"parts": len(chunks), "samples": len(samples)})
    return {"output": str(output), "samples": len(samples), "parts": len(chunks)}


def _backbone(config, device):
    from ..models.backbone import FrozenTranscriptFormer
    return FrozenTranscriptFormer(**{**config["backbone"], "device": device})


def cache_identity(path):
    return json.loads((Path(path) / "identity.json").read_text())


def cache_initialization(path):
    with h5py.File(Path(path) / "initialization.h5", "r") as handle:
        return {name: handle[name][:] for name in ("masked", "unmasked", "proportion_mean", "gene_vectors")}


class TrainingCache:
    """A complete cache held in memory; batches are assembled on the requested device."""

    def __init__(self, path):
        self.path = Path(path)
        self.identity = cache_identity(path)
        complete = json.loads((self.path / "complete.json").read_text())
        if complete["parts"] != self.identity["parts"] or complete["samples"] != self.identity["samples"]:
            raise ValueError("Cache build did not finish")
        parts = {name: [] for name in ("embeddings", "hidden_bits", "nonzero_index", "nonzero_count", "study")}
        lengths = []
        for index in range(complete["parts"]):
            with h5py.File(self.path / f"part_{index:05d}.h5", "r") as handle:
                for name in ("embeddings", "hidden_bits", "nonzero_index", "nonzero_count"):
                    parts[name].append(handle[name][:])
                parts["study"].append(handle["study"].asstr()[:])
                lengths.append(np.diff(handle["nonzero_ptr"][:]))
        self.embeddings = np.concatenate(parts["embeddings"])
        self.hidden_bits = np.concatenate(parts["hidden_bits"])
        self.nonzero_index = np.concatenate(parts["nonzero_index"])
        self.nonzero_count = np.concatenate(parts["nonzero_count"])
        self.study = np.concatenate(parts["study"])
        self.nonzero_ptr = np.concatenate([[0], np.cumsum(np.concatenate(lengths))])
        self.genes = len(self.identity["genes"])

    def __len__(self):
        return len(self.embeddings)

    def batch(self, indices, device):
        """Embeddings, dense hidden-gene counts and hidden masks for the given sample indices."""
        starts = self.nonzero_ptr[indices]
        lengths = self.nonzero_ptr[indices + 1] - starts
        offsets = np.repeat(starts - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths) + np.arange(lengths.sum())
        rows = torch.from_numpy(np.repeat(np.arange(len(indices)), lengths)).to(device)
        x = torch.zeros((len(indices), self.genes), device=device)
        x[rows, torch.from_numpy(self.nonzero_index[offsets].astype("i8")).to(device)] = (
            torch.from_numpy(self.nonzero_count[offsets]).to(device))
        hidden = torch.from_numpy(np.unpackbits(self.hidden_bits[indices], axis=1, count=self.genes)).to(device).bool()
        h = torch.from_numpy(self.embeddings[indices].astype("f4")).to(device)
        return h, x, hidden

    def initialization(self):
        return cache_initialization(self.path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True, help="Prepared atlas raw-count manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build_cache(load_config(args.config), args.manifest, args.output))


if __name__ == "__main__":
    main()

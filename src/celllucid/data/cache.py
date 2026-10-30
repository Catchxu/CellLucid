"""HDF5 feature shards carry exact dataset, mask and backbone identities."""

from pathlib import Path
import json
import os
import h5py
import numpy as np
from ..config import file_digest, fingerprint
from .dataset import CountDataset
from .masking import masked_views


def build_feature_cache(
    dataset_path,
    output_path,
    backbone,
    *,
    genes=None,
    gene_key=None,
    source="auto",
    batch_size=4,
    max_cells=None,
    mask_fraction=0,
    seed=42,
    n_views=1,
):
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    if n_views < 1 or batch_size < 1 or not 0 <= mask_fraction < 1:
        raise ValueError("Invalid feature extraction settings")
    genes = backbone.genes if genes is None else list(genes)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    if temporary.exists():
        raise FileExistsError(temporary)
    with CountDataset(dataset_path, genes=genes, gene_key=gene_key, source=source) as dataset:
        n_cells = len(dataset) if max_cells is None else min(len(dataset), int(max_cells))
        settings = {
            "format_version": 2,
            "dataset_path": dataset.path,
            "dataset_digest": file_digest(dataset.path),
            "backbone": backbone.identity(),
            "genes": genes,
            "count_source": dataset.source,
            "gene_key": gene_key,
            "mask_fraction": float(mask_fraction),
            "seed": seed,
            "n_views": n_views,
            "max_cells": n_cells,
            "feature_kind": "masked" if mask_fraction else "full",
        }
        settings["fingerprint"] = fingerprint(settings)
        skipped, n_rows = 0, 0
        with h5py.File(temporary, "w") as handle:
            handle.create_dataset(
                "metadata", data=json.dumps(settings), dtype=h5py.string_dtype("utf-8")
            )
            strings = h5py.string_dtype("utf-8")
            handle.create_dataset("genes", data=np.asarray(genes, dtype=object), dtype=strings)
            handle.create_dataset("available", data=dataset.available)
            handle.create_dataset("gene_vectors", data=backbone.gene_vectors(genes).numpy())
            features = handle.create_dataset(
                "features",
                (0, backbone.dimension),
                maxshape=(None, backbone.dimension),
                dtype="f4",
                chunks=(batch_size, backbone.dimension),
            )
            rows = handle.create_dataset("indices", (0,), maxshape=(None,), dtype="i8", chunks=True)
            views = handle.create_dataset("views", (0,), maxshape=(None,), dtype="i4", chunks=True)
            ids_out = handle.create_dataset(
                "cell_ids", (0,), maxshape=(None,), dtype=strings, chunks=True
            )
            for view in range(n_views):
                for start in range(0, n_cells, batch_size):
                    indices = np.arange(start, min(n_cells, start + batch_size))
                    counts = dataset.read(indices)
                    cell_ids = dataset.cell_ids[indices]
                    if mask_fraction:
                        visible, mask = masked_views(
                            counts,
                            cell_ids,
                            dataset.available,
                            fraction=mask_fraction,
                            seed=seed,
                            view=view,
                        )
                        good = ((counts * mask).sum(axis=1) > 0) & (visible.sum(axis=1) > 0)
                    else:
                        visible = counts
                        good = visible.sum(axis=1) > 0
                    skipped += int((~good).sum())
                    if not good.any():
                        continue
                    indices, cell_ids = indices[good], cell_ids[good]
                    embeddings = backbone.encode(visible[good], genes, dataset.obs(indices))
                    end = n_rows + len(indices)
                    features.resize((end, backbone.dimension))
                    for item in (rows, views, ids_out):
                        item.resize((end,))
                    features[n_rows:end] = embeddings.numpy()
                    rows[n_rows:end], views[n_rows:end] = indices, view
                    ids_out[n_rows:end] = cell_ids.astype(object)
                    n_rows = end
            if not n_rows:
                raise ValueError("No usable cell views for feature extraction")
            handle.attrs["skipped_views"] = skipped
            handle.attrs["complete"] = True
    os.replace(temporary, output)
    return {
        "path": str(output),
        "cell_views": n_rows,
        "skipped_views": skipped,
        "fingerprint": settings["fingerprint"],
    }


class FeatureCache:
    def __init__(self, path, *, verify_source=True):
        self.path = Path(path).resolve()
        self.handle = h5py.File(self.path, "r")
        if not self.handle.attrs.get("complete", False):
            raise ValueError("Feature cache is incomplete")
        self.metadata = json.loads(self.handle["metadata"].asstr()[()])
        recorded = self.metadata.get("fingerprint")
        unsigned = {k: v for k, v in self.metadata.items() if k != "fingerprint"}
        if fingerprint(unsigned) != recorded:
            raise ValueError("Feature-cache metadata fingerprint mismatch")

        if self.metadata.get("format_version") != 2:
            raise ValueError("Unsupported cache format")
        self.genes = self.handle["genes"].asstr()[:].tolist()
        if self.genes != self.metadata["genes"]:
            raise ValueError("Feature gene order differs from its manifest")
        self.available = self.handle["available"][:]
        self.gene_vectors = self.handle["gene_vectors"][:]
        self.dataset = CountDataset(
            self.metadata["dataset_path"],
            genes=self.genes,
            gene_key=self.metadata["gene_key"],
            source=self.metadata["count_source"],
        )
        if verify_source and file_digest(self.dataset.path) != self.metadata["dataset_digest"]:
            raise ValueError("Source dataset changed since feature extraction")
        if not np.array_equal(self.available, self.dataset.available):
            raise ValueError("Gene availability changed")
        self.indices = self.handle["indices"][:]
        self.views = self.handle["views"][:]
        self.cell_ids = self.handle["cell_ids"].asstr()[:]
        if not np.array_equal(self.cell_ids, self.dataset.cell_ids[self.indices]):
            raise ValueError("Cached cell identity differs from source rows")

    def __len__(self):
        return len(self.indices)

    def read(self, row_indices):
        row_indices = np.asarray(row_indices)
        unique, inverse = np.unique(row_indices, return_inverse=True)
        features = self.handle["features"][unique][inverse]
        source_indices = self.indices[row_indices]
        counts = self.dataset.read(source_indices)
        mask = np.zeros_like(counts, dtype=bool)
        fraction = self.metadata["mask_fraction"]
        if fraction:
            for view in np.unique(self.views[row_indices]):
                select = self.views[row_indices] == view
                _, mask[select] = masked_views(
                    counts[select],
                    self.cell_ids[row_indices][select],
                    self.available,
                    fraction=fraction,
                    seed=self.metadata["seed"],
                    view=int(view),
                )
        return features, counts, mask

    def close(self):
        self.dataset.close()
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

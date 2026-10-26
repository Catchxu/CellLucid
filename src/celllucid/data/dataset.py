"""Backed H5AD count access; densification is limited to the requested cell batch."""

from pathlib import Path
import anndata as ad
import h5py
import json
from types import SimpleNamespace
import numpy as np
from scipy import sparse


def validate_counts(matrix):
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if not np.allclose(values, np.rint(values), atol=1e-4, rtol=0):
        raise ValueError("Expression is not integer-valued counts; select a raw count source")


def gene_ids(var, gene_key=None, gene_mapping=None):
    if gene_key is None:
        gene_key = next(
            (k for k in ("feature_id", "gene_ids", "ensembl_id", "gene_id") if k in var), None
        )
    ids = np.asarray(var[gene_key] if gene_key else var.index, dtype=str)
    if gene_mapping is not None:
        mapping = json.loads(Path(gene_mapping).read_text())
        if not isinstance(mapping, dict):
            raise ValueError("Gene mapping must be a symbol-to-ID JSON object")
        ids = np.array([mapping.get(x, x) for x in ids])
    ids = np.array([x.split(".")[0] if x.startswith("ENS") else x for x in ids])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate gene IDs after normalization; collapse explicitly before use")
    if any(not x or x.lower() == "nan" for x in ids):
        raise ValueError("Missing gene identifiers")
    return ids


class CountDataset:
    def __init__(self, path, *, genes=None, gene_key=None, source="auto", gene_mapping=None):
        self.path = str(Path(path).resolve())
        handle = h5py.File(self.path, "r")
        # anndata's backed reader can eagerly load layers; read only the chosen matrix.
        if source == "auto":
            source = (
                "layers/counts"
                if "layers/counts" in handle
                else "raw"
                if "raw/X" in handle
                else "X"
            )
        matrix_key = "raw/X" if source == "raw" else source
        var_key = "raw/var" if source == "raw" else "var"
        if matrix_key not in handle or var_key not in handle:
            handle.close()
            raise ValueError(f"Missing requested count source: {source}")
        node = handle[matrix_key]
        self.matrix = ad.io.sparse_dataset(node) if isinstance(node, h5py.Group) else node
        var = ad.io.read_elem(handle[var_key])
        obs = ad.io.read_elem(handle["obs"])
        self.adata = SimpleNamespace(obs=obs, obs_names=obs.index, n_obs=len(obs), file=handle)
        self.source = source
        self.source_genes = gene_ids(var, gene_key, gene_mapping)
        self.source_symbols = np.asarray(var.index.astype(str))
        self.genes = list(self.source_genes if genes is None else genes)
        if len(set(self.genes)) != len(self.genes):
            raise ValueError("Target gene vocabulary contains duplicates")
        lookup = {g: i for i, g in enumerate(self.source_genes)}
        self.available = np.array([g in lookup for g in self.genes])
        if not self.available.any():
            raise ValueError("No genes overlap the specified vocabulary")
        self.target_columns = np.flatnonzero(self.available)
        self.source_columns = np.array([lookup[g] for g in self.genes if g in lookup])
        self.cell_ids = np.asarray(self.adata.obs_names.astype(str))
        if len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError("Cell IDs must be unique within a dataset")

    def __len__(self):
        return self.adata.n_obs

    def read(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        # h5py's backed indexing requires increasing indices. Restore caller order.
        unique, inverse = np.unique(indices, return_inverse=True)
        block = self.matrix[unique, :]
        block = block[:, self.source_columns]
        block = block.toarray() if sparse.issparse(block) else np.asarray(block)
        block = np.asarray(block, dtype=np.float32)[inverse]
        validate_counts(block)
        output = np.zeros((len(indices), len(self.genes)), dtype=np.float32)
        output[:, self.target_columns] = block
        return output

    def obs(self, indices):
        return self.adata.obs.iloc[indices].copy()

    def close(self):
        self.adata.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

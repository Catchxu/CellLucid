"""Stream counts into standardized H5AD without densifying an entire atlas."""

from pathlib import Path
import os
import anndata as ad
import h5py
import numpy as np
from scipy import sparse
from ..config import file_digest, write_json
from .dataset import CountDataset, validate_counts


def preprocess_h5ad(
    input_path,
    output_path,
    *,
    gene_key=None,
    source="auto",
    genes=None,
    batch_size=512,
    min_genes=0,
    gene_mapping=None,
    study_id=None,
):
    output = Path(output_path).resolve()
    if output == Path(input_path).resolve():
        raise ValueError("Preprocessing must not overwrite its source")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    if temporary.exists():
        raise FileExistsError(
            f"Remove incomplete preprocessing output before retrying: {temporary}"
        )
    kept = []
    with CountDataset(
        input_path, genes=genes, gene_key=gene_key, source=source, gene_mapping=gene_mapping
    ) as dataset:
        # Only source-measured genes are written; unavailable genes are not fabricated zeros.
        measured = np.flatnonzero(dataset.available)
        from pandas import DataFrame

        var = DataFrame(index=np.array(dataset.genes)[measured])
        var.index.name = None
        var["feature_id"] = var.index.astype(str)
        var["gene_symbol"] = dataset.source_symbols[dataset.source_columns]
        ad.AnnData(sparse.csr_matrix((0, len(measured)), dtype=np.float32), var=var).write_h5ad(
            temporary
        )
        n_input = len(dataset)
        with h5py.File(temporary, "r+") as handle:
            del handle["X"]
            group = handle.create_group("X")
            group.attrs["encoding-type"] = "csr_matrix"
            group.attrs["encoding-version"] = "0.1.0"
            values = group.create_dataset("data", (0,), maxshape=(None,), dtype="f4", chunks=True)
            columns = group.create_dataset(
                "indices", (0,), maxshape=(None,), dtype="i8", chunks=True
            )
            pointers = group.create_dataset(
                "indptr", data=np.array([0], dtype=np.int64), maxshape=(None,), chunks=True
            )
            n_rows, n_values = 0, 0
            for start in range(0, n_input, batch_size):
                ids = np.arange(start, min(n_input, start + batch_size))
                block = sparse.csr_matrix(dataset.read(ids)[:, measured])
                validate_counts(block)
                good = np.asarray((block > 0).sum(axis=1)).ravel() >= min_genes
                good &= np.asarray(block.sum(axis=1)).ravel() > 0
                block = block[good]
                kept.extend(ids[good].tolist())
                end = n_values + block.nnz
                values.resize((end,))
                columns.resize((end,))
                values[n_values:end] = block.data
                columns[n_values:end] = block.indices
                pointers.resize((n_rows + block.shape[0] + 1,))
                pointers[n_rows + 1 :] = block.indptr[1:] + n_values
                n_values, n_rows = end, n_rows + block.shape[0]
            if n_rows == 0:
                raise ValueError("No cells passed preprocessing")
            group.attrs["shape"] = np.array([n_rows, len(measured)], dtype=np.int64)
            del handle["obs"]
            obs = dataset.adata.obs.iloc[kept].copy()
            if study_id is not None:
                obs["study_id"] = study_id
            ad.io.write_elem(handle, "obs", obs)
            provenance = {
                "source_path": str(Path(input_path).resolve()),
                "source_digest": file_digest(input_path),
                "count_source": dataset.source,
                "gene_key": gene_key or "auto",
                "min_genes": min_genes,
                "input_cells_examined": n_input,
                "retained_cells": n_rows,
                "retained_genes": len(measured),
            }
            if study_id is not None:
                provenance["study_id"] = study_id
            if gene_mapping is not None:
                provenance["gene_mapping_digest"] = file_digest(gene_mapping)
            del handle["uns"]
            ad.io.write_elem(handle, "uns", {"celllucid": provenance})
    os.replace(temporary, output)
    write_json(str(output) + ".provenance.json", provenance)
    return provenance

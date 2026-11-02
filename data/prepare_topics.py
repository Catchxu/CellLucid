"""Prepare every file in an HCA inventory as a sparse pretraining corpus."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import h5py
import anndata as ad
from anndata._io.h5ad import read_dataframe
import numpy as np
import pandas as pd
from scipy import sparse
from celllucid.config import load_config, write_json, fingerprint


class ExcludedSource(ValueError):
    """A documented source is unsuitable for the RNA count objective."""


def frame(node, columns=None):
    if columns is None or not isinstance(node, h5py.Group):
        result = read_dataframe(node)
        return result if columns is None else result[[c for c in columns if c in result]]
    # Only training metadata is loaded from large source tables.
    index = ad.io.read_elem(node[node.attrs["_index"]]).astype(str)
    values = {}
    for column in columns:
        if column in node:
            data = ad.io.read_elem(node[column])
            if "__categories" in node and column in node["__categories"]:
                data = pd.Categorical.from_codes(
                    data, ad.io.read_elem(node["__categories"][column])
                )
            values[column] = data
    return pd.DataFrame(values, index=index)


def matrix_shape(node):
    return ad.io.sparse_dataset(node).shape if isinstance(node, h5py.Group) else node.shape


def count_source(handle, priorities, block_size):
    attempts = []
    for source in priorities:
        key = "raw/X" if source == "raw" else source
        if key not in handle:
            continue
        node = handle[key]
        values = node["data"] if isinstance(node, h5py.Group) else node
        valid = True
        # Validate the entire source, not a sample of its first cells.
        step = block_size if values.ndim == 1 else max(1, block_size // values.shape[1])
        for start in range(0, len(values), step):
            block = values[start : start + step]
            if (
                not np.isfinite(block).all()
                or (block < 0).any()
                or not np.allclose(block, np.rint(block), atol=1e-4, rtol=0)
            ):
                valid = False
                break
        attempts.append({"source": source, "integer_nonnegative": valid})
        if valid:
            return source, attempts
    raise ExcludedSource("No finite nonnegative integer count matrix: " + json.dumps(attempts))


def gene_projection(var, genes, symbols):
    lookup = {g: i for i, g in enumerate(genes)}
    candidates = [
        var[c].astype(str).to_numpy()
        for c in ["feature_id", "gene_ids", "ensembl_id", "gene_id"]
        if c in var
    ]
    candidates.append(var.index.astype(str).to_numpy())
    mapped = []
    for values in candidates:
        ids = [v.split(".")[0] if v.startswith("ENS") else symbols.get(v, "") for v in values]
        mapped.append(ids)
    # The best-covered explicit identifier field is recorded, not filled with guessed IDs.
    ids = max(mapped, key=lambda values: sum(v in lookup for v in values))
    columns = np.array([lookup.get(v, -1) for v in ids])
    present = columns >= 0
    if present.sum() < 3:
        raise ExcludedSource("Fewer than three genes align to the human backbone vocabulary")
    projection = sparse.csr_matrix(
        (np.ones(present.sum(), dtype="f4"), (np.flatnonzero(present), columns[present])),
        shape=(len(var), len(genes)),
    )
    available = np.zeros(len(genes), dtype=bool)
    available[columns[present]] = True
    return projection, available, int(present.sum() - available.sum())


def row_matrix(node):
    if not isinstance(node, h5py.Group):
        return node
    backed = ad.io.sparse_dataset(node)
    if backed.format == "csr":
        return backed
    # Convert one source in RAM. Random writes to shared-filesystem memory maps
    # cause severe write amplification. CSR inputs remain streamed from disk.
    print(f"Converting CSC to CSR in RAM: shape={backed.shape}, nnz={len(node['data'])}", flush=True)
    matrix = backed.to_memory().tocsr()
    print("CSC to CSR conversion complete", flush=True)
    return matrix


def write_shard(path, blocks, ids, assays, available, study, file_id):
    matrix = sparse.vstack(blocks, format="csr")
    with h5py.File(path, "w") as handle:
        ad.io.write_elem(handle, "X", matrix)
        handle.create_dataset(
            "cell_ids", data=np.array(ids, dtype=object), dtype=h5py.string_dtype("utf-8")
        )
        handle.create_dataset(
            "assay", data=np.array(assays, dtype=object), dtype=h5py.string_dtype("utf-8")
        )
        handle.create_dataset("available", data=available)
        handle.attrs.update(study_id=study, file_id=file_id, complete=True)
    return matrix.shape[0]


def prepare_file(record, root, genes, symbols, config, database):
    projects = record["project_ids"]
    if len(projects) != 1:
        raise ExcludedSource("A single source study is required for study-held-out splitting")
    study = projects[0]
    methods = [
        value
        for protocol in record["metadata"]["protocols"]
        for value in protocol.get("libraryConstructionApproach", [])
    ]
    if any(word in method.lower() for method in methods for word in config["exclude_methods"]):
        raise ExcludedSource("Excluded assay: " + ", ".join(methods))
    if record["format"] != "h5ad":
        raise ExcludedSource("Unsupported source format: " + record["format"])
    file_id = record["file_id"]
    target = root / "files" / file_id
    staging = root / "files" / (file_id + ".part")
    if target.exists():
        raise FileExistsError(f"Uncommitted source output exists: {target}")
    if staging.exists():
        shutil.rmtree(staging)
    with h5py.File(record["local_path"], "r") as handle:
        source, attempts = count_source(handle, config["count_sources"], config["validation_block"])
        var_key = "raw/var" if source == "raw" else "var"
        var = frame(handle[var_key])
        obs = frame(
            handle["obs"],
            [
                "organism",
                "organism_ontology_term_id",
                "is_primary_data",
                "assay",
                "donor_id",
                "donor",
            ],
        )
        node = handle["raw/X" if source == "raw" else source]
        projection, available, collapsed = gene_projection(var, genes, symbols)
        selected = np.ones(len(obs), dtype=bool)
        if "organism_ontology_term_id" in obs:
            selected &= obs["organism_ontology_term_id"].astype(str).to_numpy() == "NCBITaxon:9606"
        elif "organism" in obs:
            selected &= obs["organism"].astype(str).to_numpy() == "Homo sapiens"
        if "is_primary_data" in obs:
            selected &= obs["is_primary_data"].eq(True).to_numpy()
        if "assay" in obs:
            for keyword in config["exclude_methods"]:
                selected &= (
                    ~obs["assay"]
                    .astype(str)
                    .str.lower()
                    .str.contains(keyword, regex=False)
                    .to_numpy()
                )
        staging.mkdir(parents=True)
        matrix = row_matrix(node)
        records, blocks, cell_ids, assays = [], [], [], []
        n_buffer = retained = duplicates = qc_pass = 0
        donor_column = next((c for c in ["donor_id", "donor"] if c in obs), None)
        donors = (
            obs[donor_column].astype(str).to_numpy() if donor_column else np.repeat("", len(obs))
        )
        raw_ids = obs.index.astype(str).to_numpy()
        assay_values = (
            obs["assay"].astype(str).to_numpy()
            if "assay" in obs
            else np.repeat("unknown", len(obs))
        )
        availability_id = hashlib.blake2b(available.tobytes()).digest()

        def flush():
            nonlocal n_buffer
            name = f"shard_{len(records):05d}.h5"
            cells = write_shard(staging / name, blocks, cell_ids, assays, available, study, file_id)
            records.append(
                {
                    "path": str((target / name).resolve()),
                    "cells": cells,
                    "study_id": study,
                    "file_id": file_id,
                }
            )
            blocks.clear()
            cell_ids.clear()
            assays.clear()
            n_buffer = 0

        for start in range(0, len(obs), config["batch_size"]):
            end = min(start + config["batch_size"], len(obs))
            counts = sparse.csr_matrix(matrix[start:end, :]) @ projection
            counts.sum_duplicates()
            counts.sort_indices()
            counts.eliminate_zeros()
            good = selected[start:end] & (counts.getnnz(axis=1) >= config["min_genes"])
            qc_pass += int(good.sum())
            for row in np.flatnonzero(good):
                source_row = start + int(row)
                cell = counts[row]
                raw_id = str(raw_ids[source_row])
                donor = str(donors[source_row])
                identity = json.dumps([study, donor, raw_id])
                content = hashlib.blake2b(
                    availability_id
                    + cell.indices.astype("i8").tobytes()
                    + cell.data.astype("f4").tobytes()
                ).hexdigest()
                inserted = database.execute(
                    "INSERT OR IGNORE INTO cells VALUES (?, ?)", (identity, content)
                ).rowcount
                if not inserted:
                    duplicates += 1
                    continue
                blocks.append(cell.astype("f4"))
                cell_ids.append(f"{file_id}:{source_row}:{raw_id}")
                assays.append(str(assay_values[source_row]))
                n_buffer += 1
                retained += 1
                if n_buffer == config["shard_cells"]:
                    flush()
        if n_buffer:
            flush()
        report = {
            "file_id": file_id,
            "study_id": study,
            "status": "included" if retained else "empty",
            "source": source,
            "source_attempts": attempts,
            "input_cells": len(obs),
            "retained_cells": retained,
            "metadata_selected_cells": int(selected.sum()),
            "qc_pass_cells": qc_pass,
            "input_features": len(var),
            "unmapped_features": int(len(var) - projection.nnz),
            "duplicate_cells_removed": duplicates,
            "genes_measured": int(available.sum()),
            "duplicate_features_summed": collapsed,
            "shards": records,
        }
        write_json(staging / "source.json", report)
    os.replace(staging, target)
    return report


def prepare(manifest_path, output_dir, config, checkpoint, symbols_path):
    inventory = json.loads(Path(manifest_path).read_text())
    with h5py.File(Path(checkpoint) / "vocabs/homo_sapiens_gene.h5", "r") as handle:
        genes = handle["keys"].asstr()[:].tolist()
    symbols = json.loads(Path(symbols_path).read_text())
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    settings = {
        "config": config,
        "genes": genes,
        "annotation": fingerprint(symbols),
        "source_files": [
            (r["file_id"], r["file_version"], r["size_bytes"]) for r in inventory["records"]
        ],
    }
    settings_id = fingerprint(settings)
    identity = root / "preparation.json"
    if identity.exists():
        if json.loads(identity.read_text())["identity"] != settings_id:
            raise ValueError("Preparation inputs/settings changed; use a new output directory")
    else:
        write_json(identity, {"identity": settings_id, **settings})
    database = sqlite3.connect(root / "sources.sqlite")
    database.execute(
        "CREATE TABLE IF NOT EXISTS cells (identity TEXT, expression TEXT, PRIMARY KEY(identity, expression)) WITHOUT ROWID"
    )
    database.execute("CREATE TABLE IF NOT EXISTS sources (file_id TEXT PRIMARY KEY, report TEXT)")
    reports = []
    try:
        for index, record in enumerate(inventory["records"]):
            done = database.execute(
                "SELECT report FROM sources WHERE file_id=?", (record["file_id"],)
            ).fetchone()
            if done:
                report = json.loads(done[0])
            else:
                print(
                    f"[{index + 1}/{len(inventory['records'])}] {record['file_name']}: starting",
                    flush=True,
                )
                with database:
                    try:
                        report = prepare_file(record, root, genes, symbols, config, database)
                    except ExcludedSource as error:
                        report = {
                            "file_id": record["file_id"],
                            "status": "excluded",
                            "reason": str(error),
                            "shards": [],
                        }
                    database.execute(
                        "INSERT INTO sources VALUES (?, ?)", (record["file_id"], json.dumps(report))
                    )
            reports.append(report)
            write_json(
                root / "coverage.json",
                {"files_total": len(inventory["records"]), "reports": reports},
            )
            print(
                f"[{index + 1}/{len(inventory['records'])}] {record['file_name']}: {report['status']}",
                flush=True,
            )
    finally:
        database.close()
    shards = [s for report in reports for s in report["shards"]]
    studies = sorted({s["study_id"] for s in shards})
    if len(studies) < 2:
        raise ValueError("Atlas pretraining requires at least two studies for held-out validation")
    order = np.random.default_rng(config["split_seed"]).permutation(studies)
    n_validation = min(
        len(studies) - 1, max(1, round(len(studies) * config["validation_fraction"]))
    )
    validation = set(order[:n_validation])
    for shard in shards:
        shard["split"] = "validation" if shard["study_id"] in validation else "train"
    write_json(root / "genes.json", genes)
    result = {
        "format_version": 1,
        "kind": "raw_count_shards",
        "identity": settings_id,
        "genes_path": str((root / "genes.json").resolve()),
        "shards": shards,
        "source_inventory": str(Path(manifest_path).resolve()),
        "source_files": len(reports),
        "included_files": sum(r["status"] == "included" for r in reports),
        "excluded_files": sum(r["status"] == "excluded" for r in reports),
        "cells": sum(s["cells"] for s in shards),
        "studies": len(studies),
        "validation_studies": sorted(validation),
        "policy": config,
    }
    write_json(root / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Full HCA acquisition manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument("--checkpoint", default="assets/tf_sapiens")
    parser.add_argument("--gene-symbols", default="assets/gene_symbols.json")
    args = parser.parse_args()
    result = prepare(
        args.manifest, args.output_dir, load_config(args.config), args.checkpoint, args.gene_symbols
    )
    print(
        {
            k: result[k]
            for k in ["source_files", "included_files", "excluded_files", "cells", "studies"]
        }
    )


if __name__ == "__main__":
    main()

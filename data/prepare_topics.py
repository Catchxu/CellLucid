"""Prepare every file in an HCA inventory as a sparse pretraining corpus."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import warnings
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


def source_keys(source):
    """Matrix and var keys of a count source; anndata before 0.7 wrote raw as raw.X and raw.var."""
    if source == "raw":
        return "raw/X", "raw/var"
    if source == "raw.X":
        return "raw.X", "raw.var"
    return source, "var"


def count_source(handle, priorities, block_size):
    """The first integer source in priority order, else the first holding log1p(counts), else
    the first finite nonnegative one."""
    attempts = []
    for source in priorities:
        key = source_keys(source)[0]
        if key not in handle:
            continue
        node = handle[key]
        values = node["data"] if isinstance(node, h5py.Group) else node
        valid = integer = logged = True
        # Validate the entire source, not a sample of its first cells.
        step = block_size if values.ndim == 1 else max(1, block_size // values.shape[1])
        for start in range(0, len(values), step):
            block = values[start : start + step]
            if not np.isfinite(block).all() or (block < 0).any():
                valid = integer = logged = False
                break
            if integer and not np.allclose(block, np.rint(block), atol=1e-4, rtol=0):
                integer = False
            if logged:
                # Values beyond the float range are not log counts.
                with np.errstate(over="ignore"):
                    restored = np.rint(np.expm1(block.astype("f8")))
                if not np.allclose(np.log1p(restored), block, rtol=1e-6, atol=0):
                    logged = False
        attempts.append(
            {
                "source": source,
                "finite_nonnegative": valid,
                "integer": integer,
                "log1p_counts": logged,
            }
        )
        if integer:
            return source, attempts
    logged = next((a["source"] for a in attempts if a["log1p_counts"]), None)
    if logged is not None:
        return logged, attempts
    fallback = next((a["source"] for a in attempts if a["finite_nonnegative"]), None)
    if fallback is None:
        raise ExcludedSource("No finite nonnegative expression matrix: " + json.dumps(attempts))
    warnings.warn("No integer count matrix; using non-integer " + fallback + ": " + json.dumps(attempts))
    return fallback, attempts


def choose_source(record, config):
    """Screen an inventory record and select its count source."""
    if record["file_name"] in config["exclude_files"]:
        raise ExcludedSource("Excluded file: " + config["exclude_files"][record["file_name"]])
    if len(record["project_ids"]) != 1:
        raise ExcludedSource("A single source study is required for study-held-out splitting")
    methods = [
        value
        for protocol in record["metadata"]["protocols"]
        for value in protocol.get("libraryConstructionApproach", [])
    ]
    # HCA file records can list every method of their submission, so the RNA files of a
    # multiome project also name ATAC. A file is excluded only when all its methods are.
    if methods and all(
        any(word in method.lower() for word in config["exclude_methods"]) for method in methods
    ):
        raise ExcludedSource("Excluded assay: " + ", ".join(methods))
    if record["format"] != "h5ad":
        raise ExcludedSource("Unsupported source format: " + record["format"])
    with h5py.File(record["local_path"], "r") as handle:
        source, attempts = count_source(handle, config["count_sources"], config["validation_block"])
    attempt = next(a for a in attempts if a["source"] == source)
    return {
        "source": source,
        "attempts": attempts,
        "integer": attempt["integer"],
        "log1p_counts": attempt["log1p_counts"] and not attempt["integer"],
    }


ENSEMBL = re.compile(r"ENSG\d{11}")


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
        # Ensembl IDs may carry a version suffix or a symbol prefix (SYMBOL_ENSG...).
        matches = [ENSEMBL.search(v) for v in values]
        mapped.append([m.group(0) if m else symbols.get(v, "") for m, v in zip(matches, values)])
    # The best-covered explicit identifier field is recorded, not filled with guessed IDs.
    ids = max(mapped, key=lambda values: sum(v in lookup for v in values))
    columns = np.array([lookup.get(v, -1) for v in ids])
    present = columns >= 0
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


def cell_keys(names):
    """Hashed obs names and 16-nt barcodes; a barcode absent or repeated within the file is 0."""
    names = pd.Series(names, dtype=object)
    barcodes = names.str.extract("([ACGT]{16})", expand=False)
    unique = (barcodes.notna() & ~barcodes.duplicated(keep=False)).to_numpy()
    barcode_keys = np.zeros(len(names), dtype=np.uint64)
    barcode_keys[unique] = pd.util.hash_array(barcodes.to_numpy(dtype=object)[unique])
    return {"name": pd.util.hash_array(names.to_numpy()), "barcode": barcode_keys}


KINDS = ("name", "barcode")


def candidate_matches(keys, study, signatures):
    """Covered cells of earlier files that share an obs name, or within one study a barcode.

    Returns, sorted by row of this file, the row, the signature index, the cell index in that
    signature and the key kind of every candidate.
    """
    rows, owners, cells, kinds = ([np.zeros(0, np.int64)] for _ in range(4))
    for kind_index, kind in enumerate(KINDS):
        order = np.argsort(keys[kind], kind="stable")
        values = keys[kind][order]
        for owner, signature in enumerate(signatures):
            if kind == "barcode" and signature["study_id"] != study:
                continue
            wanted = signature[kind]
            position = np.searchsorted(values, wanted).clip(max=len(values) - 1)
            hit = (values[position] == wanted) & (wanted != 0)
            rows.append(order[position[hit]])
            cells.append(np.flatnonzero(hit))
            owners.append(np.full(int(hit.sum()), owner))
            kinds.append(np.full(int(hit.sum()), kind_index))
    rows, owners, cells, kinds = (np.concatenate(a) for a in (rows, owners, cells, kinds))
    order = np.argsort(rows, kind="stable")
    return rows[order], owners[order], cells[order], kinds[order]


def load_signature(directory):
    with np.load(directory / "signature.npz") as data:
        signature = {key: data[key] for key in data.files}
    signature["study_id"] = str(signature["study_id"])
    signature["file_id"] = directory.name
    # Expressed genes of every covered cell stay on disk and are read per candidate.
    signature["genes"] = np.load(directory / "genes.npy", mmap_mode="r")
    return signature


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


def prepare_file(record, choice, root, genes, symbols, config, signatures):
    study = record["project_ids"][0]
    file_id = record["file_id"]
    target = root / "files" / file_id
    staging = root / "files" / (file_id + ".part")
    if target.exists():
        raise FileExistsError(f"Uncommitted source output exists: {target}")
    if staging.exists():
        shutil.rmtree(staging)
    source = choice["source"]
    matrix_key, var_key = source_keys(source)
    with h5py.File(record["local_path"], "r") as handle:
        var = frame(handle[var_key])
        obs = frame(
            handle["obs"], ["organism", "organism_ontology_term_id", "is_primary_data", "assay"]
        )
        projection, available, collapsed = gene_projection(var, genes, symbols)
        if available.sum() < config["min_measured_genes"]:
            raise ExcludedSource(
                f"{available.sum()} backbone genes measured, fewer than {config['min_measured_genes']}"
            )
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
        matrix = row_matrix(handle[matrix_key])
        raw_ids = obs.index.astype(str).to_numpy()
        keys = cell_keys(raw_ids)
        rows, owners, owner_cells, kinds = candidate_matches(keys, study, signatures)
        first = np.searchsorted(rows, np.arange(len(obs)))
        last = np.searchsorted(rows, np.arange(len(obs)), side="right")
        shared = [available & signature["available"] for signature in signatures]
        removed = np.zeros((len(signatures), len(KINDS)), dtype=np.int64)

        def held_earlier(expressed, row):
            """Whether a key-matched covered cell of an earlier file expresses the same genes."""
            for i in range(first[row], last[row]):
                signature, both = signatures[owners[i]], shared[owners[i]]
                ptr = signature["ptr"]
                x = expressed[both[expressed]]
                y = signature["genes"][ptr[owner_cells[i]] : ptr[owner_cells[i] + 1]]
                y = y[both[y]]
                common = len(np.intersect1d(x, y, assume_unique=True))
                union = len(x) + len(y) - common
                # A pair expressing none of the shared genes carries no evidence either way.
                if union and common / union >= config["duplicate_jaccard"]:
                    removed[owners[i], kinds[i]] += 1
                    return True
            return False

        gene_dtype = np.min_scalar_type(len(genes) - 1)
        covered = np.zeros(len(obs), dtype=bool)
        covered_genes = []
        records, blocks, cell_ids, assays = [], [], [], []
        n_buffer = retained = duplicates = qc_pass = 0
        assay_values = (
            obs["assay"].astype(str).to_numpy()
            if "assay" in obs
            else np.repeat("unknown", len(obs))
        )

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
            block = sparse.csr_matrix(matrix[start:end, :])
            if choice["log1p_counts"]:
                # Counts are restored before features mapping to one gene are summed.
                block.data = np.rint(np.expm1(block.data.astype("f8")))
            counts = block @ projection
            counts.sum_duplicates()
            counts.sort_indices()
            counts.eliminate_zeros()
            good = selected[start:end] & (counts.getnnz(axis=1) >= config["min_genes"])
            qc_pass += int(good.sum())
            covered[start:end] = good
            for row in np.flatnonzero(good):
                source_row = start + int(row)
                cell = counts[row]
                covered_genes.append(cell.indices.astype(gene_dtype))
                if held_earlier(cell.indices, source_row):
                    duplicates += 1
                    continue
                raw_id = str(raw_ids[source_row])
                blocks.append(cell.astype("f4"))
                cell_ids.append(f"{file_id}:{source_row}:{raw_id}")
                assays.append(str(assay_values[source_row]))
                n_buffer += 1
                retained += 1
                if n_buffer == config["shard_cells"]:
                    flush()
        if n_buffer:
            flush()
        matched = np.zeros_like(removed)
        np.add.at(matched, (owners[covered[rows]], kinds[covered[rows]]), 1)
        matches = [
            {
                "file_id": signatures[owner]["file_id"],
                "key": KINDS[kind],
                "matched": int(matched[owner, kind]),
                "removed": int(removed[owner, kind]),
            }
            for owner, kind in zip(*np.nonzero(matched))
        ]
        # The cells this file covers (QC-passing, duplicates included) for later files.
        lengths = np.array([len(g) for g in covered_genes], dtype=np.int64)
        np.savez(
            staging / "signature.npz",
            study_id=study,
            name=keys["name"][covered],
            barcode=keys["barcode"][covered],
            available=available,
            ptr=np.concatenate([[0], np.cumsum(lengths)]),
        )
        np.save(
            staging / "genes.npy",
            np.concatenate(covered_genes) if covered_genes else np.zeros(0, gene_dtype),
        )
        report = {
            "file_id": file_id,
            "study_id": study,
            "status": "included" if retained else "empty",
            "source": source,
            "source_attempts": choice["attempts"],
            "integer_counts": choice["integer"] or choice["log1p_counts"],
            "log1p_counts": choice["log1p_counts"],
            "input_cells": len(obs),
            "retained_cells": retained,
            "metadata_selected_cells": int(selected.sum()),
            "qc_pass_cells": qc_pass,
            "input_features": len(var),
            "unmapped_features": int(len(var) - projection.nnz),
            "duplicate_cells_removed": duplicates,
            "duplicate_matches": matches,
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
    records = inventory["records"]
    database = sqlite3.connect(root / "sources.sqlite")
    database.execute("CREATE TABLE IF NOT EXISTS choices (file_id TEXT PRIMARY KEY, choice TEXT)")
    database.execute("CREATE TABLE IF NOT EXISTS sources (file_id TEXT PRIMARY KEY, report TEXT)")
    reports = {}
    try:
        choices = {}
        for index, record in enumerate(records):
            done = database.execute(
                "SELECT choice FROM choices WHERE file_id=?", (record["file_id"],)
            ).fetchone()
            if done:
                choices[record["file_id"]] = json.loads(done[0])
                continue
            print(f"[{index + 1}/{len(records)}] {record['file_name']}: validating counts", flush=True)
            try:
                choice = choose_source(record, config)
            except ExcludedSource as error:
                choice = {"excluded": str(error)}
            with database:
                database.execute(
                    "INSERT INTO choices VALUES (?, ?)", (record["file_id"], json.dumps(choice))
                )
            choices[record["file_id"]] = choice
        # Stored counts are prepared first, then counts restored from log1p, then normalized
        # expression, so a cell held several ways is kept in the form closest to its counts.
        def rank(record):
            choice = choices[record["file_id"]]
            if "excluded" in choice:
                return 2
            return 0 if choice["integer"] else 1 if choice["log1p_counts"] else 2

        queue = sorted(records, key=rank)
        signatures = []
        for step, record in enumerate(queue):
            file_id = record["file_id"]
            done = database.execute(
                "SELECT report FROM sources WHERE file_id=?", (file_id,)
            ).fetchone()
            if done:
                report = json.loads(done[0])
            else:
                choice = choices[file_id]
                print(f"[{step + 1}/{len(queue)}] {record['file_name']}: starting", flush=True)
                with database:
                    try:
                        if "excluded" in choice:
                            raise ExcludedSource(choice["excluded"])
                        report = prepare_file(
                            record, choice, root, genes, symbols, config, signatures
                        )
                    except ExcludedSource as error:
                        report = {
                            "file_id": file_id,
                            "status": "excluded",
                            "reason": str(error),
                            "shards": [],
                        }
                    database.execute(
                        "INSERT INTO sources VALUES (?, ?)", (file_id, json.dumps(report))
                    )
            if report["status"] != "excluded":
                signatures.append(load_signature(root / "files" / file_id))
            reports[file_id] = report
            write_json(
                root / "coverage.json",
                {
                    "files_total": len(records),
                    "reports": [reports[r["file_id"]] for r in records if r["file_id"] in reports],
                },
            )
            print(f"[{step + 1}/{len(queue)}] {record['file_name']}: {report['status']}", flush=True)
    finally:
        database.close()
    shards = [s for r in records for s in reports[r["file_id"]]["shards"]]
    studies = sorted({s["study_id"] for s in shards})
    validation = set(config["validation_studies"])
    if not validation < set(studies):
        raise ValueError(
            "Validation studies must hold shards and leave a training study: "
            + ", ".join(sorted(validation - set(studies)))
        )
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
        "included_files": sum(r["status"] == "included" for r in reports.values()),
        "excluded_files": sum(r["status"] == "excluded" for r in reports.values()),
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

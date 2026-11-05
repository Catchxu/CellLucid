"""Cells stored more than once within a study of the prepared corpus.

Some HCA projects deposit the same cells in several files: a whole atlas next to per-cell-type
subsets, or a contributor matrix next to its CELLxGENE export. Within each study, copies share
a barcode across files and have matching counts on the genes both files measure. For every
barcode found in more than one file, copies are visited from the file with the most measured
genes (then the larger file); a copy is a duplicate when its Pearson correlation with an
already kept copy on shared measured genes exceeds the threshold. Prepared shards are not
modified; the result is a mapping from each duplicate cell ID to the kept cell ID.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

from celllucid.config import write_json

THRESHOLD = 0.99
CHUNK = 4000
BINS = (-1.0, 0.5, 0.9, 0.99, 0.999, 1.0)


def collisions(files):
    """Barcodes present in more than one file of a study, with (file_id, path, row, cell_id) copies."""
    copies = defaultdict(list)
    for file_id, shards in files.items():
        for shard in shards:
            with h5py.File(shard["path"], "r") as handle:
                ids = handle["cell_ids"].asstr()[:]
            for row, cell_id in enumerate(ids):
                copies[cell_id.split(":", 2)[2]].append((file_id, shard["path"], row, cell_id))
    return [c for c in copies.values() if len({f for f, *_ in c}) > 1]


def read_rows(needed):
    rows, available = {}, {}
    by_path = defaultdict(set)
    for path, row in needed:
        by_path[path].add(row)
    for path, wanted in by_path.items():
        order = np.array(sorted(wanted))
        with h5py.File(path, "r") as handle:
            counts = ad.io.sparse_dataset(handle["X"])[order, :].toarray()
            available[path] = handle["available"][:]
        rows.update({(path, int(r)): x for r, x in zip(order, counts, strict=True)})
    return rows, available


def correlation(a, b, shared):
    x, y = a[shared].astype(np.float64), b[shared].astype(np.float64)
    x, y = x - x.mean(), y - y.mean()
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))


def study_duplicates(files):
    groups = collisions(files)
    file_cells = {f: sum(s["cells"] for s in shards) for f, shards in files.items()}
    duplicate_of, best = {}, []
    for start in range(0, len(groups), CHUNK):
        chunk = groups[start:start + CHUNK]
        rows, available = read_rows({(path, row) for group in chunk for _, path, row, _ in group})
        for group in chunk:
            order = sorted(group, key=lambda c: (-int(available[c[1]].sum()), -file_cells[c[0]], c[0]))
            kept = [order[0]]
            for copy in order[1:]:
                scores = [correlation(rows[(copy[1], copy[2])], rows[(k[1], k[2])],
                                      available[copy[1]] & available[k[1]]) for k in kept]
                match = int(np.argmax(scores))
                best.append(scores[match])
                if scores[match] > THRESHOLD:
                    duplicate_of[copy[3]] = kept[match][3]
                else:
                    kept.append(copy)
    return duplicate_of, len(groups), np.histogram(best, bins=BINS)[0].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    studies = defaultdict(lambda: defaultdict(list))
    for shard in manifest["shards"]:
        studies[shard["study_id"]][shard["file_id"]].append(shard)
    duplicate_of, summary = {}, {}
    for study, files in sorted(studies.items()):
        if len(files) < 2:
            continue
        found, barcodes, histogram = study_duplicates(files)
        duplicate_of.update(found)
        summary[study] = {
            "files": len(files),
            "cells": sum(s["cells"] for shards in files.values() for s in shards),
            "shared_barcodes": barcodes,
            "duplicates": len(found),
            "duplicates_by_file": dict(Counter(cell.split(":", 1)[0] for cell in found)),
            "best_correlation_histogram": dict(zip([f"{a}..{b}" for a, b in zip(BINS[:-1], BINS[1:])], histogram)),
        }
        print(study, {k: v for k, v in summary[study].items() if k != "duplicates_by_file"}, flush=True)
    write_json(args.output, {"manifest_identity": manifest["identity"], "threshold": THRESHOLD,
                             "rule": __doc__.strip(), "studies": summary, "duplicate_of": duplicate_of})


if __name__ == "__main__":
    main()

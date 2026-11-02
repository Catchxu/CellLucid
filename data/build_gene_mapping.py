#!/usr/bin/env python3
"""Build an explicit symbol-to-ID lookup from an annotated H5AD var table."""

import argparse
import gzip
import re
from collections import defaultdict
from pathlib import Path
import anndata as ad
import h5py
from celllucid.config import file_digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotation", help="Annotated H5AD")
    source.add_argument("--gtf", help="Pinned, gzip-compressed GENCODE gene annotation")
    parser.add_argument("--gene-id-column", default="gene_ids")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    input_path = args.gtf if args.gtf is not None else args.annotation
    if output.exists() or output.resolve() == Path(input_path).resolve():
        raise FileExistsError(output)
    if args.gtf is not None:
        symbols = defaultdict(set)
        with gzip.open(args.gtf, "rt") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.split("\t")
                if fields[2] == "gene":
                    attributes = dict(re.findall(r'(\w+) "([^"]+)"', fields[8]))
                    symbols[attributes["gene_name"]].add(attributes["gene_id"].split(".")[0])
        mapping = {name: next(iter(ids)) for name, ids in symbols.items() if len(ids) == 1}
    else:
        with h5py.File(args.annotation, "r") as handle:
            var = ad.io.read_elem(handle["var"])
        if args.gene_id_column not in var or not var.index.is_unique:
            raise ValueError("Annotation needs unique symbols and the requested gene-ID column")
        if var[args.gene_id_column].isna().any():
            raise ValueError("Annotation contains missing gene IDs")
        mapping = dict(
            zip(var.index.astype(str), var[args.gene_id_column].astype(str), strict=True)
        )
    write_json(output, mapping)
    write_json(
        str(output) + ".provenance.json",
        {
            "source": str(Path(input_path).resolve()),
            "source_digest": file_digest(input_path),
            "gene_id_column": args.gene_id_column,
            "symbols": len(mapping),
        },
    )
    print(f"Wrote {len(mapping)} symbol mappings to {output}")


if __name__ == "__main__":
    main()

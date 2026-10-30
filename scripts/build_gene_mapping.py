#!/usr/bin/env python3
"""Build an explicit symbol-to-ID lookup from an annotated H5AD var table."""

import argparse
from pathlib import Path
import anndata as ad
import h5py
from celllucid.config import file_digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--gene-id-column", default="gene_ids")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.resolve() == Path(args.annotation).resolve():
        raise FileExistsError(output)
    with h5py.File(args.annotation, "r") as handle:
        var = ad.io.read_elem(handle["var"])
    if args.gene_id_column not in var or not var.index.is_unique:
        raise ValueError("Annotation needs unique symbols and the requested gene-ID column")
    if var[args.gene_id_column].isna().any():
        raise ValueError("Annotation contains missing gene IDs")
    mapping = dict(zip(var.index.astype(str), var[args.gene_id_column].astype(str), strict=True))
    write_json(output, mapping)
    write_json(
        str(output) + ".provenance.json",
        {
            "source": str(Path(args.annotation).resolve()),
            "source_digest": file_digest(args.annotation),
            "gene_id_column": args.gene_id_column,
            "symbols": len(mapping),
        },
    )
    print(f"Wrote {len(mapping)} symbol mappings to {output}")


if __name__ == "__main__":
    main()

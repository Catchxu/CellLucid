"""Standardize raw count matrices without modifying their source files."""

import argparse
from ..config import load_config
from ..data.preprocess import preprocess_h5ad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gene-key")
    parser.add_argument("--count-source", help="X, raw, or layers/<name>")
    parser.add_argument("--gene-mapping")
    parser.add_argument("--max-cells", type=int, help="Optional limit for a small trial")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.gene_key is not None:
        config["gene_key"] = args.gene_key
    if args.count_source is not None:
        config["source"] = args.count_source
    print(
        preprocess_h5ad(
            input_path=args.input,
            output_path=args.output,
            gene_mapping=args.gene_mapping,
            max_cells=args.max_cells,
            **config,
        )
    )


if __name__ == "__main__":
    main()

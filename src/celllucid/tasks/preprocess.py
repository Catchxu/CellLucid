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
    parser.add_argument("--count-source", required=True, help="X, raw, or layers/<name>")
    parser.add_argument("--gene-mapping")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = {
        "gene_key": args.gene_key,
        "source": args.count_source,
        "batch_size": config["batch_size"],
        "min_genes": config["min_genes"],
    }
    print(
        preprocess_h5ad(
            input_path=args.input,
            output_path=args.output,
            gene_mapping=args.gene_mapping,
            **settings,
        )
    )


if __name__ == "__main__":
    main()

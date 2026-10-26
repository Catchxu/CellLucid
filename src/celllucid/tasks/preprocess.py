"""Standardize raw count matrices without modifying their source files."""

import argparse
from ..config import load_config
from ..data.preprocess import preprocess_h5ad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(preprocess_h5ad(**cfg))


if __name__ == "__main__":
    main()

"""Extract full-input or precisely reproducible masked-input backbone features."""

import argparse
from ..config import load_config
from ..models.backbone import FrozenTranscriptFormer
from ..data.cache import build_feature_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cells", type=int, help="Optional limit for a small trial")
    args = parser.parse_args()
    config = load_config(args.config)
    backbone = FrozenTranscriptFormer(**config["backbone"])
    print(
        build_feature_cache(
            dataset_path=args.input,
            output_path=args.output,
            backbone=backbone,
            max_cells=args.max_cells,
            **config["features"],
        )
    )


if __name__ == "__main__":
    main()

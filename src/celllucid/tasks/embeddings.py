"""Extract full-input or precisely reproducible masked-input backbone features."""

import argparse
from ..config import load_config
from ..models.backbone import FrozenTranscriptFormer
from ..data.cache import build_feature_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    backbone = FrozenTranscriptFormer(**config["backbone"])
    print(build_feature_cache(backbone=backbone, **config["features"]))


if __name__ == "__main__":
    main()

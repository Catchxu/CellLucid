"""Extract reproducible masked features from one dataset or a prepared cohort."""

import argparse
import json
from pathlib import Path
from ..config import load_config, write_json
from ..models.backbone import FrozenTranscriptFormer
from ..data.cache import build_feature_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input")
    inputs.add_argument("--manifest", help="Prepared cohort manifest")
    parser.add_argument("--output", required=True, help="Feature file, or directory for a cohort")
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    backbone = FrozenTranscriptFormer(**config["backbone"])
    if args.input is not None:
        print(
            build_feature_cache(
                args.input,
                args.output,
                backbone,
                max_cells=args.max_cells,
                **config["features"],
            )
        )
    else:
        cohort = json.loads(Path(args.manifest).read_text())
        records = []
        for record in cohort["records"]:
            output = Path(args.output) / (record["file_id"] + ".h5")
            result = build_feature_cache(
                record["dataset_path"],
                output,
                backbone,
                max_cells=args.max_cells,
                **config["features"],
            )
            records.append(
                {
                    "file_id": record["file_id"],
                    "study_id": record["study_id"],
                    "cache_path": result["path"],
                    "features": result,
                }
            )
            print(result, flush=True)
        write_json(Path(args.output) / "features.json", {"records": records})


if __name__ == "__main__":
    main()

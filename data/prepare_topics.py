"""Standardize an explicitly reviewed cohort for topic training."""

import argparse
import json
from pathlib import Path
from celllucid.config import load_config, write_json
from celllucid.data.preprocess import preprocess_h5ad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args()
    cohort = json.loads(Path(args.manifest).read_text())
    config = load_config(args.config)
    records = []
    for record in cohort["records"]:
        settings = dict(config)
        settings.update(source=record["count_source"], gene_key=record["gene_key"])
        output = Path(args.output_dir) / (record["file_id"] + ".h5ad")
        provenance = preprocess_h5ad(
            record["input_path"],
            output,
            study_id=record["study_id"],
            max_cells=args.max_cells,
            **settings,
        )
        records.append(
            {
                "file_id": record["file_id"],
                "study_id": record["study_id"],
                "dataset_path": str(output.resolve()),
                "provenance": provenance,
            }
        )
        print(record["file_id"], provenance, flush=True)
    write_json(Path(args.output_dir) / "prepared.json", {"records": records})


if __name__ == "__main__":
    main()

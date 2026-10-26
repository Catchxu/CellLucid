"""Inventory-first HCA downloads with explicit file and byte budgets."""

from pathlib import Path
import argparse
from ..config import load_config, write_json
from ..data.hca import HCAClient
from ..data.manifest import save_manifest, load_manifest
from ..data.download import download_file
from ..data.metadata import project_summary


def run(config, inventory_only=False):
    root = Path(config.get("output_dir", "data/hca"))
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest["selection"] != config.get("selection", {}):
            raise ValueError(
                "Existing manifest has another selection; choose a new output directory"
            )
    else:
        client = HCAClient(**config.get("api", {}))
        records = client.discover(**config.get("selection", {}))
        if not records:
            raise ValueError("No eligible matrices within the requested budgets")
        manifest = save_manifest(manifest_path, records, query=config.get("selection", {}))
    if inventory_only:
        return manifest
    total = sum(record["size_bytes"] for record in manifest["records"])
    budget = config.get("selection", {}).get("max_total_bytes", 20_000_000)
    if total > budget:
        raise ValueError("Manifest exceeds configured total byte budget")
    for record in manifest["records"]:
        destination = root / "raw" / record["file_id"] / record["file_name"]
        try:
            download_file(
                record["url"],
                destination,
                sha256=record["sha256"],
                size=record["size_bytes"],
                max_bytes=config.get("selection", {}).get("max_file_bytes", 20_000_000),
            )
            record["local_path"] = str(destination.resolve())
            record["status"] = "verified"
            record.pop("error", None)
            write_json(
                root / "metadata" / (record["file_id"] + ".json"),
                {"summary": project_summary(record), "original": record["metadata"]},
            )
        except Exception as error:
            record["status"] = "failed"
            record["error"] = str(error)
            write_json(manifest_path, manifest)
            raise
        write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    manifest = run(load_config(args.config), args.inventory_only)
    print({r["file_name"]: r["status"] for r in manifest["records"]})


if __name__ == "__main__":
    main()

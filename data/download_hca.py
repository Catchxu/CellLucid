"""Inventory-first HCA downloads with bounded concurrency and resumable transfers."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from collections import Counter
import argparse
import copy
import fcntl
import logging
import shutil
from celllucid.config import load_config, write_json
from .hca import HCAClient
from .manifest import save_manifest, load_manifest
from .download import download_file
from .metadata import project_summary

log = logging.getLogger(__name__)


@contextmanager
def _exclusive_lock(root):
    # The kernel releases the lock on exit, including Slurm cancellation/timeouts.
    with (root / ".download.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another HCA process is using {root}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _destination(root, record):
    for field in ("file_id", "file_name"):
        value = record[field]
        if not value or value in (".", "..") or Path(value).name != value:
            raise ValueError(f"Unsafe manifest path component: {field}")
    return root / "raw" / record["file_id"] / record["file_name"]


def _transfer(root, record, max_bytes):
    record = copy.deepcopy(record)
    destination = _destination(root, record)
    log.info("Checking/downloading %s (%d bytes)", record["file_name"], record["size_bytes"])
    try:
        download_file(
            record["url"],
            destination,
            size=record["size_bytes"],
            max_bytes=max_bytes,
        )
        write_json(
            root / "metadata" / (record["file_id"] + ".json"),
            {"summary": project_summary(record), "original": record["metadata"]},
        )
        record["local_path"] = str(destination.resolve())
        record["status"] = "verified"
        record.pop("error", None)
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
    return record


def run(config, inventory_only=False):
    root = Path(config.get("output_dir", "data/hca")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(root):
        return _run(config, root, inventory_only)


def _run(config, root, inventory_only):
    selection = config.get("selection", {})
    api = config.get("api", {})
    options = config.get("download", {})
    workers = options.get("workers", 1)
    reserve = options.get("min_free_bytes", 0)
    if not isinstance(workers, int) or not 1 <= workers <= 32 or reserve < 0:
        raise ValueError("Use 1..32 download workers and a nonnegative free-space reserve")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if {key: manifest["selection"][key] for key in selection} != selection:
            raise ValueError(
                "Existing manifest has another selection; choose a new output directory"
            )
        if "api" in manifest and manifest["api"] != api:
            raise ValueError(
                "Existing manifest has another API/catalog; choose a new output directory"
            )
    else:
        client = HCAClient(**api)
        records = client.discover(**selection)
        if not records:
            raise ValueError("No eligible matrices within the requested budgets")
        manifest = save_manifest(manifest_path, records, query=selection)
        manifest["api"] = api
        manifest["discovery"] = client.report
        write_json(manifest_path, manifest)
    records = manifest["records"]
    total = sum(record["size_bytes"] for record in records)
    max_bytes = selection.get("max_file_bytes", 20_000_000)
    if total > selection.get("max_total_bytes", 20_000_000):
        raise ValueError("Manifest exceeds configured total byte budget")
    if len(records) > selection.get("max_files", 1):
        raise ValueError("Manifest exceeds configured file budget")
    destinations = set()
    for record in records:
        destination = _destination(root, record)
        if destination in destinations:
            raise ValueError("Manifest contains colliding destination paths")
        destinations.add(destination)
        if not 0 < record["size_bytes"] <= max_bytes:
            raise ValueError("Manifest exceeds configured per-file byte budget")
    log.info(
        "Inventory: %d files, %.3f GB; discovery=%s",
        len(records),
        total / 1e9,
        manifest.get("discovery", "legacy manifest"),
    )
    if inventory_only:
        return manifest
    remaining = 0
    for record in records:
        destination = _destination(root, record)
        part = destination.with_name(destination.name + ".part")
        present = destination.stat().st_size if destination.exists() else 0
        if not destination.exists() and part.exists():
            present = part.stat().st_size
        remaining += max(0, record["size_bytes"] - present)
    free = shutil.disk_usage(root).free
    if free < remaining + reserve:
        raise OSError(
            f"Insufficient disk space: {free} free; {remaining} remaining + {reserve} reserve"
        )
    completed = 0
    # Only this coordinator writes the manifest; workers own disjoint files.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(_transfer, root, record, max_bytes): i for i, record in enumerate(records)
        }
        for future in as_completed(pending):
            index = pending[future]
            records[index] = future.result()
            write_json(manifest_path, manifest)
            completed += 1
            record = records[index]
            log.info(
                "[%d/%d] %s: %s%s",
                completed,
                len(records),
                record["status"],
                record["file_name"],
                ": " + record["error"] if "error" in record else "",
            )
    failed = sum(record["status"] == "failed" for record in records)
    if failed:
        raise RuntimeError(f"{failed} HCA downloads failed; rerun the same command to resume")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--output-dir", help="Override the configured acquisition directory")
    parser.add_argument("--max-files", type=int, help="Override the selected-file limit")
    parser.add_argument("--max-file-bytes", type=int, help="Override the per-file byte limit")
    parser.add_argument("--max-total-bytes", type=int, help="Override the total byte limit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    selection = config.setdefault("selection", {})
    for field in ("max_files", "max_file_bytes", "max_total_bytes"):
        value = getattr(args, field)
        if value is not None:
            selection[field] = value
    manifest = run(config, args.inventory_only)
    print(dict(Counter(r["status"] for r in manifest["records"])))


if __name__ == "__main__":
    main()

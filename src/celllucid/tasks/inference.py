"""Apply a fixed topic bank to new cells in streaming batches."""

from pathlib import Path
import argparse
import json
import os
import h5py
import numpy as np
from ..config import load_config, file_sha256
from ..data.dataset import CountDataset
from ..models.backbone import FrozenTranscriptFormer
from .topic_bank import TopicBank


def infer(config):
    bank = TopicBank(config["topic_bank"], device=config.get("device", "cpu"))
    backbone = FrozenTranscriptFormer(**config["backbone"])
    if backbone.identity()["fingerprint"] != bank.manifest["backbone"]["fingerprint"]:
        raise ValueError("Inference backbone or input settings differ from the trained topic bank")
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    if temporary.exists():
        raise FileExistsError(temporary)
    with CountDataset(
        config["dataset"],
        genes=bank.genes,
        gene_key=config.get("gene_key"),
        source=config.get("count_source", "auto"),
    ) as ds:
        n = min(len(ds), config.get("max_cells") or len(ds))
        size = int(config.get("batch_size", 4))
        with h5py.File(temporary, "w") as handle:
            theta = handle.create_dataset("theta", (n, len(bank.topics)), dtype="f4")
            handle.create_dataset(
                "cell_ids", data=ds.cell_ids[:n].astype(object), dtype=h5py.string_dtype("utf-8")
            )
            handle.create_dataset("gene_available", data=ds.available)
            handle.attrs["bank_id"] = bank.manifest["bank_id"]
            handle.attrs["dataset_sha256"] = file_sha256(ds.path)
            handle.attrs["backbone_fingerprint"] = backbone.identity()["fingerprint"]
            for start in range(0, n, size):
                rows = np.arange(start, min(n, start + size))
                counts = ds.read(rows)
                if (counts.sum(axis=1) <= 0).any():
                    raise ValueError("An input cell has no expression in the bank vocabulary")
                h = backbone.encode(counts, bank.genes, ds.obs(rows))
                theta[start : start + len(rows)] = bank.transform(h).cpu().numpy()
            handle.attrs["complete"] = True
    os.replace(temporary, output)
    return {"path": str(output), "cells": n, "bank_id": bank.manifest["bank_id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(infer(load_config(args.config))))


if __name__ == "__main__":
    main()

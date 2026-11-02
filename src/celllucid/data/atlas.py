"""Bounded-memory, rank/worker-partitioned batches of raw atlas counts."""

import math
from pathlib import Path
import h5py
import anndata as ad
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info


class AtlasBatches(IterableDataset):
    def __init__(
        self, manifest, split, batch_size, seed, epoch=0, rank=0, world_size=1, start_batch=0
    ):
        self.shards = [s for s in manifest["shards"] if s["split"] == split]
        self.split = split
        self.batch_size = batch_size
        self.seed, self.epoch = seed, epoch
        self.rank, self.world_size, self.start_batch = rank, world_size, start_batch
        self.batch_counts = np.array([math.ceil(s["cells"] / batch_size) for s in self.shards])
        self.total_batches = int(self.batch_counts.sum())
        if self.total_batches == 0:
            raise ValueError(f"No raw count batches for {split}")
        self.steps = (
            math.ceil(self.total_batches / world_size)
            if split == "train"
            else len(range(rank, self.total_batches, world_size))
        )

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (0, 1) if worker is None else (worker.id, worker.num_workers)
        order = np.arange(len(self.shards))
        if self.split == "train":
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        offsets = np.cumsum(self.batch_counts[order])
        current, handle, permutation = None, None, None
        try:
            for step in range(self.start_batch + worker_id, self.steps, workers):
                position = (step * self.world_size + self.rank) % self.total_batches
                order_index = int(np.searchsorted(offsets, position, side="right"))
                shard_index = int(order[order_index])
                previous = int(offsets[order_index - 1]) if order_index else 0
                batch_index = position - previous
                record = self.shards[shard_index]
                if current != shard_index:
                    if handle is not None:
                        handle.close()
                    handle = h5py.File(Path(record["path"]), "r")
                    if not handle.attrs["complete"]:
                        raise ValueError("Incomplete raw count shard")
                    matrix = ad.io.sparse_dataset(handle["X"])
                    permutation = np.arange(record["cells"])
                    if self.split == "train":
                        np.random.default_rng(
                            np.random.SeedSequence([self.seed, self.epoch, shard_index])
                        ).shuffle(permutation)
                    current = shard_index
                rows = permutation[
                    batch_index * self.batch_size : (batch_index + 1) * self.batch_size
                ]
                sorted_rows = np.sort(rows)
                counts = matrix[sorted_rows, :].toarray().astype("f4")
                yield {
                    "counts": counts,
                    "cell_ids": handle["cell_ids"].asstr()[sorted_rows],
                    "assay": handle["assay"].asstr()[sorted_rows],
                    "available": handle["available"][:],
                    "batch_index": step,
                }
        finally:
            if handle is not None:
                handle.close()

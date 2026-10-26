"""Small explicit batch contract, independent of any query-agent schema."""

from dataclasses import dataclass
import numpy as np


@dataclass
class CellBatch:
    counts: np.ndarray
    cell_ids: list[str]
    indices: np.ndarray
    available: np.ndarray


def cell_batch(dataset, indices):
    indices = np.asarray(indices, dtype=np.int64)
    return CellBatch(
        dataset.read(indices), dataset.cell_ids[indices].tolist(), indices, dataset.available
    )

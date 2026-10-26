from pathlib import Path
import torch


def pytest_sessionstart(session):
    # Keep test artifacts inside this repository, including on a fresh clone.
    (Path(__file__).resolve().parents[1] / "runs").mkdir(exist_ok=True)
    torch.set_num_threads(2)

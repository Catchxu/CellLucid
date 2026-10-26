"""Explicit output paths and append-only metrics for resumable experiments."""

from pathlib import Path
import json
import logging
import yaml
from .config import write_json


def prepare_run(path, config):
    root = Path(path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return root


def get_logger(root, name="celllucid"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(Path(root) / "train.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_metrics(root, metrics):
    with (Path(root) / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(metrics, allow_nan=False) + "\n")


__all__ = ["prepare_run", "get_logger", "log_metrics", "write_json"]

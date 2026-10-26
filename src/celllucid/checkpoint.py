"""Topic-only checkpoints: explicit gene vectors, backbone and cache compatibility."""

from pathlib import Path
import os
import json
import torch
from safetensors.torch import save_file, load_file
from .config import write_json, file_sha256


def save_checkpoint(path, model, optimizer, *, epoch, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    # The gene vectors are a nonpersistent buffer; no encoder is registered here.
    save_file(tensors, str(path / "topics.safetensors.tmp"))
    os.replace(path / "topics.safetensors.tmp", path / "topics.safetensors")
    payload = {
        "format_version": 1,
        "model": model.hyperparameters(),
        "epoch": epoch,
        "metadata": metadata,
        "weights_sha256": file_sha256(path / "topics.safetensors"),
    }
    state = {
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "epoch": epoch,
        "metadata": metadata,
    }
    torch.save(state, path / "resume.pt.tmp")
    os.replace(path / "resume.pt.tmp", path / "resume.pt")
    write_json(path / "checkpoint.json", payload)
    return payload


def load_checkpoint(path, model, optimizer=None, *, expected_metadata=None):
    path = Path(path)
    info = json.loads((path / "checkpoint.json").read_text())
    if info.get("format_version") != 1 or info["model"] != model.hyperparameters():
        raise ValueError("Checkpoint model configuration differs")
    if expected_metadata is not None and info["metadata"] != expected_metadata:
        raise ValueError("Checkpoint data/backbone/split identity differs")
    if file_sha256(path / "topics.safetensors") != info["weights_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    model.load_state_dict(load_file(str(path / "topics.safetensors")), strict=True)
    if optimizer is not None:
        state = torch.load(path / "resume.pt", map_location="cpu", weights_only=True)
        if state["metadata"] != info["metadata"] or state["epoch"] != info["epoch"]:
            raise ValueError("Resume state and weights manifest differ")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return info

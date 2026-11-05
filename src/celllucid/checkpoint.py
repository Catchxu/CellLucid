"""Topic-model checkpoints with the training cursor and cache identity."""

from pathlib import Path
import json
import os
import torch
from safetensors.torch import save_file, load_file
from .config import write_json, file_digest

FORMAT_VERSION = 5


def save_checkpoint(path, model, optimizer, *, step, metadata):
    """Write weights, optimizer and RNG state atomically; step counts completed updates."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    # Gene vectors are nonpersistent; the frozen backbone is not part of this head.
    save_file(tensors, str(path / "topics.safetensors.tmp"))
    os.replace(path / "topics.safetensors.tmp", path / "topics.safetensors")
    state = {
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "step": step,
    }
    torch.save(state, path / "resume.pt.tmp")
    os.replace(path / "resume.pt.tmp", path / "resume.pt")
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.hyperparameters(),
        "step": step,
        "metadata": metadata,
        "weights_digest": file_digest(path / "topics.safetensors"),
        "resume_digest": file_digest(path / "resume.pt"),
    }
    write_json(path / "checkpoint.json", payload)
    return payload


def load_checkpoint(path, model, optimizer=None, *, expected_metadata=None):
    path = Path(path)
    info = json.loads((path / "checkpoint.json").read_text())
    if info["format_version"] != FORMAT_VERSION or info["model"] != model.hyperparameters():
        raise ValueError("Checkpoint model configuration differs")
    if expected_metadata is not None and info["metadata"] != expected_metadata:
        raise ValueError("Checkpoint cache/backbone/training identity differs")
    if file_digest(path / "topics.safetensors") != info["weights_digest"]:
        raise ValueError("Checkpoint checksum mismatch")
    model.load_state_dict(load_file(str(path / "topics.safetensors")), strict=True)
    if optimizer is not None:
        if file_digest(path / "resume.pt") != info["resume_digest"]:
            raise ValueError("Resume state checksum mismatch")
        state = torch.load(path / "resume.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng"])
    return info

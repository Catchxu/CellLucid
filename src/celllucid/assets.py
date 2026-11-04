"""Resolve upstream assets without copying frozen weights into topic checkpoints."""

from pathlib import Path
import json
from .config import file_digest, fingerprint


class BackboneAssets:
    def __init__(self, path):
        self.root = Path(path).expanduser().resolve()
        self.config = self.root / "config.json"
        self.weights = self.root / "model_weights.pt"
        self.vocabs = self.root / "vocabs"
        for item in (self.config, self.weights, self.vocabs):
            if not item.exists():
                raise FileNotFoundError(f"Missing TranscriptFormer asset: {item}")

    def identity(self):
        # Hash the actual files; a directory name or mtime is not an identity.
        files = [self.config, self.weights]
        files.extend(sorted(p for p in self.vocabs.rglob("*") if p.is_file()))
        hashes = {str(p.relative_to(self.root)): file_digest(p) for p in files}
        return {"format_version": 2, "files": hashes, "fingerprint": fingerprint(hashes)}

    def read_config(self):
        return json.loads(self.config.read_text())


TOPIC_BANK_FILES = ("topics.safetensors", "genes.json", "manifest.json")


def download_topic_bank(repo_id, *, repo_type="space", revision="main", local_dir="assets"):
    """Download only the topic-bank package into a local assets directory."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=["topic_bank/" + name for name in TOPIC_BANK_FILES],
    )
    return Path(root) / "topic_bank"

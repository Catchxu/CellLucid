"""Portable topic definitions and cell assignment without frozen backbone weights.

A bank folder holds topics.safetensors, genes.json (Ensembl IDs in backbone input order),
symbols.json (gene symbols aligned with genes.json) and manifest.json (settings and checksums).
"""

from pathlib import Path
import argparse
import json
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from ..config import file_digest, write_json, fingerprint

FORMAT_VERSION = 5
FILES = ("topics.safetensors", "genes.json", "symbols.json", "manifest.json")


def export_topic_bank(model, genes, symbols, output_dir, *, backbone, training=None):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    if len(genes) != len(model.gene_vectors) or len(set(genes)) != len(genes) or len(symbols) != len(genes):
        raise ValueError("Invalid ordered gene vocabulary")
    output.mkdir(parents=True)
    with torch.no_grad():
        tensors = {
            "topics": F.normalize(model.topics, dim=-1).detach().cpu().contiguous(),
            "dictionary": model.dictionary().detach().cpu().contiguous(),
            "gene_prior": model.gene_prior.detach().cpu().contiguous(),
            "inference_mean": model.inference_mean.detach().cpu().contiguous(),
            "train_mean": model.train_mean.detach().cpu().contiguous(),
            "gene_map": model.gene_map.weight.detach().cpu().contiguous(),
        }
    save_file(tensors, str(output / "topics.safetensors"))
    write_json(output / "genes.json", list(genes))
    write_json(output / "symbols.json", list(symbols))
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.hyperparameters(),
        "backbone": backbone,
        "training": training or {},
        "gene_representation": "projected_input",
        "cell_input": "unmasked counts, centered by inference_mean",
        "files": {
            name: file_digest(output / name) for name in FILES[:-1]
        },
    }
    payload["bank_id"] = fingerprint(payload)
    write_json(output / "manifest.json", payload)
    return output


class TopicBank:
    def __init__(self, path="assets/topic_banks/default", device="cpu"):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        unsigned = {k: v for k, v in self.manifest.items() if k != "bank_id"}
        if fingerprint(unsigned) != self.manifest.get("bank_id"):
            raise ValueError("Topic-bank manifest fingerprint mismatch")
        if self.manifest["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported topic bank")
        for name, expected in self.manifest["files"].items():
            if file_digest(self.path / name) != expected:
                raise ValueError(f"Topic bank checksum mismatch: {name}")
        self.genes = json.loads((self.path / "genes.json").read_text())
        self.symbols = json.loads((self.path / "symbols.json").read_text())
        tensors = load_file(str(self.path / "topics.safetensors"), device=str(device))
        self.topics = tensors["topics"]
        self.dictionary = tensors["dictionary"]
        self.gene_prior = tensors["gene_prior"]
        self.inference_mean = tensors["inference_mean"]
        self.temperature = self.manifest["model"]["temperature"]
        if self.dictionary.shape != (len(self.topics), len(self.genes)):
            raise ValueError("Dictionary shape does not match ordered genes/topics")
        if (
            not torch.isfinite(self.dictionary).all()
            or (self.dictionary < 0).any()
            or not torch.allclose(
                self.dictionary.sum(-1),
                torch.ones(len(self.topics), device=self.dictionary.device),
                atol=1e-5,
            )
        ):
            raise ValueError("Invalid dictionary probabilities")

    @torch.no_grad()
    def transform(self, embeddings):
        """Topic usage of cells encoded from unmasked counts."""
        h = torch.as_tensor(embeddings, device=self.topics.device, dtype=torch.float32)
        if h.ndim != 2 or h.shape[1] != self.topics.shape[1] or not torch.isfinite(h).all():
            raise ValueError("Cell embeddings have invalid shape or values")
        scores = F.normalize(h - self.inference_mean, dim=-1) @ self.topics.T
        return torch.softmax(scores / self.temperature, dim=-1)

    def top_genes(self, n=20):
        values, indices = self.dictionary.topk(min(n, len(self.genes)), dim=-1)
        return [
            [
                {"gene": self.symbols[j], "gene_id": self.genes[j], "weight": float(v)}
                for j, v in zip(ii.tolist(), vv.tolist(), strict=True)
            ]
            for ii, vv in zip(indices, values, strict=True)
        ]


def main():
    parser = argparse.ArgumentParser(description="Export a pretrained topic checkpoint as a topic bank")
    parser.add_argument("--checkpoint", default="runs/hca_pretraining/checkpoints/latest")
    parser.add_argument("--cache", required=True, help="Training cache the checkpoint was trained on")
    parser.add_argument("--output", default="runs/hca_pretraining/topic_bank")
    args = parser.parse_args()
    from ..checkpoint import load_checkpoint
    from ..data.cache import cache_identity, cache_initialization
    from ..models.initialization import empty_model, gene_symbols

    info = json.loads((Path(args.checkpoint) / "checkpoint.json").read_text())
    identity = cache_identity(args.cache)
    if info["metadata"]["cache"]["manifest_identity"] != identity["manifest_identity"]:
        raise ValueError("Checkpoint was trained on a different cache")
    model = empty_model(cache_initialization(args.cache)["gene_vectors"], identity["genes"], info["metadata"]["model"],
                        "cpu", False)
    load_checkpoint(args.checkpoint, model)
    symbols = gene_symbols(identity["genes"], info["metadata"]["model"]["gene_symbols"])
    print(export_topic_bank(model, identity["genes"], symbols, args.output,
                            backbone={"fingerprint": identity["backbone_fingerprint"]},
                            training=info["metadata"]))


if __name__ == "__main__":
    main()

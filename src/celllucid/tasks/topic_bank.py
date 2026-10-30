"""Portable, immutable topic definitions without encoder weights or pathway labels."""

from pathlib import Path
import argparse
import json
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from ..config import file_digest, write_json, fingerprint


def export_topic_bank(model, genes, output_dir, *, backbone, training=None):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    if len(genes) != len(model.gene_vectors) or len(set(genes)) != len(genes):
        raise ValueError("Invalid ordered gene vocabulary")
    output.mkdir(parents=True)
    with torch.no_grad():
        tensors = {
            "topics": model.topics.detach().cpu().contiguous(),
            "dictionary": model.dictionary().detach().cpu().contiguous(),
            "gene_prior": model.gene_prior.detach().cpu().contiguous(),
        }
    save_file(tensors, str(output / "topics.safetensors"))
    write_json(output / "genes.json", list(genes))
    payload = {
        "format_version": 2,
        "model": model.hyperparameters(),
        "backbone": backbone,
        "training": training or {},
        "gene_representation": "projected_input",
        "files": {
            name: file_digest(output / name) for name in ("topics.safetensors", "genes.json")
        },
    }
    payload["bank_id"] = fingerprint(payload)
    write_json(output / "manifest.json", payload)
    return output


class TopicBank:
    def __init__(self, path, device="cpu"):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        unsigned = {k: v for k, v in self.manifest.items() if k != "bank_id"}
        if fingerprint(unsigned) != self.manifest.get("bank_id"):
            raise ValueError("Topic-bank manifest fingerprint mismatch")

        if self.manifest.get("format_version") != 2:
            raise ValueError("Unsupported topic bank")
        for name, expected in self.manifest["files"].items():
            if file_digest(self.path / name) != expected:
                raise ValueError(f"Topic bank checksum mismatch: {name}")
        self.genes = json.loads((self.path / "genes.json").read_text())
        tensors = load_file(str(self.path / "topics.safetensors"), device=str(device))
        self.topics = tensors["topics"]
        self.dictionary = tensors["dictionary"]
        self.gene_prior = tensors["gene_prior"]
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
        h = torch.as_tensor(embeddings, device=self.topics.device, dtype=torch.float32)
        if h.ndim != 2 or h.shape[1] != self.topics.shape[1] or not torch.isfinite(h).all():
            raise ValueError("Cell embeddings have invalid shape or values")
        if (h.norm(dim=-1) == 0).any():
            raise ValueError("Cell embeddings must be nonzero")
        return torch.softmax(
            F.normalize(h, dim=-1) @ F.normalize(self.topics, dim=-1).T / self.temperature, dim=-1
        )

    def top_genes(self, n=20):
        values, indices = self.dictionary.topk(min(n, len(self.genes)), dim=-1)
        return [
            [
                {"gene": self.genes[j], "weight": float(v)}
                for j, v in zip(ii.tolist(), vv.tolist(), strict=True)
            ]
            for ii, vv in zip(indices, values, strict=True)
        ]


def main():
    parser = argparse.ArgumentParser(description="Export a trained topic checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from ..data.cache import FeatureCache
    from ..models.topic import TopicModel
    from ..checkpoint import load_checkpoint
    from ..trainer.topic import array_hash

    info = json.loads((Path(args.checkpoint) / "checkpoint.json").read_text())
    with FeatureCache(args.cache) as cache:
        if (
            info["metadata"]["genes"] != cache.genes
            or info["metadata"]["gene_vectors_digest"] != array_hash(cache.gene_vectors)
            or info["metadata"]["backbone"]["fingerprint"]
            != cache.metadata["backbone"]["fingerprint"]
        ):
            raise ValueError("Cache is incompatible with the checkpoint")
        model = TopicModel(cache.gene_vectors, **info["model"])
        load_checkpoint(args.checkpoint, model)
        print(
            export_topic_bank(
                model,
                cache.genes,
                args.output,
                backbone=cache.metadata["backbone"],
                training=info["metadata"],
            )
        )


if __name__ == "__main__":
    main()

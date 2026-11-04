"""Portable topic definitions and cell projection without frozen backbone weights."""

from pathlib import Path
import argparse
import json
import torch
from ..models.topic import ResidualCellProjection, cell_topic_usage
from ..assets import download_topic_bank
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
        tensors.update({
            "projection." + name: value.detach().cpu().contiguous()
            for name, value in model.projection.state_dict().items()
        })
    save_file(tensors, str(output / "topics.safetensors"))
    write_json(output / "genes.json", list(genes))
    payload = {
        "format_version": 3,
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
    def __init__(self, path="assets/topic_bank", device="cpu"):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        unsigned = {k: v for k, v in self.manifest.items() if k != "bank_id"}
        if fingerprint(unsigned) != self.manifest.get("bank_id"):
            raise ValueError("Topic-bank manifest fingerprint mismatch")

        if self.manifest["format_version"] != 3:
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
        self.projection = ResidualCellProjection(
            self.topics.shape[1], self.manifest["model"]["projection_hidden"]
        ).to(device)
        self.projection.load_state_dict({
            name: tensors["projection." + name]
            for name in self.projection.state_dict()
        }, strict=True)
        self.projection.eval().requires_grad_(False)
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

    @classmethod
    def from_pretrained(cls, repo_id, *, repo_type="space", revision="main", local_dir="assets", device="cpu"):
        path = download_topic_bank(
            repo_id, repo_type=repo_type, revision=revision, local_dir=local_dir
        )
        return cls(path, device=device)

    @torch.no_grad()
    def transform(self, embeddings):
        h = torch.as_tensor(embeddings, device=self.topics.device, dtype=torch.float32)
        if h.ndim != 2 or h.shape[1] != self.topics.shape[1] or not torch.isfinite(h).all():
            raise ValueError("Cell embeddings have invalid shape or values")
        if (h.norm(dim=-1) == 0).any():
            raise ValueError("Cell embeddings must be nonzero")
        return cell_topic_usage(h, self.topics, self.projection, self.temperature)

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
    parser = argparse.ArgumentParser(description="Export an online-pretraining topic checkpoint")
    parser.add_argument("--checkpoint", default="runs/hca_pretraining/checkpoints/best")
    parser.add_argument("--config", default="configs/topic.yaml")
    parser.add_argument("--output", default="runs/hca_pretraining/topic_bank")
    args = parser.parse_args()
    from ..config import load_config
    from ..models.backbone import FrozenTranscriptFormer
    from ..models.topic import TopicModel
    from ..checkpoint import load_checkpoint

    info = json.loads((Path(args.checkpoint) / "checkpoint.json").read_text())
    backbone = FrozenTranscriptFormer(**load_config(args.config)["backbone"])
    if backbone.identity() != info["metadata"]["backbone"]:
        raise ValueError("Backbone differs from the pretraining checkpoint")
    genes = info["metadata"]["genes"]
    model = TopicModel(backbone.gene_vectors(genes), **info["model"])
    load_checkpoint(args.checkpoint, model)
    print(
        export_topic_bank(
            model, genes, args.output, backbone=backbone.identity(), training=info["metadata"]
        )
    )


if __name__ == "__main__":
    main()

"""Frozen adapter around the unmodified upstream TranscriptFormer implementation."""

from contextlib import nullcontext
import importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch import nn
from ..assets import BackboneAssets
from ..config import fingerprint, file_digest
from pathlib import Path
import importlib.util


class FrozenTranscriptFormer(nn.Module):
    def __init__(
        self,
        checkpoint,
        *,
        device="cuda",
        max_length=None,
        compile_attention=False,
        precision="float32",
    ):
        super().__init__()
        try:
            from omegaconf import OmegaConf
            from transcriptformer.model.model import Transcriptformer
            from transcriptformer.tokenizer.vocab import load_vocabs_and_embeddings
        except ImportError as error:
            raise ImportError(
                "Install the pinned TranscriptFormer dependency; see README.md"
            ) from error
        self.assets = BackboneAssets(checkpoint)
        config = OmegaConf.create(self.assets.read_config())
        config.model.data_config.aux_vocab_path = str(self.assets.vocabs)
        config.model.data_config.esm2_mappings_path = str(self.assets.vocabs)
        model_config = config.model.model_config
        # Runtime sequence length and attention compilation do not change learned weights.
        if max_length is not None:
            if max_length < 2 or max_length > model_config.seq_len:
                raise ValueError("max_length must be between 2 and the checkpoint sequence length")
            model_config.seq_len = int(max_length)
        model_config.compile_block_mask = bool(compile_attention)
        (gene_vocab, aux_vocab), matrix = load_vocabs_and_embeddings(config)
        model = Transcriptformer(
            data_config=config.model.data_config,
            model_config=model_config,
            loss_config=config.model.loss_config,
            inference_config=None,
            gene_vocab_dict=gene_vocab,
            aux_vocab_dict=aux_vocab,
            emb_matrix=matrix,
        )
        weights = torch.load(self.assets.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        del weights, matrix
        self.model = model.requires_grad_(False).to(device)
        self.device = torch.device(device)
        self.precision = precision
        if precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        self.gene_vocab = gene_vocab
        self.aux_vocab = aux_vocab
        self.max_length = int(model_config.seq_len)
        self.dimension = int(model_config.embed_dim)
        self.genes = [
            g for g, _ in sorted(gene_vocab.items(), key=lambda x: x[1]) if g.startswith("ENS")
        ]
        self._identity = None
        self.train(False)

    def train(self, mode=True):
        # nn.Module.train() on an enclosing pipeline must never enable encoder dropout.
        super().train(False)
        return self

    def identity(self):
        if self._identity is None:
            runtime = {
                "max_length": self.max_length,
                "precision": self.precision,
                "data_config": dict(self.model.data_config),
                "pooling": "upstream mean_embeddings(gene_output, shifted_padding_mask)",
                "gene_representation": "projected_input",
                "upstream_version": importlib.metadata.version("transcriptformer"),
            }
            # OmegaConf lists must be converted to native containers.
            from omegaconf import OmegaConf

            runtime["data_config"] = OmegaConf.to_container(self.model.data_config, resolve=True)
            # Location is provenance, not model identity: identical assets may be relocated.
            for key in ("aux_vocab_path", "esm2_mappings_path"):
                runtime["data_config"][key] = "@backbone/vocabs"
            source_root = Path(
                importlib.util.find_spec("transcriptformer").submodule_search_locations[0]
            )
            source_files = [
                "model/model.py",
                "model/layers.py",
                "model/masks.py",
                "data/dataclasses.py",
                "data/dataloader.py",
                "tokenizer/tokenizer.py",
                "tokenizer/vocab.py",
                "utils/utils.py",
            ]
            runtime["upstream_sources"] = {
                name: file_digest(source_root / name) for name in source_files
            }
            runtime["torch_version"] = str(torch.__version__)

            assets = self.assets.identity()
            self._identity = {
                "assets": assets,
                "runtime": runtime,
                "fingerprint": fingerprint({"assets": assets, "runtime": runtime}),
            }
        return self._identity

    @torch.no_grad()
    def gene_vectors(self, genes, batch_size=1024):
        if any(g not in self.gene_vocab for g in genes):
            raise ValueError("Requested gene is not in the backbone vocabulary")
        chunks = []
        for start in range(0, len(genes), batch_size):
            tokens = torch.tensor(
                [self.gene_vocab[g] for g in genes[start : start + batch_size]], device=self.device
            )
            chunks.append(self.model.gene_embeddings(tokens).float().cpu())
        return torch.cat(chunks)

    @torch.no_grad()
    def encode(self, counts, genes, obs=None):
        from transcriptformer.data.dataloader import process_batch
        from transcriptformer.data.dataclasses import BatchData
        from transcriptformer.tokenizer.tokenizer import BatchGeneTokenizer, BatchObsTokenizer

        self.train(False)
        counts = np.asarray(counts, dtype=np.float32)
        if counts.ndim != 2 or counts.shape[1] != len(genes):
            raise ValueError("Counts and ordered genes do not align")
        if not np.isfinite(counts).all() or (counts < 0).any() or (counts.sum(axis=1) <= 0).any():
            raise ValueError(
                "Encoder input must have finite nonnegative expression and visible counts"
            )
        if any(g not in self.gene_vocab for g in genes):
            raise ValueError("Encoder inputs must use the backbone gene vocabulary")
        obs = (
            pd.DataFrame(index=range(len(counts)))
            if obs is None
            else obs.reset_index(drop=True).copy()
        )
        if self.aux_vocab:
            for column in self.aux_vocab:
                if column not in obs:
                    obs[column] = "unknown"
                obs[column] = obs[column].astype(str)
        cfg = self.model.data_config
        tokens = process_batch(
            counts,
            obs,
            np.asarray(genes),
            BatchGeneTokenizer(self.gene_vocab),
            BatchObsTokenizer(self.aux_vocab) if self.aux_vocab else None,
            cfg.sort_genes,
            cfg.randomize_genes,
            self.max_length,
            cfg.pad_zeros,
            cfg.gene_pad_token,
            self.gene_vocab,
            cfg.normalize_to_scale,
            cfg.clip_counts,
            self.aux_vocab,
        )
        batch = BatchData(**{k: v.to(self.device) for k, v in tokens.items()})
        context = (
            torch.autocast(self.device.type, dtype=torch.bfloat16)
            if self.precision == "bfloat16"
            else nullcontext()
        )
        with context:
            result = self.model(batch, embed=True)
        embeddings = result["embeddings"].float().detach().cpu()
        if not torch.isfinite(embeddings).all():
            raise FloatingPointError("Frozen backbone returned nonfinite embeddings")
        return embeddings

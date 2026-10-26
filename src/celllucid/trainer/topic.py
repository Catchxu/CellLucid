"""Train only shared topic prototypes from reproducible masked feature caches."""

from contextlib import ExitStack
from pathlib import Path
import argparse
import hashlib
import numpy as np
import pandas as pd
import torch
from ..config import load_config, seed_everything, fingerprint, write_json
from ..checkpoint import save_checkpoint, load_checkpoint
from ..experiment import prepare_run, get_logger, log_metrics
from ..data.cache import FeatureCache
from ..data.splits import split_indices
from ..models.topic import TopicModel
from ..losses.masked import masked_compositional_loss


def array_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def prepare_rows(caches, split_config):
    unique, key_to_index, pairs = [], {}, []
    for ci, cache in enumerate(caches):
        for row, source_index in enumerate(cache.indices):
            key = (cache.metadata["dataset_sha256"], int(source_index))
            if key not in key_to_index:
                record = cache.dataset.adata.obs.iloc[int(source_index)].to_dict()
                record["_dataset"] = cache.metadata["dataset_sha256"]
                record["_cell"] = cache.cell_ids[row]
                key_to_index[key] = len(unique)
                unique.append(record)
            pairs.append((ci, row, key_to_index[key]))
    obs = pd.DataFrame(unique)
    train_cells, val_cells = split_indices(obs, **split_config)
    train_set, val_set = set(train_cells), set(val_cells)
    train = np.array([p[:2] for p in pairs if p[2] in train_set], dtype=np.int64)
    val = np.array([p[:2] for p in pairs if p[2] in val_set], dtype=np.int64)
    return (
        train,
        val,
        {
            "train_unique_cells": len(train_cells),
            "validation_unique_cells": len(val_cells),
            "policy": split_config,
            "train_identity": fingerprint(train.tolist()),
            "validation_identity": fingerprint(val.tolist()),
        },
    )


def batches(caches, pairs, batch_size, device):
    for start in range(0, len(pairs), batch_size):
        part = pairs[start : start + batch_size]
        features, counts, masks = [], [], []
        for ci, cache in enumerate(caches):
            rows = part[part[:, 0] == ci, 1]
            if len(rows):
                h, x, m = cache.read(rows)
                features.append(h)
                counts.append(x)
                masks.append(m)
        yield tuple(
            torch.as_tensor(np.concatenate(value), device=device)
            for value in (features, counts, masks)
        )


@torch.no_grad()
def evaluate(model, caches, pairs, batch_size, device, max_batches=None):
    model.eval()
    beta = model.dictionary()
    total, n = 0.0, 0
    for step, (h, x, hidden) in enumerate(batches(caches, pairs, batch_size, device)):
        if max_batches is not None and step >= max_batches:
            break
        loss = masked_compositional_loss(model.theta(h) @ beta, x, hidden)
        total += float(loss) * len(h)
        n += len(h)
    if not n:
        raise ValueError("No validation batches")
    return total / n


def train(config, resume=None):
    output_path = Path(config["output_dir"]).resolve()
    if not resume and (output_path / "checkpoints/latest/checkpoint.json").exists():
        raise FileExistsError(
            "Run already has a checkpoint; use --resume or a new output directory"
        )
    if int(config.get("epochs", 10)) < 1 or not config.get("caches"):
        raise ValueError("Training requires positive epochs and at least one masked cache")
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    root = prepare_run(config["output_dir"], config)
    logger = get_logger(root)
    device = torch.device(config.get("device", "cpu"))
    with ExitStack() as stack:
        caches = [stack.enter_context(FeatureCache(path)) for path in config["caches"]]
        first = caches[0]
        vectors_hash = array_hash(first.gene_vectors)
        for cache in caches:
            if cache.metadata["feature_kind"] != "masked":
                raise ValueError("Masked training cannot use full-input feature caches")
            if (
                cache.genes != first.genes
                or array_hash(cache.gene_vectors) != vectors_hash
                or cache.metadata["backbone"]["fingerprint"]
                != first.metadata["backbone"]["fingerprint"]
            ):
                raise ValueError("Caches must share ordered genes and an identical frozen backbone")
        split_config = config.get("split", {"validation_fraction": 0.2, "seed": seed})
        train_rows, val_rows, split_info = prepare_rows(caches, split_config)
        if not split_config.get("group_key") and not split_config.get("split_key"):
            logger.warning("Cell-level development split; this is not cross-study validation")
        model = TopicModel(first.gene_vectors, seed=seed, **config.get("model", {})).to(device)
        if list(dict(model.named_parameters())) != ["topics"]:
            raise RuntimeError("Only topic embeddings may enter the optimizer")
        optimizer_config = config.get("optimizer", {"lr": 0.005})
        optimizer = torch.optim.AdamW([model.topics], **optimizer_config)
        metadata = {
            "genes": first.genes,
            "gene_vectors_sha256": vectors_hash,
            "backbone": first.metadata["backbone"],
            "cache_fingerprints": [c.metadata["fingerprint"] for c in caches],
            "split": split_info,
            "seed": seed,
            "optimizer": optimizer_config,
            "batch_size": config.get("batch_size", 32),
            "max_steps_per_epoch": config.get("max_steps_per_epoch"),
            "grad_clip": config.get("grad_clip", 1.0),
        }
        write_json(root / "training_manifest.json", metadata)
        start_epoch = 0
        if resume:
            info = load_checkpoint(resume, model, optimizer, expected_metadata=metadata)
            start_epoch = int(info["epoch"])
            if start_epoch >= int(config.get("epochs", 10)):
                raise ValueError("Resume epochs must exceed the completed checkpoint epoch")

        batch_size = int(config.get("batch_size", 32))
        initial_val = evaluate(model, caches, val_rows, batch_size, device)
        logger.info(
            "Initial validation CE %.6f; train views %d; val views %d",
            initial_val,
            len(train_rows),
            len(val_rows),
        )
        best_val = initial_val
        initial_topics = model.topics.detach().clone()
        for epoch in range(start_epoch, int(config.get("epochs", 10))):
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(len(train_rows))
            total, n = 0.0, 0
            for step, (h, x, hidden) in enumerate(
                batches(caches, train_rows[order], batch_size, device)
            ):
                cap = config.get("max_steps_per_epoch")
                if cap is not None and step >= cap:
                    break
                optimizer.zero_grad(set_to_none=True)
                predicted, _, _ = model(h)
                loss = masked_compositional_loss(predicted, x, hidden)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                if model.topics.grad is None or not torch.isfinite(model.topics.grad).all():
                    raise FloatingPointError("Invalid topic gradient")
                torch.nn.utils.clip_grad_norm_([model.topics], config.get("grad_clip", 1.0))
                optimizer.step()
                total += float(loss.detach()) * len(h)
                n += len(h)
            if not n:
                raise ValueError("No training batches")
            val = evaluate(model, caches, val_rows, batch_size, device)
            metrics = {
                "epoch": epoch + 1,
                "train_ce": total / n,
                "validation_ce": val,
                "topic_update_norm": float((model.topics.detach() - initial_topics).norm()),
            }
            logger.info("%s", metrics)
            log_metrics(root, metrics)
            save_checkpoint(
                root / "checkpoints/latest", model, optimizer, epoch=epoch + 1, metadata=metadata
            )
            if val < best_val:
                best_val = val
                save_checkpoint(
                    root / "checkpoints/best", model, optimizer, epoch=epoch + 1, metadata=metadata
                )
        from ..tasks.topic_bank import export_topic_bank

        bank_path = root / "topic_bank"
        if bank_path.exists():
            bank_path = root / f"topic_bank_epoch_{int(config.get('epochs', 10)):04d}"
        bank = export_topic_bank(
            model,
            first.genes,
            bank_path,
            backbone=first.metadata["backbone"],
            training=metadata,
        )
        return {
            "initial_validation_ce": initial_val,
            "best_validation_ce": best_val,
            "bank": str(bank),
            "topic_update_norm": float((model.topics.detach() - initial_topics).norm()),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(train(load_config(args.config), args.resume))


if __name__ == "__main__":
    main()

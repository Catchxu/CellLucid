"""Atlas pretraining: fresh masks, frozen backbone forwards and topic updates."""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from ..config import load_config, seed_everything, write_json
from ..checkpoint import save_checkpoint, load_checkpoint
from ..experiment import prepare_run, get_logger, log_metrics
from ..data.atlas import AtlasBatches
from ..data.masking import masked_views
from ..models.backbone import FrozenTranscriptFormer
from ..models.topic import TopicModel
from ..tasks.topic_bank import export_topic_bank
from ..losses.masked import masked_compositional_loss


def loader(manifest, split, config, epoch, rank, world_size, start=0):
    dataset = AtlasBatches(
        manifest, split, config["batch_size"], config["seed"], epoch, rank, world_size, start
    )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=config["workers"],
        multiprocessing_context="spawn" if config["workers"] else None,
    )


def forward_batch(backbone, batch, genes, config, epoch, validation=False):
    counts = batch["counts"].numpy()
    available = batch["available"].numpy()
    visible, hidden = masked_views(
        counts,
        batch["cell_ids"],
        available,
        fraction=config["mask_fraction"],
        seed=config["seed"],
        view=0 if validation else epoch + 1,
    )
    good = (visible.sum(1) > 0) & ((counts * hidden).sum(1) > 0)
    indices = np.flatnonzero(good)
    features = []
    for start in range(0, len(indices), config["encoder_batch_size"]):
        rows = indices[start : start + config["encoder_batch_size"]]
        obs = pd.DataFrame({"assay": np.asarray(batch["assay"])[rows]})
        features.append(backbone.encode(visible[rows], genes, obs))
    h = (
        torch.cat(features).to(backbone.device)
        if features
        else torch.empty((0, backbone.dimension), device=backbone.device)
    )
    return (
        h,
        torch.as_tensor(counts[good], device=backbone.device),
        torch.as_tensor(hidden[good], device=backbone.device),
    )


def reduce_pair(loss, count, device, distributed):
    values = torch.tensor([loss, count], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(values)
    return values.tolist()


@torch.no_grad()
def evaluate(model, backbone, manifest, genes, config, rank, world_size):
    model.eval()
    beta = model.dictionary()
    total, cells = 0.0, 0
    for batch in loader(manifest, "validation", config, 0, rank, world_size):
        h, x, hidden = forward_batch(backbone, batch, genes, config, 0, validation=True)
        if len(h):
            loss = masked_compositional_loss(model.theta(h) @ beta, x, hidden)
            total += float(loss) * len(h)
            cells += len(h)
    total, cells = reduce_pair(total, cells, backbone.device, world_size > 1)
    if not cells:
        raise ValueError("No valid validation cells")
    return total / cells


def train(config, manifest_path, output_dir, epochs, resume=None, max_steps=None, backbone=None):
    distributed = "RANK" in os.environ
    rank = int(os.environ["RANK"]) if distributed else 0
    world_size = int(os.environ["WORLD_SIZE"]) if distributed else 1
    local_rank = int(os.environ["LOCAL_RANK"]) if distributed else 0
    settings = config["training"]
    if distributed:
        backend = "nccl" if config["backbone"]["device"] == "cuda" else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
    device = torch.device(f"cuda:{local_rank}" if config["backbone"]["device"] == "cuda" else "cpu")
    seed_everything(settings["seed"])
    root = Path(output_dir).resolve()
    if not resume and (root / "checkpoints/latest/checkpoint.json").exists():
        raise FileExistsError("Run exists; use --resume or a new output directory")
    if rank == 0:
        prepare_run(
            root,
            {
                **config,
                "manifest": str(Path(manifest_path).resolve()),
                "epochs": epochs,
                "max_steps": max_steps,
            },
        )
    if distributed:
        dist.barrier()
    logger = get_logger(root, name=f"celllucid.rank{rank}") if rank == 0 else None
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["kind"] != "raw_count_shards":
        raise ValueError("Pretraining requires the atlas raw-count manifest")
    genes = json.loads(Path(manifest["genes_path"]).read_text())
    backbone_settings = dict(config["backbone"])
    backbone_settings["device"] = str(device)
    backbone = FrozenTranscriptFormer(**backbone_settings) if backbone is None else backbone
    if backbone.genes != genes:
        raise ValueError("Corpus gene order differs from the frozen backbone")
    vectors = backbone.gene_vectors(genes)
    model = TopicModel(vectors, seed=settings["seed"], **config["model"]).to(device)
    if list(dict(model.named_parameters())) != ["topics"]:
        raise RuntimeError("Only topic embeddings may enter the optimizer")
    optimizer = torch.optim.AdamW([model.topics], **settings["optimizer"])
    metadata = {
        "corpus_identity": manifest["identity"],
        "genes": genes,
        "backbone": backbone.identity(),
        "training": settings,
        "world_size": world_size,
        "masking": "fresh_per_cell_per_epoch",
        "distributed_sampling": "Repeat initial batches to fill the final distributed step",
    }
    if rank == 0:
        write_json(root / "training_manifest.json", metadata)
    start_epoch = start_batch = step = 0
    best = float("inf")
    if resume:
        info = load_checkpoint(resume, model, optimizer, expected_metadata=metadata)
        start_epoch = info["epoch"]
        start_batch = info["batch_in_epoch"]
        step = info["step"]
        best = float("inf") if info["best_validation"] is None else info["best_validation"]
    if start_epoch >= epochs:
        raise ValueError("Requested epochs must exceed the completed checkpoint epoch")
    if max_steps is not None and step >= max_steps:
        raise ValueError("max_steps must exceed the checkpoint step")
    parallel = (
        DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if distributed
        else model
    )
    initial_topics = model.topics.detach().clone()
    stopped = False
    try:
        for epoch in range(start_epoch, epochs):
            stream = loader(
                manifest,
                "train",
                settings,
                epoch,
                rank,
                world_size,
                start_batch if epoch == start_epoch else 0,
            )
            model.train()
            loss_sum = cell_sum = 0.0
            cursor = start_batch if epoch == start_epoch else 0
            for batch in stream:
                h, x, hidden = forward_batch(backbone, batch, genes, settings, epoch)
                predicted, _, _ = parallel(h)
                count = len(h)
                loss = (
                    masked_compositional_loss(predicted, x, hidden)
                    if count
                    else predicted.sum() * 0
                )
                global_count = torch.tensor(float(count), device=device)
                if distributed:
                    dist.all_reduce(global_count)
                if global_count == 0:
                    raise ValueError("No usable masked cells in the global batch")
                scaled_loss = loss * (world_size * count / global_count)
                optimizer.zero_grad(set_to_none=True)
                scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [model.topics], settings["grad_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                step += 1
                cursor = int(batch["batch_index"]) + 1
                loss_sum += float(loss.detach()) * count
                cell_sum += count
                if rank == 0 and step % settings["log_every_steps"] == 0:
                    logger.info(
                        "epoch=%d step=%d batch=%d train_ce=%.6f",
                        epoch + 1,
                        step,
                        cursor,
                        loss_sum / cell_sum if cell_sum else 0,
                    )
                if step % settings["save_every_steps"] == 0 or (
                    max_steps is not None and step >= max_steps
                ):
                    if rank == 0:
                        save_checkpoint(
                            root / "checkpoints/latest",
                            model,
                            optimizer,
                            epoch=epoch,
                            metadata=metadata,
                            step=step,
                            batch_in_epoch=cursor,
                            best_validation=best,
                        )
                    if distributed:
                        dist.barrier()
                if max_steps is not None and step >= max_steps:
                    stopped = True
                    break
            if stopped:
                break
            train_sum, train_cells = reduce_pair(loss_sum, cell_sum, device, distributed)
            validation = evaluate(model, backbone, manifest, genes, settings, rank, world_size)
            improved = validation < best
            best = min(best, validation)
            if rank == 0:
                metrics = {
                    "epoch": epoch + 1,
                    "step": step,
                    "train_ce": train_sum / train_cells if train_cells else None,
                    "validation_ce": validation,
                    "topic_update_norm": float((model.topics.detach() - initial_topics).norm()),
                }
                logger.info("%s", metrics)
                log_metrics(root, metrics)
                save_checkpoint(
                    root / "checkpoints/latest",
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    metadata=metadata,
                    step=step,
                    batch_in_epoch=0,
                    best_validation=best,
                )
                if improved:
                    save_checkpoint(
                        root / "checkpoints/best",
                        model,
                        optimizer,
                        epoch=epoch + 1,
                        metadata=metadata,
                        step=step,
                        batch_in_epoch=0,
                        best_validation=best,
                    )
            if distributed:
                dist.barrier()
        if rank == 0:
            export_topic_bank(
                model,
                genes,
                root / f"topic_bank_step_{step:09d}",
                backbone=backbone.identity(),
                training=metadata,
            )
        return {
            "step": step,
            "output_dir": str(root),
            "topic_update_norm": float((model.topics.detach() - initial_topics).norm()),
        }
    finally:
        if distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True, help="Prepared atlas raw-count manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, help="Cells per optimizer step per GPU")
    parser.add_argument("--encoder-batch-size", type=int, help="Frozen encoder microbatch per GPU")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.encoder_batch_size is not None:
        config["training"]["encoder_batch_size"] = args.encoder_batch_size
    print(train(config, args.manifest, args.output_dir, args.epochs, args.resume, args.max_steps))


if __name__ == "__main__":
    main()

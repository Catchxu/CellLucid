"""Topic pretraining on an embedding cache (single process, one device).

Training is one pass over the cache in a fixed random order: every eligible cell is used
once, with its own mask, in batches of training.batch_size; the last partial batch is
dropped. Topics and the gene map follow the gradient of masked CE, which reaches the gene
map through the Sinkhorn dictionary. The learning rate warms up linearly to its peak and then
follows a cosine decay to training.min_lr at the last step. Training logs a few drift
indicators; periodic evaluation scores the fixed pools. Checkpoints hold weights, optimizer,
RNG and the step.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..checkpoint import load_checkpoint, save_checkpoint
from ..config import load_config, seed_everything, write_json
from ..data.cache import TrainingCache
from ..evaluation import EvaluationPools
from ..experiment import get_logger, log_metrics, prepare_run
from ..losses.masked import masked_compositional_loss
from ..models.initialization import empty_model, initial_model


def learning_rate(step, steps, peak, warmup_steps, min_lr):
    """Learning rate of update `step` (0-based): linear warmup, then cosine decay to min_lr."""
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (steps - 1 - warmup_steps)
    return min_lr + (peak - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def monitor(model, h, x, hidden, predicted, theta, ce):
    """Training-batch indicators of reconstruction, flattening, drift and concentration."""
    background = masked_compositional_loss(model.gene_prior.expand_as(predicted).contiguous(), x, hidden)
    entropy = torch.special.entr(theta).sum(1)
    return {
        "ce": float(ce),
        "ce_gain_over_background": float(background - ce),
        "cell_effective_topics": float(entropy.exp().mean()),
        "cell_topic_cosine": float(model.scores(h).max(dim=1).values.mean()),
        "max_topic_mass": float(theta.mean(0).max()),
    }


def train(config, cache_dir, output_dir, *, resume=False, max_steps=None):
    settings, model_config = config["training"], config["model"]
    device = torch.device(settings["device"])
    seed_everything(settings["seed"])
    root = Path(output_dir).resolve()
    latest = root / "checkpoints/latest"
    if not resume and latest.exists():
        raise FileExistsError("Run exists; use --resume or a new output directory")
    prepare_run(root, config)
    logger = get_logger(root)
    cache = TrainingCache(cache_dir)
    batch_size = settings["batch_size"]
    steps = len(cache) // batch_size
    metadata = json.loads(json.dumps({
        "cache": {k: cache.identity[k] for k in ("manifest_identity", "backbone_fingerprint", "evaluation_cells_digest",
                                                 "settings", "views", "samples")},
        "model": model_config,
        "training": {**{k: settings[k] for k in ("seed", "batch_size", "optimizer", "warmup_steps", "min_lr",
                                                 "grad_clip")}, "steps": steps},
    }))
    initialization = cache.initialization()
    if resume:
        model = empty_model(initialization["gene_vectors"], cache.identity["genes"], model_config, device,
                            settings["compile_sinkhorn"])
    else:
        model = initial_model(initialization, cache.identity["genes"], model_config, settings["seed"], device,
                              settings["compile_sinkhorn"])
    optimizer = torch.optim.AdamW(model.parameters(), **settings["optimizer"])
    step = 0
    if resume:
        step = load_checkpoint(latest, model, optimizer, expected_metadata=metadata)["step"]
    else:
        write_json(root / "training_manifest.json", metadata)
    pools = (None if config["evaluation"]["pools"] is None else
             EvaluationPools(config["evaluation"]["pools"], device, cache.identity["settings"]["mask_fraction"]))
    order = np.random.default_rng(np.random.SeedSequence([settings["seed"], len(cache)])).permutation(len(cache))
    end = steps if max_steps is None else min(steps, max_steps)

    def evaluate():
        key, results = pools.evaluate(model)
        log_metrics(root, {"event": "evaluation", "step": step, **key, "pools": results})
        logger.info("evaluation step=%d %s", step, " ".join(f"{k}={v:.4f}" for k, v in key.items()))

    if pools is not None and step == 0:
        evaluate()
    while step < end:
        h, x, hidden = cache.batch(order[step * batch_size:(step + 1) * batch_size], device)
        hidden &= model.target_genes
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step, steps, settings["optimizer"]["lr"], settings["warmup_steps"], settings["min_lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        predicted, theta = model(h)
        ce = masked_compositional_loss(predicted, x, hidden)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step % settings["log_every_steps"] == 0 or step == 1:
            record = {"event": "train", "step": step, "lr": lr, **monitor(model, h, x, hidden, predicted, theta, ce)}
            log_metrics(root, record)
            logger.info(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
        if pools is not None and (step % settings["eval_every_steps"] == 0 or step == end):
            evaluate()
        if step % settings["save_every_steps"] == 0 or step == end:
            save_checkpoint(latest, model, optimizer, step=step, metadata=metadata)
    return {"step": step, "output_dir": str(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True, help="Completed training cache directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true", help="Continue from <output-dir>/checkpoints/latest")
    parser.add_argument("--max-steps", type=int, help="Stop early at this step; the schedule stays unchanged")
    args = parser.parse_args()
    config = load_config(args.config)
    print(train(config, args.cache, args.output_dir, resume=args.resume, max_steps=args.max_steps))


if __name__ == "__main__":
    main()

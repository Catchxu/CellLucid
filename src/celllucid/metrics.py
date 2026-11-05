"""Detached diagnostics for cell-topic usage."""

import torch


@torch.no_grad()
def topic_usage_sums(theta):
    """Additive sufficient statistics: topic masses, cell entropies, counts."""
    probabilities = theta.detach().double()
    entropy = torch.special.entr(probabilities).sum(1)
    return torch.cat((
        probabilities.sum(0),
        entropy.sum().reshape(1),
        entropy.exp().sum().reshape(1),
        probabilities.new_tensor([len(probabilities)]),
    ))


@torch.no_grad()
def topic_usage(statistics):
    """Summarize statistics after summing across all cells and DDP ranks."""
    count = statistics[-1]
    mean_usage = statistics[:-3] / count
    aggregate_entropy = torch.special.entr(mean_usage).sum()
    mean_cell_entropy = statistics[-3] / count
    return {
        "usage_cells": int(count),
        "usage_effective_topics": float(aggregate_entropy.exp()),
        "cell_effective_topics_mean": float(statistics[-2] / count),
        "usage_max_probability": float(mean_usage.max()),
        "usage_low_mass_fraction": float(
            (mean_usage < 0.1 / len(mean_usage)).double().mean()
        ),
        "cell_topic_mutual_information": float(aggregate_entropy - mean_cell_entropy),
        "topic_mean_usage": mean_usage.cpu().tolist(),
    }

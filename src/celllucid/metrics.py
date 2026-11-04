"""Detached diagnostics for topic geometry, gene programs and cell usage."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def topic_diversity(topics, dictionary, gene_prior):
    """Compute geometry and gene-program statistics for one model state."""
    unit = F.normalize(topics.detach().double(), dim=1)
    gram = unit @ unit.T
    indices = torch.triu_indices(len(unit), len(unit), 1, device=unit.device)
    pairs = gram[indices[0], indices[1]]
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    spectrum = eigenvalues / eigenvalues.sum()
    beta = dictionary.detach().double()
    mean_beta = beta.mean(0)
    return {
        "topic_cosine_mean": float(pairs.mean()) if pairs.numel() else None,
        "topic_cosine_p95": float(torch.quantile(pairs, 0.95)) if pairs.numel() else None,
        "topic_duplicate_pair_fraction_099": float((pairs > 0.99).double().mean())
        if pairs.numel() else None,
        "topic_effective_dimension": float(torch.special.entr(spectrum).sum().exp()),
        "dictionary_gene_marginal_max_error": float(
            (mean_beta - gene_prior.detach().double()).abs().max()
        ),
        "gene_dictionary_jsd": float(
            torch.special.entr(mean_beta).sum() - torch.special.entr(beta).sum(1).mean()
        ),
    }


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

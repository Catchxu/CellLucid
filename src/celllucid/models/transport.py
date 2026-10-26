"""Entropy-regularized OT with fixed marginals and differentiable log-Sinkhorn."""

import torch


def sinkhorn(cost, topic_prior=None, gene_prior=None, *, epsilon=0.1, iterations=80):
    if cost.ndim != 2 or min(cost.shape) < 1:
        raise ValueError("cost must be a nonempty K by G matrix")
    if not torch.isfinite(cost).all():
        raise ValueError("Nonfinite transport cost")
    if epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon and iterations must be positive")
    dtype = torch.float64 if cost.dtype == torch.float64 else torch.float32
    cost = cost.to(dtype)
    k, g = cost.shape
    a = (
        torch.full((k,), 1 / k, device=cost.device, dtype=dtype)
        if topic_prior is None
        else topic_prior.to(device=cost.device, dtype=dtype)
    )
    b = (
        torch.full((g,), 1 / g, device=cost.device, dtype=dtype)
        if gene_prior is None
        else gene_prior.to(device=cost.device, dtype=dtype)
    )
    for prior, size in ((a, k), (b, g)):
        if prior.shape != (size,) or not torch.isfinite(prior).all() or (prior <= 0).any():
            raise ValueError("OT priors must be finite, strictly positive and correctly shaped")
        if not torch.allclose(prior.sum(), prior.new_tensor(1), atol=1e-6):
            raise ValueError("OT priors must sum to one")
        if prior.requires_grad:
            raise ValueError("Marginals are fixed hyperparameters, not trainable")
    log_kernel = -cost / epsilon
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)
    for _ in range(iterations):
        u = a.log() - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = b.log() - torch.logsumexp(log_kernel + u[:, None], dim=0)
    plan = torch.exp(log_kernel + u[:, None] + v[None, :])
    if not torch.isfinite(plan).all():
        raise FloatingPointError("Nonfinite OT plan")
    return plan


def gene_dictionary(topics, gene_vectors, gene_prior=None, *, epsilon=0.1, iterations=80):
    topics = torch.nn.functional.normalize(topics.float(), dim=-1)
    genes = torch.nn.functional.normalize(gene_vectors.float(), dim=-1)
    if topics.shape[1] != genes.shape[1]:
        raise ValueError("Topic and gene embedding dimensions differ")
    plan = sinkhorn(
        1 - topics @ genes.T, gene_prior=gene_prior, epsilon=epsilon, iterations=iterations
    )
    return plan / plan.sum(dim=1, keepdim=True)

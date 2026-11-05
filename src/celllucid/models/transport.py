"""Entropy-regularized OT with fixed marginals and the topic-gene dictionary built from it."""

import torch


def _iteration(log_kernel, v, log_a, log_b):
    u = log_a - torch.logsumexp(log_kernel + v[None, :], dim=1)
    v = log_b - torch.logsumexp(log_kernel + u[:, None], dim=0)
    return u, v


# Fusing one log-domain iteration avoids materializing the broadcast sums; the unrolled
# loop itself stays in Python so compilation time does not grow with the iteration count.
_compiled_iteration = torch.compile(_iteration)


def sinkhorn(cost, a, b, *, epsilon, iterations, compiled=False):
    """Log-domain Sinkhorn plan with row marginal a and column marginal b, differentiable in cost."""
    step = _compiled_iteration if compiled else _iteration
    log_kernel = -cost / epsilon
    log_a, log_b = a.log(), b.log()
    v = torch.zeros_like(b)
    for _ in range(iterations):
        u, v = step(log_kernel, v, log_a, log_b)
    return torch.exp(log_kernel + u[:, None] + v[None, :])


def topic_gene_dictionary(topics, mapped_genes, gene_prior, *, epsilon, iterations, compiled=False):
    """Row-normalized OT plan between unit topics and unit mapped genes, cost 1 - cosine.

    The topic marginal is uniform and the gene marginal is gene_prior, so the topic-averaged
    dictionary equals gene_prior. The plan is a constrained normalization of the cosines, not
    an objective: reconstruction gradients reach the topics and the gene map through it.
    """
    cost = 1 - topics @ mapped_genes.T
    uniform = torch.full((len(topics),), 1 / len(topics), device=cost.device)
    plan = sinkhorn(cost, uniform, gene_prior, epsilon=epsilon, iterations=iterations, compiled=compiled)
    return plan / plan.sum(dim=1, keepdim=True)

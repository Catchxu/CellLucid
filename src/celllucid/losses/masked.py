"""Masked reconstruction of conditional expression proportions on measured genes."""

import torch


def masked_compositional_loss(probabilities, counts, hidden):
    if probabilities.shape != counts.shape or hidden.shape != counts.shape:
        raise ValueError("Probabilities, counts and masks must have identical N by G shapes")
    if hidden.dtype != torch.bool:
        raise ValueError("hidden must be boolean")
    if not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Targets must be finite nonnegative counts")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Predictions must be finite nonnegative probabilities")
    selected = counts * hidden
    mass = selected.sum(dim=-1)
    valid = (mass > 0) & (hidden.sum(dim=-1) >= 2)
    if not valid.any():
        raise ValueError("No cells have positive hidden expression mass and at least two hidden gene slots")
    p = probabilities[valid].float()
    mask = hidden[valid]
    target = selected[valid] / mass[valid, None]
    log_p = p.clamp_min(torch.finfo(p.dtype).tiny).log()
    normalizer = torch.logsumexp(log_p.masked_fill(~mask, -torch.inf), dim=-1, keepdim=True)
    log_q = (log_p - normalizer).masked_fill(~mask, 0)
    return -(target * log_q).sum(dim=-1).mean()

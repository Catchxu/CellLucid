"""Initial topic model from the cache's study-balanced initialization sample.

Topics start at spherical k-means centers of the centered masked embeddings. The training
mean comes from the masked encodings, the inference mean from the unmasked encodings of the
same cells, and the gene marginal is their mean expression proportion with a small uniform
share so every gene stays strictly positive.
"""

import torch
from torch.nn import functional as F

from .topic import TopicModel


def spherical_kmeans(x, k, iterations, seed):
    """k-means++ seeding and Lloyd updates on unit vectors with cosine similarity."""
    generator = torch.Generator(device=x.device).manual_seed(seed)
    centers = x[torch.randint(len(x), (1,), generator=generator, device=x.device)]
    for _ in range(1, k):
        distance = (1 - x @ centers.T).min(dim=1).values.clamp_min(0)
        centers = torch.cat([centers, x[torch.multinomial(distance, 1, generator=generator)]])
    for _ in range(iterations):
        assignment = (x @ centers.T).argmax(dim=1)
        assert (torch.bincount(assignment, minlength=k) > 0).all()
        centers = F.normalize(torch.zeros_like(centers).index_add_(0, assignment, x), dim=-1)
    return centers


def gene_prior(proportion_mean, uniform_share):
    prior = (1 - uniform_share) * torch.as_tensor(proportion_mean, dtype=torch.float64) + uniform_share / len(proportion_mean)
    return (prior / prior.sum()).float()


def initial_model(initialization, model_config, seed, device, compile_sinkhorn):
    masked = torch.as_tensor(initialization["masked"], device=device)
    unmasked = torch.as_tensor(initialization["unmasked"], device=device)
    train_mean = masked.mean(0)
    topics = spherical_kmeans(F.normalize(masked - train_mean, dim=-1), model_config["n_topics"],
                              model_config["kmeans_iterations"], seed)
    return TopicModel(
        initialization["gene_vectors"], topics, train_mean, unmasked.mean(0),
        gene_prior(initialization["proportion_mean"], model_config["gene_prior_uniform_share"]),
        temperature=model_config["temperature"], ot_epsilon=model_config["ot_epsilon"],
        ot_iterations=model_config["ot_iterations"],
        compile_sinkhorn=compile_sinkhorn,
    ).to(device)


def empty_model(gene_vectors, model_config, device, compile_sinkhorn):
    """A model of the right shapes whose state is filled by load_checkpoint."""
    dimension = gene_vectors.shape[1]
    return TopicModel(
        gene_vectors, torch.ones(model_config["n_topics"], dimension), torch.zeros(dimension), torch.zeros(dimension),
        torch.full((len(gene_vectors),), 1 / len(gene_vectors)),
        temperature=model_config["temperature"], ot_epsilon=model_config["ot_epsilon"],
        ot_iterations=model_config["ot_iterations"],
        compile_sinkhorn=compile_sinkhorn,
    ).to(device)

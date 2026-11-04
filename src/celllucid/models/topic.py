"""Residual cell inference and a shared, OT-clustered gene-program dictionary."""

import torch
from torch import nn
from torch.nn import functional as F
from .transport import gene_dictionary_and_loss


class ResidualCellProjection(nn.Module):
    """A two-layer GELU residual map, initialized as the identity."""

    def __init__(self, dimension, hidden):
        super().__init__()
        self.input = nn.Linear(dimension, hidden)
        self.output = nn.Linear(hidden, dimension)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, cells):
        return cells + self.output(F.gelu(self.input(cells)))


def cell_topic_usage(cells, topics, projection, temperature):
    projected = F.normalize(projection(cells.float()), dim=-1)
    scores = projected @ F.normalize(topics, dim=-1).T
    return torch.softmax(scores / temperature, dim=-1)


class TopicModel(nn.Module):
    def __init__(
        self,
        gene_vectors,
        n_topics=128,
        *,
        temperature=0.1,
        ot_epsilon=0.03,
        ot_iterations=200,
        projection_hidden=512,
        gene_prior=None,
        seed=42,
    ):
        super().__init__()
        vectors = torch.as_tensor(gene_vectors, dtype=torch.float32).detach().clone()
        if vectors.ndim != 2 or not torch.isfinite(vectors).all():
            raise ValueError("Gene vectors must be a finite G by D matrix")
        if (vectors.norm(dim=1) == 0).any():
            raise ValueError("Gene vectors must be nonzero")
        if not 1 <= n_topics <= len(vectors) or temperature <= 0:
            raise ValueError("Invalid topic count or temperature")
        self.register_buffer("gene_vectors", vectors, persistent=False)
        prior = (
            torch.full((len(vectors),), 1 / len(vectors))
            if gene_prior is None
            else torch.as_tensor(gene_prior, dtype=torch.float32).detach().clone()
        )
        if prior.shape != (len(vectors),) or (prior <= 0).any():
            raise ValueError("Invalid gene prior")
        prior = prior / prior.sum()
        self.register_buffer("gene_prior", prior)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(vectors), generator=generator)[:n_topics]
        initial = F.normalize(vectors.cpu()[indices], dim=-1)
        noise = torch.randn(initial.shape, generator=generator) * 0.01
        self.topics = nn.Parameter(initial + noise)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.projection = ResidualCellProjection(vectors.shape[1], projection_hidden)
        self.projection_hidden = int(projection_hidden)
        self.n_topics = n_topics
        self.temperature = float(temperature)
        self.ot_epsilon = float(ot_epsilon)
        self.ot_iterations = int(ot_iterations)

    def theta(self, cells):
        if cells.ndim != 2 or cells.shape[1] != self.topics.shape[1]:
            raise ValueError("Cell embeddings must be N by D")
        if not torch.isfinite(cells).all() or (cells.norm(dim=1) == 0).any():
            raise ValueError("Cell embeddings must be finite and nonzero")
        return cell_topic_usage(cells, self.topics, self.projection, self.temperature)

    def dictionary_with_loss(self):
        return gene_dictionary_and_loss(
            self.topics,
            self.gene_vectors,
            self.gene_prior,
            epsilon=self.ot_epsilon,
            iterations=self.ot_iterations,
        )

    def dictionary(self):
        return self.dictionary_with_loss()[0]

    def forward(self, cells):
        theta = self.theta(cells)
        beta, ot_loss = self.dictionary_with_loss()
        return theta @ beta, theta, beta, ot_loss

    def hyperparameters(self):
        return {
            "n_topics": self.n_topics,
            "temperature": self.temperature,
            "ot_epsilon": self.ot_epsilon,
            "ot_iterations": self.ot_iterations,
            "projection_hidden": self.projection_hidden,
        }

"""Topic model on frozen cell embeddings with an OT topic-gene dictionary.

Cells keep their frozen TranscriptFormer geometry: after subtracting a fixed mean they are
scored against unit topic vectors by cosine and a softmax at the stated temperature. Training
inputs are masked, inference inputs are not, so the model keeps one mean for each. A gene
tower carries the frozen gene vectors into the topic space through an orthogonal linear map
(identity at initialization), which preserves every cosine between genes. Entropic OT with
a uniform topic marginal and an empirical gene marginal turns cos(topic, mapped gene) into
the dictionary, so the topic-averaged dictionary equals the gene marginal. The OT plan only
normalizes the dictionary; training minimizes reconstruction alone.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrizations

from .transport import topic_gene_dictionary


class TopicModel(nn.Module):
    def __init__(self, gene_vectors, topics, train_mean, inference_mean, gene_prior, *, temperature,
                 ot_epsilon, ot_iterations, compile_sinkhorn=False):
        super().__init__()
        vectors = torch.as_tensor(gene_vectors, dtype=torch.float32).detach().clone()
        self.register_buffer("gene_vectors", vectors, persistent=False)
        self.register_buffer("train_mean", torch.as_tensor(train_mean, dtype=torch.float32).clone())
        self.register_buffer("inference_mean", torch.as_tensor(inference_mean, dtype=torch.float32).clone())
        self.register_buffer("gene_prior", torch.as_tensor(gene_prior, dtype=torch.float32).clone())
        self.topics = nn.Parameter(F.normalize(torch.as_tensor(topics, dtype=torch.float32), dim=-1).clone())
        dimension = vectors.shape[1]
        self.gene_map = nn.Linear(dimension, dimension, bias=False)
        nn.init.eye_(self.gene_map.weight)
        parametrizations.orthogonal(self.gene_map)
        self.n_topics = len(self.topics)
        self.temperature = float(temperature)
        self.ot_epsilon = float(ot_epsilon)
        self.ot_iterations = int(ot_iterations)
        self.compile_sinkhorn = bool(compile_sinkhorn)

    def scores(self, cells, inference=False):
        mean = self.inference_mean if inference else self.train_mean
        return F.normalize(cells.float() - mean, dim=-1) @ F.normalize(self.topics, dim=-1).T

    def theta(self, cells, inference=False):
        return torch.softmax(self.scores(cells, inference) / self.temperature, dim=-1)

    def mapped_genes(self):
        return F.normalize(self.gene_map(self.gene_vectors), dim=-1)

    def dictionary(self):
        return topic_gene_dictionary(
            F.normalize(self.topics, dim=-1), self.mapped_genes(), self.gene_prior,
            epsilon=self.ot_epsilon, iterations=self.ot_iterations, compiled=self.compile_sinkhorn,
        )

    def forward(self, cells):
        theta = self.theta(cells)
        return theta @ self.dictionary(), theta

    def hyperparameters(self):
        return {
            "n_topics": self.n_topics,
            "temperature": self.temperature,
            "ot_epsilon": self.ot_epsilon,
            "ot_iterations": self.ot_iterations,
        }

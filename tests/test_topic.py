import math
import pytest
import torch
from torch.nn import functional as F
import json
from celllucid.models.initialization import gene_prior, target_genes
from celllucid.models.topic import TopicModel
from celllucid.losses.masked import masked_compositional_loss
from celllucid.tasks.topic_bank import export_topic_bank, TopicBank


def model(genes=20, dimension=8, topics=4, seed=5):
    generator = torch.Generator().manual_seed(seed)
    prior = torch.rand(genes, generator=generator) + 0.1
    return TopicModel(
        torch.randn(genes, dimension, generator=generator), torch.randn(topics, dimension, generator=generator),
        torch.randn(dimension, generator=generator), torch.randn(dimension, generator=generator), prior / prior.sum(),
        torch.ones(genes, dtype=torch.bool), temperature=0.1, ot_epsilon=0.1, ot_iterations=200,
    )


def test_conditional_loss_and_invalid_targets():
    p = torch.tensor([[0.3, 0.1, 0.6]], requires_grad=True)
    x = torch.tensor([[3.0, 1.0, 900.0]])
    mask = torch.tensor([[True, True, False]])
    loss = masked_compositional_loss(p, x, mask)
    assert float(loss.detach()) == pytest.approx(-0.75 * math.log(0.75) - 0.25 * math.log(0.25))
    loss.backward()
    assert p.grad[0, 2] == 0
    with pytest.raises(ValueError):
        masked_compositional_loss(p, torch.zeros_like(x), mask)


def test_dictionary_marginal_centering_and_bank_roundtrip(tmp_path):
    m = model()
    h = torch.randn(9, 8)
    p, theta = m(h)
    beta = m.dictionary()
    torch.testing.assert_close(p.sum(1), torch.ones(9))
    torch.testing.assert_close(theta.sum(1), torch.ones(9))
    # Uniform topic marginal: the topic-averaged dictionary equals the gene marginal.
    torch.testing.assert_close(beta.mean(0), m.gene_prior, rtol=0, atol=1e-6)
    expected = torch.softmax(F.normalize(h - m.inference_mean, dim=-1) @ F.normalize(m.topics, dim=-1).T / 0.1, -1)
    torch.testing.assert_close(m.theta(h, inference=True), expected)
    assert not torch.allclose(m.theta(h, inference=True), theta)
    path = export_topic_bank(m, [f"g{i}" for i in range(20)], tmp_path / "bank", backbone={"fingerprint": "test"})
    bank = TopicBank(path)
    torch.testing.assert_close(bank.dictionary, beta)
    torch.testing.assert_close(bank.transform(h), m.theta(h, inference=True))
    assert len(bank.top_genes(3)) == 4
    with pytest.raises(FileExistsError):
        export_topic_bank(m, bank.genes, path, backbone={})
    (path / "genes.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        TopicBank(path)


def test_gradient_training_keeps_gene_geometry_and_frozen_inputs():
    m = model()
    before = m.topics.detach().clone()
    raw = F.normalize(m.gene_vectors, dim=-1)
    frozen = {name: getattr(m, name).clone() for name in ("gene_vectors", "gene_prior", "train_mean", "inference_mean")}
    optimizer = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(5):
        optimizer.zero_grad()
        p, _ = m(torch.randn(8, 8))
        masked_compositional_loss(p, torch.randint(1, 4, (8, 20)).float(), torch.ones(8, 20, dtype=torch.bool)).backward()
        optimizer.step()
    assert not torch.equal(before, m.topics)
    mapped = m.mapped_genes()
    assert not torch.allclose(mapped, raw)
    torch.testing.assert_close(mapped @ mapped.T, raw @ raw.T, rtol=0, atol=1e-5)
    for name, value in frozen.items():
        torch.testing.assert_close(getattr(m, name), value, rtol=0, atol=0)


def test_dictionary_passes_gradients_to_topics_and_gene_map():
    m = model()
    (m.dictionary() * torch.arange(20)).sum().backward()
    assert m.topics.grad.norm() > 0
    assert all(p.grad.norm() > 0 for p in m.gene_map.parameters())



def test_excluded_genes_leave_the_loss_and_keep_only_the_uniform_share(tmp_path):
    symbols = tmp_path / "symbols.json"
    symbols.write_text(json.dumps({"MT-CO1": "E1", "RPL3": "E2", "RPS6KA1": "E3", "CD3E": "E4", "RPS4X": "E5"}))
    genes = ["E1", "E2", "E3", "E4", "E5", "E6"]
    config = {"excluded_genes": "^(MT-|RP[LS][0-9]+[AXYL]?[0-9]*$)", "gene_symbols": str(symbols)}
    targets = target_genes(genes, config)
    assert targets.tolist() == [False, False, True, True, False, True]
    assert target_genes(genes, {"excluded_genes": None}).all()
    prior = gene_prior([0.4, 0.3, 0.1, 0.1, 0.05, 0.05], targets, 0.06)
    torch.testing.assert_close(prior[~targets], torch.full((3,), 0.01))
    torch.testing.assert_close(prior[targets], torch.tensor([0.94 * 0.4 + 0.01, 0.94 * 0.4 + 0.01, 0.94 * 0.2 + 0.01]))

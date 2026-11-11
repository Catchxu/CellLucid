import math
import pytest
import torch
from torch.nn import functional as F
import json
from celllucid.models.initialization import gene_prior, target_genes
from celllucid.models.topic import TopicModel
from celllucid.losses.masked import masked_compositional_loss
from celllucid.tasks.publish import promote_topic_bank
from celllucid.tasks.topic_bank import add_gene_scale, export_topic_bank, TopicBank


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
    genes = [f"ENSG{i:011d}" for i in range(20)]
    path = export_topic_bank(m, genes, [f"G{i}" for i in range(20)], tmp_path / "bank", backbone={"fingerprint": "test"})
    bank = TopicBank(path)
    torch.testing.assert_close(bank.dictionary, beta)
    torch.testing.assert_close(bank.transform(h), m.theta(h, inference=True))
    top = bank.top_genes(3)
    assert len(top) == 4 and all(g["gene"] == "G" + str(genes.index(g["gene_id"])) for g in top[0])
    promoted = promote_topic_bank(path, "default", tmp_path / "assets")
    assert promoted == (tmp_path / "assets/topic_banks/default").resolve()
    assert TopicBank(promoted).manifest["bank_id"] == bank.manifest["bank_id"]
    with pytest.raises(FileExistsError):
        export_topic_bank(m, bank.genes, bank.symbols, path, backbone={})
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


def test_gene_space_inference_recovers_usage(tmp_path):
    m = model()
    genes = [f"ENSG{i:011d}" for i in range(20)]
    path = export_topic_bank(m, genes, [f"G{i}" for i in range(20)], tmp_path / "bank", backbone={"fingerprint": "test"})
    bank = TopicBank(path)
    assert bank.targets.all() and bank.gene_scale is None
    usage = torch.tensor([[0.7, 0.2, 0.1, 0.0], [0.25, 0.25, 0.25, 0.25], [0.0, 0.1, 0.3, 0.6]])
    counts = usage @ bank.dictionary * 1e5  # expected counts, so the usage is exactly recoverable
    torch.testing.assert_close(bank.fold_in(counts, iterations=20000, tolerance=0), usage, rtol=0, atol=1e-3)
    with pytest.raises(ValueError, match="gene_scale"):
        bank.project(counts)
    scale = (counts.double() / counts.double().sum(1, keepdim=True) * 1e6).std(0, unbiased=False)
    scaled = TopicBank(add_gene_scale(path, tmp_path / "scaled", scale, {"cells": 3}))
    assert scaled.manifest["gene_scale"] == {"cells": 3} and scaled.manifest["bank_id"] != bank.manifest["bank_id"]
    for name in ("topics", "dictionary", "gene_prior", "inference_mean"):
        torch.testing.assert_close(getattr(scaled, name), getattr(bank, name), rtol=0, atol=0)
    exact = scaled.project(counts, max_value=float("inf"), iterations=20000, tolerance=0)
    torch.testing.assert_close(exact, usage, rtol=0, atol=1e-3)
    clipped = scaled.project(counts * torch.tensor([1.0] * 19 + [50.0]))  # one gene far above its scale
    torch.testing.assert_close(clipped.sum(1), torch.ones(3))
    assert (clipped >= 0).all()
    with pytest.raises(ValueError, match="target genes"):
        bank.fold_in(torch.zeros(1, 20))


def test_target_genes_follow_the_manifest_exclusions(tmp_path):
    m = model()
    genes = [f"ENSG{i:011d}" for i in range(20)]
    symbols = ["MT-CO1", "RPL3"] + [f"G{i}" for i in range(2, 20)]
    training = {"model": {"excluded_genes": "^(MT-|RP[LS][0-9]+)"}}
    bank = TopicBank(export_topic_bank(m, genes, symbols, tmp_path / "bank", backbone={"fingerprint": "test"},
                                       training=training))
    assert bank.targets.tolist() == [False, False] + [True] * 18

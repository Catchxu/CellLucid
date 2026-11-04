import math
import pytest
import torch
from celllucid.models.topic import TopicModel
from celllucid.losses.masked import masked_compositional_loss
from celllucid.tasks.topic_bank import export_topic_bank, TopicBank


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


def test_shared_dictionary_and_bank_roundtrip(tmp_path):
    torch.manual_seed(5)
    model = TopicModel(torch.randn(20, 8), n_topics=4)
    h = torch.randn(9, 8)
    p, theta, beta, ot_loss = model(h)
    torch.testing.assert_close(p.sum(1), torch.ones(9))
    torch.testing.assert_close(theta.sum(1), torch.ones(9))
    torch.testing.assert_close(beta.sum(1), torch.ones(4))
    _, _, second_beta, _ = model(h[:2])
    torch.testing.assert_close(beta, second_beta)
    assert torch.isfinite(ot_loss)
    with torch.no_grad():
        model.projection.output.weight.add_(0.05)
    theta = model.theta(h)
    path = export_topic_bank(
        model, [f"g{i}" for i in range(20)], tmp_path / "bank", backbone={"fingerprint": "test"}
    )
    bank = TopicBank(path)
    torch.testing.assert_close(bank.dictionary, beta)
    torch.testing.assert_close(bank.transform(h), theta)
    assert len(bank.top_genes(3)) == 4
    with pytest.raises(FileExistsError):
        export_topic_bank(model, bank.genes, path, backbone={})
    (path / "genes.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        TopicBank(path)


def test_topics_and_projection_change_during_optimization():
    model = TopicModel(torch.randn(15, 7), n_topics=3)
    before = model.topics.detach().clone()
    genes = model.gene_vectors.clone()
    prior = model.gene_prior.clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(4):
        optimizer.zero_grad()
        p, _, _, ot_loss = model(torch.randn(8, 7))
        loss = masked_compositional_loss(
            p, torch.randint(0, 4, (8, 15)).float(), torch.ones(8, 15, dtype=torch.bool)
        )
        (loss + ot_loss).backward()
        optimizer.step()
    assert not torch.equal(before, model.topics)
    assert model.projection.output.weight.count_nonzero() > 0
    torch.testing.assert_close(genes, model.gene_vectors, rtol=0, atol=0)
    torch.testing.assert_close(prior, model.gene_prior, rtol=0, atol=0)

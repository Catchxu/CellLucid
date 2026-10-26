import torch
import pytest
from celllucid.models.topic import TopicModel
from celllucid.checkpoint import save_checkpoint, load_checkpoint


def test_resume_reproduces_next_optimizer_step(tmp_path):
    torch.manual_seed(2)
    vectors = torch.randn(10, 6)
    a = TopicModel(vectors, n_topics=3)
    optimizer = torch.optim.AdamW([a.topics], lr=0.01)
    h = torch.randn(5, 6)

    def step(model, opt):
        opt.zero_grad()
        p, _, _ = model(h)
        loss = -(p[:, 0].log().mean())
        loss.backward()
        opt.step()

    step(a, optimizer)
    save_checkpoint(tmp_path / "ckpt", a, optimizer, epoch=1, metadata={"version": "test"})
    b = TopicModel(vectors, n_topics=3)
    opt_b = torch.optim.AdamW([b.topics], lr=0.01)
    load_checkpoint(tmp_path / "ckpt", b, opt_b, expected_metadata={"version": "test"})
    step(a, optimizer)
    step(b, opt_b)
    torch.testing.assert_close(a.topics, b.topics, rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(tmp_path / "ckpt", b, expected_metadata={"version": "different"})
    assert list(a.state_dict()) == ["topics", "gene_prior"]

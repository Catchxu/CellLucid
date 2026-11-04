import torch
import pytest
from celllucid.models.topic import TopicModel
from celllucid.checkpoint import save_checkpoint, load_checkpoint


def test_resume_reproduces_next_optimizer_step(tmp_path):
    torch.manual_seed(2)
    vectors = torch.randn(10, 6)
    a = TopicModel(vectors, n_topics=3)
    optimizer = torch.optim.AdamW(a.parameters(), lr=0.01)
    h = torch.randn(5, 6)

    def step(model, opt):
        opt.zero_grad()
        p, _, _, ot_loss = model(h)
        loss = -(p[:, 0].log().mean()) + ot_loss
        loss.backward()
        opt.step()

    step(a, optimizer)
    save_checkpoint(tmp_path / "ckpt", a, optimizer, epoch=1, metadata={"version": "test"})
    b = TopicModel(vectors, n_topics=3)
    opt_b = torch.optim.AdamW(b.parameters(), lr=0.01)
    load_checkpoint(tmp_path / "ckpt", b, opt_b, expected_metadata={"version": "test"})
    step(a, optimizer)
    step(b, opt_b)
    torch.testing.assert_close(a.topics, b.topics, rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(tmp_path / "ckpt", b, expected_metadata={"version": "different"})
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
    assert "projection.output.weight" in a.state_dict()
    assert "gene_vectors" not in a.state_dict()

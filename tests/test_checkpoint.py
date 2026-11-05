import torch
import pytest
from celllucid.models.topic import TopicModel
from celllucid.checkpoint import save_checkpoint, load_checkpoint


def make():
    generator = torch.Generator().manual_seed(2)
    return TopicModel(
        torch.randn(10, 6, generator=generator), torch.randn(3, 6, generator=generator), torch.zeros(6), torch.ones(6),
        torch.full((10,), 0.1), temperature=0.1, ot_epsilon=0.1, ot_iterations=50,
    )


def test_resume_reproduces_next_optimizer_step(tmp_path):
    torch.manual_seed(2)
    h = torch.randn(5, 6)

    def step(model, opt):
        opt.zero_grad()
        p, _ = model(h)
        (-(p[:, 0].log().mean())).backward()
        opt.step()

    a = make()
    optimizer = torch.optim.AdamW(a.parameters(), lr=0.01)
    step(a, optimizer)
    save_checkpoint(tmp_path / "ckpt", a, optimizer, step=1, metadata={"version": "test"})
    b = make()
    opt_b = torch.optim.AdamW(b.parameters(), lr=0.01)
    assert load_checkpoint(tmp_path / "ckpt", b, opt_b, expected_metadata={"version": "test"})["step"] == 1
    step(a, optimizer)
    step(b, opt_b)
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(tmp_path / "ckpt", b, expected_metadata={"version": "different"})
    assert "gene_vectors" not in a.state_dict()

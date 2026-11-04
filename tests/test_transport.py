import pytest
import torch
from celllucid.models.transport import sinkhorn
from celllucid.models.topic import TopicModel


def test_nonuniform_marginals_and_gradients():
    torch.manual_seed(3)
    cost = torch.rand(3, 7, dtype=torch.float64, requires_grad=True)
    a = torch.tensor([0.2, 0.3, 0.5], dtype=cost.dtype)
    b = torch.arange(1, 8, dtype=cost.dtype)
    b = b / b.sum()
    plan = sinkhorn(cost, a, b, epsilon=0.3, iterations=100)
    torch.testing.assert_close(plan.sum(1), a)
    torch.testing.assert_close(plan.sum(0), b)
    (plan.square().sum()).backward()
    assert torch.isfinite(cost.grad).all() and cost.grad.norm() > 0
    assert torch.autograd.gradcheck(
        lambda c: sinkhorn(c, a, b, epsilon=0.3, iterations=30),
        (cost.detach().requires_grad_(True),),
        fast_mode=True,
    )


@pytest.mark.parametrize("epsilon", [0, -1])
def test_bad_regularization(epsilon):
    with pytest.raises(ValueError):
        sinkhorn(torch.ones(2, 3), epsilon=epsilon)


def test_dictionary_branch_updates_topics():
    torch.manual_seed(4)
    model = TopicModel(torch.randn(12, 5), n_topics=3)
    # Isolate the OT branch: a nonconstant function of B must reach topic prototypes.
    beta = model.dictionary()
    (beta * torch.arange(12)).sum().backward()
    assert model.topics.grad is not None and model.topics.grad.norm() > 0
    assert model.projection.output.weight.grad is None
    _, ot_loss = model.dictionary_with_loss()
    model.zero_grad()
    ot_loss.backward()
    assert torch.isfinite(model.topics.grad).all()
    assert model.topics.grad.norm() > 0
    assert model.projection.output.weight.grad is None
    assert "gene_vectors" not in model.state_dict()

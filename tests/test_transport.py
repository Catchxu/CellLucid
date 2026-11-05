import torch
from torch.nn import functional as F
from celllucid.models.transport import sinkhorn, topic_gene_dictionary


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


def test_dictionary_rows_and_gene_marginal():
    torch.manual_seed(4)
    genes = F.normalize(torch.randn(12, 5, dtype=torch.float64), dim=-1)
    prior = torch.rand(12, dtype=torch.float64) + 0.1
    prior = prior / prior.sum()
    topics = F.normalize(torch.randn(3, 5, dtype=torch.float64), dim=-1)
    beta = topic_gene_dictionary(topics, genes, prior, epsilon=0.2, iterations=500)
    torch.testing.assert_close(beta.sum(1), torch.ones(3, dtype=beta.dtype))
    torch.testing.assert_close(beta.mean(0), prior)
    # Shifting one topic's cosines by a constant leaves its row unchanged; only relative cosines matter.
    plan = sinkhorn(1 - topics @ genes.T - torch.tensor([[0.3], [0.0], [0.0]], dtype=prior.dtype),
                    torch.full((3,), 1 / 3, dtype=prior.dtype), prior, epsilon=0.2, iterations=500)
    torch.testing.assert_close(plan / plan.sum(1, keepdim=True), beta)

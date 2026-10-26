import os
import pytest
import numpy as np
import torch
from celllucid.models.backbone import FrozenTranscriptFormer


@pytest.mark.integration
def test_real_backbone_freeze_projection_and_determinism():
    checkpoint = os.environ.get("CELLLUCID_TF_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set CELLLUCID_TF_CHECKPOINT to run the real backbone contract test")
    model = FrozenTranscriptFormer(checkpoint, device="cuda", max_length=127)
    assert all(not p.requires_grad for p in model.parameters())
    model.train(True)
    assert all(not module.training for module in model.modules())
    versions = [p._version for p in model.parameters()]
    genes = model.genes[:64]
    x = np.zeros((1, 64), dtype="f4")
    x[0, :20] = np.arange(1, 21)
    a = model.encode(x, genes)
    b = model.encode(x, genes)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    vectors = model.gene_vectors(genes)
    assert vectors.shape == (64, model.dimension)
    assert not vectors.requires_grad
    assert [p._version for p in model.parameters()] == versions

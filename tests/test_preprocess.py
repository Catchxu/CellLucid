import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from celllucid.data.dataset import CountDataset
from celllucid.data.preprocess import preprocess_h5ad


def write_fixture(path):
    x = np.array([[0, 0, 0, 0], [2, 0, 3, 0], [1, 2, 3, 4], [4, 2, 0, 0]], dtype="f4")
    a = ad.AnnData(
        sparse.csr_matrix(x),
        obs=pd.DataFrame({"donor": ["a", "a", "b", "b"]}, index=["c0", "c1", "c2", "c3"]),
        var=pd.DataFrame(
            {"gene_ids": ["ENSG1.1", "ENSG2.2", "ENSG3", "ENSG4"]},
            index=["one", "two", "three", "four"],
        ),
    )
    a.write_h5ad(path)
    return x


def test_streaming_alignment_missing_genes_and_preprocessing(tmp_path):
    source, out = tmp_path / "source.h5ad", tmp_path / "processed.h5ad"
    x = write_fixture(source)
    with CountDataset(source, genes=["ENSG3", "ENSG1", "ENSGmissing"], gene_key="gene_ids") as ds:
        np.testing.assert_array_equal(ds.available, [True, True, False])
        np.testing.assert_array_equal(
            ds.read([3, 1]), np.column_stack([x[[3, 1], 2], x[[3, 1], 0], [0, 0]])
        )
    stats = preprocess_h5ad(source, out, gene_key="gene_ids", batch_size=2, min_genes=2)
    result = ad.read_h5ad(out)
    assert stats["retained_cells"] == 3
    np.testing.assert_array_equal(result.X.toarray(), x[1:])
    assert result.obs_names.tolist() == ["c1", "c2", "c3"]
    assert result.var_names.tolist() == ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]


def test_count_layer_is_selected_without_eager_other_layers(tmp_path):
    a = ad.AnnData(
        np.log1p(np.ones((3, 4), dtype="f4")), var=pd.DataFrame(index=["g0", "g1", "g2", "g3"])
    )
    a.layers["counts"] = sparse.csr_matrix(np.ones((3, 4), dtype="f4"))
    p = tmp_path / "layers.h5ad"
    a.write_h5ad(p)
    with CountDataset(p) as ds:
        np.testing.assert_array_equal(ds.read([0]), np.ones((1, 4)))
    with CountDataset(p, source="X") as ds:
        with pytest.raises(ValueError, match="integer"):
            ds.read([0])


def test_explicit_symbol_mapping_and_provenance(tmp_path):
    import json

    source = tmp_path / "symbols.h5ad"
    out = tmp_path / "mapped.h5ad"
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"IL7R": "ENSG1", "CCR7": "ENSG2"}))
    ad.AnnData(
        np.array([[2, 4, 1]], dtype="f4"), var=pd.DataFrame(index=["IL7R", "CCR7", "unmapped"])
    ).write_h5ad(source)
    preprocess_h5ad(source, out, gene_mapping=mapping)
    a = ad.read_h5ad(out)
    assert a.var_names.tolist() == ["ENSG1", "ENSG2", "unmapped"]
    assert a.var.gene_symbol.tolist() == ["IL7R", "CCR7", "unmapped"]
    assert "gene_mapping_digest" in a.uns["celllucid"]

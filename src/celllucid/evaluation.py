"""Fixed evaluation pools: cached frozen embeddings, labels and assignment metrics.

Every model is scored on the same cells, mask views and frozen TranscriptFormer embeddings.
The pool directory holds cells.json (study declarations and per-cell records),
embeddings.pt (masked views 0 and 7) and embeddings_unmasked_compiled.pt (unmasked inputs).
Masked views are read with the training mean; unmasked inputs with the inference mean.
"""

import json
import math
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from torch.nn import functional as F

from .data.masking import PROBE_VIEW, STABILITY_VIEW, masked_views
from .losses.masked import masked_compositional_loss
from .metrics import topic_usage, topic_usage_sums

SEED = 42
VIEWS = (PROBE_VIEW, STABILITY_VIEW)
SHUFFLE_REPEATS = 5
SPEARMAN_PAIRS = 200_000


def read_rows(path, rows, fraction):
    rows = np.sort(rows)
    with h5py.File(path, "r") as handle:
        counts = ad.io.sparse_dataset(handle["X"])[rows, :].toarray().astype("f4")
        ids = handle["cell_ids"].asstr()[rows]
        assay = handle["assay"].asstr()[rows]
        available = handle["available"][:]
    views = [masked_views(counts, ids, available, fraction=fraction, seed=SEED, view=v) for v in VIEWS]
    return rows, counts, ids, assay, views


def auc(close, far):
    """Probability that a pair expected to be far is farther than a pair expected to be close."""
    ranks = rankdata(np.concatenate([close, far]))
    return float((ranks[len(close):].sum() - len(far) * (len(far) + 1) / 2) / (len(close) * len(far)))


class Pool:
    def __init__(self, name, studies, cells, embeddings, unmasked, device):
        self.name = name
        self.index = np.array([i for i, c in enumerate(cells) if c["pool"] == name])
        self.cells = [cells[i] for i in self.index]
        self.studies = list(studies[name])
        self.study = np.array([self.studies.index(c["study"]) for c in self.cells])
        self.study_index = [np.flatnonzero(self.study == s) for s in range(len(self.studies))]
        self.raw = embeddings[self.index].to(device)  # cells x views x D
        self.embeddings = F.normalize(self.raw, dim=-1)
        self.unmasked = unmasked[self.index].to(device)
        n = len(self.cells)
        upper = torch.triu_indices(n, n, 1, device=device)
        self.i, self.j = upper[0], upper[1]
        study = torch.as_tensor(self.study, device=device)
        label_names = sorted({c["label"] for c in self.cells if c["label"]})
        cl_names = sorted({c["cl"] for c in self.cells if c["cl"]})
        label = torch.tensor([label_names.index(c["label"]) if c["label"] else -1 for c in self.cells], device=device)
        cl = torch.tensor([cl_names.index(c["cl"]) if c["cl"] else -1 for c in self.cells], device=device)
        si, sj, li, lj, ci, cj = study[self.i], study[self.j], label[self.i], label[self.j], cl[self.i], cl[self.j]
        within = (si == sj) & (li >= 0) & (lj >= 0)
        cross = (si != sj) & (ci >= 0) & (cj >= 0)
        self.pairs = {"within_same": within & (li == lj), "within_diff": within & (li != lj),
                      "cross_same": cross & (ci == cj), "cross_diff": cross & (ci != cj)}
        self.pair_study = si
        self.labeled_studies = [s for s in range(len(self.studies))
                                if bool((self.pairs["within_same"] & (si == s)).any())
                                and bool((self.pairs["within_diff"] & (si == s)).any())]
        self.label, self.cl = label, cl
        generator = torch.Generator().manual_seed(SEED)
        self.sample = torch.randint(len(self.i), (SPEARMAN_PAIRS,), generator=generator).to(device)
        tissues = [studies[name][s]["tissue"] for s in self.studies]
        tissue = torch.as_tensor(np.unique([tissues[s] for s in self.study], return_inverse=True)[1], device=device)
        same_tissue = tissue[self.i] == tissue[self.j]
        # Cross-study type pairs split by tissue; a split is defined only where both groups have pairs.
        self.cross_tissue_pairs = {}
        for split, mask in (("same_tissue", same_tissue), ("different_tissue", ~same_tissue)):
            close, far = self.pairs["cross_same"] & mask, self.pairs["cross_diff"] & mask
            if close.any() and far.any():
                self.cross_tissue_pairs[split] = (close, far)
        self.study_pairs = [(a, b, tissues[a] == tissues[b])
                            for a in range(len(self.studies)) for b in range(a + 1, len(self.studies))]

    def load_reconstruction(self, fraction):
        """Counts and view-0 hidden masks, re-read from the shards in pool order."""
        parts = {}
        for position, cell in enumerate(self.cells):
            parts.setdefault(cell["path"], []).append(position)
        x, hidden = None, None
        for path, positions in parts.items():
            rows = np.array([self.cells[p]["row"] for p in positions])
            sorted_rows, counts, ids, _, views = read_rows(path, rows, fraction)
            if x is None:
                x = np.empty((len(self.cells), counts.shape[1]), dtype="f4")
                hidden = np.empty((len(self.cells), counts.shape[1]), dtype=bool)
            lookup = {int(r): p for r, p in zip(rows, positions, strict=True)}
            targets = [lookup[int(r)] for r in sorted_rows]
            assert all(self.cells[t]["cell_id"] == i for t, i in zip(targets, ids, strict=True))
            x[targets], hidden[targets] = counts, views[0][1]
        self.x = torch.from_numpy(x).to(self.raw.device)
        self.hidden = torch.from_numpy(hidden).to(self.raw.device)

    def target_profile(self):
        """Mean hidden-gene target distribution over the pool's cells."""
        selected = self.x * self.hidden
        return (selected / selected.sum(1, keepdim=True)).mean(0)

    def pair_metrics(self, distance, view_distance):
        """Pair-level structure for one N x N distance matrix and per-cell view distances."""
        values = distance[self.i, self.j]
        groups = {k: values[m].cpu().numpy() for k, m in self.pairs.items()}
        out = {f"median_{k}": float(np.median(v)) for k, v in groups.items() if len(v)}
        within = [auc(values[self.pairs["within_same"] & (self.pair_study == s)].cpu().numpy(),
                      values[self.pairs["within_diff"] & (self.pair_study == s)].cpu().numpy())
                  for s in self.labeled_studies]
        out["auc_within_study"] = float(np.mean(within))
        out["auc_within_study_by_study"] = {self.studies[s]: a for s, a in zip(self.labeled_studies, within, strict=True)}
        out["auc_cross_study"] = auc(groups["cross_same"], groups["cross_diff"])
        for split, (close, far) in self.cross_tissue_pairs.items():
            out[f"auc_cross_{split}"] = auc(values[close].cpu().numpy(), values[far].cpu().numpy())
        view = float(view_distance.median())
        out["median_view"] = view
        out["view_ratio_within"] = view / out["median_within_diff"]
        out["view_ratio_cross"] = view / out["median_cross_diff"]
        out["same_type_cross_ratio"] = out["median_cross_same"] / out["median_cross_diff"]
        reference = 1 - (self.embeddings[:, 0] @ self.embeddings[:, 0].T)
        sample = self.sample
        out["spearman_with_embedding"] = float(spearmanr(
            values[sample].cpu().numpy(), reference[self.i[sample], self.j[sample]].cpu().numpy()).statistic)
        return out

    def usage_metrics(self, rows):
        p = rows[:, 0]
        out = topic_usage(topic_usage_sums(p))
        del out["topic_mean_usage"]
        entropy = lambda q: float(torch.special.entr(q).sum())  # noqa: E731
        means = [p[idx].mean(0) for idx in self.study_index]
        pool_entropy = entropy(p.mean(0))
        within = float(np.mean([entropy(m) for m in means]))
        out["within_study_entropy"] = within
        out["mi_study_topic"] = pool_entropy - within
        out["study_effective_topics"] = {s: math.exp(entropy(m)) for s, m in zip(self.studies, means, strict=True)}
        label_mi = []
        for s in self.labeled_studies:
            idx = torch.as_tensor(self.study_index[s], device=p.device)
            labels = self.label[idx]
            keep = labels >= 0
            q, labels = p[idx][keep], labels[keep]
            parts = [(float((labels == v).double().mean()), entropy(q[labels == v].mean(0))) for v in labels.unique()]
            label_mi.append(entropy(q.mean(0)) - sum(w * e for w, e in parts))
        out["mi_label_topic_within_study"] = float(np.mean(label_mi))
        keep = self.cl >= 0
        q, cl = p[keep], self.cl[keep]
        out["mi_cl_topic"] = entropy(q.mean(0)) - sum(
            float((cl == v).double().mean()) * entropy(q[cl == v].mean(0)) for v in cl.unique())
        tv = [(0.5 * float((means[a] - means[b]).abs().sum()), same) for a, b, same in self.study_pairs]
        out["study_tv_same_tissue"] = float(np.mean([t for t, same in tv if same]))
        out["study_tv_different_tissue"] = float(np.mean([t for t, same in tv if not same]))
        return out

    def reconstruction(self, rows, beta):
        def ce(theta):
            total = 0.0
            for start in range(0, len(theta), 512):
                part = slice(start, start + 512)
                total += float(masked_compositional_loss(
                    theta[part].float() @ beta, self.x[part], self.hidden[part])) * len(theta[part])
            return total / len(theta)
        p = rows[:, 0]
        raw = ce(p)
        shuffled = []
        for repeat in range(SHUFFLE_REPEATS):
            generator = torch.Generator().manual_seed(42 + repeat)
            perm = torch.arange(len(p))
            for idx in self.study_index:
                idx = torch.as_tensor(idx)
                perm[idx] = idx[torch.randperm(len(idx), generator=generator)]
            shuffled.append(ce(p[perm.to(p.device)]))
        return {"ce": raw, "shuffled_minus_raw_ce": float(np.mean(shuffled) - raw)}

    def background_ce(self, profile):
        """Masked CE when every cell predicts the same gene profile (no cell information)."""
        total = 0.0
        for start in range(0, len(self.cells), 512):
            part = slice(start, start + 512)
            n = len(self.x[part])
            total += float(masked_compositional_loss(
                profile.expand(n, -1).contiguous(), self.x[part], self.hidden[part])) * n
        return total / len(self.cells)

    def evaluate(self, rows, beta):
        distance = 0.5 * torch.cdist(rows[:, 0].float(), rows[:, 0].float(), p=1)
        view_distance = 0.5 * (rows[:, 0] - rows[:, 1]).abs().sum(-1)
        return {**self.usage_metrics(rows), **self.reconstruction(rows, beta),
                **self.pair_metrics(distance, view_distance)}


class EvaluationPools:
    def __init__(self, directory, device, mask_fraction):
        directory = Path(directory)
        selection = json.loads((directory / "cells.json").read_text())
        embeddings = torch.load(directory / "embeddings.pt", weights_only=True)
        unmasked = torch.load(directory / "embeddings_unmasked_compiled.pt", weights_only=True)
        self.pools = {name: Pool(name, selection["studies"], selection["cells"], embeddings, unmasked, device)
                      for name in selection["studies"]}
        for pool in self.pools.values():
            pool.load_reconstruction(mask_fraction)

    @torch.no_grad()
    def evaluate(self, model):
        """Full metrics per pool, and the key metrics used for monitoring."""
        beta = model.dictionary()
        results, key = {}, {}
        for name, pool in self.pools.items():
            rows = torch.stack([model.theta(pool.raw[:, v]) for v in range(pool.raw.shape[1])], 1).double()
            metrics = pool.evaluate(rows, beta)
            metrics["ce_gain_over_background"] = pool.background_ce(model.gene_prior) - metrics["ce"]
            inference = model.theta(pool.unmasked, inference=True).double()
            distance = 0.5 * torch.cdist(inference.float(), inference.float(), p=1)
            shift = 0.5 * (inference - rows[:, 0]).abs().sum(-1)
            pairs = pool.pair_metrics(distance, shift)
            metrics["inference"] = {"auc_within_study": pairs["auc_within_study"],
                                    "auc_cross_study": pairs["auc_cross_study"],
                                    "median_tv_to_masked": pairs["median_view"]}
            results[name] = metrics
            key[f"{name}_auc_within"] = metrics["auc_within_study"]
            key[f"{name}_auc_cross"] = metrics["auc_cross_study"]
            key[f"{name}_cell_mi"] = metrics["cell_topic_mutual_information"]
        key["heldout_ce_gain"] = results["heldout"]["ce_gain_over_background"]
        return key, results

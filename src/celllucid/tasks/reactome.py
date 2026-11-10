"""Map each topic to one Reactome pathway.

A topic is represented by its 100 highest-weight genes (ties broken by gene symbol). Every
human pathway except disease pathways (Reactome's isInDisease flag, or the top level "Disease")
is a candidate, and candidates are tested in an order that puts each pathway before all of its
ancestors. For each topic, a candidate's genes minus its delete list are compared with the
topic's genes by a hypergeometric test; if p < 0.01, the candidate's genes are added to the
delete list of all its ancestors. The topic's mapping is the candidate with the smallest p
(ties: the earlier candidate). Genes are counted within the universe of input genes that belong
to at least one human Reactome pathway.

The mapping itself needs only NumPy, SciPy and pandas, so it also runs on gene programs from
other models (a topics x genes weight matrix with gene symbols).
"""

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, hypergeom

TOP_GENES = 100
P_CUT = 0.01


class Pathways:
    """Human Reactome pathways from the table written by data/reactome.py."""

    def __init__(self, path):
        self.info = {r["pathway_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
        self.genes = {p: set(r["genes"].split()) for p, r in self.info.items()}
        self.all_genes = set().union(*self.genes.values())
        parents = {p: [x for x in r["parent_ids"].split("; ") if x] for p, r in self.info.items()}
        children = {p: [x for x in r["child_ids"].split("; ") if x] for p, r in self.info.items()}

        self.ancestors = {}

        def ancestors(p):
            if p not in self.ancestors:
                found = set()
                for q in parents[p]:
                    found |= {q} | ancestors(q)
                self.ancestors[p] = found
            return self.ancestors[p]

        order, seen = [], set()

        def visit(p):
            if p in seen:
                return
            seen.add(p)
            for c in children[p]:
                visit(c)
            order.append(p)  # after all of its descendants

        for p in self.info:
            ancestors(p)
            visit(p)
        disease = {
            p for p, r in self.info.items() if r["is_in_disease"] == "True" or r["top_level"] == "Disease"
        }
        self.candidates = [p for p in order if p not in disease]


def map_topics(weights, symbols, pathways):
    """Map every row of a topics x genes weight matrix to a pathway.

    Returns two tables. `mapping` has one row per topic: the chosen pathway, its p, the overlap
    with the topic's genes and the pathway size after deletion (pathway columns are empty when
    no candidate shares a gene). `tests` has one row per (topic, candidate) pair with at least one
    shared gene: p after deletion and the Benjamini-Hochberg FDR over all candidates of that
    topic; pairs not listed share no gene (p = 1, FDR = 1).
    """
    weights = np.asarray(weights)
    symbols = np.asarray(symbols, dtype=str)
    universe = set(symbols) & pathways.all_genes
    size = len(universe)
    candidates = pathways.candidates
    gene_sets = {p: pathways.genes[p] & universe for p in candidates}
    mapping, tests = [], []
    for topic, w in enumerate(weights):
        top = np.lexsort((symbols, -w))[:TOP_GENES]
        drawn = set(symbols[top]) & universe
        zero = set(symbols[top][w[top] == 0])
        delete = defaultdict(set)
        result = {}
        for p in candidates:
            remaining = gene_sets[p] - delete[p]
            hit = drawn & remaining
            pv = float(hypergeom.sf(len(hit) - 1, size, len(remaining), len(drawn))) if hit else 1.0
            result[p] = (pv, hit, len(remaining))
            if pv < P_CUT:
                for a in pathways.ancestors[p]:
                    delete[a] |= gene_sets[p]

        fdr = false_discovery_control([result[p][0] for p in candidates], method="bh")
        for p, q in zip(candidates, fdr, strict=True):
            pv, hit, n_used = result[p]
            if hit:
                info = pathways.info[p]
                tests.append({"topic": topic, "pathway_id": p, "pathway_name": info["pathway_name"],
                              "depth": int(info["depth"]), "p": pv, "fdr": float(q),
                              "overlap": len(hit), "pathway_size_used": n_used})

        best = min(candidates, key=lambda p: result[p][0])
        pv, hit, n_used = result[best]
        row = {"topic": topic}
        if hit:
            info = pathways.info[best]
            row.update({"pathway_id": best, "pathway_name": info["pathway_name"],
                        "depth": int(info["depth"]), "top_level": info["top_level"], "p": pv,
                        "overlap": len(hit), "pathway_size_used": n_used,
                        "overlap_genes": " ".join(sorted(hit))})
        row.update({"n_significant": sum(result[p][0] < P_CUT for p in candidates),
                    "n_top_in_reactome": len(drawn), "n_zero_weight_in_top100": len(zero),
                    "n_zero_weight_in_overlap": len(hit & zero)})
        mapping.append(row)
    columns = ["topic", "pathway_id", "pathway_name", "depth", "top_level", "p", "overlap",
               "pathway_size_used", "overlap_genes", "n_significant", "n_top_in_reactome",
               "n_zero_weight_in_top100", "n_zero_weight_in_overlap"]
    return pd.DataFrame(mapping, columns=columns), pd.DataFrame(tests)


def main():
    parser = argparse.ArgumentParser(description="Map the topics of a topic bank to Reactome pathways")
    parser.add_argument("--topic-bank", default="assets/topic_banks/default")
    parser.add_argument("--pathways", default="assets/reactome/human_pathways.csv")
    parser.add_argument("--output", required=True, help="New folder for the mapping tables")
    args = parser.parse_args()
    from ..config import file_digest, write_json
    from .topic_bank import TopicBank

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    bank = TopicBank(args.topic_bank)
    mapping, tests = map_topics(bank.dictionary.numpy(), bank.symbols, Pathways(args.pathways))
    output.mkdir(parents=True)
    mapping.to_csv(output / "topic_to_reactome.csv", index=False)
    tests.to_csv(output / "topic_pathway_tests.csv", index=False)
    write_json(output / "manifest.json", {
        "bank_id": bank.manifest["bank_id"],
        "pathways_digest": file_digest(args.pathways),
        "top_genes": TOP_GENES,
        "p_cut": P_CUT,
    })
    print(json.dumps({"path": str(output), "topics": len(mapping),
                      "mapped_p_below_cut": int((mapping["p"] < P_CUT).sum())}))


if __name__ == "__main__":
    main()

"""
Step 2: map every topic in scvote_k100.sqlite to one Reactome biological pathway (bp), following the method as specified:

  1. For each topic, take the 100 genes with the highest weight in topic_gene.
  2. One candidate bp list for all topics: every human Reactome bp except disease bps, ordered so that each
     bp comes before all of its parents (and their parents, up to the top level).
     Disease bp = flagged isInDisease in Reactome, or under the top-level "Disease" (790 bps).
  3. For each topic, go through the candidate list in order. For each bp:
       - remove the genes in its delete gene list (genes of its significant descendants)
       - count how many of the remaining genes are among the topic's top 100 genes, and compute the
         probability that a random draw of the same number of genes overlaps at least that much (hypergeometric)
       - if p < 0.01, add this bp's genes to the delete gene list of all of its ancestors
     Every topic starts with empty delete gene lists.
  4. After all candidate bps are checked, the bp with the smallest p is the topic's mapping.

Choices the method does not fix:
  - genes counted in the probability: genes of the topic's own dataset that are in at least one human
    Reactome bp (the universe); a bp's size and the topic's top genes are both counted within it
  - ties in step 1 (equal weights, e.g. weight 0): broken by gene symbol
  - ties in step 4 (equal smallest p): the bp that comes first in the candidate list
  - if no bp shares any gene with the topic (all p = 1), the topic gets no mapping

Inputs:
  output/reactome_human_pathways.csv   from step 1
  input/nodes_pathway.csv              isInDisease flag of each bp (from step 0)
  scvote_k100.sqlite                   the topic store (table topic_gene); 2.8 GB, not in this folder
Run with a Python that has scipy:
  python step2_map_topics.py [path/to/scvote_k100.sqlite]
  (default path: ../data/scvote_k100/scvote_k100.sqlite)
Outputs (output/):
  candidate_bps.csv          the ordered candidate list
  topic_top100_genes.csv     the top 100 genes of every topic
  topic_to_reactome.csv      one row per topic: the chosen bp
  topic_bp_results.csv       every (topic, bp) pair with at least one shared gene: p after elim, FDR, overlap.
                             FDR = Benjamini-Hochberg over all candidate bps of that topic; pairs not listed
                             have no shared gene (p = 1, FDR = 1). Used for the dot plot.
"""
import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import false_discovery_control, hypergeom

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
REACTOME = OUT / "reactome_human_pathways.csv"
NODES = HERE / "input" / "nodes_pathway.csv"      # has the isInDisease flag
DB = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "scvote_k100" / "scvote_k100.sqlite"
TOP_N = 100
P_CUT = 0.01
sys.setrecursionlimit(10000)

# ---------------- Reactome: gene sets and hierarchy ----------------
R = {r["pathway_id"]: r for r in csv.DictReader(open(REACTOME, encoding="utf-8"))}
genes_of = {p: set(r["genes"].split()) for p, r in R.items()}
parents = {p: [x for x in r["parent_ids"].split("; ") if x] for p, r in R.items()}
children = {p: [x for x in r["child_ids"].split("; ") if x] for p, r in R.items()}
reactome_genes = set().union(*genes_of.values())
in_disease = {r["pathway_id"]: r["is_in_disease"] == "True" for r in csv.DictReader(open(NODES, encoding="utf-8"))}
is_disease = {p: in_disease[p] or R[p]["top_level"] == "Disease" for p in R}

ancestors = {}
def get_ancestors(p):
    if p not in ancestors:
        a = set()
        for q in parents[p]:
            a.add(q)
            a |= get_ancestors(q)
        ancestors[p] = a
    return ancestors[p]
for p in R:
    get_ancestors(p)

# ---------------- step 2: one ordered candidate list ----------------
order, seen = [], set()
def visit(p):
    if p in seen:
        return
    seen.add(p)
    for c in children[p]:
        visit(c)
    order.append(p)               # added only after all of its descendants
for p in R:
    visit(p)
candidates = [p for p in order if not is_disease[p]]
pos = {p: i for i, p in enumerate(candidates)}
assert all(pos[p] < pos[a] for p in candidates for a in ancestors[p] if a in pos)

with open(OUT / "candidate_bps.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["processing_order", "pathway_id", "pathway_name", "depth", "top_level", "n_genes", "parent_names"])
    for i, p in enumerate(candidates, 1):
        w.writerow([i, p, R[p]["pathway_name"], R[p]["depth"], R[p]["top_level"], len(genes_of[p]), R[p]["parent_names"]])

# ---------------- topics ----------------
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
datasets = [d for (d,) in con.execute("select distinct dataset_name from topic_gene order by 1")]

gene_rows, rows, pair_rows = [], [], []
for ds in datasets:
    universe = {g for (g,) in con.execute("select distinct gene_symbol from topic_gene where dataset_name=?", (ds,))} & reactome_genes
    U = len(universe)
    gset = {p: genes_of[p] & universe for p in candidates}
    topics = [t for (t,) in con.execute("select distinct topic_index from topic_gene where dataset_name=? order by 1", (ds,))]

    for t in topics:
        # ---- step 1: top 100 genes ----
        top = con.execute("select gene_symbol, weight from topic_gene where dataset_name=? and topic_index=? "
                          "order by weight desc, gene_symbol limit ?", (ds, t, TOP_N)).fetchall()
        for rank, (g, wt) in enumerate(top, 1):
            gene_rows.append([ds, t, rank, g, f"{wt:.6g}", int(g in universe)])
        S = {g for g, _ in top} & universe
        n = len(S)
        zero = {g for g, wt in top if wt == 0}

        # ---- step 3: go through the candidate list ----
        delete = defaultdict(set)        # bp -> genes of its significant descendants
        result = {}
        for p in candidates:
            remaining = gset[p] - delete[p]
            hit = S & remaining
            k, K = len(hit), len(remaining)
            pv = hypergeom.sf(k - 1, U, K, n) if k > 0 else 1.0
            result[p] = (pv, hit, K)
            if pv < P_CUT:
                for a in ancestors[p]:
                    delete[a] |= gset[p]

        # ---- FDR over all candidate bps of this topic (for the dot plot) ----
        fdr = dict(zip(candidates, false_discovery_control([result[p][0] for p in candidates], method="bh")))
        for p in candidates:
            if result[p][1]:
                pair_rows.append([ds, t, p, R[p]["pathway_name"], R[p]["depth"], f"{result[p][0]:.6g}",
                                  f"{fdr[p]:.6g}", len(result[p][1]), result[p][2]])

        # ---- step 4: smallest p (ties: earlier in the candidate list) ----
        best = min(candidates, key=lambda p: (result[p][0], pos[p]))
        pv, hit, K = result[best]
        row = {"dataset_name": ds, "topic_index": t, "n_top_in_reactome": n, "n_zero_weight_in_top100": len(zero),
               "n_significant": sum(1 for p in candidates if result[p][0] < P_CUT)}
        if pv < 1.0:
            row.update({"pathway_id": best, "pathway_name": R[best]["pathway_name"], "depth": R[best]["depth"],
                        "top_level": R[best]["top_level"], "p": pv, "overlap": len(hit), "pathway_size_used": K,
                        "overlap_genes": " ".join(sorted(hit)),
                        "n_zero_weight_in_overlap": len(hit & zero)})
        rows.append(row)
    print(f"{ds}: done (universe {U:,} genes)")

with open(OUT / "topic_top100_genes.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset_name", "topic_index", "rank", "gene_symbol", "weight", "in_reactome"])
    w.writerows(gene_rows)

with open(OUT / "topic_bp_results.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset_name", "topic_index", "pathway_id", "pathway_name", "depth", "p", "fdr", "overlap",
                "pathway_size_used"])
    w.writerows(pair_rows)

cols = ["dataset_name", "topic_index", "pathway_id", "pathway_name", "depth", "top_level", "p", "overlap",
        "pathway_size_used", "overlap_genes", "n_significant", "n_top_in_reactome",
        "n_zero_weight_in_top100", "n_zero_weight_in_overlap"]
with open(OUT / "topic_to_reactome.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"saved {len(candidates)} candidate bps, {len(gene_rows)} gene rows, {len(rows)} topics, {len(pair_rows)} topic-bp pairs")

"""
Step 1: merge three Reactome download files (release 97) into one table of human bps.

Inputs (input/):
  ReactomePathways.txt           pathway ID, name, species (all species)
  ReactomePathwaysRelation.txt   parent ID -> child ID (all species)
  ReactomePathways.gmt           human gene sets: name, ID, gene symbols
Output (output/):
  reactome_human_pathways.csv    one row per human bp (2,883): pathway_id, pathway_name, top_level, depth,
                                 parent_ids, parent_names, child_ids, n_children, n_genes, genes
                                 depth 0 = top level; a bp with two parents gets the smaller depth
Run: python step1_build_reactome_table.py
"""
import collections
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN = HERE / "input"
OUT = HERE / "output"

names = {}
for l in open(IN / "ReactomePathways.txt", encoding="utf-8"):
    i, n, s = l.rstrip("\n").split("\t")
    if s == "Homo sapiens":
        names[i] = n
parents = collections.defaultdict(list)
children = collections.defaultdict(list)
for l in open(IN / "ReactomePathwaysRelation.txt", encoding="utf-8"):
    p, c = l.split()
    if p in names and c in names:
        parents[c].append(p)
        children[p].append(c)
genes = {}
for l in open(IN / "ReactomePathways.gmt", encoding="utf-8"):
    f = l.rstrip("\n").split("\t")
    genes[f[1]] = sorted(set(f[2:]))

def tops(i, seen=None):
    seen = seen or set()
    if not parents[i]:
        return {i}
    out = set()
    for p in parents[i]:
        if p not in seen:
            out |= tops(p, seen | {i})
    return out

def depth(i, seen=None):
    seen = seen or set()
    if not parents[i]:
        return 0
    return 1 + min(depth(p, seen | {i}) for p in parents[i] if p not in seen)

rows = []
for i, n in names.items():
    t = sorted(tops(i))
    rows.append({"pathway_id": i, "pathway_name": n, "top_level": "; ".join(names[x] for x in t), "depth": depth(i),
                 "parent_ids": "; ".join(sorted(parents[i])),
                 "parent_names": "; ".join(names[p] for p in sorted(parents[i])),
                 "child_ids": "; ".join(sorted(children[i])), "n_children": len(children[i]),
                 "n_genes": len(genes.get(i, [])), "genes": " ".join(genes.get(i, []))})
rows.sort(key=lambda r: (r["top_level"], r["depth"], r["pathway_name"]))

OUT.mkdir(exist_ok=True)
with open(OUT / "reactome_human_pathways.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"saved {len(rows)} human bps")

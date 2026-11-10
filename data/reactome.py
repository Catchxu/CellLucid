#!/usr/bin/env python3
"""Build the table of human Reactome pathways that topic mapping reads.

Inputs, all from one Reactome release, in --input-dir:
  ReactomePathways.txt           pathway ID, name, species (all species)
  ReactomePathwaysRelation.txt   parent ID, child ID (all species)
  ReactomePathways.gmt           human gene sets: name, ID, gene symbols
  nodes_pathway.csv              human pathways exported from the Reactome graph database,
                                 with Reactome's isInDisease flag (see docs/data.md)
Output: one row per human pathway with pathway_id, pathway_name, top_level, depth (0 = top
level; a pathway with two parents gets the smaller depth), parent_ids, parent_names, child_ids,
n_children, n_genes, genes (space-separated symbols) and is_in_disease, plus
<output>.provenance.json with the release and the digests of the inputs.
"""

import argparse
import collections
import csv
from pathlib import Path
from celllucid.config import file_digest, write_json

SOURCES = ("ReactomePathways.txt", "ReactomePathwaysRelation.txt", "ReactomePathways.gmt",
           "nodes_pathway.csv")
FLAGS = {"True": True, "False": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--release", type=int, required=True, help="Reactome release of the inputs")
    parser.add_argument("--output", default="assets/reactome/human_pathways.csv")
    args = parser.parse_args()
    source = Path(args.input_dir)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    names = {}
    for line in open(source / "ReactomePathways.txt", encoding="utf-8"):
        pathway, name, species = line.rstrip("\n").split("\t")
        if species == "Homo sapiens":
            names[pathway] = name
    parents = collections.defaultdict(list)
    children = collections.defaultdict(list)
    for line in open(source / "ReactomePathwaysRelation.txt", encoding="utf-8"):
        parent, child = line.split()
        if parent in names and child in names:
            parents[child].append(parent)
            children[parent].append(child)
    genes = {}
    for line in open(source / "ReactomePathways.gmt", encoding="utf-8"):
        fields = line.rstrip("\n").split("\t")
        genes[fields[1]] = sorted(set(fields[2:]))
    in_disease = {
        row["pathway_id"]: FLAGS[row["is_in_disease"]]
        for row in csv.DictReader(open(source / "nodes_pathway.csv", encoding="utf-8"))
    }

    def tops(pathway, seen=frozenset()):
        if not parents[pathway]:
            return {pathway}
        out = set()
        for parent in parents[pathway]:
            if parent not in seen:
                out |= tops(parent, seen | {pathway})
        return out

    def depth(pathway, seen=frozenset()):
        if not parents[pathway]:
            return 0
        return 1 + min(depth(p, seen | {pathway}) for p in parents[pathway] if p not in seen)

    rows = []
    for pathway, name in names.items():
        rows.append({
            "pathway_id": pathway,
            "pathway_name": name,
            "top_level": "; ".join(names[x] for x in sorted(tops(pathway))),
            "depth": depth(pathway),
            "parent_ids": "; ".join(sorted(parents[pathway])),
            "parent_names": "; ".join(names[p] for p in sorted(parents[pathway])),
            "child_ids": "; ".join(sorted(children[pathway])),
            "n_children": len(children[pathway]),
            "n_genes": len(genes.get(pathway, [])),
            "genes": " ".join(genes.get(pathway, [])),
            "is_in_disease": in_disease[pathway],
        })
    rows.sort(key=lambda r: (r["top_level"], r["depth"], r["pathway_name"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(str(output) + ".provenance.json", {
        "reactome_release": args.release,
        "sources": {name: file_digest(source / name) for name in SOURCES},
        "pathways": len(rows),
    })
    print(f"Wrote {len(rows)} human pathways to {output}")


if __name__ == "__main__":
    main()

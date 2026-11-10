"""
Step 3: compare the step-2 mapping with the topic_pathway table already in the topic store.

topic_pathway lists, for each topic, the bps found by an earlier method, each with a p value (topgo_pval).
For each topic, sort its topic_pathway rows by p and find where the bp chosen in step 2 is:
  rank1                 same p as the smallest p of that topic (ties count as rank 1)
  2-5                   among the first 5 rows, but not rank1
  >5                    in the table, after the first 5 rows
  absent                not in that topic's rows
  topic not in table    the topic has no rows in topic_pathway
Top 5 = rank1 + 2-5.

Inputs:  output/topic_to_reactome.csv (step 2), scvote_k100.sqlite (table topic_pathway)
Output:  output/compare_topic_pathway.csv, one row per topic with its category
Run: python step3_compare_topic_pathway.py [path/to/scvote_k100.sqlite]
"""
import csv
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
DB = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "scvote_k100" / "scvote_k100.sqlite"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
tp = defaultdict(list)
for ds, t, pid, p in con.execute("select dataset_name, topic_index, pathway_id, topgo_pval from topic_pathway"):
    tp[(ds, str(t))].append((p, pid))

def category(k, pid):
    rows = sorted(tp.get(k, []))
    if not rows:
        return "topic not in table"
    p_of = {x[1]: x[0] for x in rows}
    if pid not in p_of:
        return "absent"
    if p_of[pid] == rows[0][0]:
        return "rank1"
    return "2-5" if [x[1] for x in rows].index(pid) < 5 else ">5"

out = []
for r in csv.DictReader(open(OUT / "topic_to_reactome.csv", encoding="utf-8")):
    out.append([r["dataset_name"], r["topic_index"], r["pathway_id"], r["pathway_name"],
                category((r["dataset_name"], r["topic_index"]), r["pathway_id"])])

with open(OUT / "compare_topic_pathway.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset_name", "topic_index", "pathway_id", "pathway_name", "category"])
    w.writerows(out)
c = Counter(x[-1] for x in out)
for k in ["rank1", "2-5", ">5", "absent", "topic not in table"]:
    print(f"{k:20s} {c[k]}")
print(f"{'top 5':20s} {c['rank1'] + c['2-5']}")

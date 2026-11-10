# Topic → Reactome bp mapping

给 topic store（`scvote_k100.sqlite`，5 个数据集 × 100 个 topic）里的每个 topic 找一个 Reactome 生物通路（bp）。

## 运行

```bash
python step1_build_reactome_table.py
python step2_map_topics.py   [path/to/scvote_k100.sqlite]
python step3_compare_topic_pathway.py   [path/to/scvote_k100.sqlite]
```

- 需要 Python 3 和 scipy（≥ 1.11）。
- topic store 有 2.8 GB，不放在这个文件夹里。不给路径时，默认读 `../data/scvote_k100/scvote_k100.sqlite`。
- step0 不需要重跑，它的输出已经放在 `input/` 里。

## 文件说明（按流程顺序）

### Step 0：取 bp 的疾病标记（已完成）

| 文件 | 作用 / 内容 |
|---|---|
| `step0_export_reactome_csv.py` | 从 Reactome 的 Neo4j 图数据库导出人类部分的 CSV（共 11 个）。需要本地运行 Neo4j，在 Windows 上跑过一次。topic mapping 只用其中的 `nodes_pathway.csv`。 |
| `input/nodes_pathway.csv` | 每个人类 bp 一行：pathway_id、name、schema_class、is_in_disease、has_diagram。用到的列是 `is_in_disease`，即这个 bp 是否被 Reactome 标记为疾病 bp。 |

### Step 1：整理 Reactome 表

| 文件 | 作用 / 内容 |
|---|---|
| `input/ReactomePathways.txt` | Reactome 官方下载（release 97，2026-10-11）。每行一个 bp：ID、名字、物种（包含所有物种）。 |
| `input/ReactomePathwaysRelation.txt` | 同上。每行一条上下级关系：上级 bp ID、下级 bp ID。 |
| `input/ReactomePathways.gmt` | 同上。每行一个人类 bp 的基因集：名字、ID、基因名。 |
| `step1_build_reactome_table.py` | 把上面三个文件合并成一张人类 bp 表。 |
| `output/reactome_human_pathways.csv` | 2,883 个人类 bp，每行一个：pathway_id、pathway_name、top_level（所属最顶层大类）、depth（第几层，顶层为 0）、parent_ids / parent_names（直接上级）、child_ids（直接下级）、n_children、n_genes、genes（空格分隔的基因名）。 |

### Step 2：topic mapping

| 文件 | 作用 / 内容 |
|---|---|
| `scvote_k100.sqlite`（不在文件夹内） | topic store。只用 `topic_gene` 表：dataset_name、topic_index、gene_symbol、weight（这个 topic 在这个基因上的权重）。 |
| `step2_map_topics.py` | mapping 的全部代码，步骤见下方。 |
| `output/candidate_bps.csv` | 候选 bp 列表：去掉 790 个疾病 bp 后剩下的 2,093 个，已排好序，每个 bp 都在它所有上级之前。列：processing_order（检查顺序）、pathway_id、pathway_name、depth、top_level、n_genes、parent_names。 |
| `output/topic_top100_genes.csv` | 每个 topic 权重最高的 100 个基因，共 50,000 行。列：dataset_name、topic_index、rank、gene_symbol、weight、in_reactome（这个基因是否参与计算，1 = 是）。 |
| `output/topic_to_reactome.csv` | **最终结果**，每个 topic 一行，共 500 行，其中 497 个 p < 0.01。列：dataset_name、topic_index、pathway_id、pathway_name、depth、top_level、p、overlap（重合基因数）、pathway_size_used（bp 删减后的基因数）、overlap_genes（重合的基因）、n_significant（这个 topic 有几个 bp 的 p < 0.01）、n_top_in_reactome（top 100 里参与计算的基因数）、n_zero_weight_in_top100、n_zero_weight_in_overlap（权重为 0 的基因数）。 |
| `output/topic_bp_results.csv` | 每个 topic 和每个至少重合 1 个基因的 bp 的计算结果，共 177,289 行，供画图用。列：dataset_name、topic_index、pathway_id、pathway_name、depth、p、fdr、overlap、pathway_size_used。 |

`step2_map_topics.py` 的步骤：

1. 去掉疾病 bp：`is_in_disease` 为 True 的，或者属于 Disease 大类的。剩下的 bp 排序，保证每个 bp 都排在它所有上级之前。
2. 每个数据集确定"全部基因"：这个数据集里出现过、并且至少属于一个 Reactome bp 的基因。算概率时只在这个范围内计数。
3. 每个 topic 取权重最高的 100 个基因。
4. 每个 topic 的 delete gene list 从空开始，按候选列表的顺序逐个检查 bp：
   - 从 bp 的基因中去掉它的 delete gene list；
   - 数剩下的基因和 top 100 基因重合几个，用超几何检验算出 p，即随机抽同样多的基因、至少重合这么多的概率；
   - 如果 p < 0.01，把这个 bp 的基因加进它所有上级（一直到顶层）的 delete gene list。
5. 选 p 最小的 bp 作为这个 topic 的 mapping。另外对每个 topic 的 2,093 个 p 做 BH 校正得到 FDR，只用于画图，不影响 mapping。

### Step 3：与 topic store 里已有的 topic_pathway 表比较

| 文件 | 作用 / 内容 |
|---|---|
| `step3_compare_topic_pathway.py` | topic store 的 `topic_pathway` 表里，每个 topic 有一组用另一种方法得到的 bp 和 p 值（topgo_pval）。脚本把每个 topic 的这组 bp 按 p 排序，看 step 2 选出的 bp 排在第几名，并打印各类别的计数。 |
| `output/compare_topic_pathway.csv` | 每个 topic 一行：dataset_name、topic_index、pathway_id、pathway_name、category。category 有五种：rank1（第一名，p 并列也算）、2-5、>5、absent（不在这个 topic 的表里）、topic not in table（topic_pathway 里没有这个 topic）。 |

结果：rank1 有 186 个，2-5 有 157 个，排在前五名的一共 343 个；>5 有 82 个，absent 有 66 个，topic not in table 有 9 个。

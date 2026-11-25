# MetaLead extraction test set (31 papers)

This folder holds the data, the scorer, and our system outputs for the result extraction
experiments. The task: read a machine learning paper and list every reported result as a
record (method, task, dataset, metric, value).

## What the dataset is

MetaLead is a public, hand-annotated dataset by Timmer, Bölücü, and Wan (EACL 2026,
https://aclanthology.org/2026.eacl-long.196.pdf, code and annotations at
https://github.com/RoelTim/metalead). For each paper, the MetaLead annotators listed the
results the paper reports.

We use the 31 MetaLead papers that have LaTeX source on arXiv. The annotation file covers
43 papers. The other 12 have no arXiv LaTeX source.

| Item | Count |
| --- | --- |
| Papers | 31 |
| Reference results | 2,613 |
| Reference results per paper | 12 to 249, median 61 |
| Tasks | 21 (the largest are summarization, named entity recognition, intent detection and slot filling, machine translation) |
| Results marked "proposed method" | 606 |
| Results marked "variation of proposed method" | 593 |
| Results marked "baseline" | 1,414 |

The papers are NLP papers from 2015 to 2022.

## What is in this folder

| Path | Content |
| --- | --- |
| `data/papers_latex/<arxiv id>/` | LaTeX source of each paper (the input) |
| `data/metalead_annotations/metalead_annotations.json` | The MetaLead reference results, unchanged |
| `data/metalead_annotations/key_corrections.yaml` | One spelling fix applied at scoring time (ROGUE to ROUGE, paper 1912.08777) |
| `splits/metalead_31_papers.json` | The 31 paper ids |
| `scripts/score.py` | The scorer |
| `scripts/entity_match_*.py`, `scripts/dataset_task_normalizer.py`, `config/` | The matching rules the scorer uses |
| `our_outputs/` | Extracted records from five systems, one JSON file per paper |

No file here contains labels made by us. The only human labels are the public MetaLead
annotations.

## Format of a reference result

Each paper in `metalead_annotations.json` is keyed as `<arxiv id>.pdf` and has a list `TDMs`:

```json
{"Task": "Summarization", "Train-Dataset": "Gigaword", "Test-Dataset": "Gigaword",
 "Metric": "ROUGE-2", "Result": 14.51, "Experiment-Type": "variation of proposed method",
 "Notes": "Table 3 - row a"}
```

Only 213 of the 2,613 reference results name the method.

## Format the scorer expects from a system

One JSON file per paper, named `<arxiv id>.json`, in one folder:

```json
{"paper_id": "1511.08308",
 "tuples": [
   {"method": {"name": "BLSTM-CNN"},
    "task": {"name": "Named Entity Recognition"},
    "datasets": [{"name": "CoNLL-2003"}],
    "metrics": [{"name": "F1", "value": 91.62}]}
 ]}
```

Other fields are ignored by the scorer. A paper with no file counts as zero records.

## How to score

Python 3.10 or later, with `pyyaml` and `scipy`. Run from this folder:

```bash
python scripts/score.py --pred our_outputs/ours_deepseek_inside
python scripts/score.py --pred our_outputs/gpt_alone --prefix metalead_
python scripts/score.py --pred my_run --baseline our_outputs/gpt_alone --baseline-prefix metalead_
python scripts/score.py --pred my_run --require-value --per-paper
```

`--baseline` adds a paired bootstrap over papers (10,000 resamples) for the F1 difference.

### What counts as a match

The scorer pairs each extracted record with at most one reference result, and each
reference result with at most one record. By default a pair is allowed when:

- the dataset names match after normalization (`config/dataset_task_aliases.yaml`);
- the metric names match (exact, alias, or a short metric name from
  `config/short_metric_whitelist.yaml` inside a longer one);
- the method names match, for the reference results that name a method.

The default rule does not compare the numeric value. `--require-value` adds that check:
the extracted value must equal the reference value to two decimals (0.9121 and 91.21 count
as equal).

## Our systems and their scores

"Ours" is table-aware extraction: one pass over the paper collects its vocabulary (methods,
datasets, metrics), then a model reads each results table three times, and a record is kept
when at least two readings agree. "Alone" is one call to the model with the whole paper and
a few examples.

Default matching (the numbers reported so far):

| Folder | System | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| `ours_claude_inside` | Ours, Claude Opus 5.5 inside | 0.617 | 0.462 | 0.528 |
| `ours_deepseek_inside` | Ours, DeepSeek V4 Pro inside | 0.623 | 0.454 | 0.525 |
| `ours_gpt_inside` | Ours, GPT-6.1 Terra Pro inside | 0.407 | 0.416 | 0.411 |
| `gpt_alone` | GPT-6.1 Terra Pro alone | 0.447 | 0.356 | 0.396 |
| `claude_alone` | Claude Opus 5.5 alone | 0.568 | 0.272 | 0.368 |

With `--require-value`:

| Folder | Precision | Recall | F1 |
| --- | --- | --- | --- |
| `ours_claude_inside` | 0.531 | 0.398 | 0.455 |
| `ours_deepseek_inside` | 0.531 | 0.387 | 0.448 |
| `ours_gpt_inside` | 0.338 | 0.345 | 0.342 |
| `gpt_alone` | 0.438 | 0.349 | 0.389 |
| `claude_alone` | 0.533 | 0.255 | 0.345 |

The files in `gpt_alone` and `claude_alone` are named `metalead_<arxiv id>.json`, so they
need `--prefix metalead_`. `claude_alone` has no records for one paper (1812.09471).

Records in `our_outputs/ours_*` carry more fields than the scorer reads: the quoted table
cell, the agreement between readings (`confidence`), split, metric variant, evaluation
setting, and origin of the result.

## Graph-of-graphs extraction (`gog/`)

An extractor built to test whether similar chunks of different papers help each other.
The top graph links papers by embedding similarity and shared names (kNN graph, Louvain
groups). Each group then gets a chunk graph over its papers' tables and sections. Each
chunk is extracted with its closest chunks from other papers shown as reference only. A
record is kept only when its value is printed in its own chunk. Results that several
papers report raise each other's confidence. Settings are in `config/gog.yaml`; the
pipeline never reads the annotations or the scorer's configs.

```bash
python -m gog.run --condition A3 --model <openai model> --out runs/x --dry-run   # cost estimate, no calls
python -m gog.run --condition A4 --model <ollama tag> --base-url http://localhost:11434/v1 --out runs/y
python scripts/score.py --pred runs/y --require-value --baseline runs/y_A1
python scripts/score_extra.py --pred runs/y --require-value   # macro scores, ET-Accuracy, per group
```

Conditions: A1 per paper only; A2 + group vocabulary; A3 + neighbor chunks; A4 + corroboration
and recall probe; C3 = A3 with random groups and neighbors (control). Responses are cached
in `runs/_cache`, so re-running a condition is free. For Ollama, set `OLLAMA_CONTEXT_LENGTH=8192`
or higher, since the default context is shorter than the prompts. Prices for the cost estimate
and the budget guard (`budget_usd`) are set under `llm` in `config/gog.yaml`.

The scorer reads its files as UTF-8. Before this was explicit, it scored about 6 F1 points
low on Windows machines with a cp1252 locale.

## Notes on use

The LaTeX sources come from arXiv and each paper keeps its own license. Please keep this
folder among the co-authors.

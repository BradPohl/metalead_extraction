"""Chunk-level result extraction.

One call per candidate chunk. The prompt holds the extraction guidelines (a paraphrase of
the MetaLead annotation guidelines), optionally the group vocabulary, optionally the
neighbor chunks from other papers (reference only), and the target chunk.

Grounding filter: a record is kept only when its value occurs as a number in the target
chunk (as printed, or scaled by 100 / 0.01). This drops values copied from neighbors.
"""
from __future__ import annotations
import re

from gog.chunk import Chunk, numbers_in
from gog.config import approx_tokens
from gog.llm import LLM, parse_json
from gog.vocab import FIELDS

EXPERIMENT_TYPES = ('proposed method', 'variation of proposed method', 'baseline')

SYSTEM = """You extract experimental results from machine learning papers written in LaTeX. Reply with JSON only.

Extract every result reported in the TARGET excerpt: one record per (method, dataset, metric, value).
- Include the proposed method, its variants and ablations, and baselines or prior work that appear in the excerpt.
- Only evaluation scores of models (e.g. F1, accuracy, BLEU, ROUGE, perplexity, precision, recall, exact match).
  Skip dataset statistics, hyperparameters, model sizes, speeds, training times, and human evaluation scores.
- Copy the value exactly as printed (e.g. 91.21, not 0.9121). Drop "%" and "±" parts: "91.2 ± 0.1" -> 91.2.
- Name the test dataset with its split when it is not the test set, e.g. "CoNLL-2003 (dev)". Give the
  language pair for translation, e.g. "WMT'14 EN-DE". The train dataset is usually the same as the test dataset.
- "metric" is the evaluation measure itself (e.g. "F1", "Accuracy", "BLEU", "ROUGE-L", "Perplexity"), not a
  table column header. Spell abbreviations out ("Acc" -> "Accuracy"). When a header names a sub-task and a
  measure, e.g. "Retrieval (R@10)", the metric is the measure ("Recall@10") and the sub-task goes in "task".
- If a table cell combines metrics (e.g. "R-1/R-2/R-L"), emit one record per metric.
- experiment_type is one of:
  "proposed method": the authors' main method (the final configuration used for the main test results);
  "variation of proposed method": ablations, other configurations, or ensembles of the authors' method;
  "baseline": prior work and comparison systems (including the authors' re-implementations of them).
- Never extract from REFERENCE excerpts; they are from other papers and only help you understand names.
- If the target excerpt reports no results, return {"results": []}.

Reply as {"results": [{"method": str, "task": str, "train_dataset": str, "test_dataset": str,
"metric": str, "value": number, "experiment_type": str, "quote": str}]}
where quote is the exact table cell or phrase that contains the value."""


def _block(title: str, text: str) -> str:
    return f'### {title}\n{text.strip()}\n'


def vocab_block(group_vocab: dict, max_tokens: int) -> str:
    lines = []
    for f in FIELDS:
        names = group_vocab.get(f, [])
        if names:
            lines.append(f'{f.capitalize()}: ' + '; '.join(names))
    text = '\n'.join(lines)
    while lines and approx_tokens(text) > max_tokens:
        text = text[: int(len(text) * 0.9)]
    return text


def build_messages(target: Chunk, title: str, group_vocab: dict | None, neighbors: list[tuple[Chunk, str]],
                   vocab_tokens: int, neighbor_tokens: int, hint: str = '') -> list[dict]:
    parts = []
    if group_vocab:
        parts.append(_block('Names used by this paper and related papers', vocab_block(group_vocab, vocab_tokens)))
    for i, (c, ptitle) in enumerate(neighbors, 1):
        text = c.text[: int(neighbor_tokens * 3.5)]
        where = c.caption or c.section_path
        parts.append(_block(f'REFERENCE excerpt {i} (other paper: "{ptitle[:80]}"; {where[:120]}) - do not extract', text))
    where = f'table "{target.caption}"' if target.kind == 'table' else f'section "{target.section_path}"'
    head = f'TARGET excerpt (paper: "{title[:120]}"; {where[:200]}' + ('; appendix' if target.in_appendix else '') + ')'
    parts.append(_block(head, target.text))
    if hint:
        parts.append(_block('Hint', hint))
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': '\n'.join(parts)}]


def to_float(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r'-?\d+(?:\.\d+)?', str(v or '').replace(',', ''))
    return float(m.group(0)) if m else None


def grounded(value: float, numbers: set[float]) -> bool:
    return any(abs(value * s - n) < 1e-6 for n in numbers for s in (1, 100, 0.01))


def _s(x) -> str:
    return str(x).strip() if isinstance(x, (str, int, float)) and str(x).strip() else ''


def parse_records(text: str, chunk: Chunk) -> tuple[list[dict], dict]:
    """Model reply -> scorer-format records, plus counts of what was dropped and why."""
    data = parse_json(text)
    rows = data.get('results') if isinstance(data, dict) else data if isinstance(data, list) else None
    stats = {'parsed': 0, 'ungrounded': 0, 'invalid': 0, 'parse_error': int(rows is None)}
    nums = numbers_in(chunk.text)
    out = []
    for r in rows or []:
        if not isinstance(r, dict):
            stats['invalid'] += 1; continue
        v = to_float(r.get('value'))
        test_ds, metric = _s(r.get('test_dataset')) or _s(r.get('dataset')), _s(r.get('metric'))
        if v is None or not test_ds or not metric:
            stats['invalid'] += 1; continue
        stats['parsed'] += 1
        if not grounded(v, nums):
            stats['ungrounded'] += 1; continue
        et = _s(r.get('experiment_type')).lower()
        out.append({
            'method': {'name': _s(r.get('method'))},
            'task': {'name': _s(r.get('task'))},
            'datasets': [{'name': test_ds}],
            'train_dataset': _s(r.get('train_dataset')) or test_ds,
            'metrics': [{'name': metric, 'value': v}],
            'experiment_type': et if et in EXPERIMENT_TYPES else '',
            'source': {'chunk_id': chunk.chunk_id, 'kind': chunk.kind, 'quote': _s(r.get('quote'))[:200]},
        })
    return out, stats


def extract_chunk(llm: LLM, target: Chunk, title: str, group_vocab: dict | None,
                  neighbors: list[tuple[Chunk, str]], cfg: dict, hint: str = '') -> tuple[list[dict], dict]:
    msgs = build_messages(target, title, group_vocab, neighbors, cfg['extract']['vocab_tokens'],
                          cfg['chunk_graph']['neighbor_tokens'], hint)
    reply = llm.chat(msgs, max_tokens=cfg['extract']['max_output_tokens'])
    return parse_records(reply, target)

"""One LLM call per paper: the tasks, datasets, metrics and methods the paper evaluates.

Input is the title, abstract, section headings, and each table's caption and first rows,
capped at about 3k tokens. The result feeds the paper-graph Jaccard term, the chunk
mentions, and the group vocabulary shown to the extractor.
"""
from __future__ import annotations
import re
from collections import Counter

from gog.chunk import Chunk
from gog.config import approx_tokens
from gog.latex import Paper
from gog.llm import LLM, parse_json
from gog.names import name_key

FIELDS = ('tasks', 'datasets', 'metrics', 'methods')

SYSTEM = ('You read excerpts of a machine learning paper and list the evaluation vocabulary it uses. '
          'Reply with JSON only.')

PROMPT = """List the names this paper uses for:
- "tasks": tasks evaluated (e.g. named entity recognition, summarization)
- "datasets": evaluation datasets, with the language pair or version when stated (e.g. WMT'14 EN-DE, CoNLL-2003)
- "metrics": evaluation metrics (e.g. F1, BLEU, ROUGE-L, accuracy)
- "methods": the proposed method(s) and the baseline methods compared in result tables

Use the surface forms the paper uses. At most 25 items per list.
Reply as {{"tasks": [...], "datasets": [...], "metrics": [...], "methods": [...]}}.

{excerpt}"""


def vocab_excerpt(paper: Paper, chunks: list[Chunk], max_tokens: int = 3000) -> str:
    heads = sorted({c.section_path for c in chunks if c.section_path})
    parts = [f'Title: {paper.title}', f'Abstract: {paper.abstract}', 'Sections: ' + '; '.join(heads)]
    for c in chunks:
        if c.kind != 'table':
            continue
        rows = [r.strip() for r in re.split(r'\\\\', c.text) if r.strip()][:6]
        parts.append(f'Table "{c.caption}":\n' + '\n'.join(rows)[:900])
    out, used = [], 0
    for p in parts:
        t = approx_tokens(p)
        if used + t > max_tokens:
            break
        out.append(p); used += t
    return '\n\n'.join(out)


def paper_vocab(llm: LLM, paper: Paper, chunks: list[Chunk]) -> dict[str, list[str]]:
    msg = [{'role': 'system', 'content': SYSTEM},
           {'role': 'user', 'content': PROMPT.format(excerpt=vocab_excerpt(paper, chunks))}]
    data = parse_json(llm.chat(msg, max_tokens=1500)) or {}
    out = {}
    for f in FIELDS:
        vals = data.get(f) if isinstance(data, dict) else None
        out[f] = [str(v).strip() for v in (vals or []) if isinstance(v, (str, int, float)) and str(v).strip()][:25]
    return out


def merge_vocab(vocabs: list[dict]) -> dict[str, list[str]]:
    """Group vocabulary: names ordered by how many papers in the group use them."""
    out = {}
    for f in FIELDS:
        count, surface = Counter(), {}
        for v in vocabs:
            for k, name in {name_key(x): x for x in v.get(f, [])}.items():
                if k:
                    count[k] += 1
                    surface.setdefault(k, name)
        out[f] = [surface[k] for k, _ in count.most_common()]
    return out

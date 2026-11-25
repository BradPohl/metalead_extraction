"""Name keys for comparing tasks, datasets, metrics and methods across papers.

Deliberately simple and independent of the scorer's normalizer: lowercase, unify
dashes/apostrophes, drop filler words, sort tokens. 'WMT'14 EN-DE' and 'EN-DE WMT14'
get the same key; 'CoNLL-2003 (dev)' keeps its split.
"""
from __future__ import annotations
import re
import unicodedata

_FILLER = {'the', 'dataset', 'datasets', 'corpus', 'benchmark', 'task', 'set', 'data', 'score', 'of', 'on'}


def name_key(s) -> str:
    s = unicodedata.normalize('NFKC', str(s or '')).lower()
    s = s.replace('’', "'").replace('‘', "'").replace('–', '-').replace('—', '-')
    s = re.sub(r"(\w)'(\d\d)\b", r'\1\2', s)            # wmt'14 -> wmt14
    s = re.sub(r'\\[a-z]+', ' ', s)                     # leftover latex commands
    toks = [t for t in re.split(r'[^a-z0-9.+]+', s) if t and t not in _FILLER]
    return ' '.join(sorted(toks))


def key_set(names) -> set[str]:
    return {k for k in (name_key(n) for n in names or []) if k}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def text_mentions(text: str, names: list[str]) -> set[str]:
    """Keys of the names in `names` that occur in `text` (case-insensitive, punctuation-insensitive)."""
    flat = re.sub(r'[^a-z0-9]+', '', text.lower())
    out = set()
    for n in names:
        probe = re.sub(r'[^a-z0-9]+', '', str(n).lower())
        if len(probe) >= 3 and probe in flat:
            out.add(name_key(n))
    return out

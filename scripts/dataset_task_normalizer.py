r"""v6c dataset/task name normalizer.

Idempotent, deterministic, documented. See EXECUTOR_PROMPT_V6C_NORMALIZER.md §1.

Pipeline (applied in order; the same normalization is applied to alias-map keys):

  1. Unicode normalize (NFKC); replace typographic apostrophes/dashes/arrows.
  2. Strip LaTeX leftovers (\textsc{...}, \emph{...}, $...$, ~, \&, \%).
  3. Lowercase.
  4. Apostrophe removal (wmt'14 -> wmt14).
  5. Strip gold's split/qualifier suffixes (' - english', ' dev', '(test)', ...).
  6. Expand long language names and arrow-direction phrases
     (english-french -> en-fr; english to french -> en-fr;
      en<->de -> en-de with bidirectional=True).
  7. Punctuation and whitespace collapse (remove .,;:()[]{}/\\\"; squeeze -).
  8. Extract language-direction token (en-de) into meta.
  9. Extract year (4-digit or 2-digit after letters) into meta.
 10. Duplicate-token collapse (wmt wmt14 -> wmt14).
 11. Alias expansion (fix-point) from config/dataset_task_aliases.yaml.
 12. Final canonical = `{head}+{lang_dir}` when lang_dir is present.
"""
from __future__ import annotations
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import yaml

CFG = Path(__file__).resolve().parent.parent / 'config' / 'dataset_task_aliases.yaml'


class CanonMeta(NamedTuple):
    canonical: str
    year_hint: str        # '', '14', '03', '2003'
    lang_dir: str         # '', 'en-de', 'de-en'
    bidirectional: bool
    raw: str


_LANG_CODES = ['en', 'de', 'fr', 'zh', 'ro', 'cs', 'ru', 'ja', 'ko', 'es', 'it',
               'pt', 'nl', 'fi', 'tr', 'ar', 'hi', 'vi']

_LANG_LONG = {
    'english': 'en', 'german': 'de', 'french': 'fr', 'chinese': 'zh',
    'romanian': 'ro', 'czech': 'cs', 'russian': 'ru', 'japanese': 'ja',
    'korean': 'ko', 'spanish': 'es', 'italian': 'it', 'portuguese': 'pt',
    'dutch': 'nl', 'finnish': 'fi', 'turkish': 'tr', 'arabic': 'ar',
    'hindi': 'hi', 'vietnamese': 'vi',
}


def _step1_unicode(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    for a, b in [('\u2019', "'"), ('\u2018', "'"), ('\u02bc', "'"),
                 ('\u2013', '-'), ('\u2014', '-'), ('\u2212', '-'),
                 ('\u2011', '-'), ('\u2192', ' to '), ('\u21d2', ' to '),
                 ('\u2194', ' <-> '), ('\u21d4', ' <-> ')]:
        s = s.replace(a, b)
    return s


def _step2_latex(s: str) -> str:
    s = re.sub(r'\\textsc\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\emph\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\textbf\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\texttt\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\$([^$]*)\$', r'\1', s)
    return (s.replace('~', ' ').replace('\\&', '&')
             .replace('\\%', '%').replace('\\textregistered', ''))


def _step3_lower(s: str) -> str:
    return s.lower()


def _step4_apostrophe(s: str) -> str:
    return s.replace("'", '')


def _step5_strip_qualifiers(s: str) -> str:
    """Strip gold's trailing qualifier markers."""
    s = re.sub(r'\s*-\s*english(\s+dev|\s+test)?\s*$', '', s)
    s = re.sub(r'\s*\((test|dev|val|validation)\)\s*$', '', s)
    s = re.sub(r'\s+(dev|test|val|validation|training)\s*$', '', s)
    # WMT-specific split markers: " newstest2014", " newstest 2014"
    s = re.sub(r'\s+newstest\s*(19|20)?\d{2}\s*$', '', s)
    s = re.sub(r'\s+newstest\s*(19|20)?\d{2}\b', ' ', s)
    return s.strip()


def _step6_expand_lang(s: str) -> str:
    """Convert long-form language names and arrow-direction phrases to
    hyphenated ISO code pairs."""
    # "english to french" / "english-french" / "english - french"
    long_re = '|'.join(sorted(_LANG_LONG.keys(), key=len, reverse=True))
    s = re.sub(
        rf'\b({long_re})\s*(?:to|-|<->)\s*({long_re})\b',
        lambda m: f'{_LANG_LONG[m.group(1)]}-{_LANG_LONG[m.group(2)]}',
        s,
    )
    # Singleton long language names at phrase end (e.g. "wmt 2014 french") -> lang code
    s = re.sub(rf'\b({long_re})\b',
               lambda m: _LANG_LONG.get(m.group(1), m.group(1)), s)
    # ISO pairs "en to de" / "en -> de" / "en <-> de"
    code_re = '|'.join(_LANG_CODES)
    s = re.sub(rf'\b({code_re})\s*(?:to|->|<->)\s*({code_re})\b',
               lambda m: f'{m.group(1)}-{m.group(2)}', s)
    return s


def _step7_collapse_punct(s: str) -> str:
    s = re.sub(r"[.,;:()\[\]\{\}/\\\"&!?]+", ' ', s)
    s = re.sub(r'-+', '-', s)
    s = re.sub(r'\s+', ' ', s).strip(' -')
    return s


def _step8_extract_lang_dir(s: str) -> tuple[str, str, bool]:
    code_re = '|'.join(_LANG_CODES)
    bi = re.search(rf'\b({code_re})\s*<->\s*({code_re})\b', s)
    if bi:
        a, b = bi.group(1), bi.group(2)
        stripped = (s[:bi.start()] + s[bi.end():]).strip(' -')
        return stripped, f'{a}-{b}', True
    m = re.search(rf'\b({code_re})[\s-]+({code_re})\b', s)
    if m:
        a, b = m.group(1), m.group(2)
        stripped = (s[:m.start()] + s[m.end():]).strip(' -')
        return stripped, f'{a}-{b}', False
    return s, '', False


def _step9_extract_year(s: str) -> tuple[str, str]:
    m4 = re.search(r'\b(19|20)(\d{2})\b', s)
    if m4:
        year = m4.group(2)  # 2-digit
        stripped = (s[:m4.start()] + ' ' + s[m4.end():]).strip(' -')
        stripped = re.sub(r'\s+', ' ', stripped)
        return stripped, year
    # 2-digit directly attached to letters (wmt14)
    m2 = re.search(r'([a-z])(\d{2})\b', s)
    if m2:
        year = m2.group(2)
        pre = s[:m2.start(2)]  # keep the letter
        post = s[m2.end(2):]
        stripped = (pre + post).strip(' -')
        return stripped, year
    # 2-digit after space
    m2b = re.search(r'\b([a-z]+)\s+(\d{2})\b', s)
    if m2b:
        year = m2b.group(2)
        stripped = (s[:m2b.start()] + m2b.group(1) + s[m2b.end():]).strip(' -')
        return stripped, year
    return s, ''


def _step10_dedupe(s: str) -> str:
    toks = s.split()
    if len(toks) < 2:
        return s
    out = [toks[0]]
    for t in toks[1:]:
        if t == out[-1]:
            continue
        if t.startswith(out[-1]) and len(t) > len(out[-1]):
            out[-1] = t
        elif out[-1].startswith(t):
            continue
        else:
            out.append(t)
    return ' '.join(out)


def _prepipeline(s: str) -> tuple[str, str, str, bool]:
    """Run steps 1-10. Returns (core_head, year_hint, lang_dir, bidirectional)."""
    s = _step1_unicode(s)
    s = _step2_latex(s)
    s = _step3_lower(s)
    s = _step4_apostrophe(s)
    s = _step5_strip_qualifiers(s)
    s = _step6_expand_lang(s)
    s = _step7_collapse_punct(s)
    s, lang, bi = _step8_extract_lang_dir(s)
    s, year = _step9_extract_year(s)
    s = _step10_dedupe(s)
    s = re.sub(r'\s+', ' ', s).strip(' -')
    return s, year, lang, bi


def _load_tables():
    d = yaml.safe_load(CFG.read_text(encoding='utf-8'))
    out = {}
    for kind in ('datasets', 'tasks'):
        table = d.get(kind, {}) or {}
        alias_map = {}
        canon_set = set()
        for canon, aliases in table.items():
            canon_l = canon.lower().strip()
            canon_set.add(canon_l)
            for a in (aliases or []):
                a_key, _, _, _ = _prepipeline(str(a))
                alias_map[a_key] = canon_l
            # Also map the canonical head's prepipeline form to itself
            canon_key, _, _, _ = _prepipeline(canon)
            alias_map[canon_key] = canon_l
        # Fix-point: resolve chains
        changed = True; iters = 0
        while changed and iters < 10:
            changed = False; iters += 1
            for a, c in list(alias_map.items()):
                if c in alias_map and alias_map[c] != c:
                    alias_map[a] = alias_map[c]; changed = True
        out[kind] = (alias_map, canon_set)
    return out


_TABLES = _load_tables()


def _alias_expand(s: str, kind: str) -> str:
    if not s:
        return s
    # kind is 'dataset' or 'task'; _TABLES keys are 'datasets' / 'tasks'
    plural = 'datasets' if kind == 'dataset' else 'tasks'
    amap, canon_set = _TABLES[plural]
    cur = s
    prev = None; iters = 0
    while cur != prev and iters < 6:
        prev = cur; iters += 1
        if cur in amap:
            cur = amap[cur]; continue
        # Try with spaces removed
        no_sp = re.sub(r'\s+', '', cur)
        if no_sp in amap:
            cur = amap[no_sp]; continue
        # Try splitting at the first space: longest-prefix-wins
        parts = cur.split()
        matched = False
        for cut in range(len(parts), 0, -1):
            prefix = ' '.join(parts[:cut])
            if prefix in amap:
                cur = amap[prefix] + (' ' + ' '.join(parts[cut:]) if cut < len(parts) else '')
                matched = True; break
        if not matched:
            break
    return cur


@lru_cache(maxsize=8192)
def canonicalize_with_meta(name: str, kind: str) -> CanonMeta:
    if kind not in ('dataset', 'task'):
        raise ValueError(kind)
    raw = name or ''
    core, year, lang, bi = _prepipeline(raw)
    head = _alias_expand(core, kind) if core else ''
    # Re-dedupe after alias expansion (handles e.g. "switchboard dialog act corpus swda"
    # -> alias-expand to "swda swda" -> dedupe to "swda").
    head = _step10_dedupe(head)
    head = re.sub(r'\s+', ' ', head).strip(' -')
    canon = head + ('+' + lang if lang else '')
    return CanonMeta(canonical=canon, year_hint=year, lang_dir=lang,
                     bidirectional=bi, raw=raw)


def canonicalize(name: str, kind: str) -> str:
    return canonicalize_with_meta(name, kind).canonical


if __name__ == '__main__':
    tests = [
        ("WMT'14 EN-FR", 'dataset'),
        ("WMT'14 EN-DE", 'dataset'),
        ("WMT14 En-Fr", 'dataset'),
        ("WMT 2014 English-French", 'dataset'),
        ("WMT'14 En-Fr newstest2014", 'dataset'),
        ("WMT'14 EN-FR", 'dataset'),
        ("CNN/DailyMail", 'dataset'),
        ("CNNDM", 'dataset'),
        ("CNN-DailyMail", 'dataset'),
        ("CoNLL-2003 - English", 'dataset'),
        ("CoNLL-2003", 'dataset'),
        ("CoNLL03 NER", 'dataset'),
        ("CoNLL-03", 'dataset'),
        ("CoNLL++", 'dataset'),
        ("IWSLT'14 DE-EN", 'dataset'),
        ("IWSLT'15 DE-EN", 'dataset'),
        ("Switchboard Dialog Act Corpus (SWDA)", 'dataset'),
        ("SwDA", 'dataset'),
        ("Ontonotes v5 - English", 'dataset'),
        ("OntoNotes 5.0", 'dataset'),
        ("WNUT-17 - English", 'dataset'),
        ("WNUT-17", 'dataset'),
        ("Machine Translation", 'task'),
        ("Neural Machine Translation", 'task'),
        ("NMT", 'task'),
        ("Summarization", 'task'),
        ("Abstractive Summarization", 'task'),
        ("single-document summarization", 'task'),
        ("Named Entity Recognition (NER)", 'task'),
        ("named entity recognition", 'task'),
        ("NER", 'task'),
        ("Intent Detection and Slot Filling", 'task'),
        ("Slot filling", 'task'),
        ("Intent Detection", 'task'),
    ]
    for name, kind in tests:
        m = canonicalize_with_meta(name, kind)
        print(f'[{kind}] {name!r:50} -> canon={m.canonical!r:28} year={m.year_hint!r:4} dir={m.lang_dir!r}')

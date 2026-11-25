"""v11 entity matcher — v9 with the short-metric comparator fix (v6e Fix 1).

Edge rule: same dataset-canonical, same method surname (if gold states one),
and the metric comparator accepts either:
  - v9's full exact / alias / ≥3-char-substring rule, OR
  - a token-aligned short-head substring where the short head is in the
    whitelist `config/short_metric_whitelist.yaml`.

"Token-aligned" = the short head appears in the longer form bounded by
non-alphanumeric characters or string boundaries. "F1" matches "entity-level F1"
(space before) and "F1-mix" (hyphen after); it does NOT match "F12" or "BF1"
(no boundary).

Keep v9 untouched for strict-mode comparability.
"""
from __future__ import annotations
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from entity_match_v9 import (
    dataset_match, collision_suspect, _nm_of, _value_eq, nm_method,
)

_WL_PATH = Path(__file__).resolve().parent.parent / 'config' / 'short_metric_whitelist.yaml'
_WHITELIST = set((yaml.safe_load(_WL_PATH.read_text(encoding='utf-8')) or {}).get('short_metrics', []))

METRIC_ALIAS_SRC = {
    'accuracy':        ['percentage correct', 'top 1 accuracy', 'top-1 accuracy',
                        'classification accuracy', 'acc'],
    'error rate':      ['percentage error', 'classification error', 'error',
                        'top-1 error', 'top 1 error'],
    'inception score': ['is', 'inception'],
    'f1':              ['f 1', 'f measure', 'f1 score'],
    'bleu':            ['bleu 4', 'bleu score', 'bleu-4', 'bleu4'],
    # ROUGE-N F1 is the default ROUGE-N reporting; the *f1 arm spellings are the
    # same quantity as the gold's bare ROUGE-N head.
    'rouge l':         ['rougel', 'rouge-l', 'rouge l', 'rougel f1', 'rouge-l f1',
                        'rouge l f1', 'rougelf1'],
    'rouge 1':         ['rouge-1', 'rouge1', 'rouge1 f1', 'rouge-1 f1',
                        'rouge 1 f1', 'rouge1f1'],
    'rouge 2':         ['rouge-2', 'rouge2', 'rouge2 f1', 'rouge-2 f1',
                        'rouge 2 f1', 'rouge2f1'],
    'map':             ['mean average precision', 'm ap', 'average precision'],
    'perplexity':      ['ppl'],
    'mae':             ['mean absolute error'],
}


def _norm_simple(s):
    s = unicodedata.normalize('NFKC', str(s or '').strip()).lower()
    s = re.sub(r"['\-/_]+", ' ', s)
    s = re.sub(r"[^a-z0-9. ]+", ' ', s)
    return re.sub(r"\s+", ' ', s).strip()


METRIC_ALIAS = {_norm_simple(k): [_norm_simple(v) for v in vs]
                for k, vs in METRIC_ALIAS_SRC.items()}


def _norm_pair(s):
    if s is None: return ('', '')
    s = unicodedata.normalize('NFKC', str(s).strip())
    for a, b in [('’', "'"), ('‘', "'"), ('→', ' '),
                 ('–', '-'), ('—', '-')]:
        s = s.replace(a, b)
    s = s.lower()
    s = re.sub(r"\(.*?\)", '', s)
    s_sub = re.sub(r"\s+", ' ',
                   re.sub(r"[^a-z0-9. ]+", ' ', re.sub(r"['\-/_]+", ' ', s))).strip()
    s_h = re.sub(r"\s+", ' ', re.sub(r"[^a-z0-9.\-' ]+", ' ', s)).strip()
    return (s_sub, s_h)


def _token_aligned(short: str, long: str) -> bool:
    """`short` appears in `long` bounded by non-alphanumeric chars or boundaries."""
    if not short or not long: return False
    if short == long: return True
    # Build boundary-anchored regex. Use \b wasn't enough because '-' counts
    # as a non-word char and interacts with alphanumerics; use explicit [^a-z0-9]
    # or string boundary.
    pat = rf'(?:^|[^a-z0-9]){re.escape(short)}(?:$|[^a-z0-9])'
    return bool(re.search(pat, long))


def nm_metric_v11(a, b) -> tuple[bool, str]:
    """Return (matched, how). `how` in {'exact','alias','substring>=3','short_whitelist','no'}."""
    na, nb = _norm_pair(a), _norm_pair(b)
    if not na[0] or not nb[0]: return (not na[0] and not nb[0], 'exact' if (not na[0] and not nb[0]) else 'no')
    if na[0] == nb[0] or na[1] == nb[1]: return (True, 'exact')
    # Alias
    for k, als in METRIC_ALIAS.items():
        if k == na[0] and nb[0] in als: return (True, 'alias')
        if k == nb[0] and na[0] in als: return (True, 'alias')
        if na[0] in als and nb[0] in als: return (True, 'alias')
    # v9 ≥3 substring
    if len(na[0]) >= 3 and len(nb[0]) >= 3 and (na[0] in nb[0] or nb[0] in na[0]):
        return (True, 'substring>=3')
    # Fix 1: short-metric whitelist, token-aligned
    # Which side is the short head?
    short_head = None
    long_form = None
    if na[0] in _WHITELIST:
        short_head, long_form = na[0], nb[0]
    elif nb[0] in _WHITELIST:
        short_head, long_form = nb[0], na[0]
    if short_head and long_form and short_head != long_form:
        if _token_aligned(short_head, long_form):
            return (True, 'short_whitelist')
    return (False, 'no')


def entity_match_v11(t, g, record_suspect=None, strict_metric: bool = False) -> bool:
    """v11 match. `strict_metric=True` uses v9's metric comparator exactly
    (no short-whitelist loosening) so the caller can compare modes."""
    # Dataset
    ds_t = t.get('datasets') or (([t['dataset']]) if 'dataset' in t else [])
    pds = [_nm_of(d) for d in (ds_t or [])]
    gt_ds = g.get('Test-Dataset') or g.get('Train-Dataset') or g.get('Material', '')
    matched_ds = False; ds_meta = None; chosen_arm_ds = ''
    for p in pds:
        m, meta = dataset_match(p, gt_ds)
        if m:
            matched_ds = True; ds_meta = meta; chosen_arm_ds = p; break
    if not matched_ds: return False
    # Metric
    ms_t = t.get('metrics')
    if ms_t is None and ('metric' in t or 'value' in t):
        ms_t = [{'name': _nm_of(t.get('metric')), 'value': t.get('value')}]
    metric_how = 'no'
    for m in (ms_t or []):
        if strict_metric:
            ok, how = nm_metric_v11(_nm_of(m), g.get('Metric', ''))
            if ok and how != 'short_whitelist':
                metric_how = how; break
        else:
            ok, how = nm_metric_v11(_nm_of(m), g.get('Metric', ''))
            if ok:
                metric_how = how; break
    if metric_how == 'no': return False
    # Method
    gm = g.get('Method', '')
    if gm and not nm_method(_nm_of(t.get('method')), gm): return False
    # Year-is-soft rule
    if ds_meta and ds_meta.get('year_conflict'):
        gv = g.get('Result')
        exact_ok = any(_value_eq((m.get('value') if isinstance(m, dict) else None),
                                 gv, tol_rel=0.0) for m in (ms_t or []))
        if not exact_ok: return False
    # Record suspect
    if record_suspect is not None:
        sus = collision_suspect(ds_meta or {}, chosen_arm_ds, gt_ds)
        if sus:
            record_suspect.append({**sus,
                                   'arm_dataset_raw': chosen_arm_ds,
                                   'gold_dataset_raw': gt_ds,
                                   'metric_match_how': metric_how})
        elif metric_how == 'short_whitelist':
            # Flag short-whitelist-only matches as a new kind of suspect
            record_suspect.append({
                'reason': 'short_metric_whitelist_match',
                'canonical_head': (ds_meta or {}).get('head', ''),
                'arm_year': (ds_meta or {}).get('arm_year'),
                'gold_year': (ds_meta or {}).get('gold_year'),
                'arm_lang': (ds_meta or {}).get('arm_lang'),
                'gold_lang': (ds_meta or {}).get('gold_lang'),
                'arm_dataset_raw': chosen_arm_ds,
                'gold_dataset_raw': gt_ds,
                'metric_match_how': metric_how,
            })
    return True


if __name__ == '__main__':
    cases = [
        ('F1', 'entity-level F1', True),
        ('F1', 'segment-level micro-averaged F1 score', True),
        ('F1', 'BF1', False),
        ('F1', 'F12', False),
        ('IS', 'inception score', True),  # via alias
        ('BLEU', 'BLEU-4', True),  # via alias
        ('ROUGE-1', 'ROUGE1-F1', True),  # substring>=3
    ]
    for a, b, want in cases:
        got, how = nm_metric_v11(a, b)
        ok = '✓' if got == want else '✗'
        print(f'  {ok} {a!r:12} vs {b!r:50} -> {got} ({how})  expected {want}')

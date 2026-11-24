"""v9 entity matcher — uses v6c normalizer for dataset and task names.

Match key (brief §2):
    (method_surname, dataset_canonical, metric_canonical, value_rounded)

Task is NOT part of the key (v8 convention). Method uses surname-only matching
with the v8 `nm()` alias-aware comparator. Metric uses v8 METRIC_ALIAS.

Year-is-soft rule: if canonical dataset heads match but year_hints differ and
both are non-empty, the match is only accepted when method/metric/value match
EXACTLY (value equality to the stated precision, not the ±2% tolerance used
elsewhere). Flag such matches as collision suspects.

Collision suspect = any accepted match whose raw gold/arm dataset strings
differ in a way the normalizer may have papered over:
  - canonical heads identical AND (
       year_hints conflict (both non-empty, different)
    OR lang_dir conflicts (both non-empty, different)
    OR <3 character overlap between the raw forms
  )
"""
from __future__ import annotations
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset_task_normalizer import canonicalize_with_meta, canonicalize

# --- Reuse v8's method surname + metric alias logic ---
METRIC_ALIAS_SRC = {
    'accuracy':        ['percentage correct', 'top 1 accuracy', 'top-1 accuracy',
                        'classification accuracy', 'acc'],
    'error rate':      ['percentage error', 'classification error', 'error',
                        'top-1 error', 'top 1 error'],
    'inception score': ['is', 'inception'],
    'f1':              ['f 1', 'f measure', 'f1 score'],
    'bleu':            ['bleu 4', 'bleu score', 'bleu-4', 'bleu4'],
    'rouge l':         ['rougel', 'rouge-l', 'rougeL', 'rouge l'],
    'rouge 1':         ['rouge-1', 'rouge1'],
    'rouge 2':         ['rouge-2', 'rouge2'],
    'map':             ['mean average precision', 'm ap', 'average precision'],
    'perplexity':      ['ppl'],
    'mae':             ['mean absolute error'],
}


def _norm_s(s):
    s = unicodedata.normalize('NFKC', str(s).strip()).lower()
    s = re.sub(r"['\-/_]+", ' ', s)
    s = re.sub(r"[^a-z0-9\. ]+", ' ', s)
    return re.sub(r"\s+", ' ', s).strip()


METRIC_ALIAS = {_norm_s(k): [_norm_s(v) for v in vs] for k, vs in METRIC_ALIAS_SRC.items()}


def _norm_pair(s):
    if s is None: return ('', '')
    s = unicodedata.normalize('NFKC', str(s).strip())
    for a, b in [('’', "'"), ('‘', "'"), ('→', ' '),
                 ('–', '-'), ('—', '-')]:
        s = s.replace(a, b)
    s = s.lower()
    s = re.sub(r"\(.*?\)", '', s)
    s_sub = re.sub(r"\s+", ' ',
                   re.sub(r"[^a-z0-9\. ]+", ' ', re.sub(r"['\-/_]+", ' ', s))).strip()
    s_h = re.sub(r"\s+", ' ', re.sub(r"[^a-z0-9\.\-' ]+", ' ', s)).strip()
    return (s_sub, s_h)


def nm_metric(a, b):
    na, nb = _norm_pair(a), _norm_pair(b)
    if not na[0] or not nb[0]: return not na[0] and not nb[0]
    if na[0] == nb[0] or na[1] == nb[1]: return True
    for k, als in METRIC_ALIAS.items():
        if k == na[0] and nb[0] in als: return True
        if k == nb[0] and na[0] in als: return True
        if na[0] in als and nb[0] in als: return True
    if len(na[0]) >= 3 and len(nb[0]) >= 3 and (na[0] in nb[0] or nb[0] in na[0]): return True
    return False


def nm_method(a, b):
    na, nb = _norm_pair(a), _norm_pair(b)
    if not na[0] or not nb[0]: return not na[0] and not nb[0]
    if na[0] == nb[0] or na[1] == nb[1]: return True
    if len(na[0]) >= 3 and len(nb[0]) >= 3 and (na[0] in nb[0] or nb[0] in na[0]): return True
    return False


def _nm_of(x):
    if isinstance(x, dict): return x.get('name') or x.get('model_name') or x.get('method_name') or ''
    if isinstance(x, list) and x: return _nm_of(x[0])
    if isinstance(x, str): return x
    return ''


def _value_eq(a, b, tol_rel=0.0):
    """Strict equality unless tol_rel>0. Both must coerce to float."""
    if a is None or b is None: return False
    try:
        fa, fb = float(a), float(b)
    except Exception:
        return str(a).strip() == str(b).strip()
    if abs(fa - fb) < 1e-9: return True
    if tol_rel > 0 and abs(fa - fb) / max(abs(fb), 1e-9) <= tol_rel: return True
    return False


def dataset_match(arm_ds_raw, gold_ds_raw) -> tuple[bool, dict | None]:
    """Compare dataset names. Returns (matched_on_normalized_key, meta) where
    meta is a dict with year/lang/heads if there was a soft match, else None."""
    a = canonicalize_with_meta(arm_ds_raw or '', 'dataset')
    g = canonicalize_with_meta(gold_ds_raw or '', 'dataset')
    # Split canonical back into head + lang
    def split(c):
        if '+' in c:
            h, _, l = c.partition('+')
            return h, l
        return c, ''
    ah, alang = split(a.canonical)
    gh, glang = split(g.canonical)
    # Direct equal
    if a.canonical == g.canonical and a.canonical:
        year_conflict = (a.year_hint and g.year_hint and a.year_hint != g.year_hint)
        lang_conflict = False  # same canonical means same lang
        meta = {'arm_year': a.year_hint, 'gold_year': g.year_hint,
                'arm_lang': alang, 'gold_lang': glang,
                'arm_canon': a.canonical, 'gold_canon': g.canonical,
                'year_conflict': year_conflict,
                'lang_conflict': lang_conflict,
                'head': ah}
        return (True, meta)
    # Head matches, lang one-side-empty
    if ah and ah == gh and (not alang or not glang):
        # One side didn't state direction; accept the head match
        meta = {'arm_year': a.year_hint, 'gold_year': g.year_hint,
                'arm_lang': alang, 'gold_lang': glang,
                'arm_canon': a.canonical, 'gold_canon': g.canonical,
                'year_conflict': (a.year_hint and g.year_hint and a.year_hint != g.year_hint),
                'lang_conflict': False, 'head': ah}
        return (True, meta)
    # Head matches, year one-side-empty, same lang
    if ah and ah == gh and alang == glang and (not a.year_hint or not g.year_hint):
        meta = {'arm_year': a.year_hint, 'gold_year': g.year_hint,
                'arm_lang': alang, 'gold_lang': glang,
                'arm_canon': a.canonical, 'gold_canon': g.canonical,
                'year_conflict': False, 'lang_conflict': False, 'head': ah}
        return (True, meta)
    return (False, None)


def collision_suspect(ds_meta: dict, arm_ds_raw: str, gold_ds_raw: str) -> dict | None:
    """Apply the brief's suspect rule. Returns a dict or None."""
    if not ds_meta: return None
    year_conflict = bool(ds_meta.get('year_conflict'))
    lang_conflict = bool(ds_meta.get('lang_conflict'))
    arm_r = (arm_ds_raw or '').lower()
    gold_r = (gold_ds_raw or '').lower()
    # Character-overlap sanity check (jaccard of alnum char sets)
    def charset(s):
        return set(re.sub(r'[^a-z0-9]', '', s))
    overlap = len(charset(arm_r) & charset(gold_r))
    low_overlap = overlap < 3
    if year_conflict or lang_conflict or low_overlap:
        reasons = []
        if year_conflict: reasons.append(
            f"year {ds_meta.get('arm_year')!r} vs {ds_meta.get('gold_year')!r}")
        if lang_conflict: reasons.append(
            f"lang {ds_meta.get('arm_lang')!r} vs {ds_meta.get('gold_lang')!r}")
        if low_overlap: reasons.append(f"<3-char overlap ({overlap})")
        return {'reason': '; '.join(reasons),
                'canonical_head': ds_meta.get('head', ''),
                'arm_year': ds_meta.get('arm_year'),
                'gold_year': ds_meta.get('gold_year'),
                'arm_lang': ds_meta.get('arm_lang'),
                'gold_lang': ds_meta.get('gold_lang')}
    return None


def entity_match_v9(t, g, record_suspect=None) -> bool:
    """v9 match. If record_suspect is a list, append a collision dict when the
    match is accepted but flagged suspect."""
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
    if not any(nm_metric(_nm_of(m), g.get('Metric', '')) for m in (ms_t or [])):
        return False
    # Method (surname-aware)
    gm = g.get('Method', '')
    if gm and not nm_method(_nm_of(t.get('method')), gm): return False
    # Value — with year-is-soft rule: when year_hints differ, require exact value match
    if ds_meta and ds_meta.get('year_conflict'):
        # extract value(s) from tuple; require any to match gold's Result exactly
        gv = g.get('Result')
        mets = ms_t or []
        exact_ok = any(_value_eq((m.get('value') if isinstance(m, dict) else None),
                                 gv, tol_rel=0.0) for m in mets)
        if not exact_ok: return False
    # Record suspect
    if record_suspect is not None:
        sus = collision_suspect(ds_meta or {}, chosen_arm_ds, gt_ds)
        if sus:
            record_suspect.append({**sus,
                                   'arm_dataset_raw': chosen_arm_ds,
                                   'gold_dataset_raw': gt_ds})
    return True


if __name__ == '__main__':
    # Smoke tests
    import json
    cases = [
        ({'datasets': [{'name': "WMT14 En-De"}], 'metrics': [{'name': 'BLEU', 'value': 28.4}],
          'method': {'name': 'Transformer'}, 'task': {'name': 'MT'}},
         {'Test-Dataset': "WMT’14 EN-DE", 'Metric': 'BLEU', 'Result': 28.4,
          'Method': 'Transformer', 'Task': 'Machine Translation'}),
        ({'datasets': [{'name': "CNNDM"}], 'metrics': [{'name': 'ROUGE-1', 'value': 44.88}],
          'method': {'name': 'BART'}, 'task': {'name': 'Summarization'}},
         {'Test-Dataset': 'CNN/DailyMail', 'Metric': 'ROUGE-1', 'Result': 44.88,
          'Method': 'BART', 'Task': 'Summarization'}),
        ({'datasets': [{'name': "IWSLT'14 DE-EN"}], 'metrics': [{'name': 'BLEU', 'value': 36.3}],
          'method': {'name': 'MUSE'}, 'task': {'name': 'MT'}},
         {'Test-Dataset': "IWSLT’15 DE-EN", 'Metric': 'BLEU', 'Result': 36.3,
          'Method': 'MUSE', 'Task': 'Machine Translation'}),
    ]
    for t, g in cases:
        suspect = []
        m = entity_match_v9(t, g, suspect)
        print(f'  match={m}  suspect={suspect or None}')
        print(f'    arm={t["datasets"][0]["name"]!r}  gold={g["Test-Dataset"]!r}')

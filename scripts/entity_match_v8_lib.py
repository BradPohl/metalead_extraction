"""v8 entity matcher extracted as a library (no module-level side effects)."""
import re
import unicodedata

TASK_ALIAS = {
    'named entity recognition': ['ner', 'sentence-level ner', 'sentence level ner'],
    'dialogue act classification': ['dialogue act recognition', 'dialog act classification', 'dac'],
    'machine translation': ['mt', 'neural machine translation', 'nmt'],
}
DATASET_ALIAS = {
    'conll-2003': ['conll 2003', 'conll03', 'conll-03'],
    'switchboard dialog act corpus': ['swda', 'switchboard'],
    "wmt'14 en-fr": ['wmt14 en-fr', 'wmt 14 en-fr', 'wmt-14 en-fr'],
    'ontonotes v5': ['ontonotes 5', 'ontonotes 5.0', 'ontonotes'],
}
METRIC_ALIAS = {
    'accuracy':        ['percentage correct', 'top 1 accuracy', 'top-1 accuracy',
                        'classification accuracy', 'acc'],
    'error rate':      ['percentage error', 'classification error', 'error',
                        'top-1 error', 'top 1 error'],
    'inception score': ['is', 'inception'],
    'f1':              ['f 1', 'f measure', 'f1 score'],
    'bleu':            ['bleu 4', 'bleu score'],
    'rouge l':         ['rougel', 'rouge 1', 'rouge-1'],
    'map':             ['mean average precision', 'm ap', 'average precision'],
    'perplexity':      ['ppl'],
    'mae':             ['mean absolute error'],
}


def _norm_s(s):
    s = unicodedata.normalize('NFKC', str(s).strip()).lower()
    s = re.sub(r"['\-/_]+", ' ', s)
    s = re.sub(r"[^a-z0-9\. ]+", ' ', s)
    return re.sub(r"\s+", ' ', s).strip()


METRIC_ALIAS = {_norm_s(k): [_norm_s(v) for v in vs] for k, vs in METRIC_ALIAS.items()}


def norm(s):
    if s is None: return ('', '')
    s = unicodedata.normalize('NFKC', str(s).strip())
    for a, b in [('’', "'"), ('‘', "'"), ('→', ' '),
                 ('–', '-'), ('—', '-')]:
        s = s.replace(a, b)
    s = s.lower()
    s = re.sub(r"\(.*?\)", '', s)
    s = re.sub(r"\s*-\s*english\s*$", '', s)
    s_sub = re.sub(r"\s+", ' ', re.sub(r"[^a-z0-9\. ]+", ' ',
                   re.sub(r"['\-/_]+", ' ', s))).strip()
    s_h = re.sub(r"\s+", ' ', re.sub(r"[^a-z0-9\.\-' ]+", ' ', s)).strip()
    return (s_sub, s_h)


def nm(a, b, kind='generic'):
    na, nb = norm(a), norm(b)
    if not na[0] or not nb[0]: return not na[0] and not nb[0]
    if na[0] == nb[0] or na[1] == nb[1]: return True
    t = (TASK_ALIAS if kind == 'task' else DATASET_ALIAS if kind == 'dataset'
         else METRIC_ALIAS if kind == 'metric' else {})
    for k, als in t.items():
        if k == na[0] and nb[0] in als: return True
        if k == nb[0] and na[0] in als: return True
        if na[0] in als and nb[0] in als: return True
    if len(na[0]) >= 3 and len(nb[0]) >= 3 and (na[0] in nb[0] or nb[0] in na[0]): return True
    return False


def _nm_of(x):
    if isinstance(x, dict): return x.get('name') or x.get('model_name') or x.get('method_name') or ''
    if isinstance(x, list) and x: return _nm_of(x[0])
    if isinstance(x, str): return x
    return ''


def entity_match_v8(t, g):
    if not nm(_nm_of(t.get('task')), g.get('Task', ''), 'task'): return False
    ds = t.get('datasets')
    if ds is None and 'dataset' in t: ds = [t['dataset']]
    pds = [_nm_of(d) for d in (ds or [])]
    gt = g.get('Test-Dataset') or g.get('Train-Dataset') or g.get('Material', '')
    if not any(nm(p, gt, 'dataset') for p in pds): return False
    ms = t.get('metrics')
    if ms is None and ('metric' in t or 'value' in t):
        ms = [{'name': _nm_of(t.get('metric')), 'value': t.get('value')}]
    if not any(nm(_nm_of(m), g.get('Metric', ''), 'metric') for m in (ms or [])):
        return False
    gm = g.get('Method', '')
    if gm and not nm(_nm_of(t.get('method')), gm): return False
    return True

"""Score one extraction run against the MetaLead reference results.

Usage (run from the package root):

    python scripts/score.py --pred our_outputs/ours_deepseek_inside
    python scripts/score.py --pred our_outputs/gpt_alone --prefix metalead_
    python scripts/score.py --pred my_run --baseline our_outputs/gpt_alone --baseline-prefix metalead_

--pred       folder with one JSON file per paper: {"tuples": [ ...records... ]}
--prefix     file name prefix, if the files are named <prefix><paper_id>.json
--baseline   optional second run; prints a paired bootstrap over papers (pred minus baseline)

--require-value   also require the extracted value to equal the reference value
                  (to two decimals; a 0-1 value and its 0-100 form count as equal)

A paper with no file, or with an empty "tuples" list, counts as zero records.
The default matching is the one used for the Chapter 1 results: the v11 matcher, one-to-one
(bipartite) matching between extracted records and reference results, and the
ROGUE -> ROUGE spelling fix for paper 1912.08777. The v11 matcher compares the dataset
name and the metric name (and the method name for the few reference rows that have one).
It does not compare the numeric value. Use --require-value to add that check.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from entity_match_v11 import entity_match_v11  # noqa: E402

GOLD = json.load(open(ROOT / 'data' / 'metalead_annotations' / 'metalead_annotations.json', encoding='utf-8'))
PAPERS = json.load(open(ROOT / 'splits' / 'metalead_31_papers.json', encoding='utf-8'))
CORR = yaml.safe_load((ROOT / 'data' / 'metalead_annotations' / 'key_corrections.yaml').read_text(encoding='utf-8')) or {}


def gold_for(pid):
    rows = GOLD.get(pid + '.pdf', {}).get('TDMs', [])
    cors = [c for c in CORR.get('corrections', [])
            if c.get('paper_id') == pid and c.get('kind') == 'metric_typo']
    out = []
    for g in rows:
        g2 = dict(g)
        for c in cors:
            if g2.get('Metric') == c.get('from'):
                g2['Metric'] = c['to']; break
        out.append(g2)
    return out


def load(run_dir, prefix, pid):
    f = Path(run_dir) / f'{prefix}{pid}.json'
    if not f.exists():
        return []
    return json.loads(f.read_text(encoding='utf-8')).get('tuples', [])


def _values(t):
    ms = t.get('metrics')
    if ms is None and ('metric' in t or 'value' in t):
        ms = [{'value': t.get('value')}]
    return [m.get('value') for m in (ms or []) if isinstance(m, dict)]


def _value_equal(a, b):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        return False
    return any(abs(a * s - b) <= 0.005 + 1e-9 for s in (1, 100, 0.01))


REQUIRE_VALUE = False


def _edge(t, g):
    if not entity_match_v11(t, g):
        return False
    if REQUIRE_VALUE:
        return any(_value_equal(v, g.get('Result')) for v in _values(t))
    return True


def match(tuples, gold):
    n_g, n_t = len(gold), len(tuples)
    if n_g == 0 or n_t == 0:
        return 0, n_t, n_g
    rows, cols = [], []
    for gi, g in enumerate(gold):
        for tj, t in enumerate(tuples):
            if _edge(t, g):
                rows.append(gi); cols.append(tj)
    if not rows:
        return 0, n_t, n_g
    graph = csr_matrix(([1] * len(rows), (rows, cols)), shape=(n_g, n_t))
    m = maximum_bipartite_matching(graph, perm_type='column')
    tp = int((m >= 0).sum())
    return tp, n_t - tp, n_g - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_run(run_dir, prefix):
    return {pid: match(load(run_dir, prefix, pid), gold_for(pid)) for pid in PAPERS}


def pooled(per_paper, papers=None):
    papers = papers or PAPERS
    tp = sum(per_paper[p][0] for p in papers)
    fp = sum(per_paper[p][1] for p in papers)
    fn = sum(per_paper[p][2] for p in papers)
    return tp, fp, fn, *prf(tp, fp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--prefix', default='')
    ap.add_argument('--baseline')
    ap.add_argument('--baseline-prefix', default='')
    ap.add_argument('--per-paper', action='store_true')
    ap.add_argument('--require-value', action='store_true')
    a = ap.parse_args()
    global REQUIRE_VALUE
    REQUIRE_VALUE = a.require_value

    pp = score_run(a.pred, a.prefix)
    tp, fp, fn, p, r, f = pooled(pp)
    done = sum(1 for pid in PAPERS if pp[pid][0] + pp[pid][1] > 0)
    print(f'run: {a.pred}' + ('   (value must match)' if REQUIRE_VALUE else ''))
    print(f'papers with at least one record: {done} of {len(PAPERS)}')
    print(f'matched {tp}   extra {fp}   missed {fn}')
    print(f'precision {p:.4f}   recall {r:.4f}   F1 {f:.4f}')
    if a.per_paper:
        for pid in PAPERS:
            t, x, m = pp[pid]; pp_p, pp_r, pp_f = prf(t, x, m)
            print(f'  {pid}  matched {t:4d}  extra {x:4d}  missed {m:4d}  F1 {pp_f:.3f}')

    if a.baseline:
        bb = score_run(a.baseline, a.baseline_prefix)
        _, _, _, _, _, fb = pooled(bb)
        rng = random.Random(42)
        deltas = []
        for _ in range(10000):
            s = [rng.choice(PAPERS) for _ in PAPERS]
            deltas.append(pooled(pp, s)[5] - pooled(bb, s)[5])
        deltas.sort()
        print(f'baseline: {a.baseline}   F1 {fb:.4f}')
        print(f'F1 difference (run minus baseline): {f - fb:+.4f}')
        print(f'95% interval [{deltas[250]:+.4f}, {deltas[9750]:+.4f}]   '
              f'share of resamples <= 0: {sum(d <= 0 for d in deltas) / 10000:.4f}')


if __name__ == '__main__':
    main()

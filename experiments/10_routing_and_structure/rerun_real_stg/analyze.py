"""Paired analysis of rerun_real_stg configs. Usage: python analyze.py [ref] [cfg ...]"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
def load(c):
    d = {}
    for base in (f'{HERE}/out', f'{HERE}/../rerun_sensitivity/out', f'{HERE}/../rerun_diag/out'):
        for f in glob.glob(f'{base}/{c}/*.json'):
            r = json.load(open(f))
            if r.get('status') == 'ok' and (r['task'], r['fold']) not in d:
                d[(r['task'], r['fold'])] = (r['accuracy'], r.get('d'))
        if d:
            return d
    return d
args = sys.argv[1:] or ['ref', 'ref_nodag', 'stg', 'stg_l003', 'stg_l03', 'stg_nodag']
ref, cfgs = args[0], args[1:]
R = load(ref)
I = pd.read_csv(f'{HERE}/../rerun_interact/interaction_real.csv')
I['gap'] = I.lgbm - I.lgbm_stumps
lg = {(t, f): (a, g) for t, f, a, g in zip(I.task, I.fold, I.lgbm, I.gap)}
gap_t = I.groupby('task').gap.mean()
print(f'ref = {ref} ({len(R)} task-folds)')
def stat(x):
    x = np.asarray(x)
    p = wilcoxon(x).pvalue if (x != 0).any() and len(x) > 1 else 1.0
    return f'{100*x.mean():+6.2f} ({(x>0).sum()}/{(x<0).sum()}/{len(x)}, p={p:.3f})'
rows = []
for c in cfgs + ['lgbm']:
    C = {k: (lg[k][0], None) for k in R if k in lg} if c == 'lgbm' else load(c)
    ks = sorted(set(R) & set(C))
    x = np.array([C[k][0] - R[k][0] for k in ks])
    dd = np.array([R[k][1] for k in ks]); g = np.array([gap_t.get(k[0], np.nan) for k in ks])
    rows.append(dict(cfg=c, n=len(ks), all=stat(x), d_ge30=stat(x[dd >= 30]) if (dd >= 30).any() else '-',
                     d_lt30=stat(x[dd < 30]), gap_gt1=stat(x[g > 0.01]), gap_le1=stat(x[g <= 0.01])))
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)
print('diff vs ref, pts (wins/losses/N, Wilcoxon p)')
print(pd.DataFrame(rows).set_index('cfg').to_string())
if '--per-task' in os.environ.get('PT', '--per-task'):
    T = {}
    for c in cfgs:
        C = load(c)
        T[c] = pd.Series({k: 100 * (C[k][0] - R[k][0]) for k in set(R) & set(C)}).groupby(level=0).mean()
    T = pd.DataFrame(T)
    T['d'] = pd.Series({k[0]: v[1] for k, v in R.items()})
    T['gap'] = 100 * gap_t
    print('\nper task (pts vs ref, mean over folds), sorted by d')
    print(T.sort_values('d', ascending=False).round(2).to_string())

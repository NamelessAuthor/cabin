"""Compare rerun_headopt variants with the rerun_interact/out_highd references (paired by setting)."""
import glob, json, os, sys
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, '..', 'rerun_interact', 'out_highd')


def load(root, prefix=''):
    rows = []
    for f in glob.glob(f'{root}/*/*.json'):
        m = os.path.basename(os.path.dirname(f))
        r = json.load(open(f))
        if r.get('status') != 'ok':
            continue
        k = dict(d=r['d'], n=r['n'], lam=r['lam'], rep=r['rep'])
        if m == 'baselines':
            for b in ('lgbm', 'lgbm_stumps', 'lgbm_mb'):
                rows.append({**k, 'model': b, 'acc': r[b]})
        else:
            rows.append({**k, 'model': prefix + m, 'acc': r['acc']})
    return rows


rows = load(REF, 'ref:') + load(os.path.join(HERE, 'out'))
P = pd.DataFrame(rows).pivot_table(index=['lam', 'd', 'n', 'rep'], columns='model', values='acc')
mine = [c for c in P.columns if '__' in c]
P = P[P[mine].notna().any(axis=1)] if mine else P
refs = [c for c in ('lgbm', 'lgbm_mb', 'ref:mlp_nodag', 'ref:add_nodag', 'ref:mlp_true', 'ref:add_true') if c in P]
pd.set_option('display.width', 250)
print('mean accuracy by (lam, d, n):')
print(P[refs + sorted(mine)].groupby(['lam', 'd', 'n']).mean().T.round(3).to_string())
targets = sys.argv[1:] or ['lgbm', 'ref:mlp_nodag']
for t in targets:
    print(f'\npaired diff vs {t} (pts, wins/N, Wilcoxon p), by lam:')
    out = {}
    for m in sorted(mine):
        cell = {}
        for lam, g in P.groupby('lam'):
            x = (g[m] - g[t]).dropna()
            if len(x) == 0:
                continue
            p = wilcoxon(x).pvalue if (x != 0).any() else 1.0
            cell[f'lam{lam}'] = f'{100 * x.mean():+.1f} ({(x > 0).sum()}/{len(x)}, p={p:.3f})'
        out[m] = cell
    print(pd.DataFrame(out).T.to_string())

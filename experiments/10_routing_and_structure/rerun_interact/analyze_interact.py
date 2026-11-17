"""Analyse rerun_interact: synthetic lambda sweep + real-data interaction gap vs DAG effect."""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in glob.glob(f'{HERE}/out/*/*.json'):
    r = json.load(open(f))
    m = os.path.basename(os.path.dirname(f))
    key = dict(d=r['d'], lam=r['lam'], rep=r['rep'])
    if m == 'baselines':
        for b in ('lgbm', 'lgbm_stumps', 'logreg'):
            rows.append({**key, 'model': b, 'acc': r[b]})
    else:
        rows.append({**key, 'model': m, 'acc': r['acc'], 'best_epoch': r.get('best_epoch')})
S = pd.DataFrame(rows)
P = S.pivot_table(index=['d', 'lam', 'rep'], columns='model', values='acc')
print('== Synthetic: mean acc by lambda (both d, 10 settings each) ==')
print(P.groupby('lam').mean().T.round(3).to_string())
def cmp(a, b):
    out = {}
    for lam, g in P.groupby('lam'):
        x = g[a] - g[b]
        p = wilcoxon(x).pvalue if (x != 0).any() else 1
        out[lam] = f'{100*x.mean():+.1f} ({(x>0).sum()}/{len(x)}, p={p:.2f})'
    return out
pairs = [('mlp_true', 'mlp_nodag'), ('mlp_true', 'mlp_rand'), ('mlp_learn', 'mlp_rand'), ('mlp_learn', 'mlp_nodag'),
         ('add_true', 'add_nodag'), ('add_true', 'add_rand'), ('add_learn', 'add_rand'), ('add_learn', 'add_nodag'),
         ('mlp_true', 'add_true'), ('mlp_nodag', 'add_nodag'), ('cabin', 'mlp_nodag'), ('lgbm', 'lgbm_stumps'), ('lgbm', 'mlp_nodag')]
print('\n== Paired differences (pts, wins/10, Wilcoxon) ==')
print(pd.DataFrame({f'{a} - {b}': cmp(a, b) for a, b in pairs}).T.to_string())
print('\nby d, lam=1:')
print(P.xs(1.0, level='lam').groupby('d').mean().T.round(3).to_string())

# ---------- real data ----------
R = pd.read_csv(f'{HERE}/interaction_real.csv')
R['gap'] = R.lgbm - R.lgbm_stumps
G = R.groupby('task').agg(gap=('gap', 'mean'), d=('d', 'first'), n=('n', 'first'), in_sens=('in_sens', 'first'))
print(f'\n== Real: interaction gap (LGBM - stumps), {len(G)} tasks ==')
print(G.gap.describe().round(4).to_string())
for t in (0.005, 0.01, 0.02):
    print(f'gap > {t}: {(G.gap > t).sum()}/{len(G)}')
DIAG = f'{HERE}/../rerun_diag/out'
def load(c):
    out = {}
    for f in glob.glob(f'{DIAG}/{c}/*.json'):
        r = json.load(open(f))
        if r.get('status') == 'ok':
            out[(r['task'], r['fold'])] = r['accuracy']
    return pd.Series(out)
cfgs = ['base_ft', 'rand_ft', 'nodag_ft', 'add_ft', 'add_rand_ft', 'add_nodag_ft', 'diag_base', 'add', 'add_nodag']
D = pd.DataFrame({c: load(c) for c in cfgs})
D.index.names = ['task', 'fold']
T = D.groupby('task').mean()
eff = pd.DataFrame({
    'mlp: learn-rand (ft)': T.base_ft - T.rand_ft,
    'mlp: learn-nodag (ft)': T.base_ft - T.nodag_ft,
    'add: learn-nodag (ft)': T.add_ft - T.add_nodag_ft,
    'add: learn-rand (ft)': T.add_ft - T.add_rand_ft,
    'mlp-add (ft, learned)': T.base_ft - T.add_ft,
    'mlp-add (nodag ft)': T.nodag_ft - T.add_nodag_ft,
})
J = eff.join(G.gap, how='inner')
print(f'\n== Spearman: interaction gap vs per-task effect ({len(J)} tasks) ==')
for c in eff:
    rho, p = spearmanr(J.gap, J[c])
    hi, lo = J[J.gap > 0.01][c], J[J.gap <= 0.01][c]
    print(f'{c:24s} rho={rho:+.2f} p={p:.3f} | mean eff gap>1pt: {100*hi.mean():+.2f} (n={len(hi)})  gap<=1pt: {100*lo.mean():+.2f} (n={len(lo)})')
print('\nper task (sorted by gap):')
print((J.sort_values('gap', ascending=False) * 100).round(2).to_string())

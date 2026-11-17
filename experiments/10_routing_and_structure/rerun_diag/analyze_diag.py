#!/usr/bin/env python3
"""Summaries of rerun_diag (diagnostics) and rerun_shift (distribution shift)."""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, '..')

def load(pat):
    return pd.DataFrame([json.load(open(f)) for f in glob.glob(pat)])

D = load(os.path.join(HERE, 'out', '*', '*.json'))
S = load(os.path.join(P, 'rerun_sensitivity', 'out', 'base', '*.json'))
D = D[D.status == 'ok']
A = D.pivot_table(index=['task', 'fold'], columns='config', values='accuracy')
A['base'] = S.set_index(['task', 'fold']).accuracy
W0 = D.pivot_table(index=['task', 'fold'], columns='config', values='acc_W0')
def cmp(a, b, M=A):
    x = M[[a, b]].dropna(); d = (x[a] - x[b]) * 100
    p = wilcoxon(d).pvalue if (d != 0).any() else 1
    return f'{a:>10s} vs {b:<10s} {x[a].mean():.4f} vs {x[b].mean():.4f}  d={d.mean():+.2f}  W/L={(d>0).sum()}/{(d<0).sum()}  p={p:.3f}  n={len(x)}'
print('reproduction diag_base == base:', (A.diag_base - A.base).abs().max())
for a, b in [('selacc', 'diag_base'), ('selacc_ft', 'diag_base'), ('selacc_ft', 'selacc'),
             ('add', 'diag_base'), ('add_rand', 'add'), ('add_nodag', 'add'), ('add_nodag', 'add_rand')]:
    print(cmp(a, b))
print('\nTest-time W=0 (acc_W0 - acc):')
for c in ['diag_base', 'selacc', 'selacc_ft', 'add', 'add_rand']:
    d = (W0[c] - A[c]).dropna() * 100
    print(f'  {c:10s} mean {d.mean():+.2f}  median {d.median():+.2f}  share with drop>1pt {(d < -1).mean():.2f}')

# Restored epoch by phase
def phase(e): return 'warmup' if e < 20 else ('joint' if e < 120 else 'finetune')
print('\nRestored phase share:')
for c in ['diag_base', 'selacc', 'add']:
    print(' ', c, D[D.config == c].best_epoch.map(phase).value_counts(normalize=True).round(2).to_dict())

# Curves
B = D[D.config == 'diag_base']
rows = []
for _, r in B.iterrows():
    for c in r.curve:
        rows.append(dict(task=r.task, fold=r.fold, **c))
C = pd.DataFrame(rows)
# per-run z of val loss relative to its min; normalise accuracies relative to final
key = {'end warmup (ep 19)': 19, 'joint 25% (ep 44)': 44, 'end joint (ep 119)': 119}
print('\nMean over runs at fixed epochs (runs still going):')
for lab, e in key.items():
    x = C[C.epoch == e]
    print(f'  {lab:22s} val_loss {x.val_loss.mean():.4f}  val_acc {x.val_acc.mean():.4f}  test_acc {x.test_acc.mean():.4f}  n={len(x)}')
last = C.sort_values('epoch').groupby(['task', 'fold']).tail(1)
print(f"  {'last epoch':22s} val_loss {last.val_loss.mean():.4f}  val_acc {last.val_acc.mean():.4f}  test_acc {last.test_acc.mean():.4f}")
best = C.loc[C.groupby(['task', 'fold']).val_loss.idxmin()]
print(f"  {'restored (min loss)':22s} val_loss {best.val_loss.mean():.4f}  val_acc {best.val_acc.mean():.4f}  test_acc {best.test_acc.mean():.4f}")
bacc = C.loc[C.groupby(['task', 'fold']).test_acc.idxmax()]
print(f"  {'oracle (max test)':22s} test_acc {bacc.test_acc.mean():.4f}")
# Within joint: does val loss go up while val acc stays?
J = C[C.phase == 'joint']
g = J.groupby(['task', 'fold'])
first, lastj = g.head(1).set_index(['task', 'fold']), g.tail(1).set_index(['task', 'fold'])
for m in ['val_loss', 'val_acc', 'test_acc']:
    d = lastj[m] - first[m]
    print(f'  joint start->end {m:9s} {d.mean():+.4f}  share up {(d > 0).mean():.2f}')
W = C[C.phase == 'warmup']; g = W.groupby(['task', 'fold'])
mn = g.val_loss.idxmin(); print('  share of runs whose global min loss is in warmup:', (best.epoch < 20).mean().round(2))
C.to_csv(os.path.join(HERE, 'curves_long.csv'), index=False)

# Shift
print('\n=== Shift ===')
Sh = load(os.path.join(P, 'rerun_shift', 'out', '*', '*.json'))
Sh = Sh[Sh.status == 'ok']
Sh['setting'] = Sh.apply(lambda r: f"d{r.d}_p{r.p}_{r.nonlinear}_{r.rep}", axis=1)
rows = {}
for m, g in Sh.groupby('model'):
    if m == 'baselines':
        for b in ['lgbm', 'logreg', 'lgbm_parents']:
            rows[b] = g.set_index('setting')[[f'{b}_iid', f'{b}_do', f'{b}_flip']].set_axis(['iid', 'do', 'flip'], axis=1)
    else:
        rows[m] = g.set_index('setting')[['iid', 'do', 'flip']]
T = pd.concat({k: v.mean() for k, v in rows.items()}, axis=1).T
print(T.round(3))
for a, b in [('mlp', 'mlp_rand'), ('mlp', 'mlp_nodag'), ('add', 'add_rand'), ('add', 'add_nodag'), ('mlp', 'lgbm'), ('add', 'mlp')]:
    for k in ['iid', 'do', 'flip']:
        x = pd.concat([rows[a][k], rows[b][k]], axis=1, keys=[a, b]).dropna()
        d = (x[a] - x[b]) * 100
        p = wilcoxon(d).pvalue if (d != 0).any() else 1
        print(f'  {a:>5s} vs {b:<10s} {k:4s} d={d.mean():+.2f} W/L={(d>0).sum()}/{(d<0).sum()} p={p:.3f}')

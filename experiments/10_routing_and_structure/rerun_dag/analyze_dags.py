#!/usr/bin/env python3
"""§6.2 DAG statistics and Figure 1 correlations from the rerun (out/).

Metrics on the group-level S (k x k, off-diagonal):
  density   fraction of entries with |S_ij| > 0.1
  asymmetry ||S - S^T||_1 / ||S||_1
  active    fraction of VCUDA edges selected (binary edges, off-diagonal)
  moe_cos   mean pairwise cosine similarity among the K=4 MoE S_k
Dataset properties: d, n, C, % categorical, imbalance (max/min class count),
mean |r| (mean absolute pairwise Pearson correlation of the features).
Accuracy and gains use the tub 5-fold results of Table 2.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser('~/tub'))
sys.path.insert(0, os.path.join(HERE, '..', 'results', 'tub'))
from table2 import load
from tub.loader import load_task

def offdiag(M):
    return M[~np.eye(M.shape[-1], dtype=bool)]

def metrics(S, edges):
    o = np.abs(offdiag(S))
    return dict(density=(o > 0.1).mean(),
                asymmetry=np.abs(S - S.T).sum() / max(np.abs(S).sum(), 1e-12),
                active=offdiag(edges).mean())

rows = []
for f in sorted(glob.glob(os.path.join(HERE, 'out', 'base', '*.json'))):
    t = json.load(open(f))['task']
    r = {'task': t}
    for v in ['base', 'contrastive', 'moe']:
        j = json.load(open(os.path.join(HERE, 'out', v, t + '.json')))
        z = np.load(os.path.join(HERE, 'out', v, t + '.npz'))
        r[f'{v}_restored_untrained'] = j['joint_steps_at_restored'] == 0
        r[f'{v}_restored_frac'] = j['joint_steps_at_restored'] / max(j['joint_steps_total'], 1)
        r[f'{v}_acc_f0'] = j['accuracy']
        if v == 'moe':
            Ss, E = z['S_list'], z['edges']
            ms = [metrics(Ss[i], E[i]) for i in range(len(Ss))]
            for m in ms[0]:
                r[f'moe_{m}'] = np.mean([x[m] for x in ms])
            fl = Ss.reshape(len(Ss), -1)
            fl = fl / np.maximum(np.linalg.norm(fl, axis=1, keepdims=True), 1e-12)
            C = fl @ fl.T
            r['moe_cos'] = C[np.triu_indices(len(Ss), 1)].mean()
        else:
            for m, val in metrics(z['S'], z['edges']).items():
                r[f'{v}_{m}'] = val
        if v == 'base':
            r.update(d=j['d'], n=j['n'], C=j['n_classes'], pct_cat=j['n_categorical'] / j['d'])
    task = load_task(t)
    cnt = np.bincount(task.y)
    r['imbalance'] = cnt.max() / cnt[cnt > 0].min()
    X = task.X.copy()
    for c in X.select_dtypes(include='object').columns:
        X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    X = X.astype(float)
    X = X.loc[:, X.std() > 0]
    if len(X) > 5000:
        X = X.sample(5000, random_state=0)
    R = X.corr().abs().to_numpy()
    r['mean_abs_r'] = np.nanmean(offdiag(R)) if R.shape[0] > 1 else np.nan
    rows.append(r)

D = pd.DataFrame(rows).set_index('task')
P = pd.DataFrame({m: load(m) for m in ['cabin-v33-2slots', 'cabin-v33-moe4-shared', 'cabin-v33-contrastive']})
D['acc'] = P['cabin-v33-2slots']
D['moe_gain'] = P['cabin-v33-moe4-shared'] - P['cabin-v33-2slots']
D['con_gain'] = P['cabin-v33-contrastive'] - P['cabin-v33-2slots']
D.to_csv(os.path.join(HERE, 'dag_metrics.csv'))

print(f'{len(D)} tasks')
print('\n== Restored checkpoint (fraction of joint-phase DAG steps done at the restored best epoch)')
for v in ['base', 'contrastive', 'moe']:
    print(f"{v:12s} untrained (0 steps): {D[f'{v}_restored_untrained'].mean():.0%}   "
          f"<25% of joint: {(D[f'{v}_restored_frac'] < .25).mean():.0%}   median frac {D[f'{v}_restored_frac'].median():.2f}")
print('\n== Means (paper: base density .731±.300, 59% >0.8; contrastive density .773 asym .768; MoE density .911, cos .913±.024)')
for v in ['base', 'contrastive', 'moe']:
    print(f"{v:12s} density {D[f'{v}_density'].mean():.3f}±{D[f'{v}_density'].std():.3f}  "
          f">0.8: {(D[f'{v}_density'] > .8).mean():.0%}  asym {D[f'{v}_asymmetry'].mean():.3f}  active {D[f'{v}_active'].mean():.3f}")
print(f"moe cos {D.moe_cos.mean():.3f}±{D.moe_cos.std():.3f}")
tr = ~D.base_restored_untrained
print(f"\nbase, trained DAGs only (n={tr.sum()}): density {D.base_density[tr].mean():.3f}, >0.8 {(D.base_density[tr]>.8).mean():.0%}, asym {D.base_asymmetry[tr].mean():.3f}")
print(f"base, untrained DAGs (n={(~tr).sum()}): density {D.base_density[~tr].mean():.3f}, asym {D.base_asymmetry[~tr].mean():.3f}")
hi = D.d > 30
print(f"\nd>30 (n={hi.sum()}) vs d<=30: density {D.base_density[hi].mean():.2f} vs {D.base_density[~hi].mean():.2f}; "
      f"asym {D.base_asymmetry[hi].mean():.2f} vs {D.base_asymmetry[~hi].mean():.2f}  (paper .66 vs .92; .55 vs .32)")
print(f"\nMoE: Pearson(moe_cos, moe_gain) r={pearsonr(D.moe_cos, D.moe_gain.fillna(0))[0]:.2f} (paper r=0.14 'diversity')"
      f"; Spearman(moe_gain, base_density) {spearmanr(D.moe_gain, D.base_density, nan_policy='omit')[0]:.2f} (paper .01)"
      f"; Spearman(con_gain, acc) {spearmanr(D.con_gain, D.acc, nan_policy='omit')[0]:.2f} (paper -.72)")

props = ['d', 'n', 'C', 'pct_cat', 'imbalance', 'mean_abs_r']
cols = ['base_density', 'contrastive_density', 'moe_density', 'moe_cos', 'acc', 'moe_gain', 'con_gain']
F = pd.DataFrame({c: [spearmanr(D[p], D[c], nan_policy='omit')[0] for p in props] for c in cols}, index=props)
F.to_csv(os.path.join(HERE, 'figure1_spearman.csv'))
print('\n== Figure 1 (Spearman), rerun')
print(F.round(2).to_string())

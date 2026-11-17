"""Track E analysis: estimated graph vs references, paired per task-fold."""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
H = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(H, '..')
HIGHD = ['mfeat-zernike', 'spambase', 'splice', 'mfeat-karhunen', 'analcatdata_authorship',
         'ozone-level-8hr', 'mfeat-fourier', 'MiceProtein', 'mfeat-factors', 'semeion']

def load(d, name):
    out = {}
    for f in glob.glob(f'{d}/*.json'):
        r = json.load(open(f))
        if r.get('status') == 'ok':
            out[(r['task'], r['fold'])] = r['accuracy']
    return pd.Series(out, name=name, dtype=float)

S = {
    'add_est': load(f'{H}/out/add_est', 'add_est'),
    'mlp_est': load(f'{H}/out/mlp_est', 'mlp_est'),
    # additive head refs: rerun_diag for the 28 tasks, this dir for high-d
    'add_nodag': pd.concat([load(f'{P}/rerun_diag/out/add_nodag', ''), load(f'{H}/out/add_nodag', '')]),
    'add_learn': load(f'{P}/rerun_diag/out/add', 'add_learn'),
    'add_rand': load(f'{P}/rerun_diag/out/add_rand', 'add_rand'),
    # MLP head refs (paper config single net): diag_base on the 28, Track C ref on high-d
    'mlp_learn': pd.concat([load(f'{P}/rerun_diag/out/diag_base', ''), load(f'{P}/rerun_real_stg/out/ref', '')]),
    'mlp_nodag': load(f'{P}/rerun_real_stg/out/ref_nodag', 'mlp_nodag'),
}
D = pd.DataFrame(S)
D = D[~D.index.duplicated()]
D.index.names = ['task', 'fold']
R = pd.read_csv(f'{P}/rerun_interact/interaction_real.csv')
R['lgbm_gap'] = R.lgbm - R.lgbm_stumps
R = R.drop_duplicates(['task', 'fold']).set_index(['task', 'fold'])
D = D.join(R[['lgbm', 'lgbm_gap', 'd']], how='left')
dmeta = {}
for f in glob.glob(f'{H}/out/*/*.json'):
    r = json.load(open(f))
    if 'd' in r:
        dmeta[r['task']] = r['d']
D['d'] = [dmeta.get(t, np.nan) for t, _ in D.index]
gap = D.groupby('task').lgbm_gap.transform('mean')
D['gap_task'] = gap

def cmp(a, b, mask=None, label=''):
    x = (D[a] - D[b]) if mask is None else (D[a] - D[b])[mask]
    x = x.dropna()
    if len(x) == 0:
        return f'{label}: n=0'
    p = wilcoxon(x).pvalue if (x != 0).any() else 1.0
    return f'{label:8s} {100*x.mean():+.2f} ({(x>0).sum()}/{(x<0).sum()}/{len(x)}, p={p:.3f})'

hd = D.index.get_level_values('task').isin(HIGHD)
pairs = [('add_est', 'add_nodag'), ('add_est', 'add_learn'), ('add_est', 'add_rand'),
         ('mlp_est', 'mlp_learn'), ('mlp_est', 'mlp_nodag'), ('mlp_est', 'add_est'), ('mlp_est', 'lgbm'), ('add_est', 'lgbm')]
print('Paired diff in pts (wins/losses/N, Wilcoxon p)')
for a, b in pairs:
    print(f'{a} - {b}:'.ljust(26), ' | '.join([cmp(a, b, ~hd, '28 tasks'), cmp(a, b, hd, 'high-d'), cmp(a, b, None, 'pooled')]))
print('\nBreakdown add_est - add_nodag and mlp_est - mlp_learn:')
for lab, m in [('gap>1pt', D.gap_task > 0.01), ('gap<=1pt', D.gap_task <= 0.01), ('d>=30', D.d >= 30), ('d<30', D.d < 30)]:
    print(f'  {lab:9s}', cmp('add_est', 'add_nodag', m, 'add'), '|', cmp('mlp_est', 'mlp_learn', m, 'mlp'))
T = (D.groupby('task').mean() * 100).round(2)
T['est-nodag(add)'] = T.add_est - T.add_nodag
T['est-learn(mlp)'] = T.mlp_est - T.mlp_learn
T['est-nodag(mlp)'] = T.mlp_est - T.mlp_nodag
print('\nPer task (mean over folds, %):')
print(T[['d', 'gap_task', 'add_est', 'add_nodag', 'est-nodag(add)', 'mlp_est', 'mlp_learn', 'est-learn(mlp)', 'est-nodag(mlp)', 'lgbm']]
      .assign(d=T.d / 100).sort_values('gap_task', ascending=False).to_string())

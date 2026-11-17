"""Paired tables: nodag/randdag vs learned DAG, M=5 ensemble, 28 tasks + high-d tasks."""
import glob, json, os
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
SENS = os.path.join(HERE, '..', 'rerun_sensitivity', 'out')
CFG = ['base_ens5', 'nodag_ens5', 'randdag_ens5']
rows = []
for src, grp in ((SENS, '28'), (os.path.join(HERE, 'out'), 'highd')):
    for c in CFG:
        for f in glob.glob(f'{src}/{c}/*.json'):
            r = json.load(open(f))
            if r.get('status') == 'ok':
                rows.append(dict(group=grp, config=c, task=r['task'], fold=r['fold'], d=r['d'], acc=r['accuracy']))
D = pd.DataFrame(rows)
P = D.pivot_table(index=['group', 'task', 'fold'], columns='config', values='acc')
dmap = D.groupby('task').d.first()
lg = pd.read_csv(os.path.join(HERE, '..', 'rerun_interact', 'interaction_real.csv'))[['task', 'fold', 'lgbm']]
lg = pd.concat([lg, pd.read_csv(os.path.join(HERE, 'lgbm.csv'))[['task', 'fold', 'lgbm']]]).drop_duplicates(['task', 'fold'], keep='last')
P = P.join(lg.set_index(['task', 'fold']), on=['task', 'fold'])

def cmp(x, a, b):
    s = (x[a] - x[b]).dropna()
    if len(s) == 0:
        return 'n/a'
    p = wilcoxon(s).pvalue if (s != 0).any() else 1.0
    return f'{100*s.mean():+.2f} ({(s>0).sum()}/{(s<0).sum()}/{len(s)}, p={p:.3f})'

sets = {'28 tasks': P.xs('28', level='group'),
        'high-d (10)': P.xs('highd', level='group').drop(index='cnae-9', level='task', errors='ignore'),
        'high-d + cnae-9': P.xs('highd', level='group'),
        'pooled (all)': P.droplevel('group')}
print('Paired diff in pts (wins/losses/N, Wilcoxon) over task-folds')
print(pd.DataFrame({k: {'nodag-base': cmp(v, 'nodag_ens5', 'base_ens5'), 'rand-base': cmp(v, 'randdag_ens5', 'base_ens5'),
                        'nodag-rand': cmp(v, 'nodag_ens5', 'randdag_ens5'), 'base-lgbm': cmp(v, 'base_ens5', 'lgbm'),
                        'nodag-lgbm': cmp(v, 'nodag_ens5', 'lgbm')} for k, v in sets.items()}).to_string())
T = P.droplevel('group').groupby('task').mean()
T['d'] = dmap
T['nodag-base'] = T.nodag_ens5 - T.base_ens5
T['rand-base'] = T.randdag_ens5 - T.base_ens5
for c in ('nodag-base', 'rand-base'):
    s = T[[c, 'd']].dropna()
    rho, p = spearmanr(s.d, s[c])
    print(f'Spearman({c}, d) = {rho:+.2f} (p={p:.3f}, {len(s)} tasks)')
print('\nPer task (task means over folds, acc; diffs in pts):')
out = T[['d', 'base_ens5', 'nodag_ens5', 'randdag_ens5', 'lgbm', 'nodag-base', 'rand-base']].sort_values('d')
out[['nodag-base', 'rand-base']] *= 100
print(out.round(3).to_string())

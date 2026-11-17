"""Ablation numbers (design study) from the frozen experiment CSVs.

Each CSV row is one (config, dataset, fold). Some (config, dataset, fold)
were run more than once; duplicates are averaged. A dataset counts for a
config only with all 5 folds. Comparisons use the datasets common to all
configs being compared.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

BK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
# folder names used when the paper numbers were generated -> folders of this repo
ALIAS = {'04_D3_acyclicity': '04_acyclicity_constraint', '08_cc18_baselines': '08_baselines'}


def load(sub):
    d = pd.read_csv(os.path.join(BK, ALIAS.get(sub, sub), 'results.csv'), on_bad_lines='skip', low_memory=False)
    d = d[d.config != 'config']
    d['accuracy'] = pd.to_numeric(d.accuracy, errors='coerce')
    d['time'] = pd.to_numeric(d.time, errors='coerce')
    d['fold'] = pd.to_numeric(d.fold, errors='coerce')
    d = d.dropna(subset=['accuracy'])
    g = d.groupby(['config', 'dataset', 'fold'], as_index=False)[['accuracy', 'time']].mean()
    full = g.groupby(['config', 'dataset']).fold.nunique()
    keep = full[full == 5].index
    g = g.set_index(['config', 'dataset']).loc[keep].reset_index()
    return g


def compare(sub, configs, ref=None):
    """Mean acc, mean rank, wins vs ref, rel. time on the common datasets."""
    g = load(sub)
    g = g[g.config.isin(configs)]
    acc = g.pivot_table(index='dataset', columns='config', values='accuracy').dropna()[configs]
    tim = g.pivot_table(index='dataset', columns='config', values='time').loc[acc.index][configs]
    ranks = acc.apply(lambda r: pd.Series(rankdata(-r.values), index=r.index), axis=1)
    out = pd.DataFrame({'acc': acc.mean(), 'rank': ranks.mean(),
                        'time_rel': tim.mean() / tim[ref or configs[-1]].mean()})
    if ref:
        out['delta_pts'] = (acc.mean() - acc[ref].mean()) * 100
        out['wins_vs_ref'] = [(acc[c] > acc[ref]).sum() for c in configs]
        out['p_vs_ref'] = [np.nan if c == ref else wilcoxon(acc[c], acc[ref]).pvalue for c in configs]
    out.attrs['n'] = len(acc)
    return out, acc


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    for name, sub, cfgs, ref in [
        ('D1 DAG learning', '02_D1_dag_learning', ['cum_castle', 'dim1_golem', 'cum0_base'], 'cum0_base'),
        ('D2 parameterization', '03_D2_factored_adjacency', ['cum0_base', 'dim2_block', 'dim2_latent32', 'cum1_factored'], 'cum1_factored'),
        ('D3 acyclicity (all 4)', '04_D3_acyclicity', ['cum1_factored', 'dim3_dagma', 'dim3_dpdag', 'dim3_vcuda'], 'dim3_vcuda'),
        ('D3 acyclicity (no DAGMA)', '04_D3_acyclicity', ['cum1_factored', 'dim3_dpdag', 'dim3_vcuda'], 'dim3_vcuda'),
        ('D4 message passing', '05_D4_message_passing', ['dim4_multi2', 'dim4_multi3', 'cum1_factored'], 'cum1_factored'),
        ('D6 ensemble', '06_D6_ensemble', ['cum2_reg', 'cum3_boost', 'cum4_reweight', 'cum5_causalboost'], 'cum5_causalboost'),
        ('Cumulative', '01_cumulative_ablation', ['cum_castle', 'cum0_base', 'cum1_factored', 'cum2_reg', 'cum3_boost', 'cum4_reweight', 'cum5_causalboost'], None),
    ]:
        t, acc = compare(sub, cfgs, ref)
        print(f'== {name}  (n={t.attrs["n"]} datasets)')
        print(t.round(4).to_string())
        if name == 'Cumulative':
            prev = None
            for c in cfgs:
                if prev:
                    dlt = (acc[c].mean() - acc[prev].mean()) * 100
                    print(f'  {prev:>18s} -> {c:18s} delta {dlt:+.2f}  wins {(acc[c] > acc[prev]).sum()}/{len(acc)}  p={wilcoxon(acc[c], acc[prev]).pvalue:.3g}')
                prev = c
            print('  total', round((acc[cfgs[-1]].mean() - acc[cfgs[0]].mean()) * 100, 2))
            print('  median time per fold (s):', load(sub).groupby('config').time.median().round(0).to_dict())

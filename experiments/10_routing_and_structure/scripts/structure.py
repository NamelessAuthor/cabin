"""When does the feature graph help? Numbers, tables and figure (section 'sec:when').

Called from make_numbers.py with its macro/table writers. Read-only sources:
  synthetic, low d   ../../rerun_interact/out          d 10/20, n 2000, lam 0-1
                     ../../rerun_interact/out_n10000   same SCM, n 10000, lam 0/1
  synthetic, high d  ../../rerun_interact/out_highd    d 50/100, n 500/1000, lam 0/1
                     ../../rerun_dagopt/out            estimated graph (n 1000)
                     ../../rerun_headopt/out           stochastic gates (STG)
  real               ../../rerun_sensitivity/out       28 tasks: randdag, *_ens5
                     ../../rerun_real_stg/out          38 tasks: ref, ref_nodag, stg*
                     ../../rerun_real_estgraph/out     38 tasks: mlp_est, add_est, mlp_est_ens5
                     ../../rerun_highd_real/out        11 high-d tasks: *_ens5
                     ../../rerun_diag/out              28 tasks: additive head, acc_W0
                     ../../rerun_interact/interaction_real.csv  LightGBM / stumps, 55 tasks
All runs: seed 42 (synthetic: per-setting seed), clock pinned, restore from all
phases (the paper configuration) unless the directory says otherwise.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

WARMUP = 20  # epochs before the DAG trains


def _wil(d):
    d = np.asarray(d, float)
    return wilcoxon(d).pvalue if (d != 0).sum() > 0 else 1.0


def _stars(p):
    return '$^{**}$' if p < 0.01 else '$^{*}$' if p < 0.05 else ''


class Ctx:
    def __init__(self, paper, mac, write, f3, pts, pval):
        self.paper, self.mac, self.write, self.f3, self.pts, self.pval = paper, mac, write, f3, pts, pval

    def path(self, *p):
        return os.path.join(self.paper, *p)


# ----------------------------------------------------------------------------
# Synthetic
# ----------------------------------------------------------------------------
SYN_DIRS = [('rerun_interact/out', 'low', 2000), ('rerun_interact/out_n10000', 'low', 10000),
            ('rerun_interact/out_highd', 'high', None), ('rerun_dagopt/out', 'high', None),
            ('rerun_headopt/out', 'high', None)]
# paper-restore MLP-head variants, renamed to graph names
SYN_LOW = {'mlp_nodag_all': 'none', 'mlp_rand_all': 'random', 'cabin': 'learned', 'mlp_true_all': 'true',
           'add_nodag_all': 'add_none', 'add_rand_all': 'add_random', 'add_true_all': 'add_true'}
SYN_HIGH = {'mlp_nodag': 'none', 'mlp_rand': 'random', 'cabin': 'learned', 'mlp_true': 'true',
            'mlp_est_lgbm3': 'estimated', 'add_nodag': 'add_none', 'add_rand': 'add_random',
            'add_true': 'add_true', 'add_est_lgbm3': 'add_estimated', 'stgsum_mlp__nodag': 'stg_none',
            'stgsum_mlp__true': 'stg_true', 'mlp_rand_true': 'rand_true', 'mlp_rand_est': 'rand_est',
            'add_rand_true': 'add_rand_true', 'add_rand_est': 'add_rand_est'}
BASE = {'lgbm': 'lgbm', 'lgbm_stumps': 'stumps', 'logreg': 'logreg', 'lgbm_mb': 'lgbm_mb'}


def load_synthetic(c):
    rows = []
    for rel, regime, n_default in SYN_DIRS:
        names = SYN_LOW if regime == 'low' else SYN_HIGH
        for f in glob.glob(c.path(rel, '*', '*.json')):
            r = json.load(open(f))
            if r.get('status', 'ok') != 'ok':
                continue
            m = os.path.basename(os.path.dirname(f))
            key = dict(regime=regime, d=r['d'], n=r.get('n', n_default), lam=r['lam'], rep=r['rep'])
            if m == 'baselines':
                for b, nm in BASE.items():
                    if b in r:
                        rows.append({**key, 'model': nm, 'acc': r[b]})
            elif m in names:
                rows.append({**key, 'model': names[m], 'acc': r['acc'], 'best_epoch': r.get('best_epoch')})
    S = pd.DataFrame(rows).drop_duplicates(['regime', 'd', 'n', 'lam', 'rep', 'model'])
    return S


SYN_TABLE_ROWS = [  # (regime, n, lam, label)
    ('low', 2000, 0.0, r'$d\in\{10,20\}$, $n{=}2000$'),
    ('low', 2000, 1.0, None),
    ('low', 10000, 0.0, r'$d\in\{10,20\}$, $n{=}10^4$'),
    ('low', 10000, 1.0, None),
    ('high', 500, 0.0, r'$d\in\{50,100\}$, $n{=}500$'),
    ('high', 500, 1.0, None),
    ('high', 1000, 0.0, r'$d\in\{50,100\}$, $n{=}1000$'),
    ('high', 1000, 1.0, None),
]
SYN_COLS = ['none', 'random', 'learned', 'true', 'estimated', 'lgbm', 'lgbm_mb']
SYN_ADD_COLS = ['add_none', 'add_random', 'add_true', 'add_estimated', 'stumps']


def synthetic(c):
    S = load_synthetic(c)
    P = S.pivot_table(index=['regime', 'n', 'lam', 'd', 'rep'], columns='model', values='acc')

    def table(cols, best_of, name):
        rows = []
        for regime, n, lam, lab in SYN_TABLE_ROWS:
            if (regime, n, lam) not in P.groupby(level=[0, 1, 2]).groups:
                continue
            g = P.loc[(regime, n, lam)]
            means = {k: g[k].mean() for k in cols if k in g and g[k].notna().any()}
            if not any(k in means for k in best_of):
                continue
            best = max([means[k] for k in best_of if k in means], default=np.nan)
            cells = []
            for k in cols:
                if k not in means:
                    cells.append('--')
                else:
                    s = c.f3(means[k])
                    cells.append(r'\textbf{%s}' % s if k in best_of and np.isclose(means[k], best) else s)
            head = r'\multirow{2}{*}{%s}' % lab if lab else ''
            rows.append(f'{head} & {lam:g} & ' + ' & '.join(cells) + r' \\')
            if lab is None:
                rows.append(r'\midrule')
        c.write(name, '\n'.join(rows[:-1]) + '\n')

    table(SYN_COLS, ['none', 'random', 'learned', 'true', 'estimated', 'lgbm'], 'table_struct_synth.tex')
    table(SYN_ADD_COLS, ['add_none', 'add_random', 'add_true', 'add_estimated'], 'table_struct_synth_add.tex')

    def diff(a, b, sel, tag):
        g = P.loc[sel] if not isinstance(sel, list) else pd.concat([P.loc[s] for s in sel])
        # unit of analysis: the random graph (d, rep); its n and lambda settings share the SCM
        x = ((g[a] - g[b]).dropna() * 100).groupby(level=[0, 1]).mean()
        p = _wil(x)
        c.mac(f'Syn{tag}', c.pts(x.mean()))
        c.mac(f'Syn{tag}Abs', f'{abs(x.mean()):.2f}')
        c.mac(f'Syn{tag}P', c.pval(p))
        c.mac(f'Syn{tag}WL', f'{(x > 0).sum()}/{(x < 0).sum()}')
        c.mac(f'Syn{tag}N', str(len(x)))
        return x.mean(), p

    # Low d: identity of the graph does not matter
    low = [('low', 2000, l) for l in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for a, b, tag in [('true', 'none', 'LowTrueNone'), ('true', 'random', 'LowTrueRand'),
                      ('learned', 'none', 'LowLearnNone'), ('random', 'none', 'LowRandNone')]:
        diff(a, b, low, tag)
    # max |mean difference| between the four graph variants over all lambda (n 2000)
    spread = []
    for _, _, l in low:
        g = P.loc[('low', 2000, l)][['none', 'random', 'learned', 'true']].mean() * 100
        spread.append(g.max() - g.min())
    c.mac('SynLowSpread', f'{max(spread):.1f}')
    c.mac('SynLowNSettings', str(len(P.loc[('low', 2000, 0.0)])))
    big = [('low', 10000, 1.0)]
    diff('add_random', 'add_none', big, 'BigAddRandNone')
    diff('add_true', 'add_random', big, 'BigAddTrueRand')
    diff('true', 'random', big, 'BigTrueRand')
    diff('lgbm', 'stumps', [('low', 2000, 1.0)], 'LowLgbStumpsLamOne')
    # High d
    hi = [('high', n, l) for n in (500, 1000) for l in (0.0, 1.0)]
    diff('random', 'none', hi, 'HighRandNone')
    diff('true', 'random', hi, 'HighTrueRand')
    diff('learned', 'random', hi, 'HighLearnRand')
    diff('learned', 'none', hi, 'HighLearnNone')
    diff('true', 'none', [('high', n, 0.0) for n in (500, 1000)], 'HighTrueNoneLamZero')
    diff('true', 'none', [('high', n, 1.0) for n in (500, 1000)], 'HighTrueNoneLamOne')
    diff('add_true', 'add_none', [('high', n, 1.0) for n in (500, 1000)], 'HighAddTrueNoneLamOne')
    diff('estimated', 'none', [('high', n, 1.0) for n in (500, 1000)], 'HighEstNoneLamOne')
    diff('estimated', 'none', [('high', n, 0.0) for n in (500, 1000)], 'HighEstNoneLamZero')
    diff('estimated', 'random', hi, 'HighEstRand')
    diff('estimated', 'learned', hi, 'HighEstLearn')
    diff('true', 'estimated', hi, 'HighTrueEst')
    diff('lgbm', 'estimated', [('high', n, 1.0) for n in (500, 1000)], 'HighLgbEstLamOne')
    diff('add_estimated', 'add_none', [('high', n, 1.0) for n in (500, 1000)], 'HighAddEstNoneLamOne')
    diff('add_estimated', 'add_true', [('high', n, 0.0) for n in (500, 1000)], 'HighAddEstTrueLamZero')
    diff('lgbm', 'true', [('high', n, 1.0) for n in (500, 1000)], 'HighLgbTrueLamOne')
    diff('lgbm', 'learned', [('high', n, 1.0) for n in (500, 1000)], 'HighLgbLearnLamOne')
    diff('lgbm_mb', 'lgbm', hi, 'HighMbLgb')
    diff('lgbm_mb', 'lgbm', [('high', n, 1.0) for n in (500, 1000)], 'HighMbLgbLamOne')
    diff('stg_none', 'lgbm', [('high', 1000, 1.0)], 'HighStgLgbNThousandLamOne')
    diff('stg_none', 'lgbm', [('high', 500, 1.0)], 'HighStgLgbNFiveHundredLamOne')
    diff('stg_none', 'lgbm', [('high', n, 0.0) for n in (500, 1000)], 'HighStgLgbLamZero')
    diff('stg_true', 'stg_none', hi, 'HighStgTrueNone')
    # density-matched random controls (sparse random graphs with the edge count of the true / estimated graph)
    if 'rand_true' in P and P['rand_true'].notna().any():
        lam1 = [('high', n, 1.0) for n in (500, 1000)]
        lam0 = [('high', n, 0.0) for n in (500, 1000)]
        for a, b, tag in [('true', 'rand_true', 'TrueRandTrue'), ('estimated', 'rand_est', 'EstRandEst'),
                          ('rand_true', 'none', 'RandTrueNone'), ('rand_est', 'none', 'RandEstNone'),
                          ('rand_true', 'random', 'RandTrueRandom'), ('rand_est', 'random', 'RandEstRandom')]:
            diff(a, b, hi, 'High' + tag)
            diff(a, b, lam1, 'High' + tag + 'LamOne')
            diff(a, b, lam0, 'High' + tag + 'LamZero')
        if 'add_rand_true' in P:
            for a, b, tag in [('add_true', 'add_rand_true', 'AddTrueRandTrue'),
                              ('add_estimated', 'add_rand_est', 'AddEstRandEst'),
                              ('add_rand_true', 'add_none', 'AddRandTrueNone'),
                              ('add_rand_est', 'add_none', 'AddRandEstNone')]:
                diff(a, b, lam1, 'High' + tag + 'LamOne')
                diff(a, b, hi, 'High' + tag)
        n_ctrl = P.loc[hi[0]]['rand_true'].notna().sum() if 'rand_true' in P else 0
        c.mac('SynCtrlN', str(int(P[['rand_true']].dropna().groupby(level=[3, 4]).ngroups)))
    # early collapse: learned-DAG runs whose restored checkpoint precedes DAG training
    L = S[(S.regime == 'high') & (S.model == 'learned')]
    c.mac('SynHighWarmupRestore', f'{(L.best_epoch < WARMUP).sum()}')
    c.mac('SynHighLearnedRuns', f'{len(L)}')
    c.mac('SynHighNSettings', str(len(P.loc[('high', 1000, 0.0)])))
    N = S[(S.regime == 'high') & (S.model == 'none')]
    c.mac('SynHighEarlyRestore', f'{(N.best_epoch <= 5).mean() * 100:.0f}')
    # skeleton AUC of the estimated graph, and end-to-end attempts (rerun_dagopt)
    auc = [json.load(open(f)).get('auc_skel') for f in glob.glob(c.path('rerun_dagopt', 'out', '*_est_lgbm3', '*.json'))]
    auc = [a for a in auc if a is not None]
    c.mac('SynEstAucSkel', f'{np.mean(auc):.2f}')
    E2E = [('add_true', r'additive head, true graph (reference)', None),
           ('add_feat_ft', r'feature-level ordered graph learned end to end, finetune restore', 'add'),
           ('add_skel_est_ft', r'factored ordered graph on the estimated skeleton, finetune restore', 'add'),
           ('add_skel_est_nw_ft', r'same, no warmup (graph trains from the start)', 'add'),
           ('add_skel_est_learn_ft', r'same, edges of the skeleton learned', 'add'),
           ('add_estimated', r'estimated graph, fixed (two-stage)', 'add'),
           ('true', r'MLP head, true graph (reference)', None),
           ('mlp_skel_est_ft', r'MLP head, factored ordered graph on the estimated skeleton', 'mlp'),
           ('estimated', r'MLP head, estimated graph, fixed (two-stage)', 'mlp')]
    raw = {}
    for f in glob.glob(c.path('rerun_dagopt', 'out', '*', '*.json')):
        r = json.load(open(f))
        if r.get('status', 'ok') == 'ok':
            raw[(os.path.basename(os.path.dirname(f)), r['d'], r['n'], r['lam'], r['rep'])] = r['acc']
    Q = P.loc[('high', 1000)].copy()
    for m in ('add_feat_ft', 'add_skel_est_ft', 'add_skel_est_nw_ft', 'add_skel_est_learn_ft', 'mlp_skel_est_ft'):
        Q[m] = [raw.get((m, d, 1000, lam, rep), np.nan) for lam, d, rep in Q.index]
    rows = []
    for m, lab, ref in E2E:
        ref_col = {'add': 'add_true', 'mlp': 'true', None: None}[ref]
        cells = []
        for lam in (0.0, 1.0):
            g = Q.loc[lam]
            if m not in g or g[m].isna().all():
                cells += ['--', '--']
                continue
            if ref_col is None:
                cells += [c.f3(g[m].mean()), '--']
            else:
                pr = g[[m, ref_col]].dropna()
                cells += [c.f3(pr[m].mean()), c.pts((pr[m] - pr[ref_col]).mean() * 100)]
                if m == 'mlp_skel_est_ft':
                    tag = 'LamOne' if lam == 1.0 else 'LamZero'
                    c.mac(f'SynHighSkelEstTrue{tag}', c.pts((pr[m] - pr[ref_col]).mean() * 100))
                    c.mac(f'SynHighSkelEstTrue{tag}Abs', f'{abs((pr[m] - pr[ref_col]).mean() * 100):.2f}')
        rows.append(f'{lab} & ' + ' & '.join(cells) + r' \\')
        if m == 'add_estimated':
            rows.append(r'\midrule')
    c.write('table_struct_e2e.tex', '\n'.join(rows) + '\n')
    return P


# ----------------------------------------------------------------------------
# Real data
# ----------------------------------------------------------------------------
def load_real(c, *dirs, drop=('cnae-9',)):
    r = {}
    extra = {}
    for d in dirs:
        for f in glob.glob(c.path(d, '*.json')):
            j = json.load(open(f))
            if j.get('status') != 'ok' or j['task'] in drop:
                continue
            r[(j['task'], int(j['fold']))] = j['accuracy']
            extra[(j['task'], int(j['fold']))] = j
    s = pd.Series(r, dtype=float)
    s.index.names = ['task', 'fold']
    return s.sort_index(), extra


def real(c):
    LOW = sorted({k[0] for k in load_real(c, 'rerun_sensitivity/out/base')[0].index})
    ref, _ = load_real(c, 'rerun_real_stg/out/ref')
    HIGH = sorted(set(ref.index.get_level_values(0)) - set(LOW))
    I = pd.read_csv(c.path('rerun_interact', 'interaction_real.csv')).set_index(['task', 'fold'])
    lgb = I.lgbm

    single = dict(learned=ref, none=load_real(c, 'rerun_real_stg/out/ref_nodag')[0],
                  random=load_real(c, 'rerun_sensitivity/out/randdag')[0],
                  estimated=load_real(c, 'rerun_real_estgraph/out/mlp_est')[0])
    ens = {k: load_real(c, f'rerun_sensitivity/out/{v}', f'rerun_highd_real/out/{v}')[0]
           for k, v in dict(learned='base_ens5', none='nodag_ens5', random='randdag_ens5').items()}
    ens['estimated'] = load_real(c, 'rerun_real_estgraph/out/mlp_est_ens5')[0]
    # density-matched sparse random graph (edge count of the estimated graph)
    single['sparse'] = load_real(c, 'rerun_real_estgraph/out/mlp_randsparse')[0]
    ens['sparse'] = load_real(c, 'rerun_real_estgraph/out/mlp_randsparse_ens5')[0]

    def sub(s, tasks):
        return s[s.index.get_level_values(0).isin(tasks)]

    def complete(s, tasks):  # only report a variant once every dataset of the subset has run
        return set(sub(s, tasks).index.get_level_values(0)) >= set(tasks) and len(sub(s, tasks)) > 0

    def diff(a, b, tag=None):
        k = a.index.intersection(b.index)
        # unit of analysis: the dataset (mean over its folds); folds are not independent
        x = ((a[k] - b[k]) * 100).groupby(level=0).mean()
        p = _wil(x)
        if tag:
            c.mac(f'Real{tag}', c.pts(x.mean()))
            c.mac(f'Real{tag}Abs', f'{abs(x.mean()):.2f}')
            c.mac(f'Real{tag}P', c.pval(p))
            c.mac(f'Real{tag}WL', f'{(x > 0).sum()}/{(x < 0).sum()}')
            c.mac(f'Real{tag}N', str(len(x)))
        return x.mean(), p, len(x), (x > 0).sum(), (x < 0).sum()

    rows = []
    fig = {}
    for setlab, tasks, tagset in [(r'%d low-$d$' % len(LOW), LOW, 'Low'), (r'%d high-$d$' % len(HIGH), HIGH, 'High')]:
        for mlab, D, tagm in [('single', single, 'Single'), (r'$M{=}5$', ens, 'Ens')]:
            L = sub(D['learned'], tasks)
            cells = [c.f3(L.mean())]
            for k in ('random', 'sparse', 'none', 'estimated'):
                if not complete(D[k], tasks):
                    cells.append('--')
                    continue
                m, p, *_ = diff(sub(D[k], tasks), L, f'{tagset}{tagm}{k.capitalize()}Learned')
                cells.append(c.pts(m) + _stars(p))
            # columns: estimated - sparse random, estimated - none
            if len(sub(D['random'], tasks)):
                diff(sub(D['estimated'], tasks), sub(D['random'], tasks), f'{tagset}{tagm}EstimatedRandom')
                diff(sub(D['random'], tasks), sub(D['none'], tasks), f'{tagset}{tagm}RandomNone')
            if complete(D['sparse'], tasks):
                m, p, *_ = diff(sub(D['estimated'], tasks), sub(D['sparse'], tasks), f'{tagset}{tagm}EstimatedSparse')
                cells.append(c.pts(m) + _stars(p))
                diff(sub(D['sparse'], tasks), sub(D['none'], tasks), f'{tagset}{tagm}SparseNone')
                if len(sub(D['random'], tasks)):
                    diff(sub(D['sparse'], tasks), sub(D['random'], tasks), f'{tagset}{tagm}SparseRandom')
            else:
                cells.append('--')
            m, p, *_ = diff(sub(D['estimated'], tasks), sub(D['none'], tasks), f'{tagset}{tagm}EstimatedNone')
            cells.append(c.pts(m) + _stars(p))
            diff(sub(D['none'], tasks), sub(lgb, tasks), f'{tagset}{tagm}NoneLgb')
            diff(sub(D['estimated'], tasks), sub(lgb, tasks), f'{tagset}{tagm}EstimatedLgb')
            diff(L, sub(lgb, tasks), f'{tagset}{tagm}LearnedLgb')
            if tagm == 'Single':
                be = pd.Series({k: v.get('best_epoch', np.nan)
                                for k, v in load_real(c, 'rerun_real_stg/out/ref')[1].items() if k[0] in tasks})
                c.mac(f'{tagset}SingleWarmupRestore', f'{(be < WARMUP).mean() * 100:.0f}')
            head = r'\multirow{2}{*}{%s}' % setlab if tagm == 'Single' else ''
            rows.append(f'{head} & {mlab} & ' + ' & '.join(cells) + r' \\')
            # figure: differences to no graph, with paired bootstrap CI
            for k in ('random', 'learned', 'estimated'):
                if len(sub(D[k], tasks)):
                    a, b = sub(D[k], tasks), sub(D['none'], tasks)
                    kk = a.index.intersection(b.index)
                    fig[(tagset, tagm, k)] = ((a[kk] - b[kk]) * 100).groupby(level=0).mean().values
        rows.append(r'\midrule')
    c.write('table_struct_real.tex', '\n'.join(rows[:-1]) + '\n')
    c.mac('RealNLow', str(len(LOW)))
    c.mac('RealNHigh', str(len(HIGH)))
    dims = {t: e['d'] for t, e in load_real(c, 'rerun_real_stg/out/ref')[1].items() for t in [t[0]]}
    c.mac('RealHighDMin', str(min(dims[t] for t in HIGH)))
    c.mac('RealHighDMax', str(max(dims[t] for t in HIGH)))
    c.mac('RealLowDMax', str(max(dims[t] for t in LOW)))
    c.mac('RealHighTasks', ', '.join(r'\emph{%s}' % t.replace('_', r'\_') for t in HIGH))

    # pooled estimated graph, single and ensemble
    diff(single['estimated'], single['learned'], 'PoolSingleEstLearned')
    diff(single['estimated'], single['none'], 'PoolSingleEstNone')
    diff(ens['estimated'], ens['learned'], 'PoolEnsEstLearned')
    diff(ens['none'], ens['learned'], 'PoolEnsNoneLearned')
    diff(ens['random'], ens['learned'], 'PoolEnsRandomLearned')
    c.mac('RealPoolN', str(ens['learned'].index.get_level_values(0).nunique()))
    # estimated graph statistics
    _, ex = load_real(c, 'rerun_real_estgraph/out/mlp_est')
    deg = pd.Series({k: v['mean_degree'] for k, v in ex.items()})
    c.mac('RealEstDegree', f'{deg.mean():.1f}')
    gs = pd.Series({k: v.get('graph_seconds', np.nan) for k, v in ex.items()})
    c.mac('RealEstSecondsMedian', f'{gs.median():.1f}')
    c.mac('RealEstSecondsMax', f'{gs.max():.0f}')

    # interaction gap (LightGBM minus additive LightGBM stumps)
    gap = ((I.lgbm - I.lgbm_stumps) * 100).groupby(level=0).mean()
    common = [l.strip() for l in open(c.path('..', '..', 'paper', 'generated', 'cc18_common.txt')) if l.strip()]
    gap = gap[gap.index.isin(common)]  # the CC18 datasets of Table 1
    c.mac('IntNTasks', str(len(gap)))
    c.mac('IntNGapOne', str(int((gap > 1).sum())))
    c.mac('IntNGapTwo', str(int((gap > 2).sum())))
    c.mac('IntMedian', f'{gap.median():.2f}')
    c.mac('IntNNeg', str(int((gap < 0).sum())))
    top = gap.sort_values(ascending=False)
    c.mac('IntTop', ', '.join(r'\emph{%s} (%+.1f)' % (t.replace('_', r'\_'), v) for t, v in top.head(4).items()))
    # additive head on the 28 tasks: messages carry interactions
    add = {k: load_real(c, v)[0] for k, v in dict(learned='rerun_diag/out/add', random='rerun_diag/out/add_rand',
                                                     none='rerun_diag/out/add_nodag',
                                                     estimated='rerun_real_estgraph/out/add_est').items()}
    inter = set(gap[gap > 1].index)
    for k in ('learned', 'random', 'estimated'):
        diff(add[k], add['none'], f'Add{k.capitalize()}None')
        a = add[k][add[k].index.get_level_values(0).isin(inter)]
        diff(a, add['none'], f'AddInter{k.capitalize()}None')
        eff = ((add[k] - add['none']) * 100).dropna().groupby(level=0).mean()
        rho, p = spearmanr(eff, gap.reindex(eff.index))
        c.mac(f'Add{k.capitalize()}Rho', f'{rho:.2f}')
        c.mac(f'Add{k.capitalize()}RhoP', c.pval(p))
    diff(single['learned'], add['learned'], 'MlpMinusAddLearned')
    # same with finetune-only restore (the DAG is always trained)
    ft = {k: load_real(c, f'rerun_diag/out/{v}')[0] for k, v in dict(learned='add_ft', none='add_nodag_ft').items()}
    eff = ((ft['learned'] - ft['none']) * 100).dropna().groupby(level=0).mean()
    rho, p = spearmanr(eff, gap.reindex(eff.index))
    c.mac('AddFtRho', f'{rho:.2f}')
    c.mac('AddFtRhoP', c.pval(p))
    c.mac('AddNInter', str(len([t for t in LOW if t in inter])))
    eff = ((single['learned'] - single['none']) * 100)
    eff = eff[eff.index.get_level_values(0).isin(LOW)].groupby(level=0).mean()
    rho, p = spearmanr(eff, gap.reindex(eff.index))
    c.mac('MlpLearnedRho', f'{rho:.2f}')
    c.mac('MlpLearnedRhoP', c.pval(p))
    # messages are used: drop when W is zeroed at test time only
    _, dx = load_real(c, 'rerun_diag/out/diag_base')
    w0 = pd.Series({k: (v['accuracy'] - v['acc_W0']) * 100 for k, v in dx.items() if 'acc_W0' in v})
    c.mac('WZeroDrop', f'{w0.mean():.1f}')
    c.mac('WZeroDropMedian', f'{w0.median():.1f}')
    be = pd.Series({k: v.get('best_epoch', np.nan) for k, v in dx.items()})
    c.mac('RealWarmupRestore', f'{(be < WARMUP).mean() * 100:.0f}')

    # stochastic gates on real data
    stg = {k: load_real(c, f'rerun_real_stg/out/{k}')[0] for k in ('stg', 'stg_l003', 'stg_nodag')}
    diff(stg['stg'], single['learned'], 'StgLearned')
    diff(stg['stg_l003'], single['learned'], 'StgLThreeLearned')
    diff(stg['stg_nodag'], single['none'], 'StgNoDagNone')
    per_task = ((stg['stg'] - single['learned']) * 100).groupby(level=0).mean().sort_values()
    c.mac('StgWorst', ', '.join(r'\emph{%s}: %s' % (t.replace('_', r'\_'), c.pts(v).replace('0', '0', 1)) for t, v in per_task.head(2).items()))

    # cnae-9: message passing collapses
    cn = {}
    for k, d in dict(learned='base_ens5', random='randdag_ens5', none='nodag_ens5').items():
        s, _ = load_real(c, f'rerun_highd_real/out/{d}', drop=())
        cn[k] = s.loc['cnae-9']
    cn['lgbm'] = lgb.loc['cnae-9']
    for k, s in cn.items():
        c.mac(f'Cnae{k.capitalize()}', ' / '.join(c.f3(v) for v in s.values))
        c.mac(f'Cnae{k.capitalize()}Mean', c.f3(s.mean()))
    tub = json.load(open(c.path('results', 'tub', 'results', 'cabin-v33-2slots', 'cnae-9.json')))
    orig = [f['metrics']['accuracy'] for f in sorted(tub['fold_results'], key=lambda f: f['fold'])][:3]
    c.mac('CnaeOrig', ' / '.join(c.f3(v) for v in orig))
    c.mac('CnaeNCollapsed', str(int((cn['learned'] < 0.5).sum() + (cn['random'] < 0.5).sum())))
    return fig


# ----------------------------------------------------------------------------
# Figure: effect of the graph relative to no graph
# ----------------------------------------------------------------------------
COLORS = {'random': '#eb6834', 'learned': '#2a78d6', 'true': '#1baf7a', 'estimated': '#eda100'}
MARKERS = {'random': 'v', 'learned': 'o', 'true': 's', 'estimated': 'D'}
LABELS = {'random': 'dense random (frozen)', 'learned': 'default (Cabin)', 'true': 'true DAG (oracle)',
          'estimated': 'estimated (two-stage)'}


def _ci(x, rng):
    x = np.asarray(x)
    if len(x) < 2:
        return x.mean(), 0, 0
    b = rng.choice(x, (2000, len(x))).mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return x.mean(), x.mean() - lo, hi - x.mean()


def figure(c, P, real_fig):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.edgecolor': '#52514e',
                         'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.45, 1]))
    groups = [(('low', 2000, 0.0), 'low $d$\n\n$\\lambda{=}0$'), (('low', 2000, 1.0), 'low $d$\n\n$\\lambda{=}1$'),
              (('high', 500, 0.0), 'high $d$\n$n{=}500$\n$\\lambda{=}0$'),
              (('high', 500, 1.0), 'high $d$\n$n{=}500$\n$\\lambda{=}1$'),
              (('high', 1000, 0.0), 'high $d$\n$n{=}1000$\n$\\lambda{=}0$'),
              (('high', 1000, 1.0), 'high $d$\n$n{=}1000$\n$\\lambda{=}1$')]
    ax = axes[0]
    keys = ['random', 'learned', 'true', 'estimated']
    w = 0.18
    for gi, (sel, _) in enumerate(groups):
        g = P.loc[sel]
        for ki, k in enumerate(keys):
            if k not in g or g[k].isna().all():
                continue
            x = (g[k] - g['none']).dropna() * 100
            m, lo, hi = _ci(x, rng)
            ax.errorbar(gi + (ki - 1.5) * w, m, yerr=[[lo], [hi]], fmt=MARKERS[k], color=COLORS[k], ms=4.5,
                        elinewidth=1.2, capsize=0, mec='white', mew=0.6,
                        label=LABELS[k] if gi == 3 else None)
    ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([l for _, l in groups], fontsize=6.5)
    ax.set_ylabel('accuracy vs. no graph (points)')
    ax.set_title('(a) synthetic SCM, MLP head, single network', fontsize=8)
    ax.grid(axis='y', color='#e5e4e0', lw=0.6)
    ax.set_axisbelow(True)
    ax = axes[1]
    rgroups = [(('Low', 'Single'), 'low $d$\nsingle'), (('Low', 'Ens'), 'low $d$\n$M{=}5$'),
               (('High', 'Single'), 'high $d$\nsingle'), (('High', 'Ens'), 'high $d$\n$M{=}5$')]
    for gi, (sel, _) in enumerate(rgroups):
        for ki, k in enumerate(['random', 'learned', 'estimated']):
            if (sel + (k,)) not in real_fig:
                continue
            m, lo, hi = _ci(real_fig[sel + (k,)], rng)
            ax.errorbar(gi + (ki - 1) * w, m, yerr=[[lo], [hi]], fmt=MARKERS[k], color=COLORS[k], ms=4.5,
                        elinewidth=1.2, capsize=0, mec='white', mew=0.6)
    ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_xticks(range(len(rgroups)))
    ax.set_xticklabels([l for _, l in rgroups], fontsize=6.5)
    ax.set_title('(b) CC18 tasks, MLP head', fontsize=8)
    ax.grid(axis='y', color='#e5e4e0', lw=0.6)
    ax.set_axisbelow(True)
    for a in axes:
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'paper', 'generated', 'fig_structure.pdf'))
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'paper', 'generated', 'fig_structure.png'), dpi=160)
    plt.close(fig)


def run(paper, mac, write, f3, pts, pval):
    c = Ctx(paper, mac, write, f3, pts, pval)
    P = synthetic(c)
    rf = real(c)
    figure(c, P, rf)

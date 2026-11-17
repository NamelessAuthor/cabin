#!/usr/bin/env python3
"""Generate every number, table and figure of the CLeaR 2027 paper.

Nothing in the paper is typed by hand: the text uses the macros written to
generated/numbers.tex and \\input's the generated tables. Rerun this script
after any new result lands, then rebuild the PDF.

Sources (all read-only):
  tub        ../results/tub/results/<model>/<task>.json  (Table 2, LightGBM bins)
  bk         ../../NN_*/results.csv  (design study)
  tabarena   ../../09_tabarena/paper_results.csv
  synthetic  ../rerun_synthetic/out_k{auto,32}[_reps3-10]/summary.csv
  dag        ../rerun_dag/dag_metrics.csv  (from rerun_dag/analyze_dags.py)
  sens       ../rerun_sensitivity/out/<config>/<task>__f<fold>.json

Usage: python make_numbers.py
"""
import glob
import json
import re
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..'))
GEN = os.path.join(HERE, '..', '..', '..', 'paper', 'generated')
os.makedirs(GEN, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PAPER, 'results', 'tub'))
from ablations import compare, load as load_bk  # noqa: E402
from table2 import MODELS, load as load_tub  # noqa: E402

MACROS = {}


DIGITS = dict(zip('0123456789', ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']))


def mac(name, value):
    # LaTeX macro names are letters only: spell out digits, drop the rest.
    name = ''.join(DIGITS.get(ch, ch) for ch in name if ch.isalnum())
    MACROS[name] = value


def f3(x):           # .887
    return f'{x:.3f}'.lstrip('0') if x < 1 else f'{x:.3f}'


def f2(x):
    return f'{x:.2f}'


def pts(x, signed=True):  # accuracy points; a true minus sign in text and math
    s = f'{x:+.2f}' if signed else f'{x:.2f}'
    return s.replace('-', r'\ensuremath{-}')


def pval(p):
    if np.isnan(p):
        return '--'
    return '$<$0.001' if p < 1e-3 else f'{p:.3f}'


def write(name, text):
    with open(os.path.join(GEN, name), 'w') as fh:
        fh.write(text)


# ----------------------------------------------------------------------------
# Table 2: CC18 comparison (tub)
# ----------------------------------------------------------------------------
TYPES = {'TabICL': 'ICL', 'TabM': 'Neural', 'RealMLP': 'Neural', 'CatBoost': 'Tree',
         'XGBoost': 'Tree', 'LightGBM': 'Tree'}


def _cc18_table(P, fname, with_type):
    best = P.max(axis=1)
    is_best = P.eq(best, axis=0)
    strict = is_best & (is_best.sum(axis=1) == 1).values[:, None]
    ranks = P.apply(lambda r: pd.Series(rankdata(-r.values), index=r.index), axis=1)
    T = pd.DataFrame({'mean': P.mean(), 'std': P.std(), 'rank': ranks.mean(),
                      'wins': strict.sum()}).sort_values('mean', ascending=False)
    rows = []
    for name, r in T.iterrows():
        typ = TYPES.get(name, 'Ours')
        label = name + (r'$^\dagger$' if name == 'TabICL' else '')
        cells = [label] + ([typ] if with_type else []) + [f3(r["mean"]), f3(r["std"]), f'{r["rank"]:.2f}', str(int(r["wins"]))]
        rows.append(' & '.join(cells) + r' \\')
        if name == 'TabICL':
            rows.append(r'\midrule')
    write(fname, '\n'.join(rows) + '\n')
    return T, is_best


def table2():
    P = pd.DataFrame({name: load_tub(m) for name, m in MODELS.items()}).dropna()
    # Full table (all Cabin variants) for the appendix
    Tf, is_best_f = _cc18_table(P, 'table_cc18_full.tex', True)
    # Main table: Cabin only
    main_cols = [c for c in P.columns if c not in ('Cabin+MoE', 'Cabin+Contrastive', 'Cabin+Anneal')]
    T, is_best = _cc18_table(P[main_cols], 'table_cc18.tex', False)
    n = len(P)
    mac('NCCtasks', str(n))
    mac('NCCties', str(int((is_best.sum(axis=1) > 1).sum())))
    mac('NCCtiesFull', str(int((is_best_f.sum(axis=1) > 1).sum())))
    mac('NCCmethods', str(len(main_cols)))
    for name, key in [('TabICL', 'TabICL'), ('Cabin', 'Cabin'), ('Cabin+MoE', 'CabinMoE'),
                      ('Cabin+Contrastive', 'CabinCon'), ('Cabin+Anneal', 'CabinAnneal'),
                      ('TabM', 'TabM'), ('CatBoost', 'CatBoost'), ('RealMLP', 'RealMLP'),
                      ('XGBoost', 'XGBoost'), ('LightGBM', 'LightGBM')]:
        TT = T if name in T.index else Tf
        mac(f'CC{key}', f3(TT.loc[name, 'mean']))
        mac(f'CC{key}Wins', str(int(TT.loc[name, 'wins'])))
        mac(f'CC{key}Rank', f'{TT.loc[name, "rank"]:.2f}')
    # Paired tests Cabin vs others
    for other, key in [('TabM', 'TabM'), ('CatBoost', 'CatBoost'), ('LightGBM', 'LightGBM'),
                       ('TabICL', 'TabICL'), ('Cabin+MoE', 'CabinMoE'),
                       ('Cabin+Contrastive', 'CabinCon'), ('Cabin+Anneal', 'CabinAnneal')]:
        d = (P['Cabin'] - P[other]) * 100
        mac(f'CabinVs{key}Delta', pts(d.mean()))
        mac(f'CabinVs{key}P', pval(wilcoxon(P['Cabin'], P[other]).pvalue))
        mac(f'CabinVs{key}Better', str(int((d > 0).sum())))
        mac(f'CabinVs{key}Worse', str(int((d < 0).sum())))
    for v, key in [('Cabin+MoE', 'MoE'), ('Cabin+Contrastive', 'Con'), ('Cabin+Anneal', 'Anneal')]:
        mac(f'DSeven{key}Delta', pts((P[v] - P['Cabin']).mean() * 100))
    return P


# ----------------------------------------------------------------------------
# Cabin vs LightGBM by n and d (tub, Table 2 tasks)
# ----------------------------------------------------------------------------
def lightgbm_bins(P):
    meta = {}
    for t in P.index:
        with open(os.path.join(PAPER, 'results', 'tub', 'tasks', t, 'task.json')) as fh:
            m = json.load(fh)
        meta[t] = (m['n_samples'], m['n_features'])
    M = pd.DataFrame(meta, index=['n', 'd']).T
    diff = (P['Cabin'] - P['LightGBM']) * 100
    rows = []
    for var, edges, labels in [('n', [0, 1000, 5000, np.inf], ['$n<1{,}000$', r'$1{,}000\le n<5{,}000$', r'$n\ge 5{,}000$']),
                               ('d', [0, 20, 100, np.inf], ['$d<20$', r'$20\le d<100$', r'$d\ge 100$'])]:
        b = pd.cut(M[var], edges, right=False, labels=labels)
        for i, lab in enumerate(labels):
            s = diff[b == lab]
            rows.append(f'{lab} & {len(s)} & {pts(s.mean())} & {(s > 0).sum()}/{len(s)} \\\\')
            mac(f'LGB{var.upper()}{"abc"[i]}Delta', pts(s.mean()))
            mac(f'LGB{var.upper()}{"abc"[i]}Wins', f'{(s > 0).sum()}/{len(s)}')
        if var == 'n':
            rows.append(r'\midrule')
    write('table_lightgbm_bins.tex', '\n'.join(rows) + '\n')
    worst = diff.sort_values().head(2)
    for i, (t, v) in enumerate(worst.items()):
        mac(f'LGBWorst{"ab"[i]}Task', t.replace('_', r'\_'))
        mac(f'LGBWorst{"ab"[i]}Delta', pts(v))
    mac('CabinVsLGBRho', f2(spearmanr(np.log(M['n']), diff)[0]))


# ----------------------------------------------------------------------------
# Design study (bk CSVs)
# ----------------------------------------------------------------------------
def design_study():
    rows = []

    def block(title, sub, cfgs, ref, labels, notes):
        t, acc = compare(sub, cfgs, ref)
        n = t.attrs['n']
        first = True
        for c in cfgs:
            r = t.loc[c]
            lab = labels[c]
            if c == ref:
                lab = r'\textbf{' + lab + '}'
            dlt = '--' if c == ref else pts(r['delta_pts'])
            p = '--' if c == ref else pval(r['p_vs_ref'])
            head = f'\\multirow{{{len(cfgs)}}}{{*}}{{\\shortstack[l]{{{title}\\\\($n{{=}}{n}$)}}}}' if first else ''
            rows.append(f'{head} & {lab} & {f3(r["acc"])} & {dlt} & {r["rank"]:.2f} & {p} & {notes.get(c, "")} \\\\')
            first = False
        rows.append(r'\midrule')
        return t, n

    t, n = block('D1 DAG learning', '02_D1_dag_learning', ['cum_castle', 'dim1_golem', 'cum0_base'], 'cum0_base',
                 {'cum_castle': 'CASTLE (two-stage)', 'dim1_golem': 'GOLEM (two-stage)', 'cum0_base': 'SDCD (end-to-end)'},
                 {})
    mac('DOneN', str(n))
    for c, k in [('cum_castle', 'Castle'), ('dim1_golem', 'Golem'), ('cum0_base', 'SDCD')]:
        mac(f'DOne{k}Rank', f'{t.loc[c, "rank"]:.2f}')
        mac(f'DOne{k}Acc', f3(t.loc[c, 'acc']))
    mac('DOneDeltaCastle', pts(-t.loc['cum_castle', 'delta_pts']))
    mac('DOneCastleP', pval(t.loc['cum_castle', 'p_vs_ref']))

    t, n = block('D2 Adjacency', '03_D2_factored_adjacency', ['cum0_base', 'dim2_block', 'dim2_latent32', 'cum1_factored'], 'cum1_factored',
                 {'cum0_base': r'Full $W\in\mathbb{R}^{d\times d}$', 'dim2_block': 'Block-diagonal reg.',
                  'dim2_latent32': 'Latent nodes ($n_\\ell{=}32$)', 'cum1_factored': r'Factored $ASA^\top$, $k{=}32$'},
                 {})
    mac('DTwoN', str(n))
    for c, k in [('cum0_base', 'Full'), ('dim2_block', 'Block'), ('dim2_latent32', 'Latent'), ('cum1_factored', 'Fact')]:
        mac(f'DTwo{k}Rank', f'{t.loc[c, "rank"]:.2f}')
        mac(f'DTwo{k}Acc', f3(t.loc[c, 'acc']))
    mac('DTwoDeltaFull', pts(-t.loc['cum0_base', 'delta_pts']))
    mac('DTwoFullP', pval(t.loc['cum0_base', 'p_vs_ref']))

    t, n = block('D3 Acyclicity', '04_D3_acyclicity', ['cum1_factored', 'dim3_dpdag', 'dim3_vcuda'], 'dim3_vcuda',
                 {'cum1_factored': 'Spectral radius (SDCD)', 'dim3_dpdag': 'DP-DAG (SoftSort)', 'dim3_vcuda': 'VCUDA'},
                 {})
    mac('DThreeN', str(n))
    for c, k in [('cum1_factored', 'Spectral'), ('dim3_dpdag', 'DPDAG'), ('dim3_vcuda', 'VCUDA')]:
        mac(f'DThree{k}Acc', f3(t.loc[c, 'acc']))
    g = load_bk('04_D3_acyclicity')
    mac('DThreeDagmaN', str(g[g.config == 'dim3_dagma'].dataset.nunique()))
    t4, _ = compare('04_D3_acyclicity', ['cum1_factored', 'dim3_dagma', 'dim3_dpdag', 'dim3_vcuda'], 'dim3_vcuda')
    mac('DThreeDagmaAcc', f3(t4.loc['dim3_dagma', 'acc']))
    mac('DThreeDagmaVcudaAcc', f3(t4.loc['dim3_vcuda', 'acc']))

    t, n = block('D4 Message passing', '05_D4_message_passing', ['dim4_multi2', 'dim4_multi3', 'cum1_factored'], 'cum1_factored',
                 {'dim4_multi2': 'Two rounds ($K{=}2$)', 'dim4_multi3': 'Three rounds ($K{=}3$)', 'cum1_factored': 'One round ($K{=}1$)'},
                 {})
    mac('DFourN', str(n))
    mac('DFourKtwo', pts(t.loc['dim4_multi2', 'delta_pts']))
    mac('DFourKthree', pts(t.loc['dim4_multi3', 'delta_pts']))
    mac('DFourKtwoP', pval(t.loc['dim4_multi2', 'p_vs_ref']))

    t, n = block('D6 Ensemble', '06_D6_ensemble', ['cum2_reg', 'cum3_boost', 'cum4_reweight', 'cum5_causalboost'], 'cum5_causalboost',
                 {'cum2_reg': 'Single model', 'cum3_boost': 'Boosted ($M{=}5$, $\\eta{=}0.1$)',
                  'cum4_reweight': '+ adaptive $\\alpha$', 'cum5_causalboost': '+ temperature scaling'},
                 {})
    mac('DSixN', str(n))
    mac('DSixSingle', pts(t.loc['cum2_reg', 'delta_pts']))
    write('table_design.tex', '\n'.join(rows[:-1]) + '\n')

    # Cumulative ablation on the datasets common to all steps
    cfgs = ['cum_castle', 'cum0_base', 'cum1_factored', 'cum2_reg', 'cum3_boost', 'cum4_reweight', 'cum5_causalboost']
    names = ['CASTLE+CINN (two-stage)', '+ end-to-end SDCD', r'+ factored $W{=}ASA^\top$',
             '+ label smoothing, edge dropout', '+ boosting ($M{=}5$)', '+ adaptive reweighting', '+ temperature scaling']
    t, acc = compare('01_cumulative_ablation', cfgs, None)
    n = t.attrs['n']
    rows, prev, tot = [], None, 0.0
    for i, (c, nm) in enumerate(zip(cfgs, names)):
        a = acc[c]
        if prev is None:
            rows.append(f'{i} & {nm} & {f3(a.mean())} & -- & -- & -- & -- \\\\')
        else:
            d = (a.mean() - acc[prev].mean()) * 100
            tot += d
            p = wilcoxon(a, acc[prev]).pvalue
            rows.append(f'{i} & {nm} & {f3(a.mean())} & {pts(d)} & {pts(tot)} & {(a > acc[prev]).sum()}/{n} & {pval(p)} \\\\')
            mac(f'Cum{"ABCDEFG"[i]}Delta', pts(d))
            mac(f'Cum{"ABCDEFG"[i]}Wins', f'{(a > acc[prev]).sum()}/{n}')
            mac(f'Cum{"ABCDEFG"[i]}P', pval(p))
        prev = c
    write('table_cumulative.tex', '\n'.join(rows) + '\n')
    mac('CumN', str(n))
    mac('CumStart', f3(acc[cfgs[0]].mean()))
    mac('CumEnd', f3(acc[cfgs[-1]].mean()))
    mac('CumTotal', pts(tot))
    g = load_bk('01_cumulative_ablation')
    med = g.groupby('config').time.median()
    mac('TimeSingle', f'{med["cum2_reg"]:.0f}')
    mac('TimeEnsemble', f'{med["cum5_causalboost"]:.0f}')


# ----------------------------------------------------------------------------
# TabArena
# ----------------------------------------------------------------------------
def tabarena():
    T = pd.read_csv(os.path.join(PAPER, '..', '09_tabarena', 'paper_results.csv'))
    pre = {'TabICLv2', 'RealTabPFN', 'Mitra', 'TabICL', 'TabPFNv2', 'TabDPT'}
    rows = []
    for i, r in T.iterrows():
        nm = r.method + (r'$^\dagger$' if r.method in pre else '')
        if r.method.startswith('Cabin'):
            nm = r'\textbf{' + nm + '}'
        rows.append(f'{i + 1} & {nm} & {r.elo} & {r.avg_rank:.1f} & {r.winrate:.1f} \\\\')
    write('table_tabarena.tex', '\n'.join(rows) + '\n')
    get = lambda m, c: T.set_index('method').loc[m, c]
    mac('TAcabin', str(get('Cabin (k=32)', 'elo')))
    mac('TAcabinRank', str(T.index[T.method == 'Cabin (k=32)'][0] + 1))
    mac('TAautok', str(get('Cabin (auto-k)', 'elo')))
    mac('TAautokRank', str(T.index[T.method == 'Cabin (auto-k)'][0] + 1))
    mac('TAtabicl', str(get('TabICL', 'elo')))
    mac('TAcatboost', str(get('CatBoost', 'elo')))
    mac('TAN', str(len(T)))
    ds = set(pd.read_csv(os.path.join(PAPER, 'post-paper', 'compare', 'results_per_split.csv')).dataset.astype(str))
    mac('TAdatasets', str(len(ds)))
    mac('TAnotCC', str(len(ds - set(load_tub('xgboost').index))))


# ----------------------------------------------------------------------------
# Synthetic recovery (Table 5)
# ----------------------------------------------------------------------------
def synthetic():
    def load_k(k):
        fs = sorted(glob.glob(os.path.join(PAPER, 'rerun_synthetic', f'out_k{k}*', 'summary.csv')))
        return pd.concat([pd.read_csv(f) for f in fs]) if fs else None
    A, K = load_k('auto'), load_k('32')
    cols = ['cabin_acc', 'skel_F1_flat', 'skel_F1_grouped', 'dir_F1_flat', 'dir_F1_grouped']
    g = ['d', 'edge_p', 'nonlinear']
    ta = A.groupby(g)[cols].mean()
    nrep = A.groupby(g).size()
    rows = []
    for (d, p, nl), r in ta.iterrows():
        rows.append(f'{d} & {p} & {"nonlin." if nl else "lin."} & ' + ' & '.join(f2(r[c]) for c in cols) + ' \\\\')
    rows.append(r'\midrule')
    for lab, D in [('mean, auto-$k$', A), ('mean, $k{=}32$', K)]:
        m = D[cols].mean()
        rows.append(f'\\multicolumn{{3}}{{l}}{{{lab}}} & ' + ' & '.join(f2(m[c]) for c in cols) + ' \\\\')
    write('table_synthetic.tex', '\n'.join(rows) + '\n')
    mac('SynReps', str(int(nrep.min())))
    mac('SynRuns', str(len(A)))
    for lab, D in [('Auto', A), ('KTT', K)]:
        m = D[cols].mean()
        for c, k in zip(cols, ['Acc', 'SkelFlat', 'SkelGrp', 'DirFlat', 'DirGrp']):
            mac(f'Syn{lab}{k}', f2(m[c]))
    mac('SynAutoSkelGrpMinDfifty', f2(ta.loc[50].skel_F1_grouped.min()))
    mac('SynAutoSkelGrpMaxDten', f2(ta.loc[10].skel_F1_grouped.max()))
    mac('SynAutoDirGrpMin', f2(ta.dir_F1_grouped.min()))
    mac('SynAutoDirGrpMax', f2(ta.dir_F1_grouped.max()))
    # Chance baseline for the grouped scores: a random S thresholded at the same edge count
    # against the true DAG collapsed to the learned groups (collapsing makes the target dense).
    import importlib.util, pickle
    from scipy.stats import wilcoxon as _wx
    spec = importlib.util.spec_from_file_location('synb', os.path.join(PAPER, 'rerun_synthetic', 'run_synthetic_benchmark_v2.py'))
    sb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sb)
    arts = []
    for f in sorted(glob.glob(os.path.join(PAPER, 'rerun_synthetic', 'out_kauto*', 'artifacts.pkl'))):
        arts += pickle.load(open(f, 'rb'))
    rng = np.random.RandomState(0)
    rb = []
    for a_ in arts:
        k_ = a_['A_avg'].shape[1]
        G = sb.collapse_true_dag_to_groups(a_['M_true_features'], a_['group_assign'], k_)
        ne = int(G.sum())
        if ne == 0:
            continue
        fs, fd = [], []
        for _ in range(200):
            R_ = rng.rand(k_, k_); np.fill_diagonal(R_, -1)
            St = sb.threshold_topk(R_, ne)
            fs.append(sb.compute_skel_f1(St, G)); fd.append(sb.compute_directed_f1(St, G))
        rb.append((a_['skel_F1_grouped'], np.mean(fs), a_['dir_F1_grouped'], np.mean(fd)))
    rb = np.array(rb)
    mac('SynRandSkelGrp', f2(rb[:, 1].mean()))
    mac('SynRandDirGrp', f2(rb[:, 3].mean()))
    mac('SynLearnSkelGrpB', f2(rb[:, 0].mean()))
    mac('SynLearnDirGrpB', f2(rb[:, 2].mean()))
    mac('SynRandSkelPe', ('$p<0.001$' if _wx(rb[:, 0], rb[:, 1]).pvalue < 0.001 else f'$p={_wx(rb[:, 0], rb[:, 1]).pvalue:.3f}$'))
    mac('SynRandDirPe', f'$p={_wx(rb[:, 2], rb[:, 3]).pvalue:.2f}$')
    mac('SynRandN', str(len(rb)))
    cc = A.groupby(['d', 'edge_p']).frac_causally_connected.mean()
    mac('SynConnDfiftyPfour', f'{cc.loc[(50, 0.4)] * 100:.0f}')
    mac('SynConnDtwentyPfour', f'{cc.loc[(20, 0.4)] * 100:.0f}')
    dist = A[(A.edge_p == 0.4) & (A.d >= 20)].mean_grouped_dist
    mac('SynDistMean', f'{dist.mean():.1f}')
    lin = A[~A.nonlinear].groupby(['d', 'edge_p']).cabin_acc.mean()
    mac('SynAccLinDten', f2(lin.loc[10].mean()))
    mac('SynAccLinDfiftyPfour', f2(lin.loc[(50, 0.4)]))


# ----------------------------------------------------------------------------
# DAG structure on real data (rerun_dag)
# ----------------------------------------------------------------------------
def common_tasks():
    # the CC18 datasets of Table 1 (every method finished); written by per_dataset()
    return [l.strip() for l in open(os.path.join(GEN, 'cc18_common.txt')) if l.strip()]


def dag_structure():
    D = pd.read_csv(os.path.join(PAPER, 'rerun_dag', 'dag_metrics.csv'), index_col=0)
    D = D[D.index.isin(common_tasks())]
    mac('DagN', str(len(D)))
    for v, k in [('base', 'Base'), ('contrastive', 'Con'), ('moe', 'MoE')]:
        mac(f'Dag{k}Untrained', f'{D[f"{v}_restored_untrained"].mean() * 100:.0f}')
        mac(f'Dag{k}Density', f3(D[f'{v}_density'].mean()))
        mac(f'Dag{k}DensitySd', f3(D[f'{v}_density'].std()))
        mac(f'Dag{k}Asym', f2(D[f'{v}_asymmetry'].mean()))
    mac('DagBaseEarly', f'{(D.base_restored_frac < .25).mean() * 100:.0f}')
    u = D.base_restored_untrained
    mac('DagDensUntrained', f2(D.base_density[u].mean()))
    mac('DagDensTrained', f2(D.base_density[~u].mean()))
    mac('DagAsymUntrained', f2(D.base_asymmetry[u].mean()))
    mac('DagAsymTrained', f2(D.base_asymmetry[~u].mean()))
    um = D.moe_restored_untrained
    mac('DagCos', f2(D.moe_cos.mean()))
    mac('DagCosUntrained', f2(D.moe_cos[um].mean()))
    mac('DagCosTrained', f2(D.moe_cos[~um].mean()))
    mac('DagRhoDensFrac', f2(spearmanr(D.base_density, D.base_restored_frac)[0]))
    mac('DagRhoNFrac', f2(spearmanr(D.n, D.base_restored_frac)[0]))
    mac('DagRhoConAcc', f2(spearmanr(D.con_gain, D.acc, nan_policy='omit')[0]))
    mac('DagRhoMoECos', f2(spearmanr(D.moe_gain, D.moe_cos, nan_policy='omit')[0]))

    # Figure: Spearman matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    props = [('d', '$d$'), ('n', '$n$'), ('C', '$C$'), ('pct_cat', '% categ.'),
             ('imbalance', 'Imbalance'), ('mean_abs_r', r'Mean $|r|$')]
    cols = [('base_restored_frac', 'DAG trained\n(fraction)'), ('base_density', 'Density'),
            ('moe_density', 'MoE\ndensity'), ('moe_cos', 'MoE\ncosine'), ('acc', 'Accuracy'),
            ('moe_gain', 'MoE\ngain'), ('con_gain', 'Contr.\ngain')]
    M = np.array([[spearmanr(D[p], D[c], nan_policy='omit')[0] for c, _ in cols] for p, _ in props])
    for (c, _), j in zip(cols, range(len(cols))):
        mac(f'FigRhoN{c.replace("_", "")}', f2(M[1, j]))
        mac(f'FigRhoD{c.replace("_", "")}', f2(M[0, j]))
        mac(f'FigRhoC{c.replace("_", "")}', f2(M[2, j]))
        mac(f'FigRhoR{c.replace("_", "")}', f2(M[5, j]))
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), [l for _, l in cols], fontsize=8)
    ax.set_yticks(range(len(props)), [l for _, l in props], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(M[i, j]) > .5 else 'black')
    fig.colorbar(im, ax=ax, fraction=.03, pad=.02).ax.tick_params(labelsize=7)
    ax.set_title(f'Spearman correlation across {len(D)} CC18 tasks', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(GEN, 'fig_dag_spearman.pdf'))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Sensitivity (rerun_sensitivity)
# ----------------------------------------------------------------------------
SENS_ROWS = [
    ('Groups $k$', [('k4', '4'), ('k8', '8'), ('k16', '16'), ('base', r'\textbf{32}'), ('k64', '64'),
                    ('kauto', r'auto ($\lfloor d/3\rfloor$)')]),
    ('Edge dropout', [('ed0', '0'), ('ed0.1', '0.1'), ('base', r'\textbf{0.2}'), ('ed0.3', '0.3')]),
    ('Label smoothing', [('ls0', '0'), ('ls0.05', '0.05'), ('base', r'\textbf{0.1}'), ('ls0.2', '0.2')]),
    ('Seed (noise floor)', [('base', r'\textbf{42}'), ('base_s1', '1'), ('base_s2', '2')]),
    ('Routing ablation', [('base', r'\textbf{default routing}'), ('fixrestore', 'default, restore fix'),
                          ('randdag', 'dense random (frozen)'), ('nodag', 'no message passing')]),
    ('Routing ablation, $M{=}5$', [('base_ens5', r'\textbf{default routing}'),
                                   ('randdag_ens5', 'dense random (frozen)'),
                               ('nodag_ens5', 'no message passing')], 'base_ens5'),
]
# Restore fix pooled over the three seeds: (fix config, matching default config)
FIX_SEEDS = [('fixrestore', 'base'), ('fixrestore_s1', 'base_s1'), ('fixrestore_s2', 'base_s2')]


def sensitivity():
    fs = glob.glob(os.path.join(PAPER, 'rerun_sensitivity', 'out', '*', '*.json'))
    if not fs:
        return
    R = pd.DataFrame([json.load(open(f)) for f in fs])
    R = R[R.status == 'ok']
    P = R.pivot_table(index=['task', 'fold'], columns='config', values='accuracy')
    present = [c for c in P.columns]
    rows = []
    folds = sorted(R.fold.unique())
    for title, items, *ref in SENS_ROWS:
        ref = ref[0] if ref else 'base'
        items = [(c, l) for c, l in items if c in present]
        if len(items) < 2 or ref not in present:
            continue
        first = True
        for c, lab in items:
            pair = P[[c, ref]].dropna()
            d = ((pair[c] - pair[ref]) * 100).groupby(level=0).mean()  # unit: dataset
            head = f'\\multirow{{{len(items)}}}{{*}}{{{title}}}' if first else ''
            if c == ref:
                rows.append(f'{head} & {lab} & {f3(P[ref].dropna().mean())} & -- & -- & -- \\\\')
            else:
                p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                rows.append(f'{head} & {lab} & {f3(pair[c].mean())} & {pts(d.mean())} & {(d > 0).sum()}/{(d < 0).sum()} & {pval(p)} \\\\')
                key = c.replace('.', '').replace('_', '')
                mac(f'Sens{key}Delta', pts(d.mean()))
                mac(f'Sens{key}P', pval(p))
                mac(f'Sens{key}WL', f'{(d > 0).sum()}/{(d < 0).sum()}')
            first = False
        rows.append(r'\midrule')
    write('table_sensitivity.tex', '\n'.join(rows[:-1]) + '\n')
    mac('SensTasks', str(P.index.get_level_values(0).nunique()))
    mac('SensFolds', str(len(folds)))
    mac('SensPairs', str(len(P['base'].dropna())))
    # Seed noise: mean |difference| between seeds, per (task, fold)
    seeds = [c for c in ['base', 'base_s1', 'base_s2'] if c in present]
    if len(seeds) >= 2:
        S = P[seeds].dropna()
        mac('SensSeedSd', f'{S.std(axis=1).mean() * 100:.2f}')
        mac('SensSeedMeanSpread', f'{(S.mean() * 100).max() - (S.mean() * 100).min():.2f}')
    # Restore fix pooled over seeds (paired with the default at the same seed)
    fx = [(f, b) for f, b in FIX_SEEDS if f in present and b in present]
    if len(fx) > 1:
        # unit: dataset (mean over folds and seeds)
        d = pd.concat([(P[f] - P[b]).dropna() for f, b in fx]) * 100
        d = d.groupby(level=0).mean()
        mac('SensFixSeedsN', str(len(fx)))
        mac('SensFixSeedsDelta', pts(d.mean()))
        mac('SensFixSeedsP', pval(wilcoxon(d).pvalue))
        mac('SensFixSeedsWL', f'{(d > 0).sum()}/{(d < 0).sum()}')
        mac('SensFixSeedsPerSeed', ', '.join(pts((P[f] - P[b]).dropna().mean() * 100) for f, b in fx))
    # Ensemble size
    E = R[R.config == 'ens10']
    if len(E):
        rows = []
        m1 = E['accuracy_M1'].mean()
        for M in [1, 3, 5, 10]:
            a = E[f'accuracy_M{M}']
            rows.append(f'{M} & {f3(a.mean())} & {pts((a.mean() - m1) * 100) if M > 1 else "--"} \\\\')
            mac(f'SensEns{M}', f3(a.mean()))
        write('table_ensemble_size.tex', '\n'.join(rows) + '\n')
        mac('SensEnsThreeVsFive', pts((E.accuracy_M3.mean() - E.accuracy_M5.mean()) * 100))
        mac('SensEnsTenVsFive', pts((E.accuracy_M10.mean() - E.accuracy_M5.mean()) * 100))
        mac('SensEnsN', str(len(E)))
    # k by dimensionality (low-d tasks, to test the "8 groups" explanation)
    if 'k4' in present:
        dd = R.groupby('task').d.first()
        low = dd[dd < 12].index
        for c in ['k4', 'k8', 'kauto']:
            if c in present:
                pair = P[[c, 'base']].dropna()
                pair = pair[pair.index.get_level_values(0).isin(low)]
                mac(f'SensLowD{c}Delta', pts((pair[c] - pair['base']).mean() * 100))
        mac('SensLowDN', str(len(low)))


# ----------------------------------------------------------------------------
# VSF vs linear embedding, and per-dataset appendix table
# ----------------------------------------------------------------------------
def vsf_vs_linear(P):
    """Linear-embedding Cabin (design-study pipeline, cum5) vs VSF Cabin (tub).

    The two runs use different 5-fold splits of the same datasets, so this is
    a dataset-level comparison of means, not a paired same-split comparison.
    """
    g = load_bk('01_cumulative_ablation')
    lin = g[g.config == 'cum5_causalboost'].groupby('dataset').accuracy.mean()
    vsf = load_tub('cabin-v33-2slots')
    common = lin.index.intersection(vsf.index)
    a, b = lin[common], vsf[common]
    mac('VSFN', str(len(common)))
    mac('VSFLin', f3(a.mean()))
    mac('VSFVsf', f3(b.mean()))
    mac('VSFDelta', pts((b - a).mean() * 100))
    mac('VSFP', pval(wilcoxon(b, a).pvalue))
    mac('VSFBetter', str(int((b > a).sum())))
    mac('VSFWorse', str(int((b < a).sum())))
    lgb_bk = load_bk('08_cc18_baselines')
    lgb_bk = lgb_bk[lgb_bk.config.str.contains('lightgbm', case=False)].groupby('dataset').accuracy.mean()
    lgb_tub = load_tub('lightgbm')
    c2 = lgb_bk.index.intersection(lgb_tub.index)
    if len(c2):
        mac('SplitLGBAbsDiff', f2((lgb_bk[c2] - lgb_tub[c2]).abs().mean() * 100))
        mac('SplitLGBN', str(len(c2)))


def per_dataset(P):
    meta = {}
    for t in P.index:
        with open(os.path.join(PAPER, 'results', 'tub', 'tasks', t, 'task.json')) as fh:
            m = json.load(fh)
        meta[t] = m
    cols = list(P.columns)
    rows = []
    for t in sorted(P.index, key=str.lower):
        m = meta[t]
        best = P.loc[t].max()
        cells = [(r'\textbf{%s}' % f3(P.loc[t, c])) if P.loc[t, c] == best else f3(P.loc[t, c]) for c in cols]
        rows.append(f"{t.replace('_', chr(92) + '_')} & {m['n_samples']} & {m['n_features']} & {m.get('n_classes', '')} & " + ' & '.join(cells) + ' \\\\')
    header = ' & '.join([r'\rotatebox{90}{%s}' % c for c in cols])
    write('table_per_dataset_header.tex', header + '\n')
    write('table_per_dataset.tex', '\n'.join(rows) + '\n')
    write('cc18_common.txt', '\n'.join(P.index) + '\n')


if __name__ == '__main__':
    P = table2()
    lightgbm_bins(P)
    vsf_vs_linear(P)
    per_dataset(P)
    design_study()
    tabarena()
    synthetic()
    dag_structure()
    sensitivity()
    import structure
    structure.run(PAPER, mac, write, f3, pts, pval)
    import negative
    negative.run(mac, write, f3, pts, pval)
    with open(os.path.join(GEN, 'numbers.tex'), 'w') as fh:
        fh.write('% Generated by scripts/make_numbers.py -- do not edit by hand.\n')
        for k in sorted(MACROS):
            fh.write(f'\\newcommand{{\\{k}}}{{{MACROS[k]}}}\n')
            # k + 'e': the whole test expression, '$p=0.012$' or '$p<0.001$'
            v = str(MACROS[k])
            if k.endswith('P') and (v == '$<$0.001' or re.fullmatch(r'[01]\.\d+', v)):
                expr = '$p<0.001$' if v.startswith('$') else f'$p={v}$'
                fh.write(f'\\newcommand{{\\{k}e}}{{{expr}}}\n')
    print(f'{len(MACROS)} macros, tables in {os.path.abspath(GEN)}')

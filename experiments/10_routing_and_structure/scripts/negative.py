#!/usr/bin/env python3
"""Negative-results appendix (app:negative): variants tried during development
that did not make it into Cabin.

Every row compares a variant to a reference variant on the (dataset, fold)
pairs where both have a result. Delta = mean over the common datasets of the
per-dataset mean accuracy difference (variant - reference), in points.
Duplicate rows for the same (variant, dataset, fold) (reruns after a crash)
are averaged. When the benchmark file has no in-file reference, the reference
is taken from the file that defines the base config (same 5-fold
StratifiedKFold(random_state=42) splits, but independent training seeds);
the Version column then reads "variant file / reference file".

Sources (read-only): Pipe_CINN/0[1-6]_*/ result CSVs. The docstring at the
top of each benchmark_vNN.py describes the variants.

A variant whose loss is dominated by one dataset (> COLLAPSE points) gets a
"collapses on X" note with the mean over the other datasets.

Writes generated/table_negative.tex (body rows only, 8 columns:
Dimension & Variant & Version & Reference & Suite & Delta & #datasets &
Conclusion, e.g. {llllrrrl} or a longtable) and macros
negNRows, negNWithData, negNWorse, negNNeutral, negNBetter.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))   # Pipe_CINN/

F01 = '01_foundations'
F02 = '02_cinn_architecture_experiments'
F03 = '03_dag_learning_breakthroughs'
F05 = '05_categorical_and_sdcd'
F06 = '06_champion_line'

# Dataset suites (label shown in the table).
S_EARLY = 'early core, 5-fold'          # blood, diabetes, heart, ionosphere, sonar (+1)
S_EARLY10 = '10 small, 5-fold'
S_EUC = 'eucalyptus, 5-fold'
S_CAT = '5 categorical, 5-fold'
S_CORE11 = '11 core, 5-fold'
S_GAP15 = '15 high-gap, 5-fold'
S_DEV8 = 'Dev8, 5-fold'

# Reference specs reused below: (file, column-or-variant, short name).
R_V8_FT = (f'{F01}/finetune_results.csv', 'v8', 'v8')
R_EUC = (f'{F02}/eucalyptus_ablation_results.csv', 'v15_baseline', 'v15')
R_25B = (f'{F05}/v25b_innovations_results.csv', 'baseline_acc', 'v25b')
R_28_33 = (f'{F06}/v33_results.csv', 'v28_baseline', 'v28')
R_28_35 = (f'{F06}/v35_results.csv', 'v28_baseline', 'v28')
R_28_36 = (f'{F06}/v36_results.csv', 'v28_baseline', 'v28')
R_28_37 = (f'{F06}/v37_results.csv', 'v28_baseline', 'v28')
R_28_38 = (f'{F06}/v38_results.csv', 'v28_baseline', 'v28')
R_FACT = (f'{F06}/v38_results.csv', 'v38_factored', 'v38 factored')
R_VAR42 = (f'{F06}/v42_results.csv', 'v42_variational', 'v42 variational')
R_LAT45 = (f'{F06}/v45_results.csv', 'v45_latent', 'v45 latent')
R_BH46 = (f'{F06}/v46_results.csv', 'v46_big_head', 'v46 big head')
R_MI50 = (f'{F06}/v50_results.csv', 'v50_miss_ind', 'v50 miss.\\ ind.')

# (dimension, variant label, version, suite, variant spec, reference spec, note)
# variant spec: (file, column-or-variant) or (file, [candidates]) -> best of them.
ROWS = [
    # ------------------------------------------------------------------
    ('Dim.\\ reduction before routing', 'linear bottleneck (best $k\\in\\{16,32,48\\}$)', 'v33', S_GAP15,
     (f'{F06}/v33_results.csv', ['v33a_k16', 'v33a_k32', 'v33a_k48']), R_28_33, ''),
    ('Dim.\\ reduction before routing', 'attention bottleneck (best $k$)', 'v33', S_GAP15,
     (f'{F06}/v33_results.csv', ['v33b_k16', 'v33b_k32', 'v33b_k48']), R_28_33, ''),
    ('Dim.\\ reduction before routing', 'learned linear projection', 'v38', S_DEV8,
     (f'{F06}/v38_results.csv', 'v38_linear'), R_28_38, ''),
    ('Dim.\\ reduction before routing', 'PCA-initialised projection', 'v38', S_DEV8,
     (f'{F06}/v38_results.csv', 'v38_pca_init'), R_28_38, ''),
    ('Dim.\\ reduction before routing', 'frozen PCA projection', 'v38', S_DEV8,
     (f'{F06}/v38_results.csv', 'v38_pca_fixed'), R_28_38, ''),
    ('Dim.\\ reduction before routing', 'PCA, $k$ at 95\\% variance', 'v38', S_DEV8,
     (f'{F06}/v38_results.csv', 'v38_pca_k95'), R_28_38, ''),
    ('Dim.\\ reduction before routing', 'attention grouping ($k$ queries)', 'v38', S_DEV8,
     (f'{F06}/v38_results.csv', 'v38_soft_group'), R_28_38, ''),
    # ------------------------------------------------------------------
    ('Reconstruction loss', 'SDCD reconstruction loss', 'v35', S_GAP15,
     (f'{F06}/v35_results.csv', 'v35_R1'), R_28_35, ''),
    ('Reconstruction loss', 'CASTLE reconstruction loss', 'v35', S_GAP15,
     (f'{F06}/v35_results.csv', 'v35_R2'), R_28_35, ''),
    ('Reconstruction loss', 'PCA projection + recon.\\ loss', 'v38', S_DEV8,
     (f'{F06}/v38_results.csv', 'v38_pca_recon'),
     (f'{F06}/v38_results.csv', 'v38_pca_init', 'v38 PCA init'), ''),
    ('Reconstruction loss', 'factored DAG + recon.\\ loss', 'v38', S_DEV8,
     (f'{F06}/v38_results.csv', 'v38_factored_recon'), R_FACT, ''),
    # ------------------------------------------------------------------
    ('Self-supervised pre-training', 'masked-feature SSL, boosted', 'v52/v46', S_DEV8,
     (f'{F06}/v52_results.csv', 'v52_ssl'), R_BH46, ''),
    ('Self-supervised pre-training', 'masked-feature SSL, 8 indep.\\ models', 'v52/v46', S_DEV8,
     (f'{F06}/v52_results.csv', 'v52_full'), R_BH46, ''),
    ('Self-supervised pre-training', 'correlation-clustering DAG init', 'v51/v46', S_DEV8,
     (f'{F06}/v51_results.csv', 'v51_stat_init'), R_BH46, ''),
    ('Self-supervised pre-training', 'REPTILE meta-learned message passing', 'v51/v46', S_DEV8,
     (f'{F06}/v51_results.csv', 'v51_pretrain'), R_BH46, ''),
    ('Self-supervised pre-training', 'both, halved joint epochs', 'v51/v46', S_DEV8,
     (f'{F06}/v51_results.csv', 'v51_full'), R_BH46, ''),
    # ------------------------------------------------------------------
    ('Message-passing depth', 'multi-round MP, $K{=}2$', 'v11', S_EARLY,
     (f'{F02}/v11_results.csv', 'v11_2r'), (f'{F02}/v11_results.csv', 'v8', 'v8'), ''),
    ('Message-passing depth', 'multi-round MP, $K{=}3$', 'v11', S_EARLY,
     (f'{F02}/v11_results.csv', 'v11_3r'), (f'{F02}/v11_results.csv', 'v8', 'v8'), ''),
    ('Message-passing depth', 'GOLEM + multi-round MP, $K{=}2$', 'v16', S_EARLY,
     None, (None, None, 'v15'), 'run crashed, log only'),
    ('Message-passing depth', 'multi-round MP, $K{=}2$', 'v15 abl.', S_EUC,
     (f'{F02}/eucalyptus_ablation_results.csv', 'v15+multiround_K2'), R_EUC, ''),
    ('Message-passing depth', 'multi-round MP, $K{=}3$', 'v15 abl.', S_EUC,
     (f'{F02}/eucalyptus_ablation_results.csv', 'v15+multiround_K3'), R_EUC, ''),
    ('Message-passing depth', 'multi-round MP, $K{=}2$', 'v25b', S_CAT,
     (f'{F05}/v25b_innovations_results.csv', 'multi_round_acc'), R_25B, ''),
    ('Message-passing depth', 'multi-round MP, $K{=}2$', 'v45/v42', S_DEV8,
     (f'{F06}/v45_results.csv', 'v45_multiround'), R_VAR42, ''),
    ('Message-passing depth', 'interleaved DAG+FFN layers (3)', 'v47/v46', S_DEV8,
     (f'{F06}/v47_results.csv', 'v47_interleaved'), R_BH46, ''),
    ('Message-passing depth', 'target-node readout', 'v47/v46', S_DEV8,
     (f'{F06}/v47_results.csv', 'v47_target_node'), R_BH46, ''),
    # ------------------------------------------------------------------
    ('Attention', 'attention parent aggregation', 'v10', S_EARLY,
     (f'{F02}/v10_results.csv', 'v10'), (f'{F02}/v10_results.csv', 'v8', 'v8'), ''),
    ('Attention', 'cross-node attention after MP', 'v13', S_EARLY,
     (f'{F02}/v13_results.csv', 'v13'), (f'{F02}/v13_results.csv', 'v8', 'v8'), ''),
    ('Attention', 'cross-node attention after MP', 'v15 abl.', S_EUC,
     (f'{F02}/eucalyptus_ablation_results.csv', 'v15+cross_attn'), R_EUC, ''),
    ('Attention', 'cross-node attention after MP', 'v25b', S_CAT,
     (f'{F05}/v25b_innovations_results.csv', 'attention_acc'), R_25B, ''),
    ('Attention', 'DAG-masked attention', 'v32', S_CORE11,
     (f'{F06}/v32_results.csv', 'v32a'), (f'{F06}/v32_results.csv', 'v29_scm_1.0', 'v29 SCM'), ''),
    ('Attention', 'self-attention side tower', 'v35', S_GAP15,
     (f'{F06}/v35_results.csv', 'v35_T1'), R_28_35, ''),
    ('Attention', 'transformer block on node states', 'v36', S_DEV8,
     (f'{F06}/v36_results.csv', 'v36_transformer'), R_28_36, ''),
    ('Attention', 'cross-node attention after MP', 'v45/v42', S_DEV8,
     (f'{F06}/v45_results.csv', 'v45_attention'), R_VAR42, ''),
    ('Attention', 'DAG-biased attention replacing MP', 'v46/v45', S_DEV8,
     (f'{F06}/v46_results.csv', 'v46_dag_attn'), R_LAT45, ''),
    ('Attention', 'hierarchical block DAG', 'v14', S_EARLY,
     (f'{F02}/v14_results.csv', 'v14'), (f'{F02}/v14_results.csv', 'v8', 'v8'), ''),
    # ------------------------------------------------------------------
    ('Pooled readout', '[CLS] transformer readout', 'v37', S_DEV8,
     (f'{F06}/v37_results.csv', 'v28_cls'), R_28_37, ''),
    ('Pooled readout', 'attention pooling', 'v37', S_DEV8,
     (f'{F06}/v37_results.csv', 'v28_attnpool'), R_28_37, ''),
    ('Pooled readout', 'mean+max pooling', 'v48/v46', S_DEV8,
     (f'{F06}/v48_results.csv', 'v48_stat_pool'), R_BH46, ''),
    ('Pooled readout', 'attention pooling ($k{=}8$ queries)', 'v48/v46', S_DEV8,
     (f'{F06}/v48_results.csv', 'v48_attn_pool'), R_BH46, ''),
    ('Pooled readout', 'per-node classifier vote', 'v48/v46', S_DEV8,
     (f'{F06}/v48_results.csv', 'v48_node_vote'), R_BH46, ''),
    ('Pooled readout', 'attention pooling + 3 interleaved layers', 'v48/v46', S_DEV8,
     (f'{F06}/v48_results.csv', 'v48_deep_attn'), R_BH46, ''),
    # ------------------------------------------------------------------
    ('Fine-tuned edges', 'CINN$\\to$DAG fine-tuning (v8)', 'v12', S_EARLY,
     (f'{F01}/finetune_results.csv', 'v8_ft'), R_V8_FT, ''),
    ('Fine-tuned edges', 'CINN$\\to$DAG fine-tuning (v10)', 'v12', S_EARLY,
     (f'{F01}/finetune_results.csv', 'v10_ft'), (f'{F01}/finetune_results.csv', 'v10', 'v10'), ''),
    ('Fine-tuned edges', 'CINN$\\to$DAG fine-tuning (v11)', 'v12', S_EARLY,
     (f'{F01}/finetune_results.csv', 'v11_ft'), (f'{F01}/finetune_results.csv', 'v11', 'v11'), ''),
    ('Fine-tuned edges', 'end-to-end fine-tuning', 'v15 abl.', S_EUC,
     (f'{F02}/eucalyptus_ablation_results.csv', 'v15+e2e_finetune'), R_EUC, ''),
    # ------------------------------------------------------------------
    ('RTDL embeddings', 'periodic embedding before DAG', 'v4', S_EARLY10,
     (f'{F01}/pipe_cinn_v4_full_results.csv', 'Pipe_CINN_v4_periodic'),
     (f'{F01}/pipe_cinn_v4_full_results.csv', 'Pipe_CINN_v3_ensemble', 'v3'), ''),
    ('RTDL embeddings', 'PLE embedding before DAG', 'v4', S_EARLY10,
     (f'{F01}/pipe_cinn_v4_full_results.csv', 'Pipe_CINN_v4_ple'),
     (f'{F01}/pipe_cinn_v4_full_results.csv', 'Pipe_CINN_v3_ensemble', 'v3'), ''),
    ('RTDL embeddings', 'periodic replacing linear embedding', 'v45/v42', S_DEV8,
     (f'{F06}/v45_results.csv', 'v45_periodic'), R_VAR42, ''),
    ('RTDL embeddings', 'periodic skip path', 'v50/v46', S_DEV8,
     (f'{F06}/v50_results.csv', 'v50_rtdl_skip'), R_BH46, ''),
    ('RTDL embeddings', 'periodic skip path + miss.\\ ind.', 'v50', S_DEV8,
     (f'{F06}/v50_results.csv', 'v50_both'), R_MI50, ''),
    # ------------------------------------------------------------------
    ('Voting instead of boosting', 'bagged vote (5 GOLEM + 10 random DAGs)', 'v18', S_EARLY,
     (f'{F03}/v18_core_results.csv', 'v17_hybrid_5g10r'),
     (f'{F03}/v18_core_results.csv', 'v18_boost_reweight_5r', 'v18 boost'), ''),
    ('Voting instead of boosting', 'bagging ($n{=}5$)', 'v25b', S_CAT,
     (f'{F05}/v25b_innovations_results.csv', 'bagging_acc'),
     (f'{F05}/v25b_innovations_results.csv', 'boosting_acc', 'v25b boost'), ''),
    ('Voting instead of boosting', '8 independent models, mean vote', 'v52/v46', S_DEV8,
     (f'{F06}/v52_results.csv', 'v52_diverse'), R_BH46, ''),
    ('Voting instead of boosting', 'independent ensemble', 'v53/v50', S_DEV8,
     (f'{F06}/v53_results.csv', 'v53_diverse_init'), R_MI50, ''),
    # ------------------------------------------------------------------
    ('Eval-time DAG sampling', 'MC, 8 Gumbel samples', 'v42', S_DEV8,
     (f'{F06}/v42_results.csv', 'v42_mc8'), (f'{F06}/v42_results.csv', 'v39_gumbel_vcuda', 'v39 G+VCUDA'), ''),
    ('Eval-time DAG sampling', 'MC, 32 Gumbel samples', 'v42', S_DEV8,
     (f'{F06}/v42_results.csv', 'v42_mc32'), (f'{F06}/v42_results.csv', 'v39_gumbel_vcuda', 'v39 G+VCUDA'), ''),
    ('Eval-time DAG sampling', 'variational posterior, 8 samples', 'v42', S_DEV8,
     (f'{F06}/v42_results.csv', 'v42_variational'), (f'{F06}/v42_results.csv', 'v39_gumbel_vcuda', 'v39 G+VCUDA'), ''),
    ('Eval-time DAG sampling', 'per-model Gumbel temperature', 'v42', S_DEV8,
     (f'{F06}/v42_results.csv', 'v42_diverse_tau'), (f'{F06}/v42_results.csv', 'v39_gumbel_vcuda', 'v39 G+VCUDA'), ''),
    ('Eval-time DAG sampling', 'variational + per-model temperature', 'v42', S_DEV8,
     (f'{F06}/v42_results.csv', 'v42_var_diverse'), (f'{F06}/v42_results.csv', 'v39_gumbel_vcuda', 'v39 G+VCUDA'), ''),
    ('Eval-time DAG sampling', 'confidence-weighted MC', 'v44', S_DEV8,
     (f'{F06}/v44_results.csv', 'v44_conf_weighted'), (f'{F06}/v44_results.csv', 'v44_baseline', 'v44 uniform MC'), ''),
    ('Eval-time DAG sampling', 'adaptive number of MC samples', 'v44', S_DEV8,
     (f'{F06}/v44_results.csv', 'v44_adaptive_mc'), (f'{F06}/v44_results.csv', 'v44_baseline', 'v44 uniform MC'), ''),
    ('Eval-time DAG sampling', 'test-feature DAG prior weighting', 'v44', S_DEV8,
     (f'{F06}/v44_results.csv', 'v44_test_prior'), (f'{F06}/v44_results.csv', 'v44_baseline', 'v44 uniform MC'), ''),
    ('Eval-time DAG sampling', 'per-sample DAG routing', 'v44', S_DEV8,
     (f'{F06}/v44_results.csv', 'v44_dag_routing'), (f'{F06}/v44_results.csv', 'v44_baseline', 'v44 uniform MC'), ''),
    ('Eval-time DAG sampling', 'combined weighting', 'v44', S_DEV8,
     (f'{F06}/v44_results.csv', 'v44_combined'), (f'{F06}/v44_results.csv', 'v44_baseline', 'v44 uniform MC'), ''),
    ('Eval-time DAG sampling', 'factored + Gumbel MC8', 'v45/v38', S_DEV8,
     (f'{F06}/v45_results.csv', 'v45_v38_gumbel'), R_FACT, ''),
    ('Eval-time DAG sampling', 'factored + variational MC8', 'v45/v38', S_DEV8,
     (f'{F06}/v45_results.csv', 'v45_v38_variational'), R_FACT, ''),
    # ------------------------------------------------------------------
    ('Pre-extracted DAGs', 'RCD, best of 100 DAGs', 'v43', S_DEV8,
     (f'{F06}/v43_rcd_results.csv', 'v43_rcd_best1'), (f'{F06}/v43_rcd_results.csv', 'v39_gumbel_vcuda', 'v39 G+VCUDA'), ''),
    ('Pre-extracted DAGs', 'RCD, 5 DAGs in boosting chain', 'v43', S_DEV8,
     (f'{F06}/v43_rcd_results.csv', 'v43_rcd_ensemble5'), (f'{F06}/v43_rcd_results.csv', 'v39_gumbel_vcuda', 'v39 G+VCUDA'), ''),
    ('Pre-extracted DAGs', 'RCD, 20 averaged CINNs', 'v43', S_DEV8,
     (f'{F06}/v43_rcd_results.csv', 'v43_rcd_ensemble20'), (f'{F06}/v43_rcd_results.csv', 'v39_gumbel_vcuda', 'v39 G+VCUDA'), ''),
    ('Pre-extracted DAGs', 'VCUDA, best of 100 DAGs', 'v43', S_DEV8,
     (f'{F06}/v43_vcuda_results.csv', 'v43_vcuda_best1'), (f'{F06}/v43_vcuda_results.csv', 'v39_gumbel_vcuda', 'v39 G+VCUDA'), '$d\\le 60$ only'),
    ('Pre-extracted DAGs', 'VCUDA, 5 DAGs in boosting chain', 'v43', S_DEV8,
     (f'{F06}/v43_vcuda_results.csv', 'v43_vcuda_ensemble5'), (f'{F06}/v43_vcuda_results.csv', 'v39_gumbel_vcuda', 'v39 G+VCUDA'), '$d\\le 60$ only'),
    ('Pre-extracted DAGs', 'VCUDA, 20 averaged CINNs', 'v43', S_DEV8,
     (f'{F06}/v43_vcuda_results.csv', 'v43_vcuda_ensemble20'), (f'{F06}/v43_vcuda_results.csv', 'v39_gumbel_vcuda', 'v39 G+VCUDA'), '$d\\le 60$ only'),
    ('Pre-extracted DAGs', 'VCUDA, mean posterior DAG', 'v43', S_DEV8,
     (f'{F06}/v43_vcuda_results.csv', 'v43_vcuda_posterior'), (f'{F06}/v43_vcuda_results.csv', 'v39_gumbel_vcuda', 'v39 G+VCUDA'), '$d\\le 60$ only'),
    # ------------------------------------------------------------------
    ('Pairwise delta-DAG', 'DAG over $[x_b, x_a{-}x_b, y_a]$, $k$-NN', 'v54/v50', S_DEV8,
     (f'{F06}/v54_results.csv', 'v54_delta'), R_MI50, ''),
]

_CACHE = {}


def _load(path):
    """Accuracy table: DataFrame indexed by (dataset, fold), one column per variant."""
    if path in _CACHE:
        return _CACHE[path]
    df = pd.read_csv(os.path.join(PIPE, path))
    if 'dataset' not in df.columns:            # single-dataset ablations
        df['dataset'] = 'eucalyptus'
    if 'variant' in df.columns:                # long format
        tab = df.groupby(['dataset', 'fold', 'variant'])['accuracy'].mean().unstack('variant')
    else:                                      # wide format: one column per variant
        num = df.drop(columns=[c for c in df.columns if c in ('dataset', 'fold')])
        num = num.apply(pd.to_numeric, errors='coerce')
        num[['dataset', 'fold']] = df[['dataset', 'fold']]
        tab = num.groupby(['dataset', 'fold']).mean()
    _CACHE[path] = tab
    return tab


def _series(path, col):
    tab = _load(path)
    if col not in tab.columns:
        return None
    return tab[col].dropna()


def paired_delta(var, ref):
    """Mean over common datasets of per-dataset mean(var - ref), in points."""
    common = var.index.intersection(ref.index)
    if len(common) == 0:
        return np.nan, 0, 0, None
    diff = (var.loc[common] - ref.loc[common]) * 100
    per_ds = diff.groupby(level='dataset').mean()
    collapse = None   # one dataset that breaks down (> COLLAPSE points lost)
    if len(per_ds) > 1 and per_ds.min() <= -COLLAPSE:
        collapse = (per_ds.idxmin(), per_ds.drop(per_ds.idxmin()).mean())
    return per_ds.mean(), len(per_ds), len(common), collapse


COLLAPSE = 20.0


def verdict(d):
    if np.isnan(d):
        return 'no surviving results'
    if d <= -1.0:
        return 'hurts'
    if d <= -0.3:
        return 'slightly worse'
    if d < 0.3:
        return 'neutral (within noise)'
    return 'gain within noise'


def tex(s):
    return s.replace('_', '\\_')


def compute():
    out = []
    for dim, label, version, suite, vspec, rspec, note in ROWS:
        rfile, rcol, rname = rspec
        if vspec is None:
            out.append(dict(dim=dim, label=label, version=version, suite=suite, ref=rname,
                            delta=np.nan, nds=0, nfold=0, src='--', vcol='--', rcol=rcol, note=note))
            continue
        vfile, vcols = vspec
        ref = _series(rfile, rcol)
        cands = vcols if isinstance(vcols, list) else [vcols]
        best = None
        for c in cands:
            v = _series(vfile, c)
            if v is None or ref is None:
                continue
            d, nds, nf, col = paired_delta(v, ref)
            if best is None or d > best[1]:
                best = (c, d, nds, nf, col)
        if best is None:
            best = (cands[0], np.nan, 0, 0, None)
        c, d, nds, nf, col = best
        if col is not None:
            extra = f'collapses on {col[0]} ({PTS(col[1])} without it)'
            note = f'{note}; {extra}' if note else extra
        out.append(dict(dim=dim, label=label, version=version, suite=suite, ref=rname,
                        delta=d, nds=nds, nfold=nf, src=vfile, vcol=c, rcol=rcol, note=note))
    return out


PTS = None  # set by run(): the caller's pts formatter


def run(mac, write, f3, pts, pval):
    global PTS
    PTS = pts
    rows = compute()
    lines = []
    prev = None
    groups = []
    for r in rows:
        if r['dim'] != prev:
            groups.append([])
            prev = r['dim']
        groups[-1].append(r)
    for gi, g in enumerate(groups):
        if gi:
            lines.append('\\midrule')
        for i, r in enumerate(g):
            first = r['dim'] if i == 0 else ''   # plain text: a p-column wraps it; multirow would overflow
            d = '--' if np.isnan(r['delta']) else pts(r['delta'])
            nds = '--' if r['nds'] == 0 else str(r['nds'])
            concl = verdict(r['delta'])
            if r['note']:
                concl += f'; {r["note"]}'
            lines.append(f'{first} & {r["label"]} & {r["version"]} & {tex(r["ref"])} & '
                         f'{r["suite"]} & {d} & {nds} & {concl} \\\\')
    write('table_negative.tex', '\n'.join(lines) + '\n')

    have = [r for r in rows if not np.isnan(r['delta'])]
    mac('negNRows', str(len(rows)))
    mac('negNWithData', str(len(have)))
    mac('negNWorse', str(sum(r['delta'] <= -0.3 for r in have)))
    mac('negNNeutral', str(sum(-0.3 < r['delta'] < 0.3 for r in have)))
    mac('negNBetter', str(sum(r['delta'] >= 0.3 for r in have)))
    return rows


if __name__ == '__main__':
    import sys
    macros = {}

    def _mac(k, v):
        macros[k] = v

    def _write(name, text):
        print(text)

    rows = run(_mac, _write, lambda x: f'{x:.3f}', lambda x: f'{x:+.2f}', str)
    print(macros)
    if '-v' in sys.argv:
        for r in rows:
            print(f"{r['src']:45s} {r['vcol']:24s} vs {str(r['rcol']):22s} "
                  f"d={r['delta']:+.2f} nds={r['nds']} folds={r['nfold']}")

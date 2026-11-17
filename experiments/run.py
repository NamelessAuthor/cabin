#!/usr/bin/env python3
"""
CausalBoost Paper Experiments — CLI Entry Point
================================================

Runs ablation study and baseline comparisons on OpenML-CC18 datasets
(72 classification datasets, 5-fold stratified CV).

Results are saved to a CSV file with file-locking for concurrent writes.
Already-completed experiments are skipped on re-run.

Usage:
  # Run everything (all configs, all datasets)
  python -m experiments.run --workers 4

  # Run only the cumulative ablation
  python -m experiments.run --workers 4 --groups cumulative

  # Run specific configs
  python -m experiments.run --workers 4 --configs cum0_base cum5_causalboost

  # Run only baselines (CPU models)
  python -m experiments.run --workers 4 --groups baselines

  # Print summary tables + LaTeX
  python -m experiments.run --summary

  # List all configs
  python -m experiments.run --list

  # Dry run
  python -m experiments.run --dry-run

Dependencies:
  pip install torch numpy scikit-learn scipy openml filelock pandas
  pip install xgboost catboost lightgbm  # for baselines
  pip install tabicl tabpfn              # for baselines (GPU)
"""

import argparse
import time

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_WORKERS,
    N_FOLDS,
    OUTPUT_FILE,
    get_cc18_datasets,
    get_done_keys,
    run_one_job,
)
from .configs import ALL_CONFIGS, ALL_GROUPS


# ============================================================================
# Summary & LaTeX Output
# ============================================================================

def print_group_table(df, group_name, group_configs):
    """Print a summary table for one experiment group."""
    available = [c for c in group_configs if c in df['config'].values]
    if not available:
        return

    print(f"\n{'='*100}")
    print(f"  {group_name.upper().replace('_', ' ')}")
    desc_lines = []
    for c in available:
        if c in ALL_CONFIGS:
            desc_lines.append(f"    {c:22s}  {ALL_CONFIGS[c]['description']}")
        else:
            desc_lines.append(f"    {c:22s}  (cross-ref from cumulative)")
    print('\n'.join(desc_lines))
    print(f"{'='*100}")

    datasets = sorted(df['dataset'].unique())
    col_w = max(16, max(len(c) for c in available) + 2)

    # Header
    print(f"\n{'Dataset':<25} {'n':>6} {'d':>4} {'C':>3}", end='')
    for c in available:
        print(f" {c:>{col_w}}", end='')
    print()
    print("-" * (40 + (col_w + 1) * len(available)))

    # Per-dataset rows
    ds_means = {c: [] for c in available}
    for ds_name in datasets:
        ds_df = df[df['dataset'] == ds_name]
        has_any = any(len(ds_df[ds_df['config'] == c]) > 0 for c in available)
        if not has_any:
            continue

        n_val = int(ds_df['n'].iloc[0]) if ds_df['n'].iloc[0] > 0 else 0
        d_val = int(ds_df['d'].iloc[0]) if ds_df['d'].iloc[0] > 0 else 0
        c_val = int(ds_df['classes'].iloc[0]) if ds_df['classes'].iloc[0] > 0 else 0

        print(f"{ds_name:<25} {n_val:>6} {d_val:>4} {c_val:>3}", end='')

        row_accs = {}
        for c in available:
            vd = ds_df[ds_df['config'] == c]
            if len(vd) > 0 and vd['accuracy'].notna().any():
                row_accs[c] = vd['accuracy'].mean()
        best_acc = max(row_accs.values()) if row_accs else -1

        for c in available:
            if c in row_accs:
                acc = row_accs[c]
                ds_means[c].append(acc)
                marker = '*' if abs(acc - best_acc) < 1e-6 else ' '
                print(f" {acc:>{col_w - 1}.4f}{marker}", end='')
            else:
                print(f" {'--':>{col_w}}", end='')
        print()

    # Grand mean
    print("-" * (40 + (col_w + 1) * len(available)))
    print(f"{'MEAN':<25} {'':>6} {'':>4} {'':>3}", end='')
    grand_means = {}
    for c in available:
        if ds_means[c]:
            m = np.mean(ds_means[c])
            grand_means[c] = m
            print(f" {m:>{col_w}.4f}", end='')
        else:
            print(f" {'--':>{col_w}}", end='')
    print()

    # Delta vs first config
    if available and available[0] in grand_means:
        base_mean = grand_means[available[0]]
        print(f"{'Delta':<25} {'':>6} {'':>4} {'':>3}", end='')
        for c in available:
            if c in grand_means:
                delta = grand_means[c] - base_mean
                s = f"{'+' if delta >= 0 else ''}{delta:.4f}"
                print(f" {s:>{col_w}}", end='')
            else:
                print(f" {'--':>{col_w}}", end='')
        print()

    # Datasets covered / failures
    n_datasets = len(ds_means[available[0]]) if available else 0
    print(f"\n  Datasets with results: {n_datasets}")

    fails = df[df['accuracy'].isna() & df['config'].isin(available)]
    if len(fails) > 0:
        print(f"  Failures: {len(fails)}")
        for c in available:
            cf = fails[fails['config'] == c]
            if len(cf) > 0:
                ds_list = ', '.join(sorted(cf['dataset'].unique())[:5])
                more = f" +{len(cf['dataset'].unique())-5} more" if len(cf['dataset'].unique()) > 5 else ""
                print(f"    {c}: {len(cf)} fails ({ds_list}{more})")

    # Average time per job
    print(f"\n  {'Avg time (s):':<25}", end='')
    for c in available:
        t = df[df['config'] == c]['time'].mean()
        print(f" {t:>{col_w}.0f}", end='')
    print()


def print_latex_cumulative(df):
    """Print LaTeX table for cumulative ablation."""
    cum_configs = [
        'cum_castle', 'cum0_base', 'cum1_factored', 'cum2_reg',
        'cum3_boost', 'cum4_reweight', 'cum5_causalboost',
    ]
    step_names = [
        'CASTLE+CINN (two-stage)',
        'SDCD+CINN (end-to-end)',
        '+ Factored DAG',
        '+ Label smoothing + edge dropout',
        '+ Boosted ensemble ($M{=}5$)',
        '+ Adaptive reweighting',
        '+ Temperature calibration',
    ]
    available = [c for c in cum_configs if c in df['config'].values]
    if len(available) < 2:
        return

    datasets = sorted(df['dataset'].unique())
    means = {}
    for c in available:
        accs = []
        for ds in datasets:
            vd = df[(df['config'] == c) & (df['dataset'] == ds)]
            if len(vd) > 0 and vd['accuracy'].notna().any():
                accs.append(vd['accuracy'].mean())
        if accs:
            means[c] = np.mean(accs)

    print(f"\n{'='*80}")
    print("  LaTeX: Cumulative Ablation")
    print(f"{'='*80}\n")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Cumulative ablation on CC18 (%d datasets, 5-fold CV)." % len(datasets))
    print("Each row adds one improvement to the previous configuration.}")
    print("\\label{tab:cumulative}")
    print("\\begin{tabular}{clcc}")
    print("\\toprule")
    print("Step & Component & Mean Acc.\\ & $\\Delta$ \\\\")
    print("\\midrule")

    prev_mean = None
    for i, c in enumerate(cum_configs):
        name = step_names[i] if i < len(step_names) else c
        if c not in available:
            continue
        if c in means:
            m = means[c]
            if prev_mean is None:
                delta = "---"
            else:
                d = m - prev_mean
                delta = f"${'+'if d>=0 else ''}{d:.1%}$"
            is_last = (c == available[-1])
            if is_last:
                print(f"{i} & \\textbf{{{name}}} & \\textbf{{{m:.3f}}} & {delta} \\\\")
            else:
                print(f"{i} & {name} & {m:.3f} & {delta} \\\\")
            prev_mean = m
        else:
            print(f"{i} & {name} & -- & -- \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_latex_baselines(df):
    """Print LaTeX table for baseline comparison."""
    baseline_configs = [
        'base_xgboost', 'base_catboost', 'base_lightgbm',
        'base_tabicl', 'base_tabpfn', 'cum5_causalboost',
    ]
    display_names = {
        'base_xgboost': 'XGBoost',
        'base_catboost': 'CatBoost',
        'base_lightgbm': 'LightGBM',
        'base_tabicl': 'TabICL',
        'base_tabpfn': 'TabPFN',
        'cum5_causalboost': '\\textbf{CausalBoost (ours)}',
    }
    available = [c for c in baseline_configs if c in df['config'].values]
    if len(available) < 2:
        return

    datasets = sorted(df['dataset'].unique())

    ds_means = {c: {} for c in available}
    for ds in datasets:
        for c in available:
            vd = df[(df['config'] == c) & (df['dataset'] == ds)]
            if len(vd) > 0 and vd['accuracy'].notna().any():
                ds_means[c][ds] = vd['accuracy'].mean()

    # Grand means over common datasets
    common_ds = set(datasets)
    for c in available:
        common_ds &= set(ds_means[c].keys())
    common_ds = sorted(common_ds)

    grand_means = {}
    wins = {c: 0 for c in available}
    for c in available:
        if common_ds:
            grand_means[c] = np.mean([ds_means[c][ds] for ds in common_ds])

    for ds in common_ds:
        best = max(ds_means[c].get(ds, -1) for c in available)
        for c in available:
            if abs(ds_means[c].get(ds, -1) - best) < 1e-6:
                wins[c] += 1

    print(f"\n{'='*80}")
    print("  LaTeX: Baseline Comparison")
    print(f"{'='*80}\n")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Comparison with baselines on CC18 (%d datasets, 5-fold CV).}" % len(common_ds))
    print("\\label{tab:baselines}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Method & Mean Acc.\\ & Wins \\\\")
    print("\\midrule")

    sorted_configs = sorted(available, key=lambda c: grand_means.get(c, 0), reverse=True)
    for c in sorted_configs:
        name = display_names.get(c, c)
        if c in grand_means:
            print(f"{name} & {grand_means[c]:.3f} & {wins[c]}/{len(common_ds)} \\\\")
        else:
            print(f"{name} & -- & -- \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_latex_dimension(df, group_name, group_configs, table_label, caption):
    """Print LaTeX table for a per-dimension ablation."""
    available = [c for c in group_configs if c in df['config'].values]
    if len(available) < 2:
        return

    datasets = sorted(df['dataset'].unique())
    means = {}
    for c in available:
        accs = []
        for ds in datasets:
            vd = df[(df['config'] == c) & (df['dataset'] == ds)]
            if len(vd) > 0 and vd['accuracy'].notna().any():
                accs.append(vd['accuracy'].mean())
        if accs:
            means[c] = np.mean(accs)

    print(f"\n{'='*80}")
    print(f"  LaTeX: {group_name}")
    print(f"{'='*80}\n")
    print("\\begin{table}[t]")
    print("\\centering")
    print(f"\\caption{{{caption}}}")
    print(f"\\label{{tab:{table_label}}}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Variant & Mean Acc.\\ & $\\Delta$ \\\\")
    print("\\midrule")

    base_mean = means.get(available[0])
    best_c = max(available, key=lambda c: means.get(c, 0))

    for c in available:
        desc = ALL_CONFIGS[c]['description'] if c in ALL_CONFIGS else c
        # Clean for LaTeX
        desc = desc.replace('_', '\\_').replace('+', '$+$')
        if c in means:
            m = means[c]
            if base_mean is not None:
                d = m - base_mean
                delta = f"${'+'if d>=0 else ''}{d:.1%}$" if abs(d) > 1e-6 else "---"
            else:
                delta = "---"
            if c == best_c:
                print(f"\\textbf{{{desc}}} & \\textbf{{{m:.3f}}} & {delta} \\\\")
            else:
                print(f"{desc} & {m:.3f} & {delta} \\\\")
        else:
            print(f"{desc} & -- & -- \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_summary():
    """Print all summary tables and LaTeX output."""
    try:
        df = pd.read_csv(OUTPUT_FILE)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"No results found at {OUTPUT_FILE}. Run experiments first.")
        return

    n_configs = df['config'].nunique()
    n_datasets = df['dataset'].nunique()
    n_fails = df['accuracy'].isna().sum()

    print(f"\n{'#'*80}")
    print(f"  CAUSALBOOST PAPER RESULTS")
    print(f"  {len(df)} rows | {n_configs} configs | {n_datasets} datasets | "
          f"{n_fails} failures")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"{'#'*80}")

    # Print each group table
    for group_name, group_configs in ALL_GROUPS.items():
        print_group_table(df, group_name, group_configs)

    # LaTeX tables
    print_latex_cumulative(df)
    print_latex_baselines(df)

    # Per-dimension LaTeX tables
    dim_tables = [
        ('dim1_dag_learning', 'dim1', 'DAG learning method comparison on CC18.'),
        ('dim2_dag_parameterization', 'dim2', 'DAG parameterization comparison on CC18.'),
        ('dim4_architecture', 'dim4', 'Message passing rounds comparison on CC18.'),
        ('dim6_ensemble', 'dim6', 'Ensemble strategy comparison on CC18.'),
        ('dim7_training', 'dim7', 'Training schedule comparison on CC18.'),
    ]
    for group_name, label, caption in dim_tables:
        if group_name in ALL_GROUPS:
            print_latex_dimension(df, group_name, ALL_GROUPS[group_name], label, caption)


# ============================================================================
# Config Listing
# ============================================================================

def list_configs():
    """Print all available configurations with descriptions."""
    print(f"\n{'Config Name':<22} {'Group':<12} {'Type':<7} {'M':>3} Description")
    print("-" * 100)
    for name, cfg in ALL_CONFIGS.items():
        m = cfg['kwargs'].get('n_models', 1)
        print(f"{name:<22} {cfg['group']:<12} {cfg['model']:<7} {m:>3} "
              f"{cfg['description']}")

    print(f"\nTotal: {len(ALL_CONFIGS)} unique configs")
    print(f"CC18: ~72 datasets x {N_FOLDS} folds per config")

    print("\nGroups (for --groups flag):")
    for gname, gconfigs in ALL_GROUPS.items():
        print(f"  {gname}: {gconfigs}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CausalBoost Paper Experiments on OpenML-CC18",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--workers', type=int, default=DEFAULT_WORKERS,
        help=f'Parallel workers (default: {DEFAULT_WORKERS})')
    parser.add_argument(
        '--groups', nargs='+', default=None,
        help='Run only these groups (e.g., cumulative, baselines, dim1_dag_learning)')
    parser.add_argument(
        '--configs', nargs='+', default=None,
        help='Run only these specific config names')
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        help='Run only these datasets (by OpenML name)')
    parser.add_argument(
        '--max-samples', type=int, default=None,
        help='Exclude datasets with n > this value')
    parser.add_argument(
        '--max-features', type=int, default=None,
        help='Exclude datasets with d > this value')
    parser.add_argument(
        '--output', type=str, default=None,
        help=f'Output CSV file (default: {OUTPUT_FILE})')
    parser.add_argument(
        '--summary', action='store_true',
        help='Print summary tables and exit')
    parser.add_argument(
        '--list', action='store_true',
        help='List all configs and exit')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show job plan without running')

    args = parser.parse_args()

    if args.output:
        import experiments.common as common_mod
        common_mod.OUTPUT_FILE = args.output
        common_mod.LOCK_FILE = args.output + '.lock'

    if args.list:
        list_configs()
        return

    if args.summary:
        print_summary()
        return

    # Determine which configs to run
    if args.configs:
        run_configs = []
        for c in args.configs:
            if c not in ALL_CONFIGS:
                print(f"ERROR: Unknown config '{c}'. Use --list.")
                return
            run_configs.append(c)
    elif args.groups:
        run_configs = []
        for g in args.groups:
            matched = [gn for gn in ALL_GROUPS if g in gn]
            if not matched:
                print(f"ERROR: No group matching '{g}'. "
                      f"Available: {list(ALL_GROUPS.keys())}")
                return
            for gn in matched:
                run_configs.extend(ALL_GROUPS[gn])
        # Deduplicate preserving order
        seen = set()
        run_configs = [c for c in run_configs
                       if c in ALL_CONFIGS and not (c in seen or seen.add(c))]
    else:
        run_configs = list(ALL_CONFIGS.keys())

    # Fetch CC18 datasets
    print("Fetching CC18 dataset list...", flush=True)
    try:
        all_datasets = get_cc18_datasets(
            max_samples=args.max_samples,
            max_features=args.max_features,
        )
    except Exception as e:
        print(f"FATAL: Could not fetch CC18 dataset list: {e}")
        return

    if args.datasets:
        ds_names_set = set(args.datasets)
        all_datasets = [(n, oid) for n, oid in all_datasets if n in ds_names_set]
        if not all_datasets:
            print("ERROR: No matching datasets found.")
            return

    # Build job list
    output_file = args.output or OUTPUT_FILE
    done = get_done_keys(output_file)

    jobs = []
    for config_name in run_configs:
        for ds_name, openml_id in all_datasets:
            for fold in range(N_FOLDS):
                if (ds_name, fold, config_name) not in done:
                    jobs.append((config_name, ds_name, openml_id, fold,
                                 ALL_CONFIGS, output_file))

    # Print plan
    print(f"\n{'='*80}")
    print(f"  CAUSALBOOST PAPER EXPERIMENTS")
    print(f"{'='*80}")
    print(f"  Configs:      {len(run_configs)}")
    for c in run_configs:
        m = ALL_CONFIGS[c]['kwargs'].get('n_models', 1)
        print(f"    {c:22s}  (M={m})  {ALL_CONFIGS[c]['description']}")
    print(f"  Datasets:     {len(all_datasets)}")
    print(f"  Folds:        {N_FOLDS}")
    total_possible = len(run_configs) * len(all_datasets) * N_FOLDS
    print(f"  Total jobs:   {total_possible}")
    print(f"  Done:         {total_possible - len(jobs)}")
    print(f"  Remaining:    {len(jobs)}")
    print(f"  Workers:      {args.workers}")
    print(f"  Output:       {output_file}")
    print(f"{'='*80}\n")

    if args.dry_run:
        print("DRY RUN — first 30 jobs:\n")
        for i, (cn, ds, oid, f, _, _) in enumerate(jobs[:30]):
            print(f"  {i+1:4d}. {cn:22s} | {ds:25s} (id={oid:>5d}) | fold {f+1}")
        if len(jobs) > 30:
            print(f"\n  ... and {len(jobs) - 30} more jobs")
        return

    if not jobs:
        print("All jobs already completed!")
        print_summary()
        return

    # Pre-download all datasets
    import openml as oml
    dataset_ids = sorted(set(oid for _, _, oid, _, _, _ in jobs))
    print(f"Pre-downloading {len(dataset_ids)} datasets...", flush=True)
    for oid in dataset_ids:
        try:
            oml.datasets.get_dataset(oid, download_data=True)
        except Exception as e:
            print(f"  id={oid}: download failed — {e}", flush=True)
    print("Downloads complete.\n", flush=True)

    # Run with multiprocessing
    import torch.multiprocessing as mp

    total_start = time.time()
    ctx = mp.get_context('spawn')

    with ctx.Pool(processes=args.workers) as pool:
        results = pool.map(run_one_job, jobs)

    elapsed_h = (time.time() - total_start) / 3600
    n_ok = sum(1 for r in results
               if r and not np.isnan(r.get('accuracy', np.nan)))
    n_fail = sum(1 for r in results
                 if r and np.isnan(r.get('accuracy', np.nan)))
    n_none = sum(1 for r in results if r is None)

    print(f"\n{'='*80}")
    print(f"  COMPLETE: {n_ok} succeeded, {n_fail} failed, {n_none} load errors")
    print(f"  Total runtime: {elapsed_h:.1f} hours")
    print(f"{'='*80}\n")

    print_summary()


if __name__ == '__main__':
    main()

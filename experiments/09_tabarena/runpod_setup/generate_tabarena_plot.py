#!/usr/bin/env python3
"""Convert our CausalBoost pkl results to TabArena format and generate comparison plot."""
import os
import sys
# local clone of https://github.com/autogluon/tabarena
sys.path.insert(0, os.path.expanduser('~/tabarena/tabarena'))
sys.path.insert(0, os.path.expanduser('~/tabarena/bencheval'))

import os
import pickle
import pandas as pd
import numpy as np

RESULTS_DIR = '/tmp/tabarena_results'
OUTPUT_DIR = '/tmp/tabarena_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Convert all pkl files to a single DataFrame
rows = []
for root, dirs, files in os.walk(RESULTS_DIR):
    for f in files:
        if f != 'results.pkl':
            continue
        path = os.path.join(root, f)
        with open(path, 'rb') as fh:
            data = pickle.load(fh)

        meta = data['task_metadata']
        # fold index in TabArena = split_idx (0-29 for 30-fold tasks, 0-8 for 9-fold)
        fold_idx = meta.get('split_idx', meta.get('fold', 0))

        rows.append({
            'dataset': meta['name'],
            'fold': fold_idx,
            'method': 'CausalBoost (ours)',
            'metric_error': data['metric_error'],
            'time_train_s': data['time_train_s'],
            'time_infer_s': data['time_infer_s'],
            'metric': data.get('metric', 'log_loss'),
            'problem_type': data.get('problem_type', 'binary'),
            'metric_error_val': data.get('metric_error_val', np.nan),
            'imputed': False,
            'impute_method': np.nan,
            'method_type': 'baseline',
            'method_subtype': np.nan,
            'config_type': np.nan,
            'method_metadata': np.nan,
            'ta_name': 'CausalBoost',
            'ta_suite': 'custom-2026-03-13',
        })

our_df = pd.DataFrame(rows)
print(f"Our results: {len(our_df)} rows, {our_df['dataset'].nunique()} datasets")
print(f"Datasets: {sorted(our_df['dataset'].unique())}")
print(f"Folds per dataset:")
for ds in sorted(our_df['dataset'].unique()):
    n = our_df[our_df['dataset'] == ds]['fold'].nunique()
    print(f"  {ds}: {n} folds")

# Save our results
our_df.to_csv(os.path.join(OUTPUT_DIR, 'causalboost_results.csv'), index=False)
print(f"\nSaved to {OUTPUT_DIR}/causalboost_results.csv")

# Step 2: Load baselines
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

ctx = TabArenaContext(include_unverified=False)
baseline_df = ctx.load_results_paper(download_results='auto')

# Filter baselines to our datasets only
our_datasets = set(our_df['dataset'].unique())
baseline_filtered = baseline_df[baseline_df['dataset'].isin(our_datasets)].copy()
print(f"\nBaseline results on our datasets: {len(baseline_filtered)} rows")
print(f"Baseline methods: {baseline_filtered['method'].nunique()}")

# Step 3: Use TabArenaContext.compare
print("\n=== Running TabArena Comparison ===")
leaderboard = ctx.compare(
    output_dir=OUTPUT_DIR,
    new_results=our_df,
    fillna='RF (default)',
)

print("\n=== LEADERBOARD ===")
print(leaderboard.to_string())

# Save leaderboard
leaderboard.to_csv(os.path.join(OUTPUT_DIR, 'leaderboard.csv'))
print(f"\nLeaderboard saved to {OUTPUT_DIR}/leaderboard.csv")
print(f"Plots saved to {OUTPUT_DIR}/")

# List output files
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(fpath)
    print(f"  {f}: {size:,} bytes")

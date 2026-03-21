#!/usr/bin/env python3
"""
Run missing experiments for the Cabin paper.

Missing:
  1. dim3_vcuda, dim3_dpdag — acyclicity ablation (72 CC18 datasets, 5 folds)
  2. dim5_no_missing, dim5_no_embed — feature representation ablation
  3. 10 missing TabArena classification datasets

Usage:
  # Run all missing experiments (6 workers)
  python run_missing.py --workers 6

  # Run only specific groups
  python run_missing.py --workers 6 --groups dim3
  python run_missing.py --workers 6 --groups dim5
  python run_missing.py --workers 6 --groups tabarena

  # Resume (skips already-completed jobs)
  python run_missing.py --workers 6

  # Print summary
  python run_missing.py --summary

Dependencies:
  pip install torch numpy scikit-learn scipy openml filelock pandas
"""

import os
import sys
import time
import argparse
import signal
import numpy as np
import pandas as pd
import filelock
from collections import OrderedDict
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Add model directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'model')
sys.path.insert(0, MODEL_DIR)

SEED = 42
N_FOLDS = 5
JOB_TIMEOUT = 3600  # 1 hour per job
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'missing_results.csv')
LOCK_FILE = OUTPUT_FILE + '.lock'


# ============================================================================
# Dataset Loading
# ============================================================================

def get_cc18_datasets():
    """Fetch all 72 CC18 classification datasets from OpenML (suite 99)."""
    import openml
    suite = openml.study.get_suite(99)
    all_ds = openml.datasets.list_datasets(
        data_id=suite.data, output_format='dataframe'
    )
    all_ds = all_ds.sort_values('NumberOfInstances')
    return [(row['name'], int(row['did'])) for _, row in all_ds.iterrows()]


# 10 missing TabArena classification datasets (OpenML dataset IDs)
TABARENA_MISSING = [
    ('anneal', 46906),
    ('Bioresponse', 46912),
    ('coil2000_insurance_policies', 46916),
    ('Bank_Customer_Churn', 46911),
    ('credit_card_clients_default', 46919),
    ('Amazon_employee_access', 46905),
    ('bank-marketing', 46910),
    ('kddcup09_appetency', 46939),
    ('APSFailure', 46908),
    ('customer_satisfaction_in_airline', 46920),
]


def load_dataset(openml_id):
    """Load an OpenML dataset, returning (X, y, cat_indices)."""
    import openml
    dataset = openml.datasets.get_dataset(openml_id, download_data=True)
    X, y, cat_mask, _ = dataset.get_data(
        target=dataset.default_target_attribute
    )
    X = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
    y = y.to_numpy() if hasattr(y, 'to_numpy') else np.array(y)

    cat_indices = []
    for i in range(X.shape[1]):
        if cat_mask is not None and i < len(cat_mask) and cat_mask[i]:
            cat_indices.append(i)
            X[:, i] = LabelEncoder().fit_transform(X[:, i].astype(str))

    X = X.astype(float)
    if np.isnan(X).any():
        X = SimpleImputer(strategy='median').fit_transform(X)

    X = X.astype(np.float32)
    y = LabelEncoder().fit_transform(y.astype(str))
    return X, y, cat_indices


# ============================================================================
# Experiment Configurations
# ============================================================================

def make_configs():
    configs = OrderedDict()

    # ── DIM 3: Missing acyclicity variants ─────────────────────────
    # Locked: SDCD, factored DAG W=ASA^T (k=32), single model, no reg.
    # Baseline (already run): cum1_factored (spectral radius)
    # Already run: dim3_dagma
    # MISSING: dim3_vcuda, dim3_dpdag

    configs['dim3_vcuda'] = {
        'group': 'dim3',
        'description': 'VCUDA acyclic-by-construction (priority ordering)',
        'kwargs': dict(
            dag_type='factored', n_groups=32,
            acyclicity='vcuda',
            n_models=1, calibrate=False,
            edge_dropout=0.0, label_smoothing=0.0,
        ),
    }

    configs['dim3_dpdag'] = {
        'group': 'dim3',
        'description': 'DP-DAG acyclic-by-construction (differentiable permutation)',
        'kwargs': dict(
            dag_type='factored', n_groups=32,
            acyclicity='dpdag',
            n_models=1, calibrate=False,
            edge_dropout=0.0, label_smoothing=0.0,
        ),
    }

    # ── DIM 5: Feature representation ablation ─────────────────────
    # Locked: Full Cabin (cum5_causalboost) but toggling features.
    # Baseline: cum5_causalboost (with missing indicators + entity embeddings)

    configs['dim5_no_missing'] = {
        'group': 'dim5',
        'description': 'Cabin without missingness indicators',
        'kwargs': dict(
            dag_type='factored', n_groups=32,
            acyclicity='vcuda',
            n_models=5, shrinkage=0.1, base_alpha=2.0,
            calibrate=True,
            edge_dropout=0.2, label_smoothing=0.1,
            add_missing_indicators=False,
        ),
    }

    configs['dim5_no_embed'] = {
        'group': 'dim5',
        'description': 'Cabin without entity embeddings (categoricals treated as numeric)',
        'kwargs': dict(
            dag_type='factored', n_groups=32,
            acyclicity='vcuda',
            n_models=5, shrinkage=0.1, base_alpha=2.0,
            calibrate=True,
            edge_dropout=0.2, label_smoothing=0.1,
            add_missing_indicators=True,
        ),
        'no_cat': True,  # pass cat_indices=[] to treat categoricals as numeric
    }

    # ── TABARENA: 10 missing classification datasets ───────────────
    # Full Cabin (champion config) on the 10 excluded TabArena tasks.

    configs['tabarena_cabin'] = {
        'group': 'tabarena',
        'description': 'Full Cabin on missing TabArena classification datasets',
        'kwargs': dict(
            dag_type='factored', n_groups=32,
            acyclicity='vcuda',
            n_models=5, shrinkage=0.1, base_alpha=2.0,
            calibrate=True,
            edge_dropout=0.2, label_smoothing=0.1,
            add_missing_indicators=True,
        ),
    }

    return configs


CONFIGS = make_configs()


# ============================================================================
# Result Management
# ============================================================================

def append_result(result_dict):
    lock = filelock.FileLock(LOCK_FILE, timeout=60)
    with lock:
        try:
            df = pd.read_csv(OUTPUT_FILE)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
        df.to_csv(OUTPUT_FILE, index=False)


def get_done_keys():
    lock = filelock.FileLock(LOCK_FILE, timeout=60)
    with lock:
        try:
            df = pd.read_csv(OUTPUT_FILE)
            return {
                (r['dataset'], int(r['fold']), r['config'])
                for _, r in df.iterrows()
            }
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return set()


# ============================================================================
# Job Runner
# ============================================================================

class _JobTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _JobTimeout(f"Job exceeded {JOB_TIMEOUT}s timeout")


def run_one_job(args):
    config_name, ds_name, openml_id, fold_idx = args

    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(JOB_TIMEOUT)
    except (ValueError, AttributeError):
        pass

    # Re-add model path for spawned process
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    tag = f"[W{os.getpid() % 1000:03d}]"
    config = CONFIGS[config_name]

    try:
        X, y, cat_indices = load_dataset(openml_id)
    except Exception as e:
        print(f"{tag} LOAD_FAIL {ds_name}: {e}", flush=True)
        append_result({
            'config': config_name, 'dataset': ds_name,
            'openml_id': openml_id, 'fold': fold_idx,
            'n': -1, 'd': -1, 'classes': -1,
            'accuracy': np.nan, 'time': 0, 'error': str(e)[:200],
        })
        return None

    n, d = X.shape
    n_classes = len(np.unique(y))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, te_idx = list(skf.split(X, y))[fold_idx]

    print(f"{tag} START {config_name:20s} | {ds_name:30s} | "
          f"fold {fold_idx+1}/{N_FOLDS} (n={n}, d={d}, C={n_classes})",
          flush=True)
    t0 = time.time()

    try:
        from cabin import CausalBoostClassifier
        clf = CausalBoostClassifier(**config['kwargs'])
        cat_idx = [] if config.get('no_cat', False) else cat_indices
        clf.fit(X[tr_idx], y[tr_idx], cat_indices=cat_idx)
        preds = clf.predict(X[te_idx])
        acc = accuracy_score(y[te_idx], preds)
        elapsed = time.time() - t0

        result = {
            'config': config_name, 'dataset': ds_name,
            'openml_id': openml_id, 'fold': fold_idx,
            'n': n, 'd': d, 'classes': n_classes,
            'accuracy': acc, 'time': elapsed, 'error': '',
        }
        append_result(result)
        print(f"{tag} DONE  {config_name:20s} | {ds_name:30s} | "
              f"fold {fold_idx+1} | acc={acc:.4f} | {elapsed:.1f}s",
              flush=True)
        return result

    except Exception as e:
        elapsed = time.time() - t0
        print(f"{tag} FAIL  {config_name:20s} | {ds_name:30s} | "
              f"fold {fold_idx+1} | {e}", flush=True)
        append_result({
            'config': config_name, 'dataset': ds_name,
            'openml_id': openml_id, 'fold': fold_idx,
            'n': n, 'd': d, 'classes': n_classes,
            'accuracy': np.nan, 'time': elapsed,
            'error': str(e)[:200],
        })
        return None

    finally:
        try:
            signal.alarm(0)
        except (ValueError, AttributeError):
            pass


# ============================================================================
# Main
# ============================================================================

def build_jobs(groups=None):
    """Build list of (config, dataset, openml_id, fold) jobs."""
    done = get_done_keys()
    jobs = []

    for config_name, config in CONFIGS.items():
        if groups and config['group'] not in groups:
            continue

        if config['group'] == 'tabarena':
            datasets = TABARENA_MISSING
        else:
            datasets = get_cc18_datasets()

        for ds_name, openml_id in datasets:
            for fold_idx in range(N_FOLDS):
                if (ds_name, fold_idx, config_name) not in done:
                    jobs.append((config_name, ds_name, openml_id, fold_idx))

    return jobs


def print_summary():
    """Print summary of results."""
    try:
        df = pd.read_csv(OUTPUT_FILE)
    except FileNotFoundError:
        print("No results yet.")
        return

    print(f"\n{'='*70}")
    print(f"Missing Experiments Summary ({len(df)} total rows)")
    print(f"{'='*70}\n")

    for config in df['config'].unique():
        sub = df[df['config'] == config]
        valid = sub.dropna(subset=['accuracy'])
        print(f"\n{config}:")
        print(f"  Datasets: {sub['dataset'].nunique()}")
        print(f"  Rows: {len(sub)} ({len(valid)} valid, {len(sub)-len(valid)} failed)")
        if len(valid) > 0:
            print(f"  Mean accuracy: {valid['accuracy'].mean():.4f}")
            print(f"  Mean time: {valid['time'].mean():.1f}s")

            # Per-dataset means
            ds_means = valid.groupby('dataset')['accuracy'].mean().sort_values()
            print(f"  Per-dataset range: {ds_means.iloc[0]:.4f} - {ds_means.iloc[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Run missing Cabin experiments')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--groups', nargs='+',
                        choices=['dim3', 'dim5', 'tabarena'],
                        help='Only run specific experiment groups')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary of existing results')
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    jobs = build_jobs(groups=args.groups)
    if not jobs:
        print("All jobs already completed!")
        print_summary()
        return

    # Group summary
    from collections import Counter
    by_config = Counter(j[0] for j in jobs)
    print(f"\n{len(jobs)} jobs to run:")
    for c, n in by_config.most_common():
        print(f"  {c}: {n} jobs ({n//N_FOLDS} datasets x {N_FOLDS} folds)")

    print(f"\nStarting {args.workers} workers...\n")

    if args.workers == 1:
        for j in jobs:
            run_one_job(j)
    else:
        with Pool(args.workers) as pool:
            pool.map(run_one_job, jobs)

    print("\n\nDone!")
    print_summary()


if __name__ == '__main__':
    main()

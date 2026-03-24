#!/usr/bin/env python3
"""
RCD vs Learned DAG experiment.

Two steps:
  1. Extract RCD DAGs offline (CPU, fast — seconds per dataset)
  2. Run Cabin with fixed RCD DAGs vs learned VCUDA DAGs

Variants:
  - rcd_best1:      Single best RCD DAG (selected by val loss)
  - rcd_ensemble5:  5 diverse RCD DAGs, one per boost model
  - rcd_ensemble20: 20 RCD DAGs, averaged predictions
  - vcuda_baseline: Learned VCUDA DAG (champion config, for comparison)

Usage:
  # Step 1: Extract RCD DAGs (CPU only, fast)
  python run_rcd.py --extract --workers 6

  # Step 2: Run experiments (GPU)
  python run_rcd.py --run --workers 2

  # Both steps
  python run_rcd.py --extract --run --workers 2

  # Summary
  python run_rcd.py --summary

Dependencies:
  pip install torch numpy scikit-learn scipy openml filelock pandas rcd
"""

import os
import sys
import time
import argparse
import signal
import warnings
import numpy as np
import pandas as pd
import filelock
from collections import OrderedDict
from multiprocessing import Pool, get_context
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss,
                             precision_score, recall_score, f1_score,
                             balanced_accuracy_score, matthews_corrcoef,
                             average_precision_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'model')
sys.path.insert(0, MODEL_DIR)

SEED = 42
N_FOLDS = 5
JOB_TIMEOUT = 3600
RCD_DIR = os.path.join(SCRIPT_DIR, 'rcd_dags')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'rcd_results.csv')
LOCK_FILE = OUTPUT_FILE + '.lock'
PREDICTIONS_DIR = os.path.join(SCRIPT_DIR, 'rcd_predictions')
os.makedirs(RCD_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# RCD extraction parameters
ALPHA_LEVELS = np.geomspace(0.001, 0.20, 20)
N_ORIENTATIONS = 5  # per alpha → 100 DAGs total


# ============================================================================
# Dataset Loading
# ============================================================================

def get_cc18_datasets(max_samples=20000):
    import openml
    suite = openml.study.get_suite(99)
    all_ds = openml.datasets.list_datasets(
        data_id=suite.data, output_format='dataframe'
    )
    if max_samples is not None:
        all_ds = all_ds[all_ds['NumberOfInstances'] <= max_samples]
    all_ds = all_ds.sort_values('NumberOfInstances')
    result = [(row['name'], int(row['did'])) for _, row in all_ds.iterrows()]
    print(f"Fetched {len(result)} CC18 datasets (n <= {max_samples})")
    return result


def load_dataset(openml_id):
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
# Step 1: RCD DAG Extraction
# ============================================================================

def make_ci_test(alpha=0.05):
    from scipy import stats
    def ci_test(x, y, cond_set, data):
        n_obs = data.shape[0]
        vars_all = [x, y] + list(cond_set)
        if len(vars_all) >= n_obs - 3:
            return True
        sub = data[:, vars_all]
        C = np.corrcoef(sub.T)
        try:
            P = np.linalg.inv(C)
            r = -P[0, 1] / np.sqrt(abs(P[0, 0] * P[1, 1]))
        except (np.linalg.LinAlgError, FloatingPointError):
            return True
        r = np.clip(r, -0.9999, 0.9999)
        z = 0.5 * np.log((1 + r) / (1 - r))
        z_stat = abs(z) * np.sqrt(max(n_obs - len(cond_set) - 3, 1))
        p_val = 2 * (1 - stats.norm.cdf(z_stat))
        return p_val > alpha
    return ci_test


def skeleton_to_dag(skeleton, d, rng):
    import networkx as nx
    order = rng.permutation(d)
    rank = np.empty(d, dtype=int)
    rank[order] = np.arange(d)
    adj = np.zeros((d, d), dtype=np.float32)
    for u, v in skeleton.edges():
        u, v = int(u), int(v)
        if rank[u] < rank[v]:
            adj[u, v] = 1.0
        else:
            adj[v, u] = 1.0
    return adj


def extract_dags_for_fold(X_train, d, seed=0):
    import networkx as nx
    from rcd import rsl_d

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float64)
    rng = np.random.RandomState(seed)
    dags = []

    for alpha in ALPHA_LEVELS:
        ci_fn = make_ci_test(alpha=alpha)
        try:
            skeleton = rsl_d.learn_and_get_skeleton(ci_test=ci_fn, data=X_scaled)
        except Exception:
            skeleton = nx.Graph()
            skeleton.add_nodes_from(range(d))
        for node in range(d):
            if node not in skeleton:
                skeleton.add_node(node)
        for _ in range(N_ORIENTATIONS):
            dag = skeleton_to_dag(skeleton, d, rng)
            dags.append(dag)

    return np.array(dags[:100], dtype=np.float32)


def extract_one(args):
    ds_name, openml_id, fold_idx = args
    out_path = os.path.join(RCD_DIR, f'{ds_name}_fold{fold_idx}.npy')
    if os.path.exists(out_path):
        return out_path

    tag = f"[E{os.getpid() % 1000:03d}]"
    try:
        X, y, _ = load_dataset(openml_id)
    except Exception as e:
        print(f"{tag} LOAD_FAIL {ds_name}: {e}", flush=True)
        return None

    n, d = X.shape
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, _ = list(skf.split(X, y))[fold_idx]

    print(f"{tag} EXTRACT {ds_name} fold{fold_idx+1}/{N_FOLDS} (n={len(tr_idx)}, d={d})", flush=True)
    t0 = time.time()

    try:
        dags = extract_dags_for_fold(X[tr_idx], d, seed=fold_idx)
        np.save(out_path, dags)
        n_edges = dags.sum(axis=(1, 2))
        print(f"{tag} DONE    {ds_name} fold{fold_idx+1}: "
              f"{len(dags)} DAGs, avg_edges={n_edges.mean():.1f} ({time.time()-t0:.0f}s)", flush=True)
        return out_path
    except Exception as e:
        print(f"{tag} FAIL    {ds_name} fold{fold_idx+1}: {e} ({time.time()-t0:.0f}s)", flush=True)
        return None


def run_extraction(workers=6):
    datasets = get_cc18_datasets()
    jobs = []
    for ds_name, openml_id in datasets:
        for fold in range(N_FOLDS):
            out_path = os.path.join(RCD_DIR, f'{ds_name}_fold{fold}.npy')
            if not os.path.exists(out_path):
                jobs.append((ds_name, openml_id, fold))

    print(f"\n{'='*60}")
    print(f"RCD DAG Extraction: {len(jobs)} jobs remaining")
    print(f"{'='*60}\n")

    if not jobs:
        print("All DAGs already extracted!")
        return

    if workers <= 1:
        for j in jobs:
            extract_one(j)
    else:
        ctx = get_context('spawn')
        with ctx.Pool(workers) as pool:
            pool.map(extract_one, jobs)

    print("\nExtraction complete.")


# ============================================================================
# Step 2: Run Cabin with Fixed vs Learned DAGs
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
            return {(r['dataset'], int(r['fold']), r['config']) for _, r in df.iterrows()}
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return set()


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }
    avg = 'binary' if n_classes == 2 else 'macro'
    metrics['precision'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    try:
        metrics['log_loss'] = log_loss(y_true, y_proba)
    except Exception:
        metrics['log_loss'] = np.nan
    try:
        if n_classes == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
        else:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics['pr_auc'] = np.nan
    except Exception:
        metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = np.nan
    return metrics


class _JobTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _JobTimeout("timeout")


def run_one_job(args):
    config_name, ds_name, openml_id, fold_idx = args

    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(JOB_TIMEOUT)
    except (ValueError, AttributeError):
        pass

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    tag = f"[W{os.getpid() % 1000:03d}]"

    try:
        X, y, cat_indices = load_dataset(openml_id)
    except Exception as e:
        print(f"{tag} LOAD_FAIL {ds_name}: {e}", flush=True)
        return None

    n, d = X.shape
    n_classes = len(np.unique(y))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, te_idx = list(skf.split(X, y))[fold_idx]

    print(f"{tag} START {config_name:20s} | {ds_name:30s} | "
          f"fold {fold_idx+1}/{N_FOLDS} (n={n}, d={d}, C={n_classes})", flush=True)
    t0 = time.time()

    try:
        from cabin import CausalBoostClassifier

        if config_name == 'vcuda_baseline':
            # Standard learned DAG (champion config)
            clf = CausalBoostClassifier(
                dag_type='factored', acyclicity='vcuda',
                n_models=5, shrinkage=0.1, base_alpha=2.0,
                calibrate=True, add_missing_indicators=True,
                edge_dropout=0.2, label_smoothing=0.1,
            )
            clf.fit(X[tr_idx], y[tr_idx], cat_indices=cat_indices)

        elif config_name == 'rcd_best1':
            # Load RCD DAGs, pick best by quick validation probe
            dags = np.load(os.path.join(RCD_DIR, f'{ds_name}_fold{fold_idx}.npy'))
            # Quick probe: train small model on each DAG, pick best val loss
            best_dag_idx = _select_best_dag(dags, X[tr_idx], y[tr_idx], cat_indices, n_classes)
            clf = CausalBoostClassifier(
                dag_type='factored', acyclicity='vcuda',
                n_models=5, shrinkage=0.1, base_alpha=2.0,
                calibrate=True, add_missing_indicators=True,
                edge_dropout=0.2, label_smoothing=0.1,
                fixed_dag=dags[best_dag_idx],
            )
            clf.fit(X[tr_idx], y[tr_idx], cat_indices=cat_indices)

        elif config_name == 'rcd_ensemble5':
            # 5 diverse RCD DAGs (spread across alpha levels), one per boost model
            dags = np.load(os.path.join(RCD_DIR, f'{ds_name}_fold{fold_idx}.npy'))
            indices = np.linspace(0, len(dags)-1, 5, dtype=int)
            selected_dags = [dags[i] for i in indices]
            clf = CausalBoostClassifier(
                dag_type='factored', acyclicity='vcuda',
                n_models=5, shrinkage=0.1, base_alpha=2.0,
                calibrate=True, add_missing_indicators=True,
                edge_dropout=0.2, label_smoothing=0.1,
                fixed_dags=selected_dags,
            )
            clf.fit(X[tr_idx], y[tr_idx], cat_indices=cat_indices)

        elif config_name == 'rcd_ensemble20':
            # 20 RCD DAGs, each gets its own single model, average predictions
            dags = np.load(os.path.join(RCD_DIR, f'{ds_name}_fold{fold_idx}.npy'))
            indices = np.linspace(0, len(dags)-1, 20, dtype=int)
            selected_dags = [dags[i] for i in indices]
            clf = CausalBoostClassifier(
                dag_type='factored', acyclicity='vcuda',
                n_models=20, shrinkage=0.1, base_alpha=0.0,
                calibrate=True, add_missing_indicators=True,
                edge_dropout=0.2, label_smoothing=0.1,
                fixed_dags=selected_dags,
            )
            clf.fit(X[tr_idx], y[tr_idx], cat_indices=cat_indices)

        preds = clf.predict(X[te_idx])
        proba = clf.predict_proba(X[te_idx])
        elapsed = time.time() - t0
        y_te = y[te_idx]

        # Save predictions
        pred_file = os.path.join(PREDICTIONS_DIR, f"{config_name}__{ds_name}__fold{fold_idx}.npz")
        np.savez_compressed(pred_file, y_true=y_te, y_pred=preds, y_proba=proba, test_indices=te_idx)

        metrics = compute_metrics(y_te, preds, proba, n_classes)
        result = {
            'config': config_name, 'dataset': ds_name,
            'openml_id': openml_id, 'fold': fold_idx,
            'n': n, 'd': d, 'classes': n_classes,
            **metrics, 'time': elapsed, 'error': '',
        }
        append_result(result)
        print(f"{tag} DONE  {config_name:20s} | {ds_name:30s} | "
              f"fold {fold_idx+1} | acc={metrics['accuracy']:.4f} "
              f"roc={metrics['roc_auc']:.4f} | {elapsed:.1f}s", flush=True)
        return result

    except Exception as e:
        elapsed = time.time() - t0
        print(f"{tag} FAIL  {config_name:20s} | {ds_name:30s} | "
              f"fold {fold_idx+1} | {e}", flush=True)
        append_result({
            'config': config_name, 'dataset': ds_name,
            'openml_id': openml_id, 'fold': fold_idx,
            'n': n, 'd': d, 'classes': n_classes,
            'accuracy': np.nan, 'balanced_accuracy': np.nan,
            'precision': np.nan, 'recall': np.nan, 'f1': np.nan,
            'mcc': np.nan, 'roc_auc': np.nan, 'pr_auc': np.nan,
            'log_loss': np.nan, 'time': elapsed, 'error': str(e)[:200],
        })
        return None
    finally:
        try:
            signal.alarm(0)
        except (ValueError, AttributeError):
            pass


def _select_best_dag(dags, X_train, y_train, cat_indices, n_classes):
    """Quick probe: 1-epoch training on each DAG, pick lowest val loss.

    Uses 80/20 split of training data for probe validation.
    """
    from sklearn.model_selection import train_test_split
    from cabin import CausalBoostClassifier

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )

    best_idx, best_acc = 0, -1
    # Only probe a subset (every 10th DAG) for speed
    probe_indices = list(range(0, len(dags), 10))

    for idx in probe_indices:
        try:
            clf = CausalBoostClassifier(
                dag_type='factored', acyclicity='vcuda',
                n_models=1, calibrate=False,
                edge_dropout=0.0, label_smoothing=0.0,
                fixed_dag=dags[idx],
            )
            clf.fit(X_tr, y_tr, cat_indices=cat_indices)
            acc = accuracy_score(y_val, clf.predict(X_val))
            if acc > best_acc:
                best_acc = acc
                best_idx = idx
        except Exception:
            continue

    return best_idx


# ============================================================================
# Main
# ============================================================================

CONFIGS = ['rcd_best1', 'rcd_ensemble5', 'rcd_ensemble20']


def build_run_jobs(groups=None):
    done = get_done_keys()
    datasets = get_cc18_datasets()
    jobs = []
    for config_name in CONFIGS:
        for ds_name, openml_id in datasets:
            # Skip RCD configs if DAGs not extracted
            if config_name.startswith('rcd'):
                dag_path = os.path.join(RCD_DIR, f'{ds_name}_fold0.npy')
                if not os.path.exists(dag_path):
                    continue
            for fold_idx in range(N_FOLDS):
                if (ds_name, fold_idx, config_name) not in done:
                    jobs.append((config_name, ds_name, openml_id, fold_idx))
    return jobs


def print_summary():
    try:
        df = pd.read_csv(OUTPUT_FILE)
    except FileNotFoundError:
        print("No results yet.")
        return

    print(f"\n{'='*70}")
    print(f"RCD vs Learned DAG — Summary ({len(df)} total rows)")
    print(f"{'='*70}\n")

    for config in CONFIGS:
        sub = df[df['config'] == config]
        valid = sub.dropna(subset=['accuracy'])
        if len(valid) == 0:
            continue
        print(f"{config}:")
        print(f"  Datasets: {sub['dataset'].nunique()}, Rows: {len(valid)}")
        print(f"  Mean accuracy:  {valid['accuracy'].mean():.4f} +/- {valid['accuracy'].std():.4f}")
        print(f"  Mean ROC-AUC:   {valid['roc_auc'].mean():.4f}")
        print(f"  Mean time:      {valid['time'].mean():.1f}s")
        print()


def main():
    parser = argparse.ArgumentParser(description='RCD vs Learned DAG experiment')
    parser.add_argument('--extract', action='store_true', help='Step 1: extract RCD DAGs')
    parser.add_argument('--run', action='store_true', help='Step 2: run Cabin experiments')
    parser.add_argument('--summary', action='store_true', help='Print summary')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if args.extract:
        run_extraction(workers=args.workers)

    if args.run:
        jobs = build_run_jobs()
        if not jobs:
            print("All jobs completed!")
            print_summary()
            return

        from collections import Counter
        by_config = Counter(j[0] for j in jobs)
        print(f"\n{len(jobs)} jobs to run:")
        for c, n in by_config.most_common():
            print(f"  {c}: {n}")

        if args.workers == 1:
            for j in jobs:
                run_one_job(j)
        else:
            with Pool(args.workers) as pool:
                pool.map(run_one_job, jobs)

        print("\nDone!")
        print_summary()

    if not args.extract and not args.run and not args.summary:
        parser.print_help()


if __name__ == '__main__':
    main()

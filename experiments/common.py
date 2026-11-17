"""
Shared infrastructure for CausalBoost paper experiments.

Provides:
  - CC18 dataset loading from OpenML
  - File-locked CSV result management
  - Multiprocessing job runner with timeout

All experiment configs and models are imported from sibling modules.

Dependencies:
  pip install numpy pandas scikit-learn openml filelock torch
"""

import gc
import os
import time
import traceback

import filelock
import numpy as np
import openml
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ============================================================================
# Constants
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'results.csv')
LOCK_FILE = OUTPUT_FILE + '.lock'
N_FOLDS = 5
SEED = 42
DEFAULT_WORKERS = 4
JOB_TIMEOUT = 3600  # 1 hour max per job


# ============================================================================
# CC18 Dataset List
# ============================================================================

def get_cc18_datasets(max_samples=None, max_features=None):
    """Fetch CC18 classification datasets from OpenML (suite 99).

    Returns list of (name, openml_id) tuples sorted by sample count
    (smallest first, so early results come quickly).

    Args:
        max_samples:  Exclude datasets with n > this value.
        max_features: Exclude datasets with d > this value.
    """
    suite = openml.study.get_suite(99)
    all_ds = openml.datasets.list_datasets(
        data_id=suite.data, output_format='dataframe'
    )
    total = len(all_ds)
    if max_samples is not None:
        all_ds = all_ds[all_ds['NumberOfInstances'] <= max_samples]
    if max_features is not None:
        all_ds = all_ds[all_ds['NumberOfFeatures'] <= max_features]
    all_ds = all_ds.sort_values('NumberOfInstances')
    result = [(row['name'], int(row['did'])) for _, row in all_ds.iterrows()]

    filters = []
    if max_samples:
        filters.append(f"n <= {max_samples}")
    if max_features:
        filters.append(f"d <= {max_features}")
    filt_str = f" ({', '.join(filters)})" if filters else ""
    print(f"Fetched {len(result)}/{total} CC18 datasets{filt_str}")
    return result


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(openml_id):
    """Load an OpenML dataset, returning (X, y, cat_indices).

    Categorical features are label-encoded. Missing values are median-imputed.
    """
    dataset = openml.datasets.get_dataset(openml_id, download_data=True)
    X, y, cat_mask, _ = dataset.get_data(
        target=dataset.default_target_attribute
    )
    X = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
    y = y.to_numpy() if hasattr(y, 'to_numpy') else np.array(y)

    # Label-encode categoricals
    cat_flags = []
    for i in range(X.shape[1]):
        is_cat = cat_mask is not None and i < len(cat_mask) and cat_mask[i]
        cat_flags.append(is_cat)
        if is_cat:
            X[:, i] = LabelEncoder().fit_transform(X[:, i].astype(str))

    X = X.astype(float)

    # Drop all-NaN columns BEFORE imputation (SimpleImputer drops them,
    # which would silently shift column indices and break cat_indices)
    valid_cols = ~np.isnan(X).all(axis=0)
    if not valid_cols.all():
        dropped = np.where(~valid_cols)[0].tolist()
        X = X[:, valid_cols]
        cat_flags = [cat_flags[i] for i in range(len(cat_flags)) if valid_cols[i]]

    if np.isnan(X).any():
        X = SimpleImputer(strategy='median').fit_transform(X)

    cat_indices = [i for i, is_cat in enumerate(cat_flags) if is_cat]

    X = X.astype(np.float32)
    y = LabelEncoder().fit_transform(y.astype(str))
    return X, y, cat_indices


# ============================================================================
# Helpers
# ============================================================================

def one_hot_encode(X, cat_indices):
    """One-hot encode categorical columns for old classifiers (CASTLE, GOLEM).

    Replaces each categorical column with one-hot columns, keeping numeric
    columns in their original positions.
    """
    if not cat_indices:
        return X
    cat_set = set(cat_indices)
    num_cols = [i for i in range(X.shape[1]) if i not in cat_set]
    X_num = X[:, num_cols] if num_cols else np.empty((X.shape[0], 0), dtype=X.dtype)
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=X.dtype)
    X_cat_oh = enc.fit_transform(X[:, cat_indices])
    return np.hstack([X_num, X_cat_oh])


# ============================================================================
# Result Management (file-locked CSV for concurrent writes)
# ============================================================================

def append_result(result_dict, output_file=None):
    """Append one result row to the output CSV (process-safe via filelock)."""
    out = output_file or OUTPUT_FILE
    lock = filelock.FileLock(out + '.lock', timeout=60)
    with lock:
        try:
            df = pd.read_csv(out)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
        df.to_csv(out, index=False)


def get_done_keys(output_file=None):
    """Return set of (dataset, fold, config) tuples already completed."""
    out = output_file or OUTPUT_FILE
    lock = filelock.FileLock(out + '.lock', timeout=60)
    with lock:
        try:
            df = pd.read_csv(out)
            return {
                (r['dataset'], int(r['fold']), r['config'])
                for _, r in df.iterrows()
                if pd.notna(r.get('accuracy'))
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


def _make_classifier(model_type, kwargs, X, cat_indices):
    """Instantiate the right classifier based on model type.

    Returns (classifier, X_train_transformed, X_test_fn) where X_test_fn
    transforms test data the same way as training data.
    """
    from . import models

    if model_type == 'boost':
        clf = models.CausalBoostClassifier(**kwargs)
        return clf, None, None

    elif model_type == 'single':
        clf = models.CINNClassifier(**kwargs)
        return clf, None, None

    elif model_type == 'castle':
        clf = models.CASTLEPipelineClassifier(**kwargs)
        return clf, 'one_hot', None

    elif model_type == 'golem':
        clf = models.GOLEMPipelineClassifier(**kwargs)
        return clf, 'one_hot', None

    elif model_type == 'xgboost':
        from xgboost import XGBClassifier
        clf = XGBClassifier(**kwargs)
        return clf, None, None

    elif model_type == 'catboost':
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(**kwargs, verbose=0)
        return clf, 'catboost', None

    elif model_type == 'lightgbm':
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(**kwargs, verbose=-1)
        return clf, 'lightgbm', None

    elif model_type == 'tabicl':
        from tabicl import TabICLClassifier
        clf = TabICLClassifier(**kwargs)
        return clf, None, None

    elif model_type == 'tabpfn':
        from tabpfn import TabPFNClassifier
        n = X.shape[0]
        if n > 10000:
            raise ValueError(f"TabPFN: n={n} exceeds 10K limit")
        clf = TabPFNClassifier(**kwargs)
        return clf, None, None

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_one_job(args):
    """Run one (config, dataset, fold) experiment.

    Catches ALL exceptions so the multiprocessing pool never crashes.
    Failed jobs are recorded with accuracy=NaN in the CSV.
    """
    import signal

    config_name, ds_name, openml_id, fold_idx, all_configs, output_file = args

    # Set alarm for timeout
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(JOB_TIMEOUT)
    except (ValueError, AttributeError):
        pass

    tag = f"[W{os.getpid() % 1000:03d}]"
    config = all_configs[config_name]

    # Load dataset
    try:
        X, y, cat_indices = load_dataset(openml_id)
    except Exception as e:
        msg = f"LOAD: {e}"
        print(f"{tag} LOAD_FAIL {ds_name}: {e}", flush=True)
        append_result({
            'config': config_name, 'dataset': ds_name,
            'openml_id': openml_id, 'fold': fold_idx,
            'n': -1, 'd': -1, 'classes': -1,
            'accuracy': np.nan, 'time': 0, 'error': msg[:200],
        }, output_file)
        return None

    n, d = X.shape
    n_classes = len(np.unique(y))

    # Fold split
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, te_idx = list(skf.split(X, y))[fold_idx]

    print(f"{tag} START {config_name:22s} | {ds_name:25s} | "
          f"fold {fold_idx+1}/{N_FOLDS} (n={n}, d={d}, C={n_classes})",
          flush=True)
    t0 = time.time()

    try:
        clf, transform, _ = _make_classifier(
            config['model'], config['kwargs'], X, cat_indices
        )

        # Train
        if transform == 'one_hot':
            X_oh = one_hot_encode(X, cat_indices)
            clf.fit(X_oh[tr_idx], y[tr_idx])
        elif transform == 'catboost':
            clf.fit(X[tr_idx], y[tr_idx],
                    cat_features=cat_indices if cat_indices else None)
        elif transform == 'lightgbm':
            clf.fit(X[tr_idx], y[tr_idx],
                    categorical_feature=cat_indices if cat_indices else 'auto')
        elif config['model'] in ('boost', 'single'):
            clf.fit(X[tr_idx], y[tr_idx], cat_indices=cat_indices)
        else:
            clf.fit(X[tr_idx], y[tr_idx])

        # Predict
        if transform == 'one_hot':
            preds = clf.predict(X_oh[te_idx])
        else:
            preds = clf.predict(X[te_idx])

        acc = accuracy_score(y[te_idx], preds)
        elapsed = time.time() - t0

        result = {
            'config': config_name, 'dataset': ds_name,
            'openml_id': openml_id, 'fold': fold_idx,
            'n': n, 'd': d, 'classes': n_classes,
            'accuracy': acc, 'time': elapsed, 'error': '',
        }
        append_result(result, output_file)
        print(f"{tag} DONE  {config_name:22s} | {ds_name:25s} | "
              f"fold {fold_idx+1}: acc={acc:.4f} ({elapsed:.0f}s)", flush=True)

        del clf
        torch.cuda.empty_cache()
        gc.collect()
        try:
            signal.alarm(0)
        except Exception:
            pass
        return result

    except Exception as e:
        elapsed = time.time() - t0
        try:
            signal.alarm(0)
        except Exception:
            pass
        err_type = "TIMEOUT" if isinstance(e, _JobTimeout) else "FAIL"
        print(f"{tag} {err_type:7s} {config_name:22s} | {ds_name:25s} | "
              f"fold {fold_idx+1}: {e} ({elapsed:.0f}s)", flush=True)
        if not isinstance(e, _JobTimeout):
            traceback.print_exc()

        result = {
            'config': config_name, 'dataset': ds_name,
            'openml_id': openml_id, 'fold': fold_idx,
            'n': n, 'd': d, 'classes': n_classes,
            'accuracy': np.nan, 'time': elapsed,
            'error': str(e)[:200],
        }
        append_result(result, output_file)
        torch.cuda.empty_cache()
        gc.collect()
        return result

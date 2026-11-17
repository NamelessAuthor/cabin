#!/usr/bin/env python3
"""
Hyperparameter sensitivity of Cabin (Appendix D).

One-at-a-time sweep around the paper config (cabin-v33-2slots: VSF ns=2,
k=32, VCUDA, edge dropout 0.2, label smoothing 0.1), single model unless
noted, on tub folds 0-2 of 28 small/medium CC18 tasks. Same pinned seed
for every config, so differences are paired.

  k (n_groups)     4, 8, 16, [32], 64, auto = clip(d//3, 2, 64)
  edge dropout     0, 0.1, [0.2], 0.3
  label smoothing  0, 0.05, [0.1], 0.2
  ensemble size M  one boosted M=10 run, scored after 1, 3, 5 and 10 models

Task scope: CC18 tasks of Table 2 whose single model takes <=130 s, minus
those already at >=99% accuracy (monks-problems-2, banknote-authentication,
car, MiceProtein, analcatdata_authorship, tic-tac-toe).

Model code: ../rerun_dag/cabin_v33_lib.py (verbatim tub snapshot).
Outputs: out/<config>/<task>__f<fold>.json. Finished jobs are skipped.

Usage:
  python run_sensitivity.py --workers 3
  python run_sensitivity.py --test
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback


def legacy_seed(s):
    """Model seed used for the paper runs: base seed int(s / 1000 * 1000) % 100000.

    Equal to s except for a float-rounding off-by-one for some s > 1000; kept so that
    rerunning this script reproduces the published numbers exactly."""
    return int(s / 1000 * 1000) % 100000

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, '..', 'rerun_dag')
TUB = os.path.expanduser('~/tub')
OUT = os.environ.get('SENS_OUT', os.path.join(HERE, 'out'))
SEED = 42
FOLDS = [0, 1, 2]

TASKS = ['balance-scale', 'blood-transfusion-service-center', 'analcatdata_dmft',
         'ilpd', 'breast-w', 'diabetes', 'dresses-sales', 'Australian',
         'climate-model-simulation-crashes', 'kc2', 'credit-approval',
         'eucalyptus', 'wdbc', 'vehicle', 'cmc', 'vowel', 'credit-g',
         'cylinder-bands', 'pc1', 'mfeat-morphological', 'soybean',
         'qsar-biodeg', 'kc1', 'pc4', 'steel-plates-fault', 'pc3', 'wilt',
         'segment']

BASE_KW = dict(dag_type='factored', acyclicity='vcuda', n_models=1,
               shrinkage=0.1, base_alpha=2.0, calibrate=False,
               add_missing_indicators=True, edge_dropout=0.2,
               label_smoothing=0.1, nam_shape=True, n_slots=2, n_groups=32)

CONFIGS = {
    'base': {},
    'k4': dict(n_groups=4), 'k8': dict(n_groups=8), 'k16': dict(n_groups=16),
    'k64': dict(n_groups=64), 'kauto': dict(n_groups='auto'),
    'ed0': dict(edge_dropout=0.0), 'ed0.1': dict(edge_dropout=0.1),
    'ed0.3': dict(edge_dropout=0.3),
    'ls0': dict(label_smoothing=0.0), 'ls0.05': dict(label_smoothing=0.05),
    'ls0.2': dict(label_smoothing=0.2),
    'ens10': dict(n_models=10, calibrate=True),
}
STAGES = [1, 3, 5, 10]


def preprocess(X_train, X_test):
    """Same layout as the tub wrapper: [numeric..., ordinal-encoded categorical...]."""
    import numpy as np
    from sklearn.preprocessing import OrdinalEncoder
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    num_cols = X_train.select_dtypes(exclude='object').columns.tolist()
    enc = None
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        enc.fit(X_train[cat_cols])

    def tf(X):
        parts = []
        if num_cols:
            parts.append(X[num_cols].to_numpy(dtype=np.float32))
        if cat_cols:
            parts.append(enc.transform(X[cat_cols]).astype(np.float32))
        return np.hstack(parts) if len(parts) > 1 else parts[0]

    cat_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    return tf(X_train), tf(X_test), cat_idx


def run_job(job):
    config, task_name, fold_i = job
    cdir = os.path.join(OUT, config)
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f'{task_name}__f{fold_i}.json')
    if os.path.exists(path):
        return
    import numpy as np
    import torch
    sys.path.insert(0, TUB)
    sys.path.insert(0, LIB)
    from tub.loader import load_task
    from cabin_v33_lib import CausalBoostClassifier, MissingnessHandler

    t0 = time.perf_counter()
    rec = {'config': config, 'task': task_name, 'fold': fold_i, 'seed': SEED}
    try:
        task = load_task(task_name)
        fold = task.folds[fold_i]
        tr, te = np.array(fold['train']), np.array(fold['test'])
        Xtr, Xte, cat_idx = preprocess(task.X.iloc[tr], task.X.iloc[te])
        ytr, yte = task.y[tr], task.y[te]
        kw = {**BASE_KW, **CONFIGS[config]}
        if kw['n_groups'] == 'auto':
            # Release rule (model/cabin.py of the original release): k = clip(d//3, 2, 64)
            # with d counted after the missingness indicators are added.
            d_int = MissingnessHandler().fit_transform(np.array(Xtr, dtype=float)).shape[1]
            kw['n_groups'] = min(max(2, d_int // 3), 64)
        rec['n_groups'] = kw['n_groups']
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        clf = CausalBoostClassifier(**kw)
        clf.random_state = legacy_seed(SEED)
        clf.fit(Xtr, ytr, cat_indices=cat_idx)
        rec['accuracy'] = float((clf.predict(Xte) == yte).mean())
        if kw['n_models'] > 1:
            Xp, _ = clf._preprocess_missing(np.array(Xte, dtype=float),
                                            clf._cat_indices_internal, fit=False)
            acc = 0
            for m_i, m in enumerate(clf.models, 1):
                acc = acc + clf.shrinkage * m.predict_logits(Xp)
                if m_i in STAGES:
                    rec[f'accuracy_M{m_i}'] = float((acc.argmax(1) == yte).mean())
        rec.update(n=int(len(task.y)), d=len(task.feature_columns),
                   n_classes=task.n_classes, status='ok')
    except Exception as e:
        rec.update(status='error', error=f'{type(e).__name__}: {e}',
                   traceback=traceback.format_exc())
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(f"[{rec['status']}] {config:7s} {task_name:34s} f{fold_i} "
          f"acc={rec.get('accuracy', float('nan')):.4f} {rec['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--configs', nargs='+', default=list(CONFIGS))
    p.add_argument('--test', action='store_true')
    args = p.parse_args()
    tasks, folds, configs = TASKS, FOLDS, args.configs
    if args.test:
        OUT = os.environ['SENS_OUT'] = os.path.join(HERE, 'out_test')
        tasks, folds, configs = ['balance-scale'], [0], ['base', 'kauto', 'ens10']
    # Fold-major order: every config completes fold 0 before fold 1 starts,
    # so a partial run still gives paired comparisons. ens10 runs last in each fold.
    single = [c for c in configs if c != 'ens10']
    order = single + [c for c in configs if c == 'ens10']
    jobs = [(c, t, f) for f in folds for c in order for t in tasks]
    print(f'{len(jobs)} jobs, {args.workers} workers -> {OUT}', flush=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        pool.map(run_job, jobs, chunksize=1)

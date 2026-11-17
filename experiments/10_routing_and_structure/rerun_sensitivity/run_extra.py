#!/usr/bin/env python3
"""
Extra configs on the same 28 tasks x folds 0-2 as run_sensitivity.py.

  base_s1, base_s2  paper config with seeds 1 and 2 (seed-noise floor for
                    reading the sensitivity deltas; 'base' uses seed 42)
  fixrestore        paper config with the checkpoint-restore fix (only
                    finetune-phase checkpoints can be restored, so the DAG
                    is always trained); cabin_v33_lib_fixrestore.py
  randdag           DAG frozen at its random init (w_lr=0, same schedule)
  nodag             no message passing (W forced to 0)
  fixrestore_s1/_s2 restore fix with seeds 1 and 2 (is its loss real?)
  base_ens5, randdag_ens5, nodag_ens5
                    the paper's 5-model calibrated ensemble with the learned
                    DAG, a frozen random DAG and no message passing (does the
                    DAG matter once the models are ensembled?)

Outputs: out/<config>/<task>__f<fold>.json, same format as run_sensitivity.py.
"""
import argparse, json, multiprocessing as mp, os, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_sensitivity import TASKS, FOLDS, BASE_KW, LIB, TUB, OUT, preprocess


def legacy_seed(s):
    """Model seed used for the paper runs: base seed int(s / 1000 * 1000) % 100000.

    Equal to s except for a float-rounding off-by-one for some s > 1000; kept so that
    rerunning this script reproduces the published numbers exactly."""
    return int(s / 1000 * 1000) % 100000

CONFIGS = {
    'base_s1': dict(seed=1), 'base_s2': dict(seed=2),
    'fixrestore': dict(lib='cabin_v33_lib_fixrestore'),
    'randdag': dict(kw=dict(w_lr=0.0)),
    'nodag': dict(nodag=True),
    'fixrestore_s1': dict(lib='cabin_v33_lib_fixrestore', seed=1),
    'fixrestore_s2': dict(lib='cabin_v33_lib_fixrestore', seed=2),
    'base_ens5': dict(kw=dict(n_models=5, calibrate=True)),
    'randdag_ens5': dict(kw=dict(w_lr=0.0, n_models=5, calibrate=True)),
    'nodag_ens5': dict(nodag=True, kw=dict(n_models=5, calibrate=True)),
}


def run_job(job):
    config, task_name, fold_i = job
    spec = CONFIGS[config]
    seed = spec.get('seed', 42)
    cdir = os.path.join(OUT, config)
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f'{task_name}__f{fold_i}.json')
    if os.path.exists(path):
        return
    import importlib
    import numpy as np
    import torch
    sys.path.insert(0, TUB)
    sys.path.insert(0, LIB)
    from tub.loader import load_task
    L = importlib.import_module(spec.get('lib', 'cabin_v33_lib'))
    t0 = time.perf_counter()
    rec = {'config': config, 'task': task_name, 'fold': fold_i, 'seed': seed}
    try:
        task = load_task(task_name)
        fold = task.folds[fold_i]
        tr, te = np.array(fold['train']), np.array(fold['test'])
        Xtr, Xte, cat_idx = preprocess(task.X.iloc[tr], task.X.iloc[te])
        ytr, yte = task.y[tr], task.y[te]
        if spec.get('nodag'):
            orig = L.CausalBoostNet.get_W
            L.CausalBoostNet.get_W = lambda self: orig(self) * 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        clf = L.CausalBoostClassifier(**{**BASE_KW, **spec.get('kw', {})})
        clf.random_state = legacy_seed(seed)
        clf.fit(Xtr, ytr, cat_indices=cat_idx)
        rec['accuracy'] = float((clf.predict(Xte) == yte).mean())
        rec.update(n=int(len(task.y)), d=len(task.feature_columns),
                   n_classes=task.n_classes, status='ok')
    except Exception as e:
        rec.update(status='error', error=f'{type(e).__name__}: {e}',
                   traceback=traceback.format_exc())
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(f"[{rec['status']}] {config:10s} {task_name:34s} f{fold_i} "
          f"acc={rec.get('accuracy', float('nan')):.4f} {rec['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--configs', nargs='+', default=list(CONFIGS))
    p.add_argument('--test', action='store_true')
    a = p.parse_args()
    tasks, folds = (['balance-scale'], [0]) if a.test else (TASKS, FOLDS)
    if a.test:
        OUT = os.environ['SENS_OUT'] = os.path.join(HERE, 'out_test')
    jobs = [(c, t, f) for f in folds for c in a.configs for t in tasks]
    print(f'{len(jobs)} jobs, {a.workers} workers -> {OUT}', flush=True)
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool:
        pool.map(run_job, jobs, chunksize=1)

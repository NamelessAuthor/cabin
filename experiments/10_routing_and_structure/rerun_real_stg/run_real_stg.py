#!/usr/bin/env python3
"""
Track C: do the stochastic feature gates (STG) found on the high-d synthetic
grid (paper/rerun_headopt) help on real data?

Same jobs as rerun_diag/run_diag.py: 28 tasks x folds 0-2 of rerun_sensitivity,
seed 42, clock pinned, same preprocessing and BASE_KW. Library: the
rerun_headopt copy (its directory is put FIRST on sys.path).

  ref          paper config single net (should reproduce rerun_diag diag_base)
  ref_nodag    same, W = 0
  stg          + STG gates, gate_lambda=0.01, per-feature (sum) penalty
  stg_l003     gate_lambda=0.003
  stg_l03      gate_lambda=0.03
  stg_nodag    stg + W = 0
  *_ens5       n_models=5, calibrate=True (as rerun_sensitivity base_ens5)
Outputs: out/<config>/<task>__f<fold>.json
Usage: python run_real_stg.py --workers 5 [--configs ...] [--tasks ...] [--test]
High-d extension (HIGHD below): --tasks $HIGHD, same out/ directory.
"""
import argparse, json, multiprocessing as mp, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
HEADOPT = os.path.join(HERE, '..', 'rerun_headopt')
sys.path.insert(0, os.path.join(HERE, '..', 'rerun_sensitivity'))
from run_sensitivity import TASKS, FOLDS, BASE_KW, TUB, preprocess


def legacy_seed(s):
    """Model seed used for the paper runs: base seed int(s / 1000 * 1000) % 100000.

    Equal to s except for a float-rounding off-by-one for some s > 1000; kept so that
    rerunning this script reproduces the published numbers exactly."""
    return int(s / 1000 * 1000) % 100000

OUT = os.environ.get('REALSTG_OUT', os.path.join(HERE, 'out'))
SEED = 42
STG = dict(gate='stg', gate_lambda=0.01, gate_reduce='sum', gate_sigma=0.5)
CONFIGS = {
    'ref': {},
    'ref_nodag': dict(nodag=True),
    'stg': dict(diag=STG),
    'stg_l003': dict(diag={**STG, 'gate_lambda': 0.003}),
    'stg_l03': dict(diag={**STG, 'gate_lambda': 0.03}),
    'stg_nodag': dict(diag=STG, nodag=True),
}
ENS = dict(n_models=5, calibrate=True)
CONFIGS.update({k + '_ens5': dict(v, kw=ENS) for k, v in list(CONFIGS.items())})


def run_job(job):
    config, task_name, fold_i = job
    spec = CONFIGS[config]
    cdir = os.path.join(OUT, config)
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f'{task_name}__f{fold_i}.json')
    if os.path.exists(path):
        return
    import numpy as np
    import torch
    sys.path.insert(0, TUB)
    sys.path.insert(0, HEADOPT)          # must win over any other cabin_v33_lib_diag
    from tub.loader import load_task
    import cabin_v33_lib_diag as L
    assert os.path.samefile(os.path.dirname(L.__file__), HEADOPT), L.__file__
    L.DIAG.update(spec.get('diag', {}))
    t0 = time.perf_counter()
    rec = {'config': config, 'task': task_name, 'fold': fold_i, 'seed': SEED}
    try:
        task = load_task(task_name)
        fold = task.folds[fold_i]
        tr, te = np.array(fold['train']), np.array(fold['test'])
        Xtr, Xte, cat_idx = preprocess(task.X.iloc[tr], task.X.iloc[te])
        ytr, yte = task.y[tr], task.y[te]
        if spec.get('nodag'):
            orig = L.CausalBoostNet.get_W
            L.CausalBoostNet.get_W = lambda self: orig(self) * 0
        clf = L.CausalBoostClassifier(**{**BASE_KW, **spec.get('kw', {})})
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        clf.random_state = legacy_seed(SEED)
        clf.fit(Xtr, ytr, cat_indices=cat_idx)
        rec['accuracy'] = float((clf.predict(Xte) == yte).mean())
        m = clf.models[0]
        rec['best_epoch'] = getattr(m, 'best_epoch', None)
        mu = getattr(getattr(m, 'model', None), 'gate_mu', None)
        if mu is not None:
            g = mu.detach().clamp(0, 1).cpu().numpy()
            rec['gates_open'] = int((g > 0).sum())
            rec['n_gates'] = int(len(g))
        rec.update(n=int(len(task.y)), d=len(task.feature_columns),
                   n_classes=task.n_classes, status='ok')
    except Exception as e:
        rec.update(status='error', error=f'{type(e).__name__}: {e}',
                   traceback=traceback.format_exc())
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(f"[{rec['status']}] {config:14s} {task_name:34s} f{fold_i} "
          f"acc={rec.get('accuracy', float('nan')):.4f} open={rec.get('gates_open', '-')}/{rec.get('n_gates', '-')} "
          f"be={rec.get('best_epoch')} {rec['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--configs', nargs='+', default=[c for c in CONFIGS if not c.endswith('_ens5')])
    p.add_argument('--test', action='store_true')
    p.add_argument('--tasks', nargs='+', help='override the 28 rerun_sensitivity tasks')
    a = p.parse_args()
    tasks, folds = (['balance-scale', 'credit-approval'], [0]) if a.test else (a.tasks or TASKS, FOLDS)
    if a.test:
        OUT = os.environ['REALSTG_OUT'] = os.path.join(HERE, 'out_test')
    jobs = [(c, t, f) for f in folds for c in a.configs for t in tasks]
    print(f'{len(jobs)} jobs, {a.workers} workers -> {OUT}', flush=True)
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool:
        pool.map(run_job, jobs, chunksize=1)

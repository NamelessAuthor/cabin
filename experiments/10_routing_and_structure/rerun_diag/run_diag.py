#!/usr/bin/env python3
"""
Diagnostics of how Cabin uses its learned DAG, on the 28 tasks x folds 0-2 of
rerun_sensitivity (single network, seed 42, clock pinned, same preprocessing).

Library: cabin_v33_lib_diag.py (cabin_v33_lib + DIAG switches; with default
switches it is the original library, so diag_base must reproduce 'base').

  diag_base   original config; logs val loss, val acc and test acc at every
              epoch (curve), and the restored epoch
  selacc      checkpoint selected on validation accuracy (ties: loss)
  selacc_ft   same, finetune-phase checkpoints only (DAG always trained)
  add         additive head: per-node MLPs summed, so feature interactions
              can only come from message passing along the DAG
  add_rand    additive head, DAG frozen at its random init (w_lr=0)
  add_nodag   additive head, no message passing (W=0)
  *_ft        base / rand / nodag / add / add_rand / add_nodag with
              finetune-only restore (the learned DAG is always trained)

Every config also records the test accuracy of the trained model with W set
to 0 at prediction time only (acc_W0): does the trained model rely on its
messages?

Outputs: out/<config>/<task>__f<fold>.json
Usage:   python run_diag.py --workers 3 [--configs ...] [--test]
"""
import argparse, json, multiprocessing as mp, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rerun_sensitivity'))
from run_sensitivity import TASKS, FOLDS, BASE_KW, TUB, preprocess


def legacy_seed(s):
    """Model seed used for the paper runs: base seed int(s / 1000 * 1000) % 100000.

    Equal to s except for a float-rounding off-by-one for some s > 1000; kept so that
    rerunning this script reproduces the published numbers exactly."""
    return int(s / 1000 * 1000) % 100000

OUT = os.environ.get('DIAG_OUT', os.path.join(HERE, 'out'))
SEED = 42
CONFIGS = {
    'diag_base': dict(curve=True),
    'selacc': dict(diag=dict(select='acc')),
    'selacc_ft': dict(diag=dict(select='acc', restore_from='finetune')),
    'add': dict(diag=dict(head='additive')),
    'add_rand': dict(diag=dict(head='additive'), kw=dict(w_lr=0.0)),
    'add_nodag': dict(diag=dict(head='additive'), nodag=True),
    # Same comparisons with finetune-only restore, so that the learned DAG is
    # always trained (otherwise learned and random graphs often coincide).
    'base_ft': dict(diag=dict(restore_from='finetune')),
    'rand_ft': dict(diag=dict(restore_from='finetune'), kw=dict(w_lr=0.0)),
    'nodag_ft': dict(diag=dict(restore_from='finetune'), nodag=True),
    'add_ft': dict(diag=dict(head='additive', restore_from='finetune')),
    'add_rand_ft': dict(diag=dict(head='additive', restore_from='finetune'), kw=dict(w_lr=0.0)),
    'add_nodag_ft': dict(diag=dict(head='additive', restore_from='finetune'), nodag=True),
}


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
    sys.path.insert(0, HERE)
    from tub.loader import load_task
    import cabin_v33_lib_diag as L
    L.DIAG.update(spec.get('diag', {}))
    t0 = time.perf_counter()
    rec = {'config': config, 'task': task_name, 'fold': fold_i, 'seed': SEED}
    try:
        task = load_task(task_name)
        fold = task.folds[fold_i]
        tr, te = np.array(fold['train']), np.array(fold['test'])
        Xtr, Xte, cat_idx = preprocess(task.X.iloc[tr], task.X.iloc[te])
        ytr, yte = task.y[tr], task.y[te]
        orig_get_W = L.CausalBoostNet.get_W
        if spec.get('nodag'):
            L.CausalBoostNet.get_W = lambda self: orig_get_W(self) * 0
        clf = L.CausalBoostClassifier(**{**BASE_KW, **spec.get('kw', {})})
        if spec.get('curve'):
            def hook(cinn):
                Xp, _ = clf._preprocess_missing(np.array(Xte, dtype=float),
                                                clf._cat_indices_internal)
                return float((cinn.predict(Xp) == yte).mean())
            L.DIAG['curve_hook'] = hook
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        clf.random_state = legacy_seed(SEED)
        try:
            clf.fit(Xtr, ytr, cat_indices=cat_idx)
        finally:
            L.DIAG['curve_hook'] = None
        rec['accuracy'] = float((clf.predict(Xte) == yte).mean())
        m = clf.models[0]
        rec['best_epoch'] = m.best_epoch
        if spec.get('curve'):
            rec['curve'] = m.curve
        # Test-time ablation: same trained weights, W = 0 at prediction only
        L.CausalBoostNet.get_W = lambda self: orig_get_W(self) * 0
        rec['acc_W0'] = float((clf.predict(Xte) == yte).mean())
        L.CausalBoostNet.get_W = orig_get_W
        rec.update(n=int(len(task.y)), d=len(task.feature_columns),
                   n_classes=task.n_classes, status='ok')
    except Exception as e:
        rec.update(status='error', error=f'{type(e).__name__}: {e}',
                   traceback=traceback.format_exc())
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(f"[{rec['status']}] {config:10s} {task_name:34s} f{fold_i} "
          f"acc={rec.get('accuracy', float('nan')):.4f} "
          f"W0={rec.get('acc_W0', float('nan')):.4f} {rec['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--configs', nargs='+', default=list(CONFIGS))
    p.add_argument('--test', action='store_true')
    a = p.parse_args()
    tasks, folds = (['balance-scale'], [0]) if a.test else (TASKS, FOLDS)
    if a.test:
        OUT = os.environ['DIAG_OUT'] = os.path.join(HERE, 'out_test')
    jobs = [(c, t, f) for f in folds for c in a.configs for t in tasks]
    print(f'{len(jobs)} jobs, {a.workers} workers -> {OUT}', flush=True)
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool:
        pool.map(run_job, jobs, chunksize=1)

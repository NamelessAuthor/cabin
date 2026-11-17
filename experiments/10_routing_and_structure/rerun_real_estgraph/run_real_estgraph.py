#!/usr/bin/env python3
"""
Track E: does Track A's data-only estimated graph help on REAL data?

Graph: per-node LightGBM screen on the training fold (each internal column
regressed on the others, top-3 parents by gain, OR-symmetrised), exactly
rerun_dagopt.estimate_graph('est_lgbm3'). It is estimated on the columns the
network actually sees (after MissingnessHandler: median-imputed features +
missingness indicators; ordinal-encoded categoricals treated as numeric),
standardised, then expanded to slot nodes with kron(G, 1_{n_slots}) and passed
as fixed_dag (restore 'all', harmless for a fixed graph).

Jobs as rerun_diag/run_diag.py (seed 42, clock pinned, same preprocess and
BASE_KW). Library: rerun_headopt copy (reproduces rerun_diag diag_base exactly,
checked by Track C), directory put FIRST on sys.path and asserted.

  add_est     additive head + estimated graph
  mlp_est     MLP head (paper) + estimated graph
  add_nodag   additive head, W = 0 (for the high-d tasks; rerun_diag has the 28)
Outputs: out/<config>/<task>__f<fold>.json
Usage: python run_real_estgraph.py --workers 5 [--configs ...] [--tasks ...] [--test]
"""
import argparse, json, multiprocessing as mp, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
HEADOPT = os.path.join(HERE, '..', 'rerun_headopt')
DAGOPT = os.path.join(HERE, '..', 'rerun_dagopt')
sys.path.insert(0, os.path.join(HERE, '..', 'rerun_sensitivity'))
from run_sensitivity import TASKS, FOLDS, BASE_KW, TUB, preprocess


def legacy_seed(s):
    """Model seed used for the paper runs: base seed int(s / 1000 * 1000) % 100000.

    Equal to s except for a float-rounding off-by-one for some s > 1000; kept so that
    rerunning this script reproduces the published numbers exactly."""
    return int(s / 1000 * 1000) % 100000

OUT = os.environ.get('ESTG_OUT', os.path.join(HERE, 'out'))
SEED = 42
HIGHD = ['mfeat-zernike', 'spambase', 'splice', 'mfeat-karhunen', 'analcatdata_authorship',
         'ozone-level-8hr', 'mfeat-fourier', 'MiceProtein', 'mfeat-factors', 'semeion']
CONFIGS = {
    'add_est': dict(head='additive', graph='est'),
    'mlp_est': dict(head='mlp', graph='est'),
    'add_nodag': dict(head='additive', graph='nodag'),
    # Paper model (M=5 boosted ensemble, calibrated) with the estimated graph;
    # pairs with rerun_sensitivity base_ens5 / nodag_ens5 (same seed 42).
    'mlp_est_ens5': dict(head='mlp', graph='est', kw=dict(n_models=5, calibrate=True)),
    # Density-matched control: a random symmetric graph with
    # the same number of edges as the estimated graph of the same fold.
    'mlp_randsparse': dict(head='mlp', graph='randsparse'),
    'mlp_randsparse_ens5': dict(head='mlp', graph='randsparse', kw=dict(n_models=5, calibrate=True)),
}


def estimate_lgbm3(X):
    """Copy of rerun_dagopt.run_dagopt.estimate_graph('est_lgbm3')."""
    import numpy as np
    import lightgbm as lgb
    d = X.shape[1]
    G = np.zeros((d, d), np.float32)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        if X[:, i].std() == 0:
            continue
        m = lgb.LGBMRegressor(n_estimators=100, num_leaves=8, learning_rate=0.1, verbose=-1,
                              importance_type='gain', n_jobs=1, random_state=0).fit(X[:, others], X[:, i])
        imp = m.feature_importances_
        for j in np.argsort(-imp)[:3]:
            if imp[j] > 0:
                G[others[j], i] = 1
    return np.maximum(G, G.T)


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
    sys.path.insert(0, HEADOPT)
    from tub.loader import load_task
    import cabin_v33_lib_diag as L
    assert os.path.samefile(os.path.dirname(L.__file__), HEADOPT), L.__file__
    L.DIAG.update(head=spec['head'], restore_from='all')
    t0 = time.perf_counter()
    rec = {'config': config, 'task': task_name, 'fold': fold_i, 'seed': SEED}
    try:
        task = load_task(task_name)
        fold = task.folds[fold_i]
        tr, te = np.array(fold['train']), np.array(fold['test'])
        Xtr, Xte, cat_idx = preprocess(task.X.iloc[tr], task.X.iloc[te])
        ytr, yte = task.y[tr], task.y[te]
        kw = {**BASE_KW, **spec.get('kw', {})}
        if spec['graph'] == 'nodag':
            orig = L.CausalBoostNet.get_W
            L.CausalBoostNet.get_W = lambda self: orig(self) * 0
        elif spec['graph'] in ('est', 'randsparse'):
            Xi = L.MissingnessHandler().fit_transform(np.array(Xtr, dtype=float))
            Xi = (Xi - Xi.mean(0)) / (Xi.std(0) + 1e-8)
            t1 = time.perf_counter()
            G = estimate_lgbm3(Xi)
            if spec['graph'] == 'randsparse':
                d = G.shape[0]
                iu = np.triu_indices(d, 1)
                m = int(G[iu].sum())
                rs = np.random.RandomState(1000 + fold_i)
                pick = rs.choice(len(iu[0]), m, replace=False)
                G = np.zeros_like(G)
                G[iu[0][pick], iu[1][pick]] = 1
                G = np.maximum(G, G.T)
            rec.update(graph_seconds=round(time.perf_counter() - t1, 1), d_internal=int(G.shape[0]),
                       n_edges=int(G.sum() / 2), mean_degree=float(G.sum(0).mean()))
            kw['fixed_dag'] = np.kron(G, np.ones((kw['n_slots'], kw['n_slots']), np.float32))
        clf = L.CausalBoostClassifier(**kw)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        clf.random_state = legacy_seed(SEED)
        clf.fit(Xtr, ytr, cat_indices=cat_idx)
        rec['accuracy'] = float((clf.predict(Xte) == yte).mean())
        rec['best_epoch'] = getattr(clf.models[0], 'best_epoch', None)
        rec.update(n=int(len(task.y)), d=len(task.feature_columns),
                   n_classes=task.n_classes, status='ok')
    except Exception as e:
        rec.update(status='error', error=f'{type(e).__name__}: {e}',
                   traceback=traceback.format_exc())
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(f"[{rec['status']}] {config:10s} {task_name:34s} f{fold_i} "
          f"acc={rec.get('accuracy', float('nan')):.4f} deg={rec.get('mean_degree', '-')} "
          f"be={rec.get('best_epoch')} {rec['seconds']:.0f}s {rec.get('error', '')}", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--configs', nargs='+', default=['add_est', 'mlp_est'])
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--test', action='store_true')
    a = p.parse_args()
    tasks, folds = (['credit-approval', 'balance-scale'], [0]) if a.test else (a.tasks or TASKS, FOLDS)
    if a.test:
        OUT = os.environ['ESTG_OUT'] = os.path.join(HERE, 'out_test')
    jobs = [(c, t, f) for f in folds for c in a.configs for t in tasks]
    print(f'{len(jobs)} jobs, {a.workers} workers -> {OUT}', flush=True)
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool:
        pool.map(run_job, jobs, chunksize=1)

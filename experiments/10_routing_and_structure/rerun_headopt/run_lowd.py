#!/usr/bin/env python3
"""Low-d sanity check of headopt variants on the rerun_interact SCM (d 10/20, n=2000).

Uses rerun_interact/run_interact.py's make_setting (same seeds), so results pair
with rerun_interact/out/{mlp_nodag_all,mlp_true_all,cabin,baselines}.
Outputs out_lowd/<variant>__<graph>/d{d}_lam{lam}_r{rep}.json
Usage: python run_lowd.py --variants stg1_mlp --graphs nodag true --lams 0 1 --reps 3
"""
import argparse, json, multiprocessing as mp, os, sys, time, traceback
import numpy as np


def legacy_seed(s):
    """Model seed used for the paper runs: base seed int(s / 1000 * 1000) % 100000.

    Equal to s except for a float-rounding off-by-one for some s > 1000; kept so that
    rerunning this script reproduces the published numbers exactly."""
    return int(s / 1000 * 1000) % 100000

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out_lowd')
sys.path.insert(0, os.path.join(HERE, '..', 'rerun_sensitivity'))
sys.path.insert(0, os.path.join(HERE, '..', 'rerun_interact'))
sys.path.insert(0, HERE)


def run_job(job):
    model, (d, lam, rep) = job
    import cabin_v33_lib_diag as L     # import first: later sys.path inserts must not shadow it
    import run_interact as RI          # make_setting (N=2000 from env default)
    from run_highd import parse_model
    odir = os.path.join(OUT, model)
    os.makedirs(odir, exist_ok=True)
    path = os.path.join(odir, RI.name(d, lam, rep) + '.json')
    if os.path.exists(path):
        return
    import torch
    assert 'gate' in L.DIAG, L.__file__
    from run_sensitivity import BASE_KW
    t0 = time.perf_counter()
    Xtr, ytr, Xte, yte, Mf, rec = RI.make_setting(d, lam, rep)
    rec['model'] = model
    try:
        spec, graph = parse_model(model)
        L.DIAG.update(restore_from='all', **spec.get('diag', {}))
        kw = dict(BASE_KW, **spec.get('kw', {}))
        if graph == 'rand':
            kw['w_lr'] = 0.0
        elif graph == 'nodag':
            orig = L.CausalBoostNet.get_W
            L.CausalBoostNet.get_W = lambda self: orig(self) * 0
        elif graph == 'true':
            kw['fixed_dag'] = np.kron(Mf.astype(np.float32), np.ones((kw['n_slots'], kw['n_slots']), np.float32))
        seed = rec['seed'] + rep        # same seeding as run_interact
        torch.manual_seed(seed)
        np.random.seed(seed)
        clf = L.CausalBoostClassifier(**kw)
        clf.random_state = legacy_seed(seed)
        clf.fit(Xtr.astype(np.float32), ytr, cat_indices=[])
        rec['acc'] = float((clf.predict(Xte.astype(np.float32)) == yte).mean())
        rec['best_epoch'] = clf.models[0].best_epoch
        rec['status'] = 'ok'
    except Exception as e:
        rec.update(status='error', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(f"[{rec['status']}] {model:18s} {RI.name(d, lam, rep)} acc={rec.get('acc', float('nan')):.3f} {rec['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--variants', nargs='+', required=True)
    ap.add_argument('--graphs', nargs='+', default=['nodag'])
    ap.add_argument('--lams', type=float, nargs='+', default=[0.0, 1.0])
    ap.add_argument('--reps', type=int, default=5)
    a = ap.parse_args()
    grid = [(d, lam, r) for d in (10, 20) for lam in a.lams for r in range(a.reps)]
    jobs = [(f'{v}__{g}', s) for s in grid for v in a.variants for g in a.graphs]
    print(f'{len(jobs)} jobs -> {OUT}', flush=True)
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool:
        pool.map(run_job, jobs, chunksize=1)

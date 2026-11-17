#!/usr/bin/env python3
"""
Does "no DAG beats the learned DAG on high-d real tasks" (Track C, single net)
hold for the paper's model (M=5 boosted ensemble, calibrated)?

Runs run_extra.py's base_ens5 / nodag_ens5 / randdag_ens5 (same library,
preprocessing, seed 42, pinned clock) on 10 high-d CC18 tasks + cnae-9,
folds 0-2, by importing run_extra.run_job with SENS_OUT redirected here.
Also LightGBM (default params) on the same folds: --lgbm.

Outputs: out/<config>/<task>__f<fold>.json, lgbm.csv
Usage:  python run_highd_real.py --workers 5 [--configs ...] [--lgbm]
"""
import argparse, multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['SENS_OUT'] = os.path.join(HERE, 'out')
sys.path.insert(0, os.path.join(HERE, '..', 'rerun_sensitivity'))
from run_extra import run_job  # noqa: E402  (reads SENS_OUT at import)
from run_sensitivity import TUB, preprocess  # noqa: E402

HIGHD = ['mfeat-zernike', 'spambase', 'splice', 'mfeat-karhunen', 'analcatdata_authorship',
         'ozone-level-8hr', 'mfeat-fourier', 'MiceProtein', 'mfeat-factors', 'semeion']
EXTRA = ['cnae-9']
FOLDS = [0, 1, 2]


def lgbm_all(tasks):
    import csv
    import numpy as np, lightgbm as lgb
    sys.path.insert(0, TUB)
    from tub.loader import load_task
    path = os.path.join(HERE, 'lgbm.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['task', 'fold', 'd', 'n', 'lgbm'])
        for t in tasks:
            task = load_task(t)
            for f in FOLDS:
                fold = task.folds[f]
                tr, te = np.array(fold['train']), np.array(fold['test'])
                Xtr, Xte, _ = preprocess(task.X.iloc[tr], task.X.iloc[te])
                clf = lgb.LGBMClassifier(verbose=-1, random_state=0, n_jobs=2)
                clf.fit(np.asarray(Xtr, float), task.y[tr])
                acc = float((clf.predict(np.asarray(Xte, float)) == task.y[te]).mean())
                w.writerow([t, f, len(task.feature_columns), len(task.y), acc]); fh.flush()
                print('lgbm', t, f, round(acc, 4), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--configs', nargs='+', default=['base_ens5', 'nodag_ens5', 'randdag_ens5'])
    p.add_argument('--tasks', nargs='+', default=HIGHD + EXTRA)
    p.add_argument('--lgbm', action='store_true')
    a = p.parse_args()
    if a.lgbm:
        lgbm_all(a.tasks)
        sys.exit()
    # Priority: base/nodag pairs on the 10 tasks, then cnae-9, then randdag.
    pri = [c for c in a.configs if c != 'randdag_ens5']
    jobs = [(c, t, f) for f in FOLDS for t in a.tasks if t not in EXTRA for c in pri]
    jobs += [(c, t, f) for f in FOLDS for t in a.tasks if t in EXTRA for c in pri]
    if 'randdag_ens5' in a.configs:
        jobs += [('randdag_ens5', t, f) for f in FOLDS for t in a.tasks]
    print(f'{len(jobs)} jobs, {a.workers} workers -> {os.environ["SENS_OUT"]}', flush=True)
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool:
        pool.map(run_job, jobs, chunksize=1)

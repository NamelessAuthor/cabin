#!/usr/bin/env python3
"""
How non-additive is each real dataset? LightGBM (default) vs LightGBM with
stumps (num_leaves=2: a boosted additive model, no interactions), on the tub
folds 0-2 and the same preprocessing as the Cabin reruns.

Tasks: the 28 rerun_sensitivity tasks (where per-dataset DAG effects were
measured) plus every task of the DAG-extraction scope (55 CC18 tasks).
Output: interaction_real.csv (task, fold, d, n, lgbm, lgbm_stumps).
"""
import glob, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rerun_sensitivity'))
sys.path.insert(0, os.path.join(HERE, '..', 'rerun_dag'))
from run_sensitivity import TASKS, FOLDS, TUB, preprocess
sys.path.insert(0, TUB)
from tub.loader import load_task
import lightgbm as lgb

dag_tasks = sorted({json.load(open(f))['task'] for f in
                    glob.glob(os.path.join(HERE, '..', 'rerun_dag', 'out', 'base', '*.json'))})
tasks = list(dict.fromkeys(list(TASKS) + dag_tasks))
out = os.path.join(HERE, 'interaction_real.csv')
done = set()
if os.path.exists(out):
    done = set(map(tuple, pd.read_csv(out)[['task', 'fold']].astype(str).values))
for t in tasks:
    task = load_task(t)
    for f in FOLDS:
        if (t, str(f)) in done:
            continue
        fold = task.folds[f]
        tr, te = np.array(fold['train']), np.array(fold['test'])
        Xtr, Xte, cat = preprocess(task.X.iloc[tr], task.X.iloc[te])
        ytr, yte = task.y[tr], task.y[te]
        r = dict(task=t, fold=f, d=Xtr.shape[1], n=len(task.y), n_classes=task.n_classes,
                 in_sens=t in TASKS)
        for m, kw in {'lgbm': {}, 'lgbm_stumps': dict(num_leaves=2, n_estimators=500)}.items():
            clf = lgb.LGBMClassifier(verbose=-1, random_state=0, n_jobs=4, **kw)
            clf.fit(Xtr, ytr, categorical_feature=cat or 'auto')
            r[m] = float((clf.predict(Xte) == yte).mean())
        pd.DataFrame([r]).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
        print(t, f, r['lgbm'], r['lgbm_stumps'], flush=True)

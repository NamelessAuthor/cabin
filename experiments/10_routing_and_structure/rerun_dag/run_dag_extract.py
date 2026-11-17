#!/usr/bin/env python3
"""
Re-run Cabin variants on CC18 to recover the learned DAG structure
(Figure 1 and §6.2 of the reviewed submission; that data was lost).

For each (variant, task), train a single model (n_models=1, i.e. the first
model of the boosted ensemble) on tub fold 0 and save the learned
group-level S (or S_1..S_K for MoE), the assignment A, the feature-level
W = A S A^T, the binary VCUDA edges, and the fold-0 test accuracy.

Variants use the exact configs of the tub runs behind Table 2:
  base         cabin-v33-2slots       (VSF ns=2, k=32, VCUDA)
  moe          cabin-v33-moe4-shared  (+ n_dags=4, shared_A)
  contrastive  cabin-v33-contrastive  (+ lambda_contrastive=0.5)

Outputs: out/<variant>/<task>.npz and out/<variant>/<task>.json.
Finished jobs are skipped, so the script can be stopped and restarted.

Usage:
  python run_dag_extract.py --workers 3
  python run_dag_extract.py --test          # one small task, all variants
"""

import argparse
import glob
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
OUT = os.path.join(HERE, 'out')
TUB = os.path.expanduser('~/tub')
TUB_RESULTS = os.path.join(HERE, '..', 'results', 'tub', 'results')
FOLD = 0
SEED = 42

BASE_KW = dict(dag_type='factored', acyclicity='vcuda', n_models=1,
               shrinkage=0.1, base_alpha=2.0, calibrate=False,
               add_missing_indicators=True, edge_dropout=0.2,
               label_smoothing=0.1, nam_shape=True, n_slots=2)
VARIANTS = {
    'base': {},
    'moe': dict(n_dags=4, shared_A=True),
    'contrastive': dict(lambda_contrastive=0.5),
}


def tasks_in_scope():
    """Tasks with a complete cabin-v33-2slots result (Table 2 scope + 3)."""
    names = []
    for f in glob.glob(os.path.join(TUB_RESULTS, 'cabin-v33-2slots', '*.json')):
        with open(f) as fh:
            names.append(json.load(fh)['task'])
    return sorted(names)


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


def extract(net):
    """Learned DAG matrices of a fitted CausalBoostNet, in eval mode."""
    import torch
    net.eval()
    out = {}
    with torch.no_grad():
        if getattr(net, 'n_dags', 1) > 1:
            Ss = [net._get_S_idx(i) for i in range(net.n_dags)]
            A = net._get_A_idx(0)
            out['S_list'] = torch.stack(Ss).cpu().numpy()
            out['A'] = A.cpu().numpy()
            out['W_list'] = torch.stack([A @ S @ A.T for S in Ss]).cpu().numpy()
            logits = torch.stack(list(net.edge_logits_list))
        else:
            S = net._get_S()
            A = torch.softmax(net.A_raw, dim=1)
            out['S'] = S.cpu().numpy()
            out['A'] = A.cpu().numpy()
            out['W'] = net.get_W().cpu().numpy()
            logits = net.edge_logits
        tau = max(float(net._tau_edge_current), 0.01)
        edges = (torch.sigmoid(logits / tau) > 0.5).float()
        edges = edges * net.no_self_loop_k
        out['edges'] = edges.cpu().numpy()
    return out


def run_job(job):
    variant, task_name = job
    vdir = os.path.join(OUT, variant)
    os.makedirs(vdir, exist_ok=True)
    stem = os.path.join(vdir, task_name)
    if os.path.exists(stem + '.json'):
        return
    import numpy as np
    import torch
    sys.path.insert(0, TUB)
    sys.path.insert(0, HERE)
    from tub.loader import load_task
    from cabin_v33_lib import CausalBoostClassifier

    t0 = time.perf_counter()
    rec = {'variant': variant, 'task': task_name, 'fold': FOLD, 'seed': SEED}
    try:
        task = load_task(task_name)
        fold = task.folds[FOLD]
        tr, te = np.array(fold['train']), np.array(fold['test'])
        Xtr, Xte, cat_idx = preprocess(task.X.iloc[tr], task.X.iloc[te])
        ytr, yte = task.y[tr], task.y[te]
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        clf = CausalBoostClassifier(**BASE_KW, **VARIANTS[variant])
        # Diagnostic only: count joint-phase steps (the only phase where the DAG
        # trains) and note how many had run when each best checkpoint was saved.
        # Early stopping restores the last one, which can predate DAG training.
        import cabin_v33_lib as L
        joint_steps, saves = [0], []
        dag_loss, state_dict = L.CausalBoostNet.dag_loss, L.CausalBoostNet.state_dict

        def counting_dag_loss(self):
            joint_steps[0] += 1
            return dag_loss(self)

        def recording_state_dict(self, *a, **k):
            saves.append(joint_steps[0])
            return state_dict(self, *a, **k)

        L.CausalBoostNet.dag_loss = counting_dag_loss
        L.CausalBoostNet.state_dict = recording_state_dict
        clf.random_state = legacy_seed(SEED)
        try:
            clf.fit(Xtr, ytr, cat_indices=cat_idx)
        finally:
            L.CausalBoostNet.dag_loss, L.CausalBoostNet.state_dict = dag_loss, state_dict
        rec.update(joint_steps_total=joint_steps[0],
                   joint_steps_at_restored=saves[-1] if saves else None)
        acc = float((clf.predict(Xte) == yte).mean())
        mats = extract(clf.models[0].model)
        np.savez_compressed(stem + '.npz', **mats)
        rec.update(accuracy=acc, n=int(len(task.y)), d=len(task.feature_columns),
                   n_classes=task.n_classes, n_categorical=len(cat_idx),
                   status='ok')
    except Exception as e:
        rec.update(status='error', error=f'{type(e).__name__}: {e}',
                   traceback=traceback.format_exc())
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    with open(stem + '.json', 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(f"[{rec['status']}] {variant:12s} {task_name:40s} "
          f"acc={rec.get('accuracy', float('nan')):.4f} {rec['seconds']:.0f}s", flush=True)


def task_size(name):
    with open(os.path.join(TUB, 'tasks', name, 'task.json')) as fh:
        m = json.load(fh)
    return m['n_samples'] * m['n_features']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--variants', nargs='+', default=list(VARIANTS))
    p.add_argument('--tasks', nargs='+', default=None)
    p.add_argument('--test', action='store_true')
    args = p.parse_args()

    tasks = args.tasks or tasks_in_scope()
    if args.test:
        tasks = ['Australian']
    tasks = sorted(tasks, key=task_size)   # small tasks first
    jobs = [(v, t) for t in tasks for v in args.variants]
    print(f'{len(jobs)} jobs ({len(tasks)} tasks x {len(args.variants)} variants), '
          f'{args.workers} workers -> {OUT}', flush=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        pool.map(run_job, jobs, chunksize=1)

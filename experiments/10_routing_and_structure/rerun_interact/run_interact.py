#!/usr/bin/env python3
"""
How much non-additivity does the data need before the DAG helps?

Synthetic SCM on an Erdos-Renyi DAG (edge j -> i for j > i). Every node with
parents P is generated as
    X_i = sqrt(1-lam) * z(sum_j w_j tanh(X_j))
        + sqrt(lam)   * z(sum_{j<k in P} c_jk X_j X_k)  + noise(0.5)
where z() standardizes to unit variance, so lam in [0, 1] moves the mechanisms
from purely additive (lam=0, as in rerun_synthetic) to purely pairwise
interactions between co-parents (lam=1). Nodes with one parent stay additive.
The target is the most-parented node with >= 2 parents and >= 1 child, binned
into 3 quantile classes. Predicting it from the other variables requires
combining its parents with each other (co-parent interactions, not along
feature edges) and each child with the child's other parents (along the
feature edges p -> child), so the true feature DAG is the right wiring for
the second kind.

Models (single network, paper config, finetune-only restore so the learned
DAG is always trained, unless stated):
  {mlp,add}_{learn,rand,nodag,true}   head x graph: learned / frozen random /
                                       none / true feature DAG (oracle, fixed)
  cabin                                paper config as published (mlp head,
                                       learned DAG, default restore)
  baselines                            LightGBM, LightGBM stumps (additive),
                                       logistic regression
Grid: d in {10, 20}, p = 0.3, lam in {0, .25, .5, .75, 1}, reps 0-4,
n = 2000 train / 2000 test (--n to change; output then goes to out_n<N>).
Outputs: out/<model>/<setting>.json.  Usage: python run_interact.py --workers 3 [--test]
"""
import argparse, json, multiprocessing as mp, os, sys, time, traceback
import numpy as np


def legacy_seed(s):
    """Model seed used for the paper runs: base seed int(s / 1000 * 1000) % 100000.

    Equal to s except for a float-rounding off-by-one for some s > 1000; kept so that
    rerunning this script reproduces the published numbers exactly."""
    return int(s / 1000 * 1000) % 100000

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('INTERACT_OUT', os.path.join(HERE, 'out'))
sys.path.insert(0, os.path.join(HERE, '..', 'rerun_sensitivity'))
sys.path.insert(0, os.path.join(HERE, '..', 'rerun_diag'))
N = int(os.environ.get('INTERACT_N', 2000))
LAMS = tuple(float(x) for x in os.environ.get('INTERACT_LAMS', '0,0.25,0.5,0.75,1').split(','))
GRID = [(d, lam, r) for d in (10, 20) for lam in LAMS for r in range(5)]
NETS = {f'{h}_{g}': dict(head='mlp' if h == 'mlp' else 'additive', graph=g, restore='finetune')
        for h in ('mlp', 'add') for g in ('learn', 'rand', 'nodag', 'true')}
NETS['cabin'] = dict(head='mlp', graph='learn', restore='all')
# Fixed graphs need no finetune-only restore (the graph never changes), so
# compare them in the stronger default-restore regime as well.
NETS.update({f'mlp_{g}_all': dict(head='mlp', graph=g, restore='all') for g in ('rand', 'nodag', 'true')})
NETS.update({f'add_{g}_all': dict(head='additive', graph=g, restore='all') for g in ('rand', 'nodag', 'true')})
# Column-normalised W: each node receives the weighted MEAN of its parents, so
# a dense random graph and a sparse binary true graph send messages of the same scale.
NETS.update({f'{h}_{g}_all_norm': dict(head='mlp' if h == 'mlp' else 'additive', graph=g, restore='all', norm=True)
             for h in ('mlp', 'add') for g in ('rand', 'true')})


def sample_scm(d, p, seed):
    rng = np.random.RandomState(seed)
    M = np.zeros((d, d), bool)
    for i in range(d):
        for j in range(i + 1, d):
            M[j, i] = rng.rand() < p
    w = rng.uniform(0.3, 1.0, (d, d)) * rng.choice([-1, 1], (d, d))
    c = rng.uniform(0.3, 1.0, (d, d)) * rng.choice([-1, 1], (d, d))
    return M, w, c


def sample_data(M, w, c, lam, n, seed, noise=0.5):
    rng = np.random.RandomState(seed)
    d = M.shape[0]
    X = np.zeros((n, d))
    z = lambda v: (v - v.mean()) / (v.std() + 1e-8)
    for i in range(d - 1, -1, -1):
        P = np.where(M[:, i])[0]
        if len(P) == 0:
            X[:, i] = rng.randn(n)
            continue
        add = z(np.tanh(X[:, P]) @ w[P, i])
        if len(P) >= 2 and lam > 0:
            inter = sum(c[j, k] * X[:, j] * X[:, k] for a, j in enumerate(P) for k in P[a + 1:])
            sig = np.sqrt(1 - lam) * add + np.sqrt(lam) * z(inter)
        else:
            sig = add
        X[:, i] = sig + rng.randn(n) * noise
        X[:, i] = z(X[:, i])
    return X


def make_setting(d, lam, rep):
    seed = 9000 + rep + d * 1000
    p = 0.3
    for tries in range(50):           # same graph for every lam (seed independent of lam)
        M, w, c = sample_scm(d, p, seed + 100 * tries)
        npar, nch = M.sum(0), M.sum(1)
        cand = [i for i in range(d) if npar[i] >= 2 and nch[i] >= 1]
        if cand:
            break
    t = max(cand, key=lambda i: (npar[i], -i))
    X = sample_data(M, w, c, lam, 2 * N, seed + 1)
    Xtr, Xte = X[:N], X[N:]
    q = np.quantile(Xtr[:, t], [1 / 3, 2 / 3])
    lab = lambda Z: np.digitize(Z[:, t], q).astype(int)
    feats = [i for i in range(d) if i != t]
    info = dict(d=d, lam=lam, rep=rep, seed=seed, target=int(t), n_parents=int(npar[t]),
                n_children=int(nch[t]), n_edges_feats=int(M[np.ix_(feats, feats)].sum()))
    return Xtr[:, feats], lab(Xtr), Xte[:, feats], lab(Xte), M[np.ix_(feats, feats)], info


def name(d, lam, rep):
    return f'd{d}_lam{lam}_r{rep}'


def run_job(job):
    model, (d, lam, rep) = job
    odir = os.path.join(OUT, model)
    os.makedirs(odir, exist_ok=True)
    path = os.path.join(odir, name(d, lam, rep) + '.json')
    if os.path.exists(path):
        return
    import torch
    t0 = time.perf_counter()
    Xtr, ytr, Xte, yte, Mf, rec = make_setting(d, lam, rep)
    rec['model'] = model
    try:
        if model == 'baselines':
            import lightgbm as lgb
            from sklearn.linear_model import LogisticRegression
            for m, clf in {'lgbm': lgb.LGBMClassifier(verbose=-1, random_state=0, n_jobs=2),
                           'lgbm_stumps': lgb.LGBMClassifier(verbose=-1, random_state=0, n_jobs=2, num_leaves=2,
                                                             n_estimators=500),
                           'logreg': LogisticRegression(max_iter=2000)}.items():
                clf.fit(Xtr, ytr)
                rec[m] = float((clf.predict(Xte) == yte).mean())
        else:
            import cabin_v33_lib_diag as L
            from run_sensitivity import BASE_KW
            spec = NETS[model]
            L.DIAG.update(head=spec['head'], restore_from=spec['restore'])
            kw = dict(BASE_KW)
            if spec['graph'] == 'rand':
                kw['w_lr'] = 0.0
            elif spec['graph'] == 'nodag':
                orig = L.CausalBoostNet.get_W
                L.CausalBoostNet.get_W = lambda self: orig(self) * 0
            elif spec['graph'] == 'true':
                kw['fixed_dag'] = np.kron(Mf.astype(np.float32), np.ones((kw['n_slots'], kw['n_slots']), np.float32))
            if spec.get('norm'):
                orig_n = L.CausalBoostNet.get_W
                L.CausalBoostNet.get_W = lambda self: (lambda W: W / W.sum(0, keepdim=True).clamp_min(1e-6))(orig_n(self))
            seed = rec['seed'] + rep
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
    accs = ' '.join(f'{k}={rec[k]:.3f}' for k in ('acc', 'lgbm', 'lgbm_stumps', 'logreg') if k in rec)
    print(f"[{rec['status']}] {model:10s} {name(d, lam, rep):18s} {accs} {rec['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--models', nargs='+', default=['baselines'] + list(NETS))
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--n', type=int, help='train (= test) size; sets INTERACT_N for the workers')
    ap.add_argument('--lams', help='comma-separated lambdas; sets INTERACT_LAMS')
    a = ap.parse_args()
    if a.n:
        os.environ['INTERACT_N'] = str(a.n); N = a.n
        OUT = os.environ['INTERACT_OUT'] = os.environ.get('INTERACT_OUT', os.path.join(HERE, f'out_n{a.n}'))
    if a.lams:
        os.environ['INTERACT_LAMS'] = a.lams
        LAMS = tuple(float(x) for x in a.lams.split(','))
        GRID = [(d, lam, r) for d in (10, 20) for lam in LAMS for r in range(5)]
    grid = GRID
    if a.test:
        grid = [(10, 0.0, 0), (10, 1.0, 0)]
        OUT = os.environ['INTERACT_OUT'] = os.path.join(HERE, 'out_test')
    jobs = [(m, g) for g in grid for m in a.models]
    print(f'{len(jobs)} jobs, {a.workers} workers -> {OUT}', flush=True)
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool:
        pool.map(run_job, jobs, chunksize=1)

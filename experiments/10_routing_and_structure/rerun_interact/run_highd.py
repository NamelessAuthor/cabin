#!/usr/bin/env python3
"""
Can the true DAG help when there are many features and few samples?

At d in {10, 20} with n = 2000 or 10000 (run_interact.py), the true feature
DAG never beats a frozen random DAG: a dense graph can express everything a
sparse one can. The remaining hypothesis is statistical: a correct sparse
graph rules out spurious interactions, which should matter when d is large
and n small. This script tests that regime with a richer SCM.

SCM (Erdos-Renyi DAG, edge j -> i for j > i, edge prob 4/d, ~2 parents per
node on average). Each edge gets a random mechanism f_ji drawn from
    tanh, quadratic, sine, ReLU hinge, RBF bump, step (tree-like), log|x|
(the spirit of the Synthefy / TabPFN priors), standardized. A node with
parents P is
    X_i = sqrt(1-lam) * z(sum_j w_ji f_ji(X_j))
        + sqrt(lam)   * z(sum_{j<k in P} c_jk g_ji(X_j) g_ki(X_k)) + noise(0.5)
then standardized; g are independently drawn mechanisms. lam = 0 is additive,
lam = 1 pure pairwise interactions (nodes with one parent stay additive).
Target: most-parented node with >= 2 parents and >= 1 child, 3 quantile
classes. Train n in {500, 1000}, test 5000 fresh samples.

Models (single network, paper config; fixed graphs use the default restore,
which is harmless when the graph never changes):
  cabin                     paper config (MLP head, learned DAG)
  {mlp,add}_{true,rand,nodag}  MLP / additive head x true feature DAG (oracle)
                            / frozen random DAG / no message passing
  baselines                 LightGBM, LightGBM stumps (additive), logistic
                            regression, LightGBM on the target's Markov
                            blanket only (feature-selection oracle)
Grid: d in {50, 100}, n in {500, 1000}, lam in {0, 1}, reps 0-4.
Outputs: out_highd/<model>/d{d}_n{n}_lam{lam}_r{rep}.json
Usage: python run_highd.py --workers 5 [--test]
"""
import argparse, json, multiprocessing as mp, os, sys, time, traceback
import numpy as np


def legacy_seed(s):
    """Model seed used for the paper runs: base seed int(s / 1000 * 1000) % 100000.

    Equal to s except for a float-rounding off-by-one for some s > 1000; kept so that
    rerunning this script reproduces the published numbers exactly."""
    return int(s / 1000 * 1000) % 100000

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('HIGHD_OUT', os.path.join(HERE, 'out_highd'))
sys.path.insert(0, os.path.join(HERE, '..', 'rerun_sensitivity'))
sys.path.insert(0, os.path.join(HERE, '..', 'rerun_diag'))
N_TEST = 5000
GRID = [(d, n, lam, r) for d in (50, 100) for n in (500, 1000) for lam in (0.0, 1.0) for r in range(5)]
NETS = {'cabin': dict(head='mlp', graph='learn')}
NETS.update({f'{h}_{g}': dict(head='mlp' if h == 'mlp' else 'additive', graph=g)
             for h in ('mlp', 'add') for g in ('true', 'rand', 'nodag')})
MECHS = ('tanh', 'quad', 'sin', 'hinge', 'rbf', 'step', 'log')


def mech(rng):
    """A random univariate mechanism, applied to a standardized input."""
    k = MECHS[rng.randint(len(MECHS))]
    a, b = rng.uniform(0.5, 2.0), rng.uniform(-1.0, 1.0)
    return {
        'tanh': lambda x: np.tanh(a * x),
        'quad': lambda x: b * x + (x - b) ** 2,
        'sin': lambda x: np.sin(a * x + 3 * b),
        'hinge': lambda x: np.maximum(x - b, 0),
        'rbf': lambda x: np.exp(-((x - b) * a) ** 2),
        'step': lambda x: (x > b).astype(float),
        'log': lambda x: np.log1p(np.abs(x - b)),
    }[k]


def z(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def sample_scm(d, seed):
    rng = np.random.RandomState(seed)
    p = 4.0 / d
    M = np.zeros((d, d), bool)
    for i in range(d):
        for j in range(i + 1, d):
            M[j, i] = rng.rand() < p
    F, G = {}, {}
    for i in range(d):
        for j in np.where(M[:, i])[0]:
            F[j, i], G[j, i] = mech(rng), mech(rng)
    w = rng.uniform(0.3, 1.0, (d, d)) * rng.choice([-1, 1], (d, d))
    c = rng.uniform(0.3, 1.0, (d, d)) * rng.choice([-1, 1], (d, d))
    return M, F, G, w, c


def sample_data(M, F, G, w, c, lam, n, seed, noise=0.5):
    rng = np.random.RandomState(seed)
    d = M.shape[0]
    X = np.zeros((n, d))
    for i in range(d - 1, -1, -1):
        P = np.where(M[:, i])[0]
        if len(P) == 0:
            X[:, i] = rng.randn(n)
            continue
        add = z(sum(w[j, i] * z(F[j, i](X[:, j])) for j in P))
        if len(P) >= 2 and lam > 0:
            gz = {j: z(G[j, i](X[:, j])) for j in P}
            inter = sum(c[j, k] * gz[j] * gz[k] for a, j in enumerate(P) for k in P[a + 1:])
            sig = np.sqrt(1 - lam) * add + np.sqrt(lam) * z(inter)
        else:
            sig = add
        X[:, i] = z(sig + rng.randn(n) * noise)
    return X


def make_setting(d, n, lam, rep):
    seed = 11000 + rep + d * 1000      # same graph and mechanisms for every n and lam
    for tries in range(50):
        M, F, G, w, c = sample_scm(d, seed + 100 * tries)
        npar, nch = M.sum(0), M.sum(1)
        cand = [i for i in range(d) if npar[i] >= 2 and nch[i] >= 1]
        if cand:
            break
    t = max(cand, key=lambda i: (npar[i], -i))
    X = sample_data(M, F, G, w, c, lam, n + N_TEST, seed + 1)
    Xtr, Xte = X[:n], X[n:]
    q = np.quantile(Xtr[:, t], [1 / 3, 2 / 3])
    lab = lambda Z: np.digitize(Z[:, t], q).astype(int)
    feats = [i for i in range(d) if i != t]
    ch = np.where(M[t])[0]
    mb = set(np.where(M[:, t])[0]) | set(ch) | {int(j) for k in ch for j in np.where(M[:, k])[0]}
    mb.discard(t)
    mb_cols = sorted(feats.index(int(j)) for j in mb)
    info = dict(d=d, n=n, lam=lam, rep=rep, seed=seed, target=int(t), n_parents=int(npar[t]),
                n_children=int(nch[t]), mb_size=len(mb_cols), n_edges_feats=int(M[np.ix_(feats, feats)].sum()))
    return Xtr[:, feats], lab(Xtr), Xte[:, feats], lab(Xte), M[np.ix_(feats, feats)], mb_cols, info


def name(d, n, lam, rep):
    return f'd{d}_n{n}_lam{lam}_r{rep}'


def run_job(job):
    model, (d, n, lam, rep) = job
    odir = os.path.join(OUT, model)
    os.makedirs(odir, exist_ok=True)
    path = os.path.join(odir, name(d, n, lam, rep) + '.json')
    if os.path.exists(path):
        return
    import torch
    t0 = time.perf_counter()
    Xtr, ytr, Xte, yte, Mf, mb_cols, rec = make_setting(d, n, lam, rep)
    rec['model'] = model
    try:
        if model == 'baselines':
            import lightgbm as lgb
            from sklearn.linear_model import LogisticRegression
            mods = {'lgbm': (lgb.LGBMClassifier(verbose=-1, random_state=0, n_jobs=2), slice(None)),
                    'lgbm_stumps': (lgb.LGBMClassifier(verbose=-1, random_state=0, n_jobs=2, num_leaves=2,
                                                       n_estimators=500), slice(None)),
                    'logreg': (LogisticRegression(max_iter=2000), slice(None)),
                    'lgbm_mb': (lgb.LGBMClassifier(verbose=-1, random_state=0, n_jobs=2), mb_cols)}
            for m, (clf, cols) in mods.items():
                clf.fit(Xtr[:, cols], ytr)
                rec[m] = float((clf.predict(Xte[:, cols]) == yte).mean())
        else:
            import cabin_v33_lib_diag as L
            from run_sensitivity import BASE_KW
            spec = NETS[model]
            L.DIAG.update(head=spec['head'], restore_from='all')
            kw = dict(BASE_KW)
            if spec['graph'] == 'rand':
                kw['w_lr'] = 0.0
            elif spec['graph'] == 'nodag':
                orig = L.CausalBoostNet.get_W
                L.CausalBoostNet.get_W = lambda self: orig(self) * 0
            elif spec['graph'] == 'true':
                kw['fixed_dag'] = np.kron(Mf.astype(np.float32), np.ones((kw['n_slots'], kw['n_slots']), np.float32))
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
    accs = ' '.join(f'{k}={rec[k]:.3f}' for k in ('acc', 'lgbm', 'lgbm_stumps', 'logreg', 'lgbm_mb') if k in rec)
    print(f"[{rec['status']}] {model:10s} {name(d, n, lam, rep):22s} {accs} {rec['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--models', nargs='+', default=['baselines'] + list(NETS))
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    grid = GRID
    if a.test:
        grid = [(100, 1000, 1.0, 0)]
        OUT = os.environ['HIGHD_OUT'] = os.path.join(HERE, 'out_highd_test')
    jobs = [(m, g) for g in grid for m in a.models]
    print(f'{len(jobs)} jobs, {a.workers} workers -> {OUT}', flush=True)
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool:
        pool.map(run_job, jobs, chunksize=1)

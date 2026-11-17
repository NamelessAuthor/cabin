#!/usr/bin/env python3
"""
Track A: make Cabin's learned DAG reach the true-DAG oracle in the high-d
synthetic regime of rerun_interact/run_highd.py (same make_setting, seeds and
test sets, so results pair with rerun_interact/out_highd).

Library: cabin_v33_lib_diag.py here = rerun_diag copy + DAGOPT switches
(feature-level DAG, sparse edge init, soft edges). Each variant sets
DIAG / DAGOPT / classifier kwargs; see VARIANTS. Every learned-DAG run also
records how well the learned feature graph matches the true one (AUROC of
the feature-level |W| for directed edges and for the undirected skeleton).

Outputs: out/<variant>/d{d}_n{n}_lam{lam}_r{rep}.json
Usage:  python run_dagopt.py --variants v1 v2 --grid screen --workers 5
"""
import argparse, json, multiprocessing as mp, os, sys, time, traceback
import numpy as np


def legacy_seed(s):
    """Model seed used for the paper runs: base seed int(s / 1000 * 1000) % 100000.

    Equal to s except for a float-rounding off-by-one for some s > 1000; kept so that
    rerunning this script reproduces the published numbers exactly."""
    return int(s / 1000 * 1000) % 100000

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('DAGOPT_OUT', os.path.join(HERE, 'out'))
sys.path.insert(0, os.path.join(HERE, '..', 'rerun_sensitivity'))
sys.path.insert(0, HERE)   # this directory's cabin_v33_lib_diag (with DAGOPT) must win
N_TEST = 5000
GRIDS = {
    'full': [(d, n, lam, r) for d in (50, 100) for n in (500, 1000) for lam in (0.0, 1.0) for r in range(5)],
    'screen': [(d, 1000, lam, r) for d in (50, 100) for lam in (0.0, 1.0) for r in range(5)],
    'screen100': [(100, 1000, lam, r) for lam in (0.0, 1.0) for r in range(5)],
}
# Variant = head + DIAG/DAGOPT overrides + classifier kwargs.
def V(head, restore='all', **opt):
    kw = opt.pop('kw', {})
    return dict(head=head, restore=restore, dagopt=opt, kw=kw)
VARIANTS = {}
for h, hn in (('mlp', 'mlp'), ('add', 'additive')):
    VARIANTS.update({
        f'{h}_fac_ft': V(hn, 'finetune'),                                   # paper DAG, DAG always trained
        f'{h}_fac_s3': V(hn, init=-3.0),                                    # paper DAG, sparse init
        f'{h}_fac_s3_ft': V(hn, 'finetune', init=-3.0),
        f'{h}_feat': V(hn, dag='feat'),                                     # feature-level DAG, paper init
        f'{h}_feat_s3': V(hn, dag='feat', init=-3.0),                       # feature-level, sparse init
        f'{h}_feat_s3_ft': V(hn, 'finetune', dag='feat', init=-3.0),
        f'{h}_feat_s3_nw_ft': V(hn, 'finetune', dag='feat', init=-3.0, kw=dict(warmup_epochs=0, joint_epochs=120)),
        f'{h}_feat_s3_soft_ft': V(hn, 'finetune', dag='feat', init=-3.0, soft=True),
        f'{h}_feat_s3_lr5x_ft': V(hn, 'finetune', dag='feat', init=-3.0, kw=dict(w_lr=0.025)),
        f'{h}_feat_ft': V(hn, 'finetune', dag='feat'),
        # Fixed graphs (restore 'all' is harmless): oracles and graphs estimated from X
        f'{h}_true_sym': dict(V(hn), graph='true_sym'),       # true skeleton, both directions
        f'{h}_true_moral': dict(V(hn), graph='true_moral'),   # moral graph (skeleton + co-parents)
        f'{h}_true_rev': dict(V(hn), graph='true_rev'),       # true DAG reversed
        f'{h}_est_lgbm3': dict(V(hn), graph='est_lgbm3'),     # per-node LightGBM, top-3 gain, OR-symmetrised
        f'{h}_est_lasso': dict(V(hn), graph='est_lasso'),     # neighbourhood selection (LassoCV), OR rule
        # density-matched random controls
        f'{h}_rand_true': dict(V(hn), graph='rand_true'),     # random directed graph, as many edges as the true DAG
        f'{h}_rand_est': dict(V(hn), graph='rand_est'),       # random symmetric graph, as many edges as est_lgbm3
        # Skeleton given, direction learned end-to-end (feature-level VCUDA priorities)
        f'{h}_skel_est_ft': dict(V(hn, 'finetune', dag='feat'), skel='est_lgbm3'),
        f'{h}_skel_est_nw_ft': dict(V(hn, 'finetune', dag='feat', kw=dict(warmup_epochs=0, joint_epochs=120)), skel='est_lgbm3'),
        f'{h}_skel_true_ft': dict(V(hn, 'finetune', dag='feat'), skel='true_sym'),
        f'{h}_skel_est_learn_ft': dict(V(hn, 'finetune', dag='feat', init=2.0, mask_edges='learn'), skel='est_lgbm3'),
    })


def estimate_graph(kind, X, Mf):
    """Feature-level adjacency [d, d] (row j -> column i) for a fixed-graph variant."""
    d = X.shape[1]
    T = Mf.astype(np.float32)
    if kind == 'true_sym':
        return np.maximum(T, T.T)
    if kind == 'true_rev':
        return T.T.copy()
    if kind in ('rand_true', 'rand_est'):
        ref = T if kind == 'rand_true' else estimate_graph('est_lgbm3', X, Mf)
        rs = np.random.RandomState(7 * int(ref.sum()) + d)
        if kind == 'rand_true':
            off = np.where(~np.eye(d, dtype=bool).ravel())[0]
            G = np.zeros(d * d, np.float32)
            G[rs.choice(off, int(T.sum()), replace=False)] = 1
            return G.reshape(d, d)
        iu = np.triu_indices(d, 1)
        G = np.zeros((d, d), np.float32)
        pick = rs.choice(len(iu[0]), int(ref[iu].sum()), replace=False)
        G[iu[0][pick], iu[1][pick]] = 1
        return np.maximum(G, G.T)
    if kind == 'true_moral':
        G = np.maximum(T, T.T)
        for i in range(d):
            P = np.where(T[:, i])[0]
            for a in P:
                G[a, P] = 1
        np.fill_diagonal(G, 0)
        return G
    G = np.zeros((d, d), np.float32)
    if kind == 'est_lgbm3':
        import lightgbm as lgb
        for i in range(d):
            others = [j for j in range(d) if j != i]
            m = lgb.LGBMRegressor(n_estimators=100, num_leaves=8, learning_rate=0.1, verbose=-1,
                                  importance_type='gain', n_jobs=1, random_state=0).fit(X[:, others], X[:, i])
            for j in np.argsort(-m.feature_importances_)[:3]:
                G[others[j], i] = 1
    elif kind == 'est_lasso':
        from sklearn.linear_model import LassoCV
        for i in range(d):
            others = [j for j in range(d) if j != i]
            coef = LassoCV(cv=3, n_alphas=20, max_iter=2000).fit(X[:, others], X[:, i]).coef_
            for j in np.where(np.abs(coef) > 1e-6)[0]:
                G[others[j], i] = 1
    else:
        raise ValueError(kind)
    return np.maximum(G, G.T)
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
        import cabin_v33_lib_diag as L
        from run_sensitivity import BASE_KW
        spec = VARIANTS[model]
        L.DIAG.update(head=spec['head'], restore_from=spec['restore'])
        L.DAGOPT.update(dict(dag='factored', init=None, spread=1.0, soft=False, mask=None, mask_edges='fixed'))
        L.DAGOPT.update(spec['dagopt'])
        kw = {**BASE_KW, **spec['kw']}
        L.DAGOPT['mask'] = estimate_graph(spec['skel'], Xtr, Mf) if spec.get('skel') else None
        if spec.get('graph'):
            G = estimate_graph(spec['graph'], Xtr, Mf)
            kw['fixed_dag'] = np.kron(G.astype(np.float32), np.ones((kw['n_slots'], kw['n_slots']), np.float32))
            rec['graph_edges'] = int(G.sum())
        seed = rec['seed'] + rep
        torch.manual_seed(seed)
        np.random.seed(seed)
        clf = L.CausalBoostClassifier(**kw)
        clf.random_state = legacy_seed(seed)
        clf.fit(Xtr.astype(np.float32), ytr, cat_indices=[])
        rec['acc'] = float((clf.predict(Xte.astype(np.float32)) == yte).mean())
        m = clf.models[0]
        rec['best_epoch'] = m.best_epoch
        # Graph recovery: feature-level |W| (mean over slot blocks) vs the true DAG
        from sklearn.metrics import roc_auc_score
        net = m.model
        net.eval()
        with torch.no_grad():
            W = net.get_W().detach().cpu().numpy()
        ns, df = kw['n_slots'], Mf.shape[0]
        Wf = np.abs(W[:df * ns, :df * ns]).reshape(df, ns, df, ns).mean((1, 3))
        off = ~np.eye(df, dtype=bool)
        T = Mf.astype(int)
        rec['auc_dir'] = float(roc_auc_score(T[off], Wf[off]))
        rec['auc_skel'] = float(roc_auc_score((T | T.T)[off], (Wf + Wf.T)[off]))
        rec['w_density'] = float((Wf[off] > 1e-3).mean())
        rec['w_mean'] = float(Wf[off].mean())
        rec['status'] = 'ok'
    except Exception as e:
        rec.update(status='error', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=1)
    accs = ' '.join(f'{k}={rec[k]:.3f}' for k in ('acc', 'auc_dir', 'auc_skel', 'w_density') if k in rec)
    print(f"[{rec['status']}] {model:22s} {name(d, n, lam, rep):22s} {accs} {rec['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--variants', nargs='+', required=True)
    ap.add_argument('--grid', default='screen', choices=list(GRIDS))
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    grid = GRIDS[a.grid]
    if a.test:
        grid = [(100, 1000, 1.0, 0)]
        OUT = os.environ['DAGOPT_OUT'] = os.path.join(HERE, 'out_test')
    jobs = [(m, g) for g in grid for m in a.variants]
    print(f'{len(jobs)} jobs, {a.workers} workers -> {OUT}', flush=True)
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool:
        pool.map(run_job, jobs, chunksize=1)

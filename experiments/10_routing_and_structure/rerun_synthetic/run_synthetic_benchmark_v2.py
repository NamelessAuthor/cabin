"""
NOTEARS-style synthetic benchmark for Cabin structural recovery (v2).

Changes vs v1:
  1. PC baseline dropped. Cabin's k-group DAG lives in a different dimension
     than the true d-node DAG; comparing them against PC (a d-node method)
     was apples-to-oranges. We compare Cabin against the true DAG directly.
  2. Two skel-F1 scores per config:
       - skel_F1_flat: threshold(W = A·S·Aᵀ, top-n_true_edges) vs. true d×d DAG
         (heavily penalized by the factorization — every intra-group pair
         gets an edge whether or not one exists in the truth).
       - skel_F1_grouped: k×k S from Cabin vs. the true DAG "collapsed" to
         Cabin's group space via A (each feature assigned to argmax_k A[i,k]).
  3. For every pair of features that Cabin grouped together, categorize by
     their relationship in the TRUE DAG:
       parent_child      : direct edge either way in the true DAG
       ancestor_descend  : connected via directed path (length > 1)
       sibling           : share a common ancestor (no direct/indirect line)
       decorrelated      : no directed path in either direction
     Report the percentage breakdown and the mean *undirected* shortest-path
     distance in the true DAG's skeleton (∞ → set to d+1 for reporting).
  4. Persist A, S, W_true, group_assignment per config to a pickle so we
     can re-derive metrics later without another training run.

Config sweep is identical to v1:
  - d ∈ {10, 20, 50}
  - edge_p ∈ {0.2, 0.4}
  - nonlinear ∈ {False, True}
  - 3 reps each. 36 configs total.
"""

import os
import sys
from pathlib import Path
import time
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))  # pinned copy of the original model/cabin.py
from cabin import CausalBoostClassifier


def legacy_seed(s):
    """Model seed used for the paper runs: base seed int(s / 1000 * 1000) % 100000.

    Equal to s except for a float-rounding off-by-one for some s > 1000; kept so that
    rerunning this script reproduces the published numbers exactly."""
    return int(s / 1000 * 1000) % 100000

# Rerun 2026-11-01: k from argv ('auto' or an int), local output dir.
N_GROUPS = sys.argv[1] if len(sys.argv) > 1 else 'auto'
N_GROUPS = N_GROUPS if N_GROUPS == 'auto' else int(N_GROUPS)
# Optional rep range (argv[2] = 'start-end', end exclusive) for extra graphs;
# written to a separate dir so the first 3 reps are never overwritten.
REPS = range(3)
_suffix = ''
if len(sys.argv) > 2:
    _a, _b = map(int, sys.argv[2].split('-'))
    REPS, _suffix = range(_a, _b), f'_reps{_a}-{_b}'
OUT_DIR = str(Path(__file__).resolve().parent / f'out_k{N_GROUPS}{_suffix}')
os.makedirs(OUT_DIR, exist_ok=True)


# ── Ground-truth DAG sampling + data generation ────────────────────────────────

def sample_dag(d, edge_p, seed):
    rng = np.random.RandomState(seed)
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.rand() < edge_p:
                w = rng.uniform(0.3, 1.0) * (1 if rng.rand() < 0.5 else -1)
                W[j, i] = w   # edge j → i
    return W


def sample_data(W, n, nonlinear=False, noise_std=0.5, seed=0):
    rng = np.random.RandomState(seed)
    d = W.shape[0]
    X = np.zeros((n, d))
    for i in range(d - 1, -1, -1):
        parents = np.where(W[:, i] != 0)[0]
        parents = parents[parents > i]
        if len(parents) == 0:
            X[:, i] = rng.randn(n) * noise_std
        else:
            signal = (np.tanh(X[:, parents]) if nonlinear else X[:, parents]) @ W[parents, i]
            X[:, i] = signal + rng.randn(n) * noise_std
    return X


def true_dag(W):
    return (W != 0).astype(int)


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_skel_f1(M_pred, M_true):
    """Undirected skeleton F1 on the upper-triangular part."""
    Mp = np.maximum(M_pred, M_pred.T)
    Mt = np.maximum(M_true, M_true.T)
    mask = np.triu(np.ones_like(Mp), k=1).astype(bool)
    pred = Mp[mask]; true = Mt[mask]
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def compute_directed_f1(M_pred, M_true):
    """Directed F1: an edge is a hit only if both endpoints AND direction match."""
    mask = ~np.eye(M_pred.shape[0], dtype=bool)
    pred = M_pred[mask]; true = M_true[mask]
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def threshold_topk(W_mat, n_edges_target):
    """Binarize W by keeping top-n_edges entries (off-diagonal)."""
    d = W_mat.shape[0]
    W_mat = W_mat.copy()
    np.fill_diagonal(W_mat, -np.inf)
    flat = W_mat.flatten()
    if n_edges_target <= 0:
        return np.zeros((d, d), dtype=int)
    topk = np.argsort(flat)[-n_edges_target:]
    M = np.zeros((d, d), dtype=int)
    for idx in topk:
        i, j = idx // d, idx % d
        if flat[idx] > -np.inf:
            M[i, j] = 1
    return M


def all_pairs_reach(M_true):
    """Compute reachability matrix (undirected skeleton, path length in edges).
    R[i,j] = shortest path length in undirected version of true DAG,
    or -1 if disconnected.
    """
    d = M_true.shape[0]
    # Undirected adjacency
    U = np.maximum(M_true, M_true.T)
    dist = np.full((d, d), -1, dtype=int)
    for s in range(d):
        dist[s, s] = 0
        frontier = [s]
        depth = 0
        seen = {s}
        while frontier:
            depth += 1
            new_frontier = []
            for u in frontier:
                for v in np.where(U[u] == 1)[0]:
                    if v not in seen:
                        seen.add(v)
                        dist[s, v] = depth
                        new_frontier.append(v)
            frontier = new_frontier
    return dist


def all_pairs_directed_reach(M_true):
    """R[i,j] = True iff there's a directed path i -> ... -> j in M_true."""
    d = M_true.shape[0]
    R = np.eye(d, dtype=bool)
    # M_true[j,i] = 1 means j -> i. To get "i can reach j" we need to go
    # from parent to child. So use adjacency directly: adj[j,i]=1 -> from j to i.
    # DFS from each node.
    for s in range(d):
        stack = [s]
        while stack:
            u = stack.pop()
            for v in np.where(M_true[u] == 1)[0]:  # u -> v (M_true[u,v]=1)
                if not R[s, v]:
                    R[s, v] = True
                    stack.append(v)
    return R


def relationship(i, j, M_true, reach_dir, reach_undir):
    """Categorize the relation between two nodes in the true DAG.
    reach_dir: directed reachability (M_true[u,v]=1 means u->v).
    reach_undir: undirected shortest-path length (-1 if disconnected).
    """
    if i == j:
        return "same"
    if M_true[i, j] == 1 or M_true[j, i] == 1:
        return "parent_child"
    if reach_dir[i, j] or reach_dir[j, i]:
        return "ancestor_descend"
    # Common ancestor? Look for any k that reaches both i and j directly.
    ancestors_i = np.where(reach_dir[:, i])[0]
    ancestors_j = np.where(reach_dir[:, j])[0]
    if len(np.intersect1d(ancestors_i, ancestors_j)) > 0:
        return "sibling"
    return "decorrelated"


# ── Cabin DAG accessors ────────────────────────────────────────────────────────

def _get_cabin_matrices(mdl):
    """Return numpy A [d,k], S [k,k] for a single Cabin CINN model."""
    with torch.no_grad():
        A_soft = torch.softmax(mdl.A_raw, dim=-1)
        pi = mdl.priority
        pi_diff = pi.unsqueeze(1) - pi.unsqueeze(0)
        mask = torch.sigmoid(-pi_diff / 0.3)
        edge = torch.sigmoid(mdl.edge_logits)
        k = pi.shape[0]
        S = edge * mask * (1.0 - torch.eye(k, device=pi.device))
    return A_soft.cpu().numpy(), S.cpu().numpy()


def collapse_true_dag_to_groups(M_true, group_assign, k):
    """Collapse the true d×d DAG to a k×k DAG using hard group assignment.
    G[a,b] = 1 iff exists (i,j) with group_assign[i]=a, group_assign[j]=b, M_true[i,j]=1.
    Self-loops zeroed out.
    """
    G = np.zeros((k, k), dtype=int)
    d = M_true.shape[0]
    for i in range(d):
        for j in range(d):
            if M_true[i, j] == 1:
                a, b = group_assign[i], group_assign[j]
                if a != b:
                    G[a, b] = 1
    return G


# ── Config runner ──────────────────────────────────────────────────────────────

def run_config(d, edge_p, nonlinear, rep):
    seed = 100 + rep + d * 1000 + int(edge_p * 100) + (1 if nonlinear else 0) * 500
    W_true = sample_dag(d, edge_p, seed)
    X = sample_data(W_true, 2000, nonlinear=nonlinear, noise_std=0.5, seed=seed + 1)
    M_true = true_dag(W_true)
    n_true_edges = int(M_true.sum())

    # Pick the variable with the most parents as the target.
    parents_per_var = M_true.sum(axis=0)
    target_idx = int(np.argmax(parents_per_var))
    if parents_per_var[target_idx] == 0:
        target_idx = 0

    y_cont = X[:, target_idx]
    q = np.quantile(y_cont, np.linspace(0, 1, 4)[1:-1])
    y = np.digitize(y_cont, q).astype(int)

    feature_indices = [i for i in range(d) if i != target_idx]
    X_features = X[:, feature_indices]
    M_true_features = M_true[np.ix_(feature_indices, feature_indices)]
    d_feats = len(feature_indices)
    n_true_edges_features = int(M_true_features.sum())

    # Reachability structures (computed once, on features only)
    reach_dir = all_pairs_directed_reach(M_true_features)
    reach_undir = all_pairs_reach(M_true_features)

    # Cabin training
    X_scaled = StandardScaler().fit_transform(X_features).astype(np.float32)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    accs = []
    A_mats, S_mats, W_flat_mats = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X_scaled, y)):
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)
        clf = CausalBoostClassifier(
            dag_type='factored', acyclicity='vcuda', n_models=1,
            n_groups=N_GROUPS,
            calibrate=False, add_missing_indicators=False,
            joint_epochs=80, warmup_epochs=10, finetune_epochs=20,
        )
        clf.random_state = legacy_seed(seed + fold)
        clf.fit(X_scaled[tr], y[tr], cat_indices=[])
        preds = clf.predict(X_scaled[te])
        accs.append((preds == y[te]).mean())
        # There's only one boost model (n_models=1) but keep the list for safety
        for mw in clf.models:
            A, S = _get_cabin_matrices(mw.model)
            A_mats.append(A)
            S_mats.append(S)
            W_flat_mats.append(A @ S @ A.T)

    A_avg = np.mean(A_mats, axis=0)      # [d_feats, k]
    S_avg = np.mean(S_mats, axis=0)      # [k, k]
    W_flat_avg = np.mean(W_flat_mats, axis=0)  # [d_feats, d_feats]

    k = A_avg.shape[1]
    group_assign = np.argmax(A_avg, axis=1)  # [d_feats], each feature -> group id

    # (a) Flat d×d skeleton F1 + directed F1
    M_pred_flat = threshold_topk(W_flat_avg, n_true_edges_features)
    skel_F1_flat = compute_skel_f1(M_pred_flat, M_true_features)
    dir_F1_flat = compute_directed_f1(M_pred_flat, M_true_features)

    # (b) Grouped k×k skeleton F1 + directed F1
    G_true_grouped = collapse_true_dag_to_groups(M_true_features, group_assign, k)
    G_true_grouped_edges = int(G_true_grouped.sum())  # directed count
    # Threshold Cabin's S at the same edge count as the collapsed truth so the
    # comparison is edge-count-matched.
    S_thresh = threshold_topk(S_avg, G_true_grouped_edges)
    skel_F1_grouped = compute_skel_f1(S_thresh, G_true_grouped)
    dir_F1_grouped = compute_directed_f1(S_thresh, G_true_grouped)

    # (c) Grouping-quality breakdown: for every within-group feature pair,
    # look up the true-DAG relationship.
    rel_counts = Counter()
    grouped_distances = []
    for a in range(k):
        members = np.where(group_assign == a)[0]
        for ii in range(len(members)):
            for jj in range(ii + 1, len(members)):
                i, j = members[ii], members[jj]
                rel_counts[relationship(i, j, M_true_features, reach_dir, reach_undir)] += 1
                dist = reach_undir[i, j]
                grouped_distances.append(dist if dist >= 0 else d_feats + 1)
    total_pairs = sum(rel_counts.values())
    rel_frac = {r: rel_counts[r] / max(total_pairs, 1)
                for r in ["parent_child", "ancestor_descend", "sibling", "decorrelated"]}
    mean_grouped_dist = float(np.mean(grouped_distances)) if grouped_distances else np.nan
    frac_causally_connected = 1.0 - rel_frac["decorrelated"]

    # (d) Group-size distribution
    _, counts = np.unique(group_assign, return_counts=True)
    n_nonempty_groups = int((counts > 0).sum())
    n_singleton_groups = int((counts == 1).sum())

    return {
        'd': d, 'edge_p': edge_p, 'nonlinear': nonlinear, 'rep': rep,
        'n_true_edges': n_true_edges_features,
        'k': k, 'd_feats': d_feats,
        'n_nonempty_groups': n_nonempty_groups,
        'n_singleton_groups': n_singleton_groups,
        'cabin_acc': float(np.mean(accs)),
        'skel_F1_flat': skel_F1_flat,
        'skel_F1_grouped': skel_F1_grouped,
        'dir_F1_flat': dir_F1_flat,
        'dir_F1_grouped': dir_F1_grouped,
        'grouped_pair_count': total_pairs,
        'frac_parent_child': rel_frac["parent_child"],
        'frac_ancestor_descend': rel_frac["ancestor_descend"],
        'frac_sibling': rel_frac["sibling"],
        'frac_decorrelated': rel_frac["decorrelated"],
        'frac_causally_connected': frac_causally_connected,
        'mean_grouped_dist': mean_grouped_dist,
        # Serialize matrices for later ad-hoc analysis without retraining.
        'A_avg': A_avg,
        'S_avg': S_avg,
        'W_flat_avg': W_flat_avg,
        'M_true_features': M_true_features,
        'group_assign': group_assign,
        'target_idx': target_idx,
    }


def main():
    configs = [(d, p, nl, r) for d in [10, 20, 50]
                             for p in [0.2, 0.4]
                             for nl in [False, True]
                             for r in REPS]
    print(f"Total configs: {len(configs)}", flush=True)
    results = []
    t_total = time.time()

    for i, (d, edge_p, nonlinear, rep) in enumerate(configs):
        label = f"d={d} p={edge_p} {'nonlin' if nonlinear else 'lin'} rep={rep}"
        print(f"\n[{i+1}/{len(configs)}] {label}", flush=True)
        t0 = time.time()
        try:
            res = run_config(d, edge_p, nonlinear, rep)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}\n{traceback.format_exc()}", flush=True)
            continue
        elapsed = time.time() - t0
        print(f"  acc={res['cabin_acc']:.3f}  "
              f"skel_F1_flat={res['skel_F1_flat']:.3f}  "
              f"skel_F1_grouped={res['skel_F1_grouped']:.3f}", flush=True)
        print(f"  grouped-pair rel: PC={res['frac_parent_child']:.2f} "
              f"AD={res['frac_ancestor_descend']:.2f} "
              f"S={res['frac_sibling']:.2f} "
              f"D={res['frac_decorrelated']:.2f}  "
              f"mean-dist={res['mean_grouped_dist']:.2f}", flush=True)
        print(f"  ({elapsed:.0f}s, total {(time.time()-t_total)/60:.1f} min)", flush=True)
        results.append(res)

        # Save the CSV (scalars only) and the per-config pickle every iteration.
        scalar_rows = []
        for r in results:
            scalar_rows.append({k: v for k, v in r.items()
                                if k not in ('A_avg', 'S_avg', 'W_flat_avg',
                                             'M_true_features', 'group_assign')})
        pd.DataFrame(scalar_rows).to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)
        with open(os.path.join(OUT_DIR, 'artifacts.pkl'), 'wb') as f:
            pickle.dump(results, f)

    print(f"\nDONE in {(time.time()-t_total)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()

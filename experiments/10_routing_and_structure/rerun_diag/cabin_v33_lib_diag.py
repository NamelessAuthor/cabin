# Copy of ../rerun_dag/cabin_v33_lib.py with DIAG switches (search 'DIAG').
# With the default DIAG values it behaves exactly like the original.
# Copied verbatim from the model.py snapshot of tub run cabin-v33-contrastive
# (the saved model.py of the cabin-v33 CC18 run, part above
# "class Model(BaseModel)"). Superset of the cabin-v33-moe4-shared library; defaults off.
#!/usr/bin/env python3
"""
CausalBoost: DAG-Informed Neural Network for Tabular Classification
====================================================================

Self-contained single-file implementation of CausalBoost as described in:
  "How Far Can DAG-Informed Neural Networks Go?
   A Systematic Design Study for Tabular Classification"

All design dimensions from the paper are configurable:

  Dim 1 - DAG learning: end-to-end SDCD (only mode, no two-stage)
  Dim 2 - DAG parameterization: factored | standard | block | latent
  Dim 3 - Acyclicity enforcement: vcuda | spectral | dagma | dpdag
  Dim 4 - Message passing: n_rounds (1 = champion, 2-3 for ablation)
  Dim 5 - Feature representation: entity embeddings + missingness indicators
  Dim 6 - Ensemble: boosted (M models) or single, with calibration
  Dim 7 - Training: 3-phase schedule, label smoothing, edge dropout

Champion config (paper default):
  dag_type='factored', acyclicity='vcuda', n_rounds=1,
  n_models=5, shrinkage=0.1, calibrate=True, add_missing_indicators=True

Usage:
  from causal_boost import CausalBoostClassifier

  # Champion config (default)
  clf = CausalBoostClassifier()
  clf.fit(X_train, y_train, cat_indices=[2, 5])
  preds = clf.predict(X_test)

  # Single model, no ensemble
  clf = CausalBoostClassifier(n_models=1, calibrate=False)

  # Ablation: standard DAG with spectral radius
  clf = CausalBoostClassifier(dag_type='standard', acyclicity='spectral')

  # Ablation: multi-round message passing
  clf = CausalBoostClassifier(n_rounds=3)

Dependencies:
  pip install torch numpy scikit-learn scipy
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from typing import List, Optional, Dict, Tuple
import warnings

# DIAG: switches set by rerun_diag/run_diag.py. Defaults = original behaviour.
#   select       'loss' (original) or 'acc' (val accuracy, ties broken by loss)
#   restore_from 'all' (original) or 'finetune'
#   head         'mlp' (original, flattened MLP over all nodes) or 'additive'
#                (per-node MLP, summed: interactions only via message passing)
#   curve_hook   None, or f(cinn) -> test accuracy, logged with val loss/acc
DIAG = dict(select='loss', restore_from='all', head='mlp', curve_hook=None)


class AdditiveHead(nn.Module):
    """DIAG: logits = sum_i MLP_i(h_i), one small MLP per node (NAM-style)."""
    def __init__(self, n_nodes, hidden_dim, head_hidden, n_classes):
        super().__init__()
        self.n_nodes, self.hidden_dim = n_nodes, hidden_dim
        hh = max(8, head_hidden // 8)
        self.W1 = nn.Parameter(torch.randn(n_nodes, hidden_dim, hh) / hidden_dim ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(n_nodes, hh))
        self.W2 = nn.Parameter(torch.randn(n_nodes, hh, n_classes) / hh ** 0.5 / n_nodes ** 0.5)
        self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x):
        H = x.view(x.shape[0], self.n_nodes, self.hidden_dim)
        Z = torch.relu(torch.einsum('bnh,nhk->bnk', H, self.W1) + self.b1)
        return torch.einsum('bnk,nkc->bc', Z, self.W2) + self.b
warnings.filterwarnings('ignore')


# ============================================================================
# QuantumBind Module
# ============================================================================

class QuantumBindModule(nn.Module):
    """Density-matrix fingerprinting via receptor observables.

    Inspired by quantum measurement: each sample is encoded as a density matrix
    ρ (a PSD, trace-1 matrix summarizing its state), then "measured" by P learnable
    receptor observables O_p to produce fingerprints fp_p = Tr(ρ · O_p).

    Pipeline::

        H_flat ─→ pool ─→ h ─┬─→ factor_head ─→ F [B, R, K] ─→ ρ = FFᵀ/Tr(FFᵀ) [B, R, R]
                             └─→ query_head  ─→ q [B, 16]

        Receptors: L_param [P, R(R+1)/2] ─→ L [P, R, R] (lower-tri) ─→ O = LLᵀ [P, R, R]
                                                                    ─→ O̅ = O/Tr(O) [P, R, R]

        Second-order fingerprint:  fp2_p = σ( Σ_{ij} ρ_{ij} · O̅_{p,ij} )     [B, P]
        First-order fingerprint:   fp1_p = σ( q · k_p )                        [B, P]

        Family modulation (per fingerprint vector fp [B, P]):
            A = softmax(probe_to_family)                                        [P, F]
            fa = fp · A           (aggregate probes into families)              [B, F]
            fm = fa · S           (cross-family interaction)                    [B, F]
            fp' = fp + fp ⊙ (fm · Aᵀ)  (broadcast back to probes)             [B, P]

        Output: readout( [fp1' ; fp2'] )                                        [B, C]

    Notation: B=batch, R=rho_dim, K=rank, P=n_probes, F=n_families, C=n_classes.

    Parameters
    ----------
    input_dim : int
        Flattened embedding dimension (d_features * hidden_dim).
    n_classes : int
        Number of output classes.
    n_receptors : int
        Number of receptor groups (M).
    n_sites : int
        Number of binding sites per receptor (S). Total probes P = M * S.
    n_families : int
        Number of receptor families for cross-probe modulation.
    rho_dim : int
        Dimension R of the density matrix (R × R).
    rank : int
        Rank K of the factor matrix F (R × K). Controls expressiveness of ρ.
    gamma_rep : float
        Weight for receptor repulsion diversity loss.
    """

    def __init__(self, input_dim: int, n_classes: int,
                 n_receptors: int = 6, n_sites: int = 4,
                 n_families: int = 6, rho_dim: int = 6, rank: int = 3,
                 gamma_rep: float = 0.2):
        super().__init__()
        self.rho_dim = rho_dim      # R
        self.rank = rank            # K
        self.gamma_rep = gamma_rep
        n_probes = n_receptors * n_sites  # P = M * S
        self.n_probes = n_probes
        self.n_families = n_families  # F
        obs_param_dim = rho_dim * (rho_dim + 1) // 2  # lower-triangular entries

        # Encoder: H_flat [B, input_dim] → h [B, 64]
        self.pool = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        )
        self.factor_head = nn.Linear(64, rho_dim * rank)  # h → F [B, R*K]
        self.query_head = nn.Linear(64, 16)                # h → q [B, 16]

        # Receptor observables: L_param [P, R(R+1)/2] → O = LLᵀ [P, R, R]
        self.receptor_L = nn.Parameter(torch.zeros(n_probes, obs_param_dim))
        # First-order keys: k [P, 16]
        self.receptor_keys = nn.Parameter(torch.randn(n_probes, 16) * 0.3)

        # Family modulation: probe_to_family [P, F], family_interaction [F, F]
        self.probe_to_family = nn.Parameter(torch.randn(n_probes, n_families) * 0.3)
        self.family_interaction = nn.Parameter(
            torch.eye(n_families) * 0.5 + torch.randn(n_families, n_families) * 0.1)

        # Readout: [B, 2*P] → [B, C]
        self.readout = nn.Sequential(
            nn.Linear(n_probes * 2, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )

        # Smart angular initialization for receptor L
        with torch.no_grad():
            self.receptor_L.copy_(torch.tensor(
                self._init_receptor_L(n_receptors, n_sites, rho_dim), dtype=torch.float32))

        self._repulsion = torch.tensor(0.0)

    # ── Density matrix ──

    @staticmethod
    def make_rho(F_batch):
        """Low-rank factor → normalized density matrix.

            F: [B, R, K]  →  ρ = FFᵀ / Tr(FFᵀ): [B, R, R]

        ρ is PSD and trace-1 by construction.
        """
        # gram_{bij} = Σ_k F_{bik} F_{bjk}
        gram = torch.einsum('bik,bjk->bij', F_batch, F_batch)   # [B, R, R]
        tr = torch.einsum('bii->b', gram)                       # [B]
        return gram / tr[:, None, None].clamp(min=1e-9)         # [B, R, R]

    # ── Observable construction ──

    @staticmethod
    def make_observable(L_flat, d):
        """Lower-triangular params → PSD observable.

            L_flat: [P, R(R+1)/2]  →  L: [P, R, R]  →  O = LLᵀ: [P, R, R]
        """
        P = L_flat.shape[0]
        L = torch.zeros(P, d, d, dtype=L_flat.dtype, device=L_flat.device)
        idx = torch.tril_indices(d, d)
        L[:, idx[0], idx[1]] = L_flat
        return torch.einsum('pik,pjk->pij', L, L)  # [P, R, R]

    @staticmethod
    def normalize_obs(obs):
        """Normalize so Tr(O) = 1.

            obs: [P, R, R]  →  O̅: [P, R, R]  where Tr(O̅_p) = 1
        """
        tr = torch.einsum('pii->p', obs)              # [P]
        return obs / tr[:, None, None].clamp(min=1e-9)  # [P, R, R]

    # ── Diversity loss ──

    @staticmethod
    def receptor_repulsion(obs):
        """Penalize overlapping observables to encourage diversity.

            obs: [P, R, R]  →  scalar loss

        Computes normalized overlap between all pairs: (Tr(O_i O_j))² / (Tr(O_i²) Tr(O_j²))
        """
        P = obs.shape[0]
        if P < 2:
            return torch.tensor(0.0, dtype=obs.dtype, device=obs.device)
        # Tr(O_i O_j) = Σ_{kl} O_i_{kl} O_j_{kl}  (Frobenius inner product)
        obs_flat = obs.reshape(P, -1)                             # [P, R²]
        gram = torch.einsum('id,jd->ij', obs_flat, obs_flat)     # [P, P]
        # Tr(O_i²) = ||O_i||_F²
        norms_sq = torch.einsum('ii->i', gram)                   # [P]  (diagonal)
        idx_i, idx_j = torch.triu_indices(P, P, offset=1)
        overlaps = gram[idx_i, idx_j]                             # [P*(P-1)/2]
        norms = (norms_sq[idx_i] * norms_sq[idx_j]).clamp(min=1e-12)
        return (overlaps ** 2 / norms).mean()

    # ── Receptor initialization ──

    @staticmethod
    def _init_receptor_L(M, S, rho_dim, seed=42):
        """Angularly-spaced initialization for receptor L parameters.

        Each receptor gets a distinct orientation in the R×R observable space
        by rotating the diagonal/off-diagonal entries of L by angle ∝ receptor index.

            Returns: [M*S, R(R+1)/2]
        """
        rng = np.random.default_rng(seed)
        d = rho_dim
        obs_param_dim = d * (d + 1) // 2
        L_init = np.zeros((M * S, obs_param_dim))
        for idx in range(M * S):
            L_mat = np.zeros((d, d))
            angle = idx * np.pi / (M * S)
            for i in range(d):
                for j in range(i + 1):
                    if i == j:
                        L_mat[i, j] = np.cos(angle + i * np.pi / d) * 0.8
                    elif j == 0:
                        L_mat[i, j] = np.sin(angle + i * np.pi / d) * 0.4
                    else:
                        L_mat[i, j] = rng.normal(0, 0.1)
            tril_idx = np.tril_indices(d)
            L_init[idx] = L_mat[tril_idx]
        return L_init

    # ── Forward ──

    def forward(self, H_flat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        H_flat : [B, input_dim]
            Flattened feature embeddings from CausalBoostNet.

        Returns
        -------
        logits : [B, n_classes]
        """
        R, K, P, F = self.rho_dim, self.rank, self.n_probes, self.n_families

        # ── Encode ──
        h = self.pool(H_flat)                                     # [B, 64]
        F_mat = self.factor_head(h).reshape(-1, R, K)             # [B, R, K]
        q = self.query_head(h)                                    # [B, 16]

        # ── Density matrix ──
        rho = self.make_rho(F_mat)                                # [B, R, R]

        # ── Observables ──
        obs = self.normalize_obs(
            self.make_observable(self.receptor_L, R))             # [P, R, R]

        # ── Second-order fingerprint: fp2_p = σ(Tr(ρ · O̅_p)) ──
        # Tr(ρ · O) = Σ_{ij} ρ_{ij} O_{pij}
        fp_2nd = torch.sigmoid(
            torch.einsum('bij,pij->bp', rho, obs))               # [B, P]

        # ── First-order fingerprint: fp1_p = σ(q · k_p) ──
        fp_1st = torch.sigmoid(
            torch.einsum('bd,pd->bp', q, self.receptor_keys))    # [B, P]

        # ── Family modulation ──
        # Soft assignment: probes → families
        A = torch.softmax(self.probe_to_family, dim=1)            # [P, F]
        S = self.family_interaction                                # [F, F]

        def family_modulate(fp):
            # fp: [B, P]
            fa = torch.einsum('bp,pf->bf', fp, A)                # [B, F]
            fm = torch.einsum('bf,fg->bg', fa, S)                # [B, F]
            gate = torch.einsum('bf,pf->bp', fm, A)              # [B, P]
            return fp + fp * gate                                 # [B, P]

        fp = torch.cat([
            family_modulate(fp_1st),                              # [B, P]
            family_modulate(fp_2nd),                              # [B, P]
        ], dim=1)                                                 # [B, 2P]

        # ── Repulsion loss (stored for training) ──
        self._repulsion = self.receptor_repulsion(obs)

        return self.readout(fp)                                   # [B, C]

    def repulsion_loss(self) -> torch.Tensor:
        """Return the receptor repulsion loss (call after forward)."""
        return self._repulsion * self.gamma_rep


# ============================================================================
# Preprocessing
# ============================================================================

class CategoricalPreprocessor:
    """Label-encode categorical features, pass through numerics."""

    def __init__(self):
        self.cat_indices: List[int] = []
        self.label_encoders: Dict[int, LabelEncoder] = {}
        self.cardinalities: Dict[int, int] = {}
        self.num_indices: List[int] = []
        self.n_features = 0
        self.fitted = False

    def fit(self, X: np.ndarray, cat_indices: Optional[List[int]] = None):
        self.cat_indices = cat_indices or []
        n, d = X.shape
        self.n_features = d
        self.num_indices = [i for i in range(d) if i not in self.cat_indices]
        for i in self.cat_indices:
            le = LabelEncoder()
            le.fit(X[:, i].astype(str))
            self.label_encoders[i] = le
            self.cardinalities[i] = len(le.classes_)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Preprocessor not fitted")
        n = X.shape[0]
        if self.num_indices:
            X_num = X[:, self.num_indices].astype(np.float32)
        else:
            X_num = np.zeros((n, 0), dtype=np.float32)
        if self.cat_indices:
            X_cat = np.zeros((n, len(self.cat_indices)), dtype=np.int64)
            for j, i in enumerate(self.cat_indices):
                le = self.label_encoders[i]
                col = X[:, i].astype(str)
                encoded = np.zeros(n, dtype=np.int64)
                for k, val in enumerate(col):
                    if val in le.classes_:
                        encoded[k] = le.transform([val])[0]
                X_cat[:, j] = encoded
        else:
            X_cat = np.zeros((n, 0), dtype=np.int64)
        return X_num, X_cat

    def fit_transform(self, X: np.ndarray, cat_indices: Optional[List[int]] = None):
        self.fit(X, cat_indices)
        return self.transform(X)


class MissingnessHandler:
    """Detect missing values, create binary indicator columns, median-impute.

    On fit: identifies which columns have missing values in training data.
    On transform: appends binary indicator columns for those columns,
    then median-imputes all NaNs.
    """

    def __init__(self):
        self.cols_with_missing: Optional[np.ndarray] = None
        self.imputer = SimpleImputer(strategy='median')
        self.fitted = False

    def fit(self, X: np.ndarray) -> 'MissingnessHandler':
        miss_mask = np.isnan(X.astype(float))
        self.cols_with_missing = np.where(miss_mask.sum(axis=0) > 0)[0]
        self.imputer.fit(X.astype(float))
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("MissingnessHandler not fitted")
        X = X.astype(float).copy()
        if len(self.cols_with_missing) > 0:
            miss_indicators = np.isnan(X[:, self.cols_with_missing]).astype(np.float32)
            X = self.imputer.transform(X)
            X = np.hstack([X, miss_indicators])
        else:
            X = self.imputer.transform(X)
        return X.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    @property
    def n_indicator_cols(self) -> int:
        if self.cols_with_missing is None:
            return 0
        return len(self.cols_with_missing)


# ============================================================================
# DAG Constraints
# ============================================================================

def spectral_radius_acyclicity(W: torch.Tensor,
                               n_power_iter: int = 15) -> torch.Tensor:
    """SDCD acyclicity constraint via spectral radius of W^2."""
    d = W.shape[0]
    W_sq = W * W
    W_sq = W_sq * (1 - torch.eye(d, device=W.device))
    v = torch.ones(d, device=W.device) / math.sqrt(d)
    for _ in range(n_power_iter):
        v_new = W_sq @ v
        norm = torch.norm(v_new)
        if norm > 1e-8:
            v = v_new / norm
        else:
            return torch.tensor(0.0, device=W.device)
    rho = (v @ W_sq @ v) / (v @ v + 1e-8)
    return rho


def dagma_constraint(S: torch.Tensor) -> torch.Tensor:
    """DAGMA log-det acyclicity: h(S) = -log det(kI - S*S) + k*log(k)."""
    k = S.shape[0]
    M = k * torch.eye(k, device=S.device) - S * S
    sign, logabsdet = torch.slogdet(M)
    if sign.item() <= 0:
        return torch.trace(torch.matrix_exp(S * S)) - k
    return -logabsdet + k * math.log(k)


def soft_sort(scores: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """SoftSort: differentiable relaxation of argsort to a permutation matrix."""
    scores_3d = scores.unsqueeze(0).unsqueeze(-1)
    sorted_vals = scores_3d.sort(descending=True, dim=1)[0]
    diffs = (scores_3d.transpose(1, 2) - sorted_vals).abs().neg() / tau
    P = diffs.softmax(-1).squeeze(0)
    return P


# ============================================================================
# Temperature Calibration
# ============================================================================

def learn_temperature(logits: np.ndarray, labels: np.ndarray,
                      lr: float = 0.01, max_iter: int = 200) -> float:
    """Learn optimal temperature T via LBFGS to minimize NLL on logits."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_T = nn.Parameter(torch.zeros(1))
    optimizer = optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T).clamp(min=0.1)
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return torch.exp(log_T).clamp(min=0.1).item()


# ============================================================================
# Adaptive Boost Alpha
# ============================================================================

def compute_adaptive_alpha(n: int, d: int, n_classes: int,
                           base_alpha: float = 2.0) -> float:
    """Scale boost alpha based on dataset characteristics."""
    d_scale = np.clip(np.log2(max(d, 2)) / np.log2(40), 0.3, 1.2)
    class_scale = 1.0 / np.sqrt(max(n_classes / 2, 1.0))
    n_scale = np.clip(np.sqrt(n / 500), 0.5, 1.5)
    return base_alpha * d_scale * class_scale * n_scale


# ============================================================================
# CausalBoost Neural Network
# ============================================================================

class CausalBoostNet(nn.Module):
    """
    SDCD+CINN network: learns a DAG adjacency matrix end-to-end
    alongside a causality-informed message-passing classifier.

    DAG parameterizations:
      'standard' - Full W[d,d] adjacency matrix
      'factored' - W = A[d,k] @ S[k,k] @ A.T (groups correlated features)
      'block'    - Full W with block-diagonal regularization
      'latent'   - Full W with additional latent nodes

    Acyclicity enforcement (for 'factored', applied to S; otherwise to W):
      'vcuda'    - Priority scores + edge logits, acyclic by construction
      'spectral' - Spectral radius constraint via power iteration
      'dagma'    - Log-determinant constraint
      'dpdag'    - SoftSort permutation + edge logits, acyclic by construction
    """

    def __init__(self, n_num: int, cat_cardinalities: List[int],
                 n_classes: int, hidden_dim: int = 32, emb_dim: int = 8,
                 emb_dim_formula: str = 'sqrt',
                 edge_dropout: float = 0.0, n_rounds: int = 1,
                 dag_type: str = 'factored', n_latent: int = 0,
                 n_groups: int = 32, recon_loss: bool = False,
                 lambda_block: float = 0.1, lambda_recon: float = 0.1,
                 acyclicity_type: str = 'vcuda',
                 tau_vcuda: float = 0.3, tau_perm: float = 1.0,
                 tau_edge: float = 0.5,
                 nam_shape: bool = False, n_slots: int = 1,
                 sparse_dag = False,  # False | 'prox' | 'gumbel' | 'gumbel_shared' | 'anneal' | ...
                 attn_dag: bool = False,  # merge attention with DAG structure
                 # MoE-DAG: multiple DAGs with per-sample gating
                 n_dags: int = 1,  # K separate DAGs (1 = current behavior)
                 shared_A: bool = True,  # share feature-to-group mapping across DAGs
                 garbage_alpha: float = 0.0,  # >0: last DAG is dense "garbage", fixed blend weight
                 moe_repulsion: bool = False,  # cosine repulsion among specialist W_k
                 moe_l1: bool = False,  # L1 sparsity on specialist S_k
                 moe_loadbal: bool = False,  # load balancing on gate weights
                 # Temperature sharpening: anneal sigmoid temperature
                 sharpen_tau: float = 0.0,  # 0 = disabled, >0 = final tau (e.g. 0.1)
                 # QB params
                 qb_n_receptors: int = 6, qb_n_sites: int = 4,
                 qb_n_families: int = 6, qb_rho_dim: int = 6, qb_rank: int = 3,
                 # Head params
                 head_hidden: int = None, head_dropout: float = 0.1):
        super().__init__()
        self.nam_shape = nam_shape
        self.n_slots = n_slots
        self.qb_n_receptors = qb_n_receptors
        self.qb_n_sites = qb_n_sites
        self.qb_n_families = qb_n_families
        self.qb_rho_dim = qb_rho_dim
        self.qb_rank = qb_rank
        self.head_hidden = head_hidden if head_hidden is not None else hidden_dim * 2
        self.head_dropout = head_dropout
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        self.n_features = n_num + self.n_cat
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.edge_dropout = edge_dropout
        self.n_rounds = n_rounds
        self.dag_type = dag_type
        self.n_latent = n_latent
        self.n_groups = n_groups
        self.sparse_dag = sparse_dag
        self.attn_dag = attn_dag
        self.n_dags = n_dags
        self.shared_A = shared_A
        self.garbage_alpha = garbage_alpha
        self.moe_repulsion = moe_repulsion
        self.moe_l1 = moe_l1
        self.moe_loadbal = moe_loadbal
        self.sharpen_tau = sharpen_tau
        self._tau_edge_current = tau_edge  # mutable, annealed during training
        self.recon_loss_enabled = recon_loss
        self.lambda_block = lambda_block
        self.lambda_recon = lambda_recon
        self.acyclicity_type = acyclicity_type
        self.tau_vcuda = tau_vcuda
        self.tau_perm = tau_perm
        self.tau_edge = tau_edge

        d = self.n_features

        # ── Feature embeddings (n_slots per feature) ──
        if nam_shape:
            self.num_embedders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ) for _ in range(n_num * n_slots)
            ])
        else:
            self.num_embedders = nn.ModuleList([
                nn.Linear(1, hidden_dim) for _ in range(n_num * n_slots)
            ])
        self.cat_embeddings = nn.ModuleList()
        self.cat_projectors = nn.ModuleList()
        for card in cat_cardinalities:
            if emb_dim_formula == 'sqrt':
                actual_emb_dim = min(emb_dim, max(2, int(np.sqrt(card))))
            else:  # 'linear'
                actual_emb_dim = min(emb_dim, max(2, card))
            for _ in range(n_slots):
                self.cat_embeddings.append(nn.Embedding(card, actual_emb_dim))
                self.cat_projectors.append(nn.Linear(actual_emb_dim, hidden_dim))

        # DAG operates on d * n_slots nodes
        d = d * n_slots

        # ── DAG dimension ──
        if dag_type == 'latent':
            dag_d = d + n_latent
        else:
            dag_d = d
        self.dag_d = dag_d

        # Latent node embeddings
        self.latent_embeds = None
        if dag_type == 'latent' and n_latent > 0:
            self.latent_embeds = nn.Parameter(
                torch.randn(1, n_latent, hidden_dim) * 0.02)

        # ── DAG parameters ──
        k = n_groups
        if dag_type == 'factored':
            if n_dags > 1:
                # MoE-DAG: K separate DAGs with per-sample gating
                if shared_A:
                    self.A_raw = nn.Parameter(torch.randn(d, k) * 0.01)
                else:
                    self.A_raw_list = nn.ParameterList([
                        nn.Parameter(torch.randn(d, k) * 0.01) for _ in range(n_dags)])
                    self.A_raw = self.A_raw_list[0]  # default for compatibility
                # K separate S params
                self._init_S_params_multi(k, acyclicity_type, n_dags)
                # Gate: input features -> softmax over specialists
                # If garbage_alpha > 0, gate routes among K-1 specialists only
                n_gate_outputs = (n_dags - 1) if garbage_alpha > 0 else n_dags
                self.dag_gate = nn.Sequential(
                    nn.Linear(hidden_dim * d, 64),
                    nn.ReLU(),
                    nn.Linear(64, n_gate_outputs),
                )
            else:
                self.A_raw = nn.Parameter(torch.randn(d, k) * 0.01)
                self._init_S_params(k, acyclicity_type)
            # Optional sparse DAG component: S = S_base + S_sparse
            # sparse_dag='prox': proximal soft-thresholding (exact zeros)
            # sparse_dag='gumbel': Gumbel-Softmax discrete edges, independent
            # sparse_dag='gumbel_shared': Gumbel-Softmax, shares priority with S_base
            # sparse_dag='spike_slab': spike-and-slab prior (concrete relaxation)
            # sparse_dag='rcd_residual': fixed RCD DAG + learnable residual
            if sparse_dag:
                if sparse_dag in ('gumbel', 'gumbel_shared'):
                    self.S_sparse_logits = nn.Parameter(torch.zeros(k, k) - 2.0)
                    self.S_sparse_weights = nn.Parameter(torch.randn(k, k) * 0.01)
                elif sparse_dag == 'spike_slab':
                    # Spike: log-odds of edge existence (concrete/Gumbel relaxation)
                    self.S_spike_logits = nn.Parameter(torch.zeros(k, k))
                    # Slab: continuous edge weight
                    self.S_slab_weights = nn.Parameter(torch.randn(k, k) * 0.05)
                elif sparse_dag == 'anneal':
                    # Annealed sparsity: starts dense, prunes gradually
                    # S_sparse is a full weight matrix, L1 ramps up during training
                    self.S_anneal_weights = nn.Parameter(torch.randn(k, k) * 0.05)
                elif sparse_dag == 'rcd_residual':
                    # S_rcd: fixed buffer, set later via set_rcd_dag()
                    self.register_buffer('S_rcd', torch.zeros(k, k))
                    # Residual: learned corrections, L1-penalized
                    self.S_residual = nn.Parameter(torch.zeros(k, k))
                else:  # 'prox' or True
                    self.S_sparse_raw = nn.Parameter(torch.zeros(k, k))
                self.register_buffer('S_sparse_no_self_loop', 1.0 - torch.eye(k))
            # Dummy W_raw so state_dict is consistent
            self.W_raw = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.register_buffer('no_self_loop', 1.0 - torch.eye(d))
        elif dag_type == 'uv':
            # f-DAG: W = sigmoid(U) @ sigmoid(V), asymmetric by construction
            self.U_raw = nn.Parameter(torch.randn(d, k) * 0.01)
            self.V_raw = nn.Parameter(torch.randn(k, d) * 0.01)
            # Optional sparse additive component for UV: W = U@V + W_sparse
            if sparse_dag == 'anneal':
                self.W_anneal_weights = nn.Parameter(torch.zeros(d, d) * 0.01)
                self.register_buffer('W_sparse_no_self_loop', 1.0 - torch.eye(d))
            self.W_raw = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.register_buffer('no_self_loop', 1.0 - torch.eye(d))
        elif dag_type == 'asb':
            # W = softmax(A) @ S @ softmax(B) — original ASB
            self.A_raw = nn.Parameter(torch.randn(d, k) * 0.01)
            self.B_raw = nn.Parameter(torch.randn(k, d) * 0.01)
            self._init_S_params(k, acyclicity_type)
            self.W_raw = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.register_buffer('no_self_loop', 1.0 - torch.eye(d))
        elif dag_type == 'asb_sig':
            # W = sigmoid(A) @ S @ sigmoid(B) — no softmax normalization
            self.A_raw = nn.Parameter(torch.randn(d, k) * 0.01)
            self.B_raw = nn.Parameter(torch.randn(k, d) * 0.01)
            self._init_S_params(k, acyclicity_type)
            # Optional sparse S component (same as factored)
            if sparse_dag == 'anneal':
                self.S_anneal_weights = nn.Parameter(torch.randn(k, k) * 0.05)
                self.register_buffer('S_sparse_no_self_loop', 1.0 - torch.eye(k))
            self.W_raw = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.register_buffer('no_self_loop', 1.0 - torch.eye(d))
        elif dag_type == 'asb_raw':
            # W = A @ S @ B — unconstrained A, B
            self.A_raw = nn.Parameter(torch.randn(d, k) * 0.01)
            self.B_raw = nn.Parameter(torch.randn(k, d) * 0.01)
            self._init_S_params(k, acyclicity_type)
            self.W_raw = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.register_buffer('no_self_loop', 1.0 - torch.eye(d))
        elif dag_type == 'asb_vcuda':
            # W = sigmoid(A) @ S @ sigmoid(B) * vcuda_direction_mask
            self.A_raw = nn.Parameter(torch.randn(d, k) * 0.01)
            self.B_raw = nn.Parameter(torch.randn(k, d) * 0.01)
            self.S_raw_free = nn.Parameter(torch.randn(k, k) * 0.01)  # no acyclicity on S
            self.priority_d = nn.Parameter(torch.randn(d) * 0.1)
            self.W_raw = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.register_buffer('no_self_loop', 1.0 - torch.eye(d))
            self.register_buffer('no_self_loop_k', 1.0 - torch.eye(k))
        elif dag_type == 'asb_raw_vcuda':
            self.A_raw = nn.Parameter(torch.randn(d, k) * 0.01)
            self.B_raw = nn.Parameter(torch.randn(k, d) * 0.01)
            self.S_raw_free = nn.Parameter(torch.randn(k, k) * 0.01)
            self.priority_d = nn.Parameter(torch.randn(d) * 0.1)
            self.W_raw = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.register_buffer('no_self_loop', 1.0 - torch.eye(d))
            self.register_buffer('no_self_loop_k', 1.0 - torch.eye(k))
        elif dag_type == 'stacked_pair':
            # W = (A1@S1@A1^T) @ (A2@S2@A2^T) — two pairing layers
            self.A1_raw = nn.Parameter(torch.randn(d, k) * 0.01)
            self.A2_raw = nn.Parameter(torch.randn(d, k) * 0.01)
            self.S1_raw = nn.Parameter(torch.randn(k, k) * 0.01)
            self.S2_raw = nn.Parameter(torch.randn(k, k) * 0.01)
            self.W_raw = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.register_buffer('no_self_loop', 1.0 - torch.eye(d))
        elif dag_type == 'qb_hybrid':
            # Hybrid: factored DAG + QB density matrix
            self.A_raw = nn.Parameter(torch.randn(d, k) * 0.01)
            self._init_S_params(k, acyclicity_type)
            self.W_raw = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.register_buffer('no_self_loop', 1.0 - torch.eye(d))
            self.qb = QuantumBindModule(
                d * hidden_dim, n_classes, qb_n_receptors, qb_n_sites,
                qb_n_families, qb_rho_dim, qb_rank)
        elif dag_type in ('qb', 'qb_hybrid'):
            self.qb = QuantumBindModule(
                d * hidden_dim, n_classes, qb_n_receptors, qb_n_sites,
                qb_n_families, qb_rho_dim, qb_rank)
            self.W_raw = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.register_buffer('no_self_loop', 1.0 - torch.eye(d))
        else:
            self.W_raw = nn.Parameter(torch.randn(dag_d, dag_d) * 0.01)
            self.register_buffer('no_self_loop', 1.0 - torch.eye(dag_d))

        # Block mask buffer
        if dag_type == 'block':
            self.register_buffer('block_mask', torch.ones(dag_d, dag_d))

        # ── Message passing layers ──
        self.parent_aggs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_rounds)
        ])
        self.updaters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(n_rounds)
        ])

        # ── Reconstruction heads (optional) ──
        self.factored_recon_head = None
        if recon_loss and dag_type == 'factored':
            self.factored_recon_head = nn.Linear(hidden_dim, hidden_dim)

        # ── Attention-DAG: Q/K projections + learnable alpha ──
        if attn_dag:
            self.attn_q = nn.Linear(hidden_dim, hidden_dim)
            self.attn_k = nn.Linear(hidden_dim, hidden_dim)
            self.attn_alpha = nn.Parameter(torch.tensor(1.0))  # DAG bias strength

        # ── Classifier head ──
        if dag_type in ('qb', 'qb_hybrid'):
            pred_in = 0  # QB uses its own readout head
        else:
            pred_in = hidden_dim * dag_d
        self.predictor = nn.Sequential(
            nn.Linear(pred_in, self.head_hidden),
            nn.ReLU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.head_hidden, n_classes)
        )
        if DIAG['head'] == 'additive' and pred_in > 0:   # DIAG
            self.predictor = AdditiveHead(dag_d, hidden_dim, self.head_hidden, n_classes)

    def _init_S_params(self, k: int, acyclicity_type: str):
        """Initialize the group-level DAG parameters for factored mode."""
        if acyclicity_type in ('spectral', 'dagma'):
            self.S_raw = nn.Parameter(torch.randn(k, k) * 0.01)
            self.register_buffer('S_no_self_loop', 1.0 - torch.eye(k))
        elif acyclicity_type == 'vcuda':
            self.priority = nn.Parameter(torch.randn(k))
            self.edge_logits = nn.Parameter(torch.zeros(k, k))
            nn.init.uniform_(self.edge_logits, -1, 1)
            with torch.no_grad():
                self.edge_logits.diagonal().fill_(-300)
            self.register_buffer('no_self_loop_k', 1.0 - torch.eye(k))
        elif acyclicity_type == 'dpdag':
            self.perm_scores = nn.Parameter(torch.randn(k))
            self.edge_logits = nn.Parameter(torch.zeros(k, k))
            nn.init.uniform_(self.edge_logits, -1, 1)
            with torch.no_grad():
                self.edge_logits.diagonal().fill_(-300)
            self.register_buffer('triu_mask',
                                 torch.triu(torch.ones(k, k), diagonal=1))
            self.register_buffer('no_self_loop_k', 1.0 - torch.eye(k))

    def _init_S_params_multi(self, k: int, acyclicity_type: str, n_dags: int):
        """Initialize K separate sets of DAG parameters for MoE-DAG."""
        if acyclicity_type == 'vcuda':
            self.priority_list = nn.ParameterList([
                nn.Parameter(torch.randn(k)) for _ in range(n_dags)])
            self.edge_logits_list = nn.ParameterList()
            for _ in range(n_dags):
                el = nn.Parameter(torch.zeros(k, k))
                nn.init.uniform_(el, -1, 1)
                with torch.no_grad():
                    el.diagonal().fill_(-300)
                self.edge_logits_list.append(el)
            self.register_buffer('no_self_loop_k', 1.0 - torch.eye(k))
            # Aliases for backward compat (loss functions etc.)
            self.priority = self.priority_list[0]
            self.edge_logits = self.edge_logits_list[0]
        else:
            self.S_raw_list = nn.ParameterList([
                nn.Parameter(torch.randn(k, k) * 0.01) for _ in range(n_dags)])
            self.register_buffer('S_no_self_loop', 1.0 - torch.eye(k))
            self.S_raw = self.S_raw_list[0]

    # ── DAG accessors ──

    def _get_S_for(self, priority, edge_logits, S_raw=None):
        """Get S from specific parameters (for MoE-DAG).

        Uses self._tau_edge_current which can be annealed for temperature sharpening.
        """
        tau = self._tau_edge_current
        if self.acyclicity_type in ('spectral', 'dagma'):
            if self.sharpen_tau > 0 and tau < 1.0:
                S = torch.sigmoid(S_raw / tau) * self.S_no_self_loop
            else:
                S = torch.sigmoid(S_raw) * self.S_no_self_loop
        elif self.acyclicity_type == 'vcuda':
            grad_p = priority.unsqueeze(0) - priority.unsqueeze(1)
            direction = torch.sigmoid(grad_p / self.tau_vcuda)
            if self.training:
                raw = torch.stack([edge_logits, -edge_logits])
                edges = F.gumbel_softmax(raw, tau=tau,
                                         hard=True, dim=0)[0]
            else:
                edges = (torch.sigmoid(edge_logits / max(tau, 0.01)) > 0.5).float()
                edges.diagonal().fill_(0)
            S = edges * direction * self.no_self_loop_k
        elif self.acyclicity_type == 'dpdag':
            if self.training:
                gumbels = -torch.empty_like(self.perm_scores).exponential_().log()
                perturbed = self.perm_scores + gumbels
            else:
                perturbed = self.perm_scores
            P = soft_sort(perturbed, tau=self.tau_perm)
            P_hard = torch.zeros_like(P)
            P_hard.scatter_(-1, P.topk(1, -1)[1], 1.0)
            P = (P_hard - P).detach() + P
            if self.training:
                raw = torch.stack([edge_logits, -edge_logits])
                edges = F.gumbel_softmax(raw, tau=tau,
                                         hard=True, dim=0)[0]
            else:
                edges = (torch.sigmoid(edge_logits) > 0.5).float()
                edges.diagonal().fill_(0)
            order_mask = P @ self.triu_mask @ P.T
            S = edges * order_mask * self.no_self_loop_k
        return S

    def _get_S(self):
        """Get group-level DAG matrix S [k, k] (factored mode only).

        If sparse_dag=True, returns S_base + S_sparse where S_sparse
        is a separate learnable component with strong L1 penalty.
        """
        if self.acyclicity_type in ('spectral', 'dagma'):
            S = self._get_S_for(None, None, S_raw=self.S_raw)
        elif self.acyclicity_type == 'vcuda':
            S = self._get_S_for(self.priority, self.edge_logits)
        elif self.acyclicity_type == 'dpdag':
            S = self._get_S_for(None, self.edge_logits)

        # Add sparse component: S_total = S_base + S_sparse
        if self.sparse_dag:
            if self.sparse_dag == 'gumbel':
                if self.training:
                    raw = torch.stack([self.S_sparse_logits, -self.S_sparse_logits])
                    edges_sp = F.gumbel_softmax(raw, tau=self.tau_edge, hard=True, dim=0)[0]
                else:
                    edges_sp = (torch.sigmoid(self.S_sparse_logits) > 0.5).float()
                S = S + edges_sp * self.S_sparse_weights * self.S_sparse_no_self_loop
            elif self.sparse_dag == 'gumbel_shared':
                grad_p = self.priority.unsqueeze(0) - self.priority.unsqueeze(1)
                direction = torch.sigmoid(grad_p / self.tau_vcuda)
                if self.training:
                    raw = torch.stack([self.S_sparse_logits, -self.S_sparse_logits])
                    edges_sp = F.gumbel_softmax(raw, tau=self.tau_edge, hard=True, dim=0)[0]
                else:
                    edges_sp = (torch.sigmoid(self.S_sparse_logits) > 0.5).float()
                S = S + edges_sp * direction * self.S_sparse_weights * self.S_sparse_no_self_loop
            elif self.sparse_dag == 'spike_slab':
                # Concrete relaxation of spike (edge on/off)
                if self.training:
                    u = torch.rand_like(self.S_spike_logits).clamp(1e-6, 1-1e-6)
                    concrete = torch.sigmoid((torch.log(u/(1-u)) + self.S_spike_logits) / 0.2)
                else:
                    concrete = (torch.sigmoid(self.S_spike_logits) > 0.5).float()
                S = S + concrete * self.S_slab_weights * self.S_sparse_no_self_loop
            elif self.sparse_dag == 'rcd_residual':
                # Fixed RCD structure + learned residual (L1-penalized)
                S = S + (self.S_rcd + self.S_residual) * self.S_sparse_no_self_loop
            elif self.sparse_dag == 'anneal':
                # During phase 1: S_anneal is the only signal (S_base frozen)
                # During phase 2: L1 ramps up, pruning S_anneal
                # During phase 3: S_base unfreezes, absorbs residuals
                S = S + self.S_anneal_weights * self.S_sparse_no_self_loop
            else:
                # Proximal: S_sparse_raw is soft-thresholded after each step
                S = S + self.S_sparse_raw * self.S_sparse_no_self_loop

        return S

    def set_fixed_dag(self, W_numpy):
        """Fix the DAG to a pre-computed adjacency matrix (non-learnable).

        Args:
            W_numpy: [d, d] numpy array, binary or weighted adjacency matrix.
        """
        W_t = torch.tensor(W_numpy, dtype=torch.float32)
        self.register_buffer('_fixed_W', W_t)

    def set_rcd_dag(self, W_rcd: torch.Tensor):
        """Set the fixed RCD DAG for rcd_residual mode.

        W_rcd is a d_orig×d_orig feature-level DAG. We expand it to
        slot space (d_W = d_orig * n_slots) then project to group-level
        via S_rcd = A^T W_expanded A.
        """
        if hasattr(self, 'S_rcd'):
            with torch.no_grad():
                d_orig = W_rcd.shape[0]
                d_W = self.A_raw.shape[0]  # d * n_slots
                W_rcd = W_rcd.to(self.A_raw.device)

                # Expand W_rcd to slot space: each slot pair gets same weight
                if d_W > d_orig:
                    n_slots = d_W // d_orig
                    W_exp = W_rcd.repeat(n_slots, n_slots)[:d_W, :d_W]
                else:
                    W_exp = W_rcd[:d_W, :d_W]

                A = torch.softmax(self.A_raw, dim=1)  # [d_W, k]
                S_rcd = A.T @ W_exp @ A  # [k, k]
                S_rcd.diagonal().fill_(0)
                self.S_rcd.copy_(S_rcd)

    def _get_S_idx(self, idx: int):
        """Get S for the idx-th DAG in MoE-DAG mode.

        If sparse_dag='anneal', adds the shared S_anneal_weights component.
        """
        if self.acyclicity_type == 'vcuda':
            S = self._get_S_for(self.priority_list[idx], self.edge_logits_list[idx])
        else:
            S = self._get_S_for(None, None, S_raw=self.S_raw_list[idx])
        # Add anneal component if enabled
        if self.sparse_dag == 'anneal' and hasattr(self, 'S_anneal_weights'):
            S = S + self.S_anneal_weights * self.S_sparse_no_self_loop
        return S

    def _get_A_idx(self, idx: int):
        """Get A for the idx-th DAG (shared or separate)."""
        if self.shared_A:
            return torch.softmax(self.A_raw, dim=1)
        else:
            return torch.softmax(self.A_raw_list[idx], dim=1)

    def get_W_moe(self, H: torch.Tensor):
        """Compute per-sample gated W for MoE-DAG.

        If garbage_alpha > 0: last DAG is the dense "garbage" with fixed weight alpha.
        Gate routes among specialists (DAGs 0..K-2) only.
        W(x) = (1-alpha) * sum_k g_k(x) * W_k + alpha * W_garbage

        Stores self._gate_weights for load balancing loss.

        Args:
            H: [B, d, hidden_dim] - embeddings for gating
        Returns:
            W: [B, d, d] - per-sample adjacency matrices
        """
        B = H.shape[0]
        # Compute K W matrices
        W_list = []
        for idx in range(self.n_dags):
            A = self._get_A_idx(idx)
            S = self._get_S_idx(idx)
            W_k = (A @ S @ A.T) * self.no_self_loop  # [d, d]
            W_list.append(W_k)

        if self.garbage_alpha > 0 and self.n_dags >= 2:
            # Last DAG is garbage (dense, fixed weight)
            n_specialists = self.n_dags - 1
            W_garbage = W_list[-1]  # [d, d]
            W_specialists = torch.stack(W_list[:n_specialists], dim=0)  # [K-1, d, d]

            # Gate over specialists only
            gate_logits = self.dag_gate(H.view(B, -1))  # [B, K-1]
            gate_weights = F.softmax(gate_logits, dim=-1)  # [B, K-1]
            self._gate_weights = gate_weights  # store for load balancing

            # Weighted specialist mix
            W_spec = torch.einsum('bk,kij->bij', gate_weights, W_specialists)  # [B, d, d]

            # Blend: specialists + garbage
            alpha = self.garbage_alpha
            W = (1.0 - alpha) * W_spec + alpha * W_garbage.unsqueeze(0)
        else:
            # Standard MoE: gate over all DAGs
            W_stack = torch.stack(W_list, dim=0)  # [K, d, d]
            gate_logits = self.dag_gate(H.view(B, -1))  # [B, K]
            gate_weights = F.softmax(gate_logits, dim=-1)  # [B, K]
            self._gate_weights = gate_weights
            W = torch.einsum('bk,kij->bij', gate_weights, W_stack)

        return W

    def get_W(self):
        """Compute full adjacency matrix W [d, d]."""
        if hasattr(self, '_fixed_W') and self._fixed_W is not None:
            return self._fixed_W
        if self.dag_type == 'factored':
            A = torch.softmax(self.A_raw, dim=1)
            S = self._get_S()
            W = A @ S @ A.T
            return W * self.no_self_loop
        elif self.dag_type == 'uv':
            U = torch.sigmoid(self.U_raw)  # [d, k]
            V = torch.sigmoid(self.V_raw)  # [k, d]
            W = U @ V  # [d, d] — asymmetric
            # Add sparse component if enabled
            if self.sparse_dag == 'anneal' and hasattr(self, 'W_anneal_weights'):
                W = W + self.W_anneal_weights * self.W_sparse_no_self_loop
            return W * self.no_self_loop
        elif self.dag_type == 'asb':
            A = torch.softmax(self.A_raw, dim=1)
            S = self._get_S()
            B = torch.softmax(self.B_raw, dim=0)
            return (A @ S @ B) * self.no_self_loop
        elif self.dag_type == 'asb_sig':
            A = torch.sigmoid(self.A_raw)
            S = self._get_S()
            B = torch.sigmoid(self.B_raw)
            return (A @ S @ B) * self.no_self_loop
        elif self.dag_type == 'asb_raw':
            S = self._get_S()
            return (self.A_raw @ S @ self.B_raw) * self.no_self_loop
        elif self.dag_type == 'asb_vcuda':
            A = torch.sigmoid(self.A_raw)
            S = torch.sigmoid(self.S_raw_free) * self.no_self_loop_k
            B = torch.sigmoid(self.B_raw)
            W = A @ S @ B
            grad_p = self.priority_d.unsqueeze(0) - self.priority_d.unsqueeze(1)
            direction = torch.sigmoid(grad_p / self.tau_vcuda)
            return W * direction * self.no_self_loop
        elif self.dag_type == 'asb_raw_vcuda':
            S = torch.sigmoid(self.S_raw_free) * self.no_self_loop_k
            W = self.A_raw @ S @ self.B_raw
            grad_p = self.priority_d.unsqueeze(0) - self.priority_d.unsqueeze(1)
            direction = torch.sigmoid(grad_p / self.tau_vcuda)
            return W * direction * self.no_self_loop
        elif self.dag_type == 'stacked_pair':
            A1 = torch.softmax(self.A1_raw, dim=1)
            S1 = torch.sigmoid(self.S1_raw)
            W1 = A1 @ S1 @ A1.T
            A2 = torch.softmax(self.A2_raw, dim=1)
            S2 = torch.sigmoid(self.S2_raw)
            W2 = A2 @ S2 @ A2.T
            return (W1 @ W2) * self.no_self_loop
        elif self.dag_type == 'qb_hybrid':
            # Hybrid uses factored W for message passing
            A = torch.softmax(self.A_raw, dim=1)
            S = self._get_S()
            return (A @ S @ A.T) * self.no_self_loop
        elif self.dag_type == 'qb':
            return self.no_self_loop * 0  # dummy — QB skips message passing
        else:
            return torch.sigmoid(self.W_raw) * self.no_self_loop

    # ── Loss components ──

    def dag_loss(self):
        """Acyclicity constraint loss."""
        if hasattr(self, '_fixed_W') and self._fixed_W is not None:
            return torch.tensor(0.0)
        if self.dag_type == 'factored':
            if self.n_dags > 1:
                # MoE-DAG losses: acyclicity + L1 specialists + repulsion + load balancing
                dev = self.A_raw.device
                loss = torch.tensor(0.0, device=dev)
                n_specialists = (self.n_dags - 1) if self.garbage_alpha > 0 else self.n_dags

                # 1. Acyclicity on each S_k
                if self.acyclicity_type in ('spectral', 'dagma'):
                    for idx in range(self.n_dags):
                        S_k = self._get_S_idx(idx)
                        if self.acyclicity_type == 'spectral':
                            loss = loss + spectral_radius_acyclicity(S_k)
                        else:
                            loss = loss + dagma_constraint(S_k)

                # 2. L1 sparsity on specialists only (not garbage)
                if self.moe_l1:
                    for idx in range(n_specialists):
                        S_k = self._get_S_idx(idx)
                        loss = loss + 0.05 * S_k.abs().mean()

                # 3. Repulsion: push specialist W_k apart (cosine similarity)
                if self.moe_repulsion and n_specialists >= 2:
                    W_spec = []
                    for idx in range(n_specialists):
                        A = self._get_A_idx(idx)
                        S = self._get_S_idx(idx)
                        W_spec.append((A @ S @ A.T).view(-1))
                    rep_loss = torch.tensor(0.0, device=dev)
                    count = 0
                    for i in range(n_specialists):
                        for j in range(i + 1, n_specialists):
                            cos_sim = F.cosine_similarity(W_spec[i].unsqueeze(0),
                                                          W_spec[j].unsqueeze(0))
                            rep_loss = rep_loss + cos_sim.abs()
                            count += 1
                    loss = loss + 0.1 * rep_loss / max(count, 1)

                # 4. Load balancing: push gate toward uniform over specialists
                if self.moe_loadbal and hasattr(self, '_gate_weights') and self._gate_weights is not None:
                    mean_usage = self._gate_weights.mean(dim=0)  # [n_specialists]
                    target = 1.0 / max(n_specialists, 1)
                    lb_loss = ((mean_usage - target) ** 2).sum() * n_specialists
                    loss = loss + 0.1 * lb_loss

                return loss
            if self.acyclicity_type == 'spectral':
                return spectral_radius_acyclicity(self._get_S())
            elif self.acyclicity_type == 'dagma':
                return dagma_constraint(self._get_S())
            else:
                # vcuda and dpdag are acyclic by construction
                return torch.tensor(0.0, device=self.A_raw.device)
        elif self.dag_type == 'uv':
            return spectral_radius_acyclicity(self.get_W())
        elif self.dag_type in ('asb', 'asb_sig', 'asb_raw'):
            if self.acyclicity_type in ('spectral', 'dagma'):
                return spectral_radius_acyclicity(self._get_S())
            else:
                return torch.tensor(0.0, device=self.A_raw.device)
        elif self.dag_type in ('asb_vcuda', 'asb_raw_vcuda'):
            return torch.tensor(0.0, device=self.A_raw.device)  # VCUDA on W is acyclic by construction
        elif self.dag_type == 'stacked_pair':
            return spectral_radius_acyclicity(self.get_W())
        elif self.dag_type == 'qb_hybrid':
            if self.acyclicity_type in ('spectral', 'dagma'):
                return spectral_radius_acyclicity(self._get_S())
            else:
                return torch.tensor(0.0, device=self.A_raw.device)
        elif self.dag_type == 'qb':
            return torch.tensor(0.0, device=self.qb.receptor_L.device)
        else:
            return spectral_radius_acyclicity(self.get_W())

    def sparse_dag_loss(self):
        """Sparsity penalty on S_sparse + uniformity penalty on S_base.

        S_sparse is the primary interpretable structure (sparse).
        S_base is the residual (pushed toward uniform = no structure).
        """
        if not self.sparse_dag:
            return torch.tensor(0.0)

        dev = next(self.parameters()).device
        loss = torch.tensor(0.0, device=dev)

        # 1. Sparsity on S_sparse
        if self.sparse_dag in ('gumbel', 'gumbel_shared'):
            loss = loss + torch.sigmoid(self.S_sparse_logits).mean()
        elif self.sparse_dag == 'spike_slab':
            p = torch.sigmoid(self.S_spike_logits)
            prior = 0.3  # expect ~30% edges on
            kl = p * torch.log(p / prior + 1e-8) + (1-p) * torch.log((1-p) / (1-prior) + 1e-8)
            loss = loss + 0.5 * kl.mean()
        elif self.sparse_dag == 'rcd_residual':
            loss = loss + self.S_residual.abs().mean()
        elif self.sparse_dag == 'anneal':
            # L1 on anneal weights (S_anneal for factored/ASB, W_anneal for UV)
            if hasattr(self, 'S_anneal_weights'):
                loss = loss + self.S_anneal_weights.abs().mean()
            if hasattr(self, 'W_anneal_weights'):
                loss = loss + self.W_anneal_weights.abs().mean()
        elif hasattr(self, 'S_sparse_raw'):
            loss = loss + self.S_sparse_raw.abs().mean()

        # 2. Uniformity on S_base: push all entries toward the mean
        #    This forces structure into S_sparse, S_base is just background
        if self.acyclicity_type == 'vcuda':
            S_base_edges = torch.sigmoid(self.edge_logits)
            s_mean = S_base_edges.mean()
            loss = loss + (S_base_edges - s_mean).pow(2).mean()
        elif hasattr(self, 'S_raw'):
            S_base = torch.sigmoid(self.S_raw)
            s_mean = S_base.mean()
            loss = loss + (S_base - s_mean).pow(2).mean()

        # UV: push U@V toward uniform
        if self.dag_type == 'uv' and self.sparse_dag:
            U = torch.sigmoid(self.U_raw)
            V = torch.sigmoid(self.V_raw)
            W_base = U @ V
            w_mean = W_base.mean()
            loss = loss + (W_base - w_mean).pow(2).mean()

        return loss

    def sparsity_loss(self):
        """L1 sparsity on edge weights."""
        if self.dag_type == 'factored':
            if self.acyclicity_type in ('spectral', 'dagma'):
                return self._get_S().abs().mean()
            else:
                return torch.sigmoid(self.edge_logits).mean()
        elif self.dag_type == 'uv':
            return (torch.sigmoid(self.U_raw).mean() + torch.sigmoid(self.V_raw).mean()) / 2
        elif self.dag_type in ('asb', 'asb_sig', 'asb_raw'):
            if self.acyclicity_type in ('spectral', 'dagma'):
                return self._get_S().abs().mean()
            else:
                return torch.sigmoid(self.edge_logits).mean()
        elif self.dag_type in ('asb_vcuda', 'asb_raw_vcuda'):
            return torch.sigmoid(self.S_raw_free).mean()
        elif self.dag_type == 'stacked_pair':
            return (torch.sigmoid(self.S1_raw).mean() + torch.sigmoid(self.S2_raw).mean()) / 2
        elif self.dag_type == 'qb_hybrid':
            if self.acyclicity_type in ('spectral', 'dagma'):
                return spectral_radius_acyclicity(self._get_S())
            else:
                return torch.tensor(0.0, device=self.A_raw.device)
        elif self.dag_type == 'qb':
            return torch.tensor(0.0, device=self.qb.receptor_L.device)
        else:
            return self.get_W().abs().mean()

    def block_penalty(self):
        """Cross-block penalty for block-diagonal DAG regularization."""
        if self.dag_type != 'block':
            return torch.tensor(0.0, device=self.W_raw.device)
        W = self.get_W()
        cross_mask = 1.0 - self.block_mask
        return (W * cross_mask).abs().mean()

    def factored_reconstruction_loss(self, H: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss through the factored DAG."""
        if self.factored_recon_head is None:
            return torch.tensor(0.0, device=H.device)
        A = torch.softmax(self.A_raw, dim=1)
        S = self._get_S()
        H_group = torch.einsum('bdh,dk->bkh', H, A)
        H_parents = torch.einsum('ji,bjh->bih', S, H_group)
        H_recon = self.factored_recon_head(H_parents)
        return F.mse_loss(H_recon, H_group.detach())

    # ── Intervention loss ──

    def intervention_loss(self, H_pre: torch.Tensor, H_post: torch.Tensor,
                          W: torch.Tensor) -> torch.Tensor:
        """Intervention loss: mask a random feature, predict it from DAG parents.

        If the DAG is causally correct, a node's post-message-passing embedding
        should be reconstructable from its parents' pre-message-passing embeddings
        weighted by the DAG adjacency.

        Args:
            H_pre: [B, d, h] embeddings before message passing
            H_post: [B, d, h] embeddings after message passing
            W: [d, d] adjacency matrix
        """
        d = H_pre.shape[1]
        if d < 2:
            return torch.tensor(0.0, device=H_pre.device)

        # Pick a random feature to "intervene" on
        target = torch.randint(0, d, (1,)).item()

        # Parent message for the target node: weighted sum of parents' pre-DAG embeddings
        parent_weights = W[:, target]  # [d] - who are the parents of target?
        if parent_weights.sum() < 1e-6:
            return torch.tensor(0.0, device=H_pre.device)

        parent_msg = torch.einsum('j,bjh->bh', parent_weights, H_pre)  # [B, h]
        target_emb = H_post[:, target, :]  # [B, h]

        # The target's post-DAG embedding should be predictable from its parents
        return F.mse_loss(parent_msg, target_emb.detach())

    def contrastive_nam_loss(self, H_pre: torch.Tensor, labels: torch.Tensor,
                             temperature: float = 0.5) -> torch.Tensor:
        """Supervised contrastive loss on pre-DAG NAM embeddings.

        Pulls same-class embeddings close, pushes different-class apart.
        Operates on the pooled (mean over features) pre-DAG embedding.
        """
        # Pool over features: [B, d, h] -> [B, h]
        z = F.normalize(H_pre.mean(dim=1), dim=1)
        B = z.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z.device)

        # Similarity matrix
        sim = z @ z.T / temperature  # [B, B]
        # Masks (no in-place ops)
        diag_mask = 1.0 - torch.eye(B, device=z.device)
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * diag_mask

        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        # Log-sum-exp over all non-self entries
        exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True)[0].detach()) * diag_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log-prob over positive pairs
        loss = -(mask_pos * log_prob).sum() / (mask_pos.sum() + 1e-8)
        return loss

    def decorrelation_loss(self, H_pre: torch.Tensor) -> torch.Tensor:
        """Decorrelation loss on NAM embeddings (Barlow Twins style).

        Penalizes off-diagonal entries of the correlation matrix of the
        pooled (mean over features) pre-DAG embedding. Prevents NAM
        outputs from collapsing to a low-rank subspace.
        """
        # Pool over features: [B, d, h] -> [B, h]
        z = H_pre.mean(dim=1)
        B = z.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z.device)
        # Normalize
        z = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-6)
        # Correlation matrix [h, h]
        corr = (z.T @ z) / (B - 1)
        h = corr.shape[0]
        # Off-diagonal penalty
        off_diag = corr - torch.eye(h, device=corr.device)
        return (off_diag ** 2).sum() / (h * h)

    def rank_loss(self, H_pre: torch.Tensor) -> torch.Tensor:
        """Rank regularization: penalize low effective rank of embeddings.

        Encourages the NAM embedding matrix to use more dimensions by
        penalizing concentration of singular values (nuclear norm proxy).
        """
        # Pool over features: [B, d, h] -> [B, h]
        z = H_pre.mean(dim=1)
        B, h = z.shape
        if B < 2 or h < 2:
            return torch.tensor(0.0, device=z.device)
        # SVD — penalize entropy of normalized singular values
        # Low entropy = concentrated in few dims = bad
        s = torch.linalg.svdvals(z)
        s_norm = s / (s.sum() + 1e-8)
        entropy = -(s_norm * torch.log(s_norm + 1e-8)).sum()
        max_entropy = np.log(min(B, h))
        # Loss: want high entropy, so penalize (max - actual)
        return (max_entropy - entropy) / max_entropy

    def denoise_loss(self, X_num: torch.Tensor, X_cat: torch.Tensor) -> torch.Tensor:
        """Self-supervised denoising: mask one feature, reconstruct from DAG parents.

        Corrupts a random feature by replacing its value with a random sample's value,
        runs through DAG, and tries to reconstruct the original embedding.
        """
        B = X_num.shape[0] if X_num.shape[1] > 0 else X_cat.shape[0]
        d = self.n_features * self.n_slots
        if d < 2 or B < 2:
            return torch.tensor(0.0, device=X_num.device)

        # Pick a random node to corrupt
        target = torch.randint(0, d, (1,)).item()

        # Get clean embeddings
        clean_embs = []
        for i in range(self.n_num):
            for s in range(self.n_slots):
                clean_embs.append(self.num_embedders[i * self.n_slots + s](X_num[:, i:i + 1]))
        cat_idx = 0
        for i in range(self.n_cat):
            for s in range(self.n_slots):
                emb_i = self.cat_embeddings[cat_idx](X_cat[:, i])
                clean_embs.append(self.cat_projectors[cat_idx](emb_i))
                cat_idx += 1
        clean_target = clean_embs[target].detach()  # [B, h] — reconstruction target

        # Corrupt: replace target feature with shuffled version
        corrupt_embs = [e.clone() for e in clean_embs]
        perm = torch.randperm(B, device=X_num.device)
        corrupt_embs[target] = clean_embs[target][perm]  # shuffle within batch

        H = torch.stack(corrupt_embs, dim=1)  # [B, d, h]

        # Run message passing — DAG parents should help reconstruct
        W = self.get_W()
        for r in range(self.n_rounds):
            parent_msgs = torch.einsum('ji,bjh->bih', W, H)
            parent_msgs = self.parent_aggs[r](parent_msgs)
            H = self.updaters[r](torch.cat([H, parent_msgs], dim=-1))

        # Reconstruct target feature from its post-message-passing embedding
        reconstructed = H[:, target, :]  # [B, h]
        return F.mse_loss(reconstructed, clean_target)

    # ── Forward pass ──

    def forward(self, X_num: torch.Tensor, X_cat: torch.Tensor, labels: torch.Tensor = None):
        B = X_num.shape[0] if X_num.shape[1] > 0 else X_cat.shape[0]

        # Embed each feature (n_slots embeddings per feature)
        embeddings = []
        for i in range(self.n_num):
            for s in range(self.n_slots):
                embeddings.append(self.num_embedders[i * self.n_slots + s](X_num[:, i:i + 1]))
        cat_idx = 0
        for i in range(self.n_cat):
            for s in range(self.n_slots):
                emb_i = self.cat_embeddings[cat_idx](X_cat[:, i])
                embeddings.append(self.cat_projectors[cat_idx](emb_i))
                cat_idx += 1

        H = torch.stack(embeddings, dim=1)  # [B, d, hidden_dim]

        # Latent nodes
        if self.latent_embeds is not None:
            H = torch.cat([H, self.latent_embeds.expand(B, -1, -1)], dim=1)

        H_pre_dag = H.clone() if (self.recon_loss_enabled or self.training) else None

        # Feature interaction
        if self.dag_type == 'qb_hybrid':
            # Hybrid: NAM → DAG message passing → QB density matrix
            A = torch.softmax(self.A_raw, dim=1)
            S = self._get_S()
            W = (A @ S @ A.T) * self.no_self_loop
            if self.training and self.edge_dropout > 0:
                mask = (torch.rand_like(W) > self.edge_dropout).float()
                W = W * mask
            for r in range(self.n_rounds):
                parent_msgs = torch.einsum('ji,bjh->bih', W, H)
                parent_msgs = self.parent_aggs[r](parent_msgs)
                H = self.updaters[r](torch.cat([H, parent_msgs], dim=-1))
            logits = self.qb(H.view(B, -1))
        elif self.dag_type == 'qb':
            # Pure QB: NAM embeddings → density matrix (no DAG)
            logits = self.qb(H.view(B, -1))
        else:
            if self.n_dags > 1 and self.dag_type == 'factored':
                # MoE-DAG: per-sample gated adjacency
                W_per_sample = self.get_W_moe(H)  # [B, d, d]
                if self.training and self.edge_dropout > 0:
                    mask = (torch.rand(B, W_per_sample.shape[1], W_per_sample.shape[2],
                                       device=W_per_sample.device) > self.edge_dropout).float()
                    W_per_sample = W_per_sample * mask
                for r in range(self.n_rounds):
                    parent_msgs = torch.einsum('bij,bjh->bih', W_per_sample, H)
                    parent_msgs = self.parent_aggs[r](parent_msgs)
                    H = self.updaters[r](torch.cat([H, parent_msgs], dim=-1))
            else:
                W = self.get_W()
                if self.training and self.edge_dropout > 0:
                    mask = (torch.rand_like(W) > self.edge_dropout).float()
                    W = W * mask

                if self.attn_dag:
                    # Attention-DAG: per-sample attention biased by global DAG structure
                    for r in range(self.n_rounds):
                        Q = self.attn_q(H)  # [B, d, h]
                        K = self.attn_k(H)  # [B, d, h]
                        # Per-sample attention: [B, d, d]
                        attn = torch.einsum('bih,bjh->bij', Q, K) / (self.hidden_dim ** 0.5)
                        # Add global DAG bias: W [d, d] broadcast to [B, d, d]
                        attn = attn + self.attn_alpha * W.unsqueeze(0)
                        attn = torch.softmax(attn, dim=-1)  # normalize over sources j
                        # Message passing with attention weights
                        parent_msgs = torch.einsum('bij,bjh->bih', attn, H)
                        parent_msgs = self.parent_aggs[r](parent_msgs)
                        H = self.updaters[r](torch.cat([H, parent_msgs], dim=-1))
                else:
                    # Standard DAG message passing (global W, same for all samples)
                    for r in range(self.n_rounds):
                        parent_msgs = torch.einsum('ji,bjh->bih', W, H)
                        parent_msgs = self.parent_aggs[r](parent_msgs)
                        H = self.updaters[r](torch.cat([H, parent_msgs], dim=-1))

            logits = self.predictor(H.view(B, -1))

        # Reconstruction loss (stored for training loop to access)
        if self.recon_loss_enabled and self.training and H_pre_dag is not None:
            self._recon_loss = self.factored_reconstruction_loss(H_pre_dag)
        else:
            self._recon_loss = torch.tensor(0.0, device=logits.device)

        # Intervention loss (stored for training loop to access)
        if self.training and H_pre_dag is not None and self.dag_type != 'qb':
            self._intervention_loss = self.intervention_loss(H_pre_dag, H, self.get_W())
        else:
            self._intervention_loss = torch.tensor(0.0, device=logits.device)

        # Contrastive NAM loss (stored for training loop to access)
        if self.training and H_pre_dag is not None and labels is not None:
            self._contrastive_loss = self.contrastive_nam_loss(H_pre_dag, labels)
        else:
            self._contrastive_loss = torch.tensor(0.0, device=logits.device)

        # Decorrelation and rank losses (stored for training loop)
        if self.training and H_pre_dag is not None:
            self._decorrelation_loss = self.decorrelation_loss(H_pre_dag)
            self._rank_loss = self.rank_loss(H_pre_dag)
        else:
            self._decorrelation_loss = torch.tensor(0.0, device=logits.device)
            self._rank_loss = torch.tensor(0.0, device=logits.device)

        return logits


# ============================================================================
# CINNClassifier: Single SDCD+CINN Model
# ============================================================================

class CINNClassifier:
    """
    Single SDCD+CINN model with 3-phase training:
      1. Warmup (20 epochs): train CINN only, DAG frozen
      2. Joint (100 epochs): train DAG + CINN jointly, DAG penalty annealed
      3. Finetune (30 epochs): train CINN only, DAG frozen, low LR
    """

    def __init__(self, hidden_dim=32, emb_dim=8, emb_dim_formula='sqrt',
                 warmup_epochs=20, joint_epochs=100, finetune_epochs=30,
                 lr=0.001, w_lr=0.005, batch_size=32, patience=25,
                 lambda_dag=1.0, lambda_sparse=0.05, dag_gamma_max=50.0,
                 lambda_intervention=0.0, lambda_contrastive=0.0, lambda_w_l1=0.0,
                 lambda_denoise=0.0, lambda_decorrelation=0.0, lambda_rank=0.0,
                 label_smoothing=0.0, edge_dropout=0.0, n_rounds=1,
                 dag_type='factored', n_latent=0, n_groups=32,
                 recon_loss=False, lambda_block=0.1, lambda_recon=0.1,
                 block_mask=None,
                 acyclicity='vcuda',
                 tau_vcuda=0.3, tau_perm=1.0, tau_edge=0.5,
                 fixed_dag=None, nam_shape=False, n_slots=1,
                 sparse_dag=False, attn_dag=False,
                 n_dags=1, shared_A=True, garbage_alpha=0.0,
                 moe_repulsion=False, moe_l1=False, moe_loadbal=False,
                 sharpen_tau=0.0,
                 qb_n_receptors=6, qb_n_sites=4, qb_n_families=6, qb_rho_dim=6, qb_rank=3,
                 head_hidden=None, head_dropout=0.1):

        self.fixed_dag = fixed_dag
        self.nam_shape = nam_shape
        self.n_slots = n_slots
        self.sparse_dag = sparse_dag
        self.attn_dag = attn_dag
        self.n_dags = n_dags
        self.shared_A = shared_A
        self.garbage_alpha = garbage_alpha
        self.moe_repulsion = moe_repulsion
        self.moe_l1 = moe_l1
        self.moe_loadbal = moe_loadbal
        self.sharpen_tau = sharpen_tau
        self.qb_n_receptors = qb_n_receptors
        self.qb_n_sites = qb_n_sites
        self.qb_n_families = qb_n_families
        self.qb_rho_dim = qb_rho_dim
        self.qb_rank = qb_rank
        self.head_hidden = head_hidden
        self.head_dropout = head_dropout
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.emb_dim_formula = emb_dim_formula
        self.warmup_epochs = warmup_epochs
        self.joint_epochs = joint_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.w_lr = w_lr
        self.batch_size = batch_size
        self.patience = patience
        self.lambda_dag = lambda_dag
        self.lambda_sparse = lambda_sparse
        self.dag_gamma_max = dag_gamma_max
        self.lambda_intervention = lambda_intervention
        self.lambda_contrastive = lambda_contrastive
        self.lambda_w_l1 = lambda_w_l1
        self.lambda_denoise = lambda_denoise
        self.lambda_decorrelation = lambda_decorrelation
        self.lambda_rank = lambda_rank
        self.label_smoothing = label_smoothing
        self.edge_dropout = edge_dropout
        self.n_rounds = n_rounds
        self.dag_type = dag_type
        self.n_latent = n_latent
        self.n_groups = n_groups
        self.recon_loss = recon_loss
        self.lambda_block = lambda_block
        self.lambda_recon = lambda_recon
        self.block_mask = block_mask
        self.acyclicity = acyclicity
        self.tau_vcuda = tau_vcuda
        self.tau_perm = tau_perm
        self.tau_edge = tau_edge

        self.model = None
        self.preprocessor = None
        self.num_scaler = StandardScaler()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def _get_dag_param_names(self):
        """Return parameter names that belong to the DAG (frozen in warmup/finetune)."""
        if self.dag_type == 'factored':
            if self.acyclicity in ('spectral', 'dagma'):
                return {'A_raw', 'S_raw'}
            elif self.acyclicity == 'vcuda':
                return {'A_raw', 'priority', 'edge_logits'}
            elif self.acyclicity == 'dpdag':
                return {'A_raw', 'perm_scores', 'edge_logits'}
        if self.dag_type == 'uv':
            return {'U_raw', 'V_raw'}
        if self.dag_type == 'asb':
            if self.acyclicity in ('spectral', 'dagma'):
                return {'A_raw', 'B_raw', 'S_raw'}
            elif self.acyclicity == 'vcuda':
                return {'A_raw', 'B_raw', 'priority', 'edge_logits'}
            elif self.acyclicity == 'dpdag':
                return {'A_raw', 'B_raw', 'perm_scores', 'edge_logits'}
        if self.dag_type in ('asb_sig', 'asb_raw'):
            if self.acyclicity in ('spectral', 'dagma'):
                return {'A_raw', 'B_raw', 'S_raw'}
            else:
                return {'A_raw', 'B_raw', 'priority', 'edge_logits'}
        if self.dag_type in ('asb_vcuda', 'asb_raw_vcuda'):
            return {'A_raw', 'B_raw', 'S_raw_free', 'priority_d'}
        if self.dag_type == 'stacked_pair':
            return {'A1_raw', 'A2_raw', 'S1_raw', 'S2_raw'}
        if self.dag_type == 'qb':
            return {'qb'}
        if self.dag_type == 'qb_hybrid':
            dag_params = {'A_raw', 'qb'}
            if self.acyclicity in ('spectral', 'dagma'):
                dag_params.add('S_raw')
            elif self.acyclicity == 'vcuda':
                dag_params.update({'priority', 'edge_logits'})
            return dag_params
        return {'W_raw'}

    def fit(self, X, y, cat_indices=None, sample_weights=None, teacher_probas=None, X_unlabeled=None, verbose=False):
        cat_indices = cat_indices or []
        self.preprocessor = CategoricalPreprocessor()
        X_num, X_cat = self.preprocessor.fit_transform(X, cat_indices)
        if X_num.shape[1] > 0:
            X_num = self.num_scaler.fit_transform(X_num).astype(np.float32)

        n_classes = len(np.unique(y))
        cat_cards = [self.preprocessor.cardinalities[i]
                     for i in self.preprocessor.cat_indices]

        X_num_t = torch.tensor(X_num, dtype=torch.float32)
        X_cat_t = torch.tensor(X_cat, dtype=torch.long)
        y_t = torch.tensor(y, dtype=torch.long)
        sw_t = (torch.tensor(sample_weights, dtype=torch.float32)
                if sample_weights is not None else None)
        tp_t = (torch.tensor(teacher_probas, dtype=torch.float32)
                if teacher_probas is not None else None)

        n_val = max(int(0.15 * len(X)), 20)
        idx = np.random.permutation(len(X))
        tr_idx, val_idx = idx[n_val:], idx[:n_val]

        # Preprocess unlabeled data (test features)
        ul_loader = None
        if X_unlabeled is not None and (self.lambda_denoise > 0 or self.lambda_intervention > 0):
            X_ul_num, X_ul_cat = self.preprocessor.transform(X_unlabeled)
            if X_ul_num.shape[1] > 0:
                X_ul_num = self.num_scaler.transform(X_ul_num).astype(np.float32)
            ul_num_t = torch.tensor(X_ul_num, dtype=torch.float32)
            ul_cat_t = torch.tensor(X_ul_cat, dtype=torch.long)
            ul_ds = TensorDataset(ul_num_t, ul_cat_t)
            ul_loader = DataLoader(ul_ds, batch_size=self.batch_size, shuffle=True)

        tensors = [X_num_t[tr_idx], X_cat_t[tr_idx], y_t[tr_idx]]
        if sw_t is not None:
            tensors.append(sw_t[tr_idx])
        if tp_t is not None:
            tensors.append(tp_t[tr_idx])
        ds = TensorDataset(*tensors)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model = CausalBoostNet(
            n_num=X_num.shape[1], cat_cardinalities=cat_cards,
            n_classes=n_classes, hidden_dim=self.hidden_dim,
            emb_dim=self.emb_dim, emb_dim_formula=self.emb_dim_formula,
            nam_shape=self.nam_shape, n_slots=self.n_slots,
            qb_n_receptors=self.qb_n_receptors, qb_n_sites=self.qb_n_sites,
            qb_n_families=self.qb_n_families, qb_rho_dim=self.qb_rho_dim, qb_rank=self.qb_rank,
            head_hidden=self.head_hidden, head_dropout=self.head_dropout,
            edge_dropout=self.edge_dropout,
            n_rounds=self.n_rounds, dag_type=self.dag_type,
            n_latent=self.n_latent, n_groups=self.n_groups,
            recon_loss=self.recon_loss, lambda_block=self.lambda_block,
            lambda_recon=self.lambda_recon,
            acyclicity_type=self.acyclicity,
            tau_vcuda=self.tau_vcuda, tau_perm=self.tau_perm,
            tau_edge=self.tau_edge,
            sparse_dag=self.sparse_dag,
            attn_dag=self.attn_dag,
            n_dags=self.n_dags,
            shared_A=self.shared_A,
            garbage_alpha=self.garbage_alpha,
            moe_repulsion=self.moe_repulsion,
            moe_l1=self.moe_l1,
            moe_loadbal=self.moe_loadbal,
            sharpen_tau=self.sharpen_tau,
        ).to(self.device)

        # Set fixed DAG if provided (non-learnable)
        if self.fixed_dag is not None:
            self.model.set_fixed_dag(self.fixed_dag)
            self.model._fixed_W = self.model._fixed_W.to(self.device)

        # Set block mask if provided
        if self.block_mask is not None and self.dag_type == 'block':
            self.model.block_mask.copy_(
                torch.tensor(self.block_mask, dtype=torch.float32)
                .to(self.device))

        # Set RCD DAG if provided (for rcd_residual mode)
        if hasattr(self, 'rcd_dag') and self.rcd_dag is not None:
            self.model.set_rcd_dag(self.rcd_dag.to(self.device))

        val_crit = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing
            if self.label_smoothing > 0 else 0.0)
        d = self.model.n_features
        lambda_sparse = self.lambda_sparse + 0.001 * d

        best_val_loss = float('inf')
        best_score = None   # DIAG
        self.curve, self.best_epoch = [], None   # DIAG
        patience_counter = 0
        best_state = None
        total = self.warmup_epochs + self.joint_epochs + self.finetune_epochs

        dag_param_names = self._get_dag_param_names()
        dag_params, cinn_params = [], []
        for n, p in self.model.named_parameters():
            if n in dag_param_names:
                dag_params.append(p)
            else:
                cinn_params.append(p)
        cinn_params = [p for p in cinn_params if p.requires_grad]

        opt_warmup = optim.AdamW(cinn_params, lr=self.lr, weight_decay=0.01)
        for p in dag_params:
            p.requires_grad_(True)
        opt_joint = optim.AdamW([
            {'params': dag_params, 'lr': self.w_lr},
            {'params': cinn_params, 'lr': self.lr}
        ], weight_decay=0.01)
        for p in dag_params:
            p.requires_grad_(False)
        opt_finetune = optim.AdamW(cinn_params, lr=self.lr * 0.1,
                                   weight_decay=0.01)

        use_w = sw_t is not None
        use_tp = tp_t is not None
        distill_alpha = 0.3 if use_tp else 0.0
        distill_temp = 3.0

        # Find S_anneal param for anneal mode phase control
        anneal_param = None
        if self.sparse_dag == 'anneal' and hasattr(self.model, 'S_anneal_weights'):
            anneal_param = self.model.S_anneal_weights

        for epoch in range(total):
            # Temperature sharpening: anneal tau_edge from initial to sharpen_tau
            if self.sharpen_tau > 0 and total > 0:
                progress = epoch / max(total - 1, 1)
                # Keep tau=initial during warmup, anneal during joint+finetune
                if epoch >= self.warmup_epochs:
                    jp = (epoch - self.warmup_epochs) / max(self.joint_epochs + self.finetune_epochs, 1)
                    new_tau = self.tau_edge * (1.0 - jp) + self.sharpen_tau * jp
                    self.model._tau_edge_current = max(new_tau, self.sharpen_tau)
                else:
                    self.model._tau_edge_current = self.tau_edge

            if epoch < self.warmup_epochs:
                phase = 'warmup'
                for p in dag_params:
                    p.requires_grad_(False)
                if anneal_param is not None:
                    anneal_param.requires_grad_(True)  # S_sparse trains freely
                opt = opt_warmup
                gamma = 0.0
            elif epoch < self.warmup_epochs + self.joint_epochs:
                phase = 'joint'
                if self.sparse_dag == 'anneal':
                    # Anneal joint: S_base still frozen, only S_anneal + CINN
                    for p in dag_params:
                        p.requires_grad_(False)
                    if anneal_param is not None:
                        anneal_param.requires_grad_(True)
                else:
                    for p in dag_params:
                        p.requires_grad_(True)
                opt = opt_joint
                jp = ((epoch - self.warmup_epochs)
                      / max(self.joint_epochs, 1))
                gamma = self.lambda_dag * min(
                    self.dag_gamma_max,
                    self.dag_gamma_max * (2.0 ** (jp * 8) - 1)
                    / (2.0 ** 8 - 1))
            else:
                phase = 'finetune'
                if self.sparse_dag == 'anneal':
                    # Anneal finetune: UNfreeze S_base, freeze S_anneal
                    for p in dag_params:
                        p.requires_grad_(True)  # S_base learns residual
                    if anneal_param is not None:
                        anneal_param.requires_grad_(False)  # sparse structure locked
                else:
                    for p in dag_params:
                        p.requires_grad_(False)
                opt = opt_finetune
                gamma = 0.0

            self.model.train()
            for batch in loader:
                parts = [b.to(self.device) for b in batch]
                xn, xc, yb = parts[0], parts[1], parts[2]
                idx = 3
                wb = parts[idx] if use_w else None
                if use_w:
                    idx += 1
                tb = parts[idx] if use_tp else None

                opt.zero_grad()
                logits = self.model(xn, xc, labels=yb)

                if use_w:
                    loss = (F.cross_entropy(
                        logits, yb, reduction='none',
                        label_smoothing=self.label_smoothing) * wb).mean()
                else:
                    loss = F.cross_entropy(
                        logits, yb,
                        label_smoothing=self.label_smoothing)

                # Distillation loss: KL divergence with teacher soft labels
                if use_tp:
                    log_student = F.log_softmax(logits / distill_temp, dim=1)
                    teacher_soft = (tb / distill_temp).softmax(dim=1) if tb.min() < 0 else tb
                    distill_loss = F.kl_div(log_student, teacher_soft,
                                            reduction='batchmean') * (distill_temp ** 2)
                    loss = (1 - distill_alpha) * loss + distill_alpha * distill_loss

                # Sparse DAG loss
                if self.sparse_dag:
                    if self.sparse_dag == 'anneal':
                        # Anneal mode: 3 phases with different sparsity schedules
                        # Phase 1 (warmup): S_sparse trains freely, no L1
                        # Phase 2 (joint): L1 ramps from 0 to max, pruning weak edges
                        # Phase 3 (finetune): S_base unfreezes, S_sparse stays
                        if phase == 'warmup':
                            anneal_mult = 0.0  # no sparsity — learn freely
                        elif phase == 'joint':
                            jp = ((epoch - self.warmup_epochs)
                                  / max(self.joint_epochs, 1))
                            anneal_mult = jp * 20.0  # ramp L1 from 0 to 20x
                        else:
                            anneal_mult = 20.0  # keep max sparsity
                        if anneal_mult > 0:
                            loss = loss + lambda_sparse * anneal_mult * self.model.sparse_dag_loss()
                    else:
                        loss = loss + lambda_sparse * 5 * self.model.sparse_dag_loss()

                if phase == 'joint':
                    loss = loss + gamma * self.model.dag_loss()
                    if not self.sparse_dag:
                        loss = loss + lambda_sparse * self.model.sparsity_loss()
                    if self.dag_type == 'block':
                        loss = (loss
                                + self.lambda_block * self.model.block_penalty())
                    if self.recon_loss:
                        loss = (loss
                                + self.lambda_recon * self.model._recon_loss)
                    if self.lambda_intervention > 0:
                        loss = loss + self.lambda_intervention * self.model._intervention_loss
                    if self.lambda_contrastive > 0:
                        loss = loss + self.lambda_contrastive * self.model._contrastive_loss
                    if self.lambda_w_l1 > 0:
                        loss = loss + self.lambda_w_l1 * self.model.get_W().abs().mean()

                # Decorrelation and rank losses
                if self.lambda_decorrelation > 0:
                    loss = loss + self.lambda_decorrelation * self.model._decorrelation_loss
                if self.lambda_rank > 0:
                    loss = loss + self.lambda_rank * self.model._rank_loss

                # Denoise loss on training batch
                if self.lambda_denoise > 0:
                    loss = loss + self.lambda_denoise * self.model.denoise_loss(xn, xc)

                # QB receptor repulsion
                if hasattr(self.model, 'qb'):
                    loss = loss + self.model.qb.repulsion_loss()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()

                # Proximal step: soft-threshold S_sparse for exact zeros
                if self.sparse_dag and self.sparse_dag not in ('gumbel', 'gumbel_shared'):
                    if hasattr(self.model, 'S_sparse_raw'):
                        with torch.no_grad():
                            threshold = lambda_sparse * 0.5
                            p = self.model.S_sparse_raw
                            p.data = torch.sign(p) * torch.relu(p.abs() - threshold)

                # Unlabeled pass: denoise + intervention on test features
                if ul_loader is not None and phase == 'joint':
                    try:
                        ul_batch = next(ul_iter)
                    except (StopIteration, NameError):
                        ul_iter = iter(ul_loader)
                        ul_batch = next(ul_iter)
                    ul_xn, ul_xc = [b.to(self.device) for b in ul_batch]
                    opt.zero_grad()
                    ul_loss = torch.tensor(0.0, device=self.device)
                    if self.lambda_denoise > 0:
                        ul_loss = ul_loss + self.lambda_denoise * self.model.denoise_loss(ul_xn, ul_xc)
                    if self.lambda_intervention > 0:
                        self.model(ul_xn, ul_xc)
                        ul_loss = ul_loss + self.lambda_intervention * self.model._intervention_loss
                    if ul_loss.requires_grad:
                        ul_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        opt.step()

            self.model.eval()
            with torch.no_grad():
                vlogits = self.model(X_num_t[val_idx].to(self.device),
                                     X_cat_t[val_idx].to(self.device))
                yv = y_t[val_idx].to(self.device)
                vl = val_crit(vlogits, yv).item()
                vacc = (vlogits.argmax(1) == yv).float().mean().item()   # DIAG
            if DIAG['curve_hook'] is not None:   # DIAG
                self.curve.append(dict(epoch=epoch, phase=phase, val_loss=vl, val_acc=vacc,
                                       test_acc=DIAG['curve_hook'](self)))
            if DIAG['restore_from'] == 'finetune' and phase != 'finetune':   # DIAG
                continue
            score = (vacc, -vl) if DIAG['select'] == 'acc' else (-vl,)   # DIAG
            if best_score is None or score > best_score:
                best_score = score
                best_val_loss = vl
                self.best_epoch = epoch   # DIAG
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if phase == 'finetune' and patience_counter >= self.patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        X_num, X_cat = self.preprocessor.transform(X)
        if X_num.shape[1] > 0:
            X_num = self.num_scaler.transform(X_num).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(
                torch.tensor(X_num, dtype=torch.float32).to(self.device),
                torch.tensor(X_cat, dtype=torch.long).to(self.device)
            ).argmax(dim=1).cpu().numpy()

    def predict_logits(self, X):
        X_num, X_cat = self.preprocessor.transform(X)
        if X_num.shape[1] > 0:
            X_num = self.num_scaler.transform(X_num).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(
                torch.tensor(X_num, dtype=torch.float32).to(self.device),
                torch.tensor(X_cat, dtype=torch.long).to(self.device)
            ).cpu().numpy()


# ============================================================================
# CausalBoostClassifier: Boosted Ensemble (top-level API)
# ============================================================================

class CausalBoostClassifier:
    """
    CausalBoost: boosted ensemble of SDCD+CINN models.

    Training:
      1. Preprocessing: detect missing values, append binary indicators,
         median-impute, label-encode categoricals, StandardScale.
      2. Train M models sequentially with gradient-boosting-style reweighting.
      3. Post-hoc temperature calibration via LBFGS.

    Prediction:
      logits = sum(shrinkage * model_logits) / T

    Args:
      n_models:       Number of boosted models (1 = single model, no boosting).
      shrinkage:      Learning rate for logit accumulation.
      base_alpha:     Base reweighting strength (scaled adaptively per dataset).
      calibrate:      Whether to apply post-hoc temperature scaling.

      dag_type:       'factored' | 'standard' | 'block' | 'latent'
      acyclicity:     'vcuda' | 'spectral' | 'dagma' | 'dpdag'
      n_groups:       Number of groups for factored DAG (default 32).
      n_latent:       Number of latent nodes for latent DAG (default 0).
      n_rounds:       Number of message passing rounds (default 1).

      edge_dropout:   Edge dropout probability during training.
      label_smoothing: Label smoothing for cross-entropy loss.
      recon_loss:     Enable reconstruction loss (factored DAG only).
      lambda_block:   Block-diagonal penalty weight (block DAG only).

      add_missing_indicators: Append binary missingness columns before impute.

      hidden_dim:     Hidden dimension of CINN (default 32).
      emb_dim:        Max embedding dimension for categoricals (default 8).
      warmup_epochs:  Epochs for warmup phase (default 20).
      joint_epochs:   Epochs for joint DAG+CINN training (default 100).
      finetune_epochs: Epochs for finetune phase (default 30).
    """

    def __init__(self,
                 # Ensemble (Dim 6)
                 n_models=5, shrinkage=0.1, base_alpha=2.0,
                 weight_clamp=(0.5, 3.0), calibrate=True,
                 # DAG parameterization (Dim 2)
                 dag_type='factored', n_groups=32, n_latent=0,
                 recon_loss=False, lambda_block=0.1, lambda_recon=0.1,
                 # Acyclicity (Dim 3)
                 acyclicity='vcuda',
                 tau_vcuda=0.3, tau_perm=1.0, tau_edge=0.5,
                 # Message passing (Dim 4)
                 n_rounds=1,
                 # Feature representation (Dim 5)
                 add_missing_indicators=True,
                 # Training (Dim 7)
                 edge_dropout=0.2, label_smoothing=0.1,
                 hidden_dim=32, emb_dim=8, emb_dim_formula='sqrt', nam_shape=False, n_slots=1,
                 sparse_dag=False, attn_dag=False, diverse_dag=False,
                 n_dags=1, shared_A=True, garbage_alpha=0.0,
                 moe_repulsion=False, moe_l1=False, moe_loadbal=False,
                 sharpen_tau=0.0,
                 qb_n_receptors=6, qb_n_sites=4, qb_n_families=6, qb_rho_dim=6, qb_rank=3,
                 head_hidden=None, head_dropout=0.1,
                 warmup_epochs=20, joint_epochs=100, finetune_epochs=30,
                 lr=0.001, w_lr=0.005, batch_size=32, patience=25,
                 lambda_dag=1.0, lambda_sparse=0.05, dag_gamma_max=50.0,
                 lambda_intervention=0.0, lambda_contrastive=0.0, lambda_w_l1=0.0, lambda_denoise=0.0,
                 lambda_decorrelation=0.0, lambda_rank=0.0,
                 # Fixed DAG (optional — overrides learned DAG)
                 fixed_dag=None, fixed_dags=None, random_state=42):
        self.random_state = random_state

        # Fixed DAGs
        self.fixed_dag = fixed_dag    # single DAG for all models
        self.fixed_dags = fixed_dags  # list of DAGs, one per model
        # MoE-DAG / Temperature sharpening
        self.n_dags = n_dags
        self.shared_A = shared_A
        self.garbage_alpha = garbage_alpha
        self.moe_repulsion = moe_repulsion
        self.moe_l1 = moe_l1
        self.moe_loadbal = moe_loadbal
        self.sharpen_tau = sharpen_tau
        # Ensemble
        self.n_models = n_models
        self.shrinkage = shrinkage
        self.base_alpha = base_alpha
        self.weight_clamp = weight_clamp
        self.calibrate = calibrate
        # DAG
        self.dag_type = dag_type
        self.n_groups = n_groups
        self.n_latent = n_latent
        self.recon_loss = recon_loss
        self.lambda_block = lambda_block
        self.lambda_recon = lambda_recon
        # Acyclicity
        self.acyclicity = acyclicity
        self.tau_vcuda = tau_vcuda
        self.tau_perm = tau_perm
        self.tau_edge = tau_edge
        # Message passing
        self.n_rounds = n_rounds
        # Features
        self.add_missing_indicators = add_missing_indicators
        # Training
        self.edge_dropout = edge_dropout
        self.label_smoothing = label_smoothing
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.emb_dim_formula = emb_dim_formula
        self.nam_shape = nam_shape
        self.n_slots = n_slots
        self.diverse_dag = diverse_dag
        self.sparse_dag = sparse_dag
        self.attn_dag = attn_dag
        self.qb_n_receptors = qb_n_receptors
        self.qb_n_sites = qb_n_sites
        self.qb_n_families = qb_n_families
        self.qb_rho_dim = qb_rho_dim
        self.qb_rank = qb_rank
        self.head_hidden = head_hidden
        self.head_dropout = head_dropout
        self.warmup_epochs = warmup_epochs
        self.joint_epochs = joint_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.w_lr = w_lr
        self.batch_size = batch_size
        self.patience = patience
        self.lambda_dag = lambda_dag
        self.lambda_sparse = lambda_sparse
        self.dag_gamma_max = dag_gamma_max
        self.lambda_intervention = lambda_intervention
        self.lambda_contrastive = lambda_contrastive
        self.lambda_w_l1 = lambda_w_l1
        self.lambda_denoise = lambda_denoise
        self.lambda_decorrelation = lambda_decorrelation
        self.lambda_rank = lambda_rank

        # State
        self.models: List[CINNClassifier] = []
        self.n_classes = None
        self.temperature = 1.0
        self.miss_handler: Optional[MissingnessHandler] = None
        self._cat_indices_internal: List[int] = []

    # ── Missingness preprocessing ──

    def _preprocess_missing(self, X, cat_indices, fit=False):
        if not self.add_missing_indicators:
            if fit:
                self.miss_handler = None
                self._cat_indices_internal = cat_indices
            return X, cat_indices

        if fit:
            self.miss_handler = MissingnessHandler()
            X_out = self.miss_handler.fit_transform(X)
            self._cat_indices_internal = cat_indices
        else:
            X_out = self.miss_handler.transform(X)

        return X_out, self._cat_indices_internal

    def _get_fixed_dag(self, model_idx):
        """Get fixed DAG for model i, or None if learning DAG."""
        if self.fixed_dags is not None and model_idx < len(self.fixed_dags):
            return self.fixed_dags[model_idx]
        if self.fixed_dag is not None:
            return self.fixed_dag
        return None

    # ── Block mask computation ──

    def _compute_block_mask(self, X_num, d_total):
        """Compute correlation-based block mask for block DAG."""
        if X_num.shape[1] < 2:
            return np.ones((d_total, d_total), dtype=np.float32)
        corr = np.abs(np.corrcoef(X_num.T))
        corr = np.nan_to_num(corr, nan=0.0)
        d_num = X_num.shape[1]
        n_clusters = max(d_total // 4, 4)
        n_clusters = min(n_clusters, d_total)
        if d_num >= 2:
            dist = 1.0 - corr
            np.fill_diagonal(dist, 0)
            dist = np.clip(dist, 0, 2)
            dist = (dist + dist.T) / 2
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method='ward')
            clusters_num = fcluster(Z, t=n_clusters, criterion='maxclust')
        else:
            clusters_num = np.array([1])
        clusters = np.zeros(d_total, dtype=int)
        clusters[:d_num] = clusters_num
        next_cluster = clusters_num.max() + 1
        for i in range(d_num, d_total):
            clusters[i] = next_cluster
            next_cluster += 1
        return (clusters[:, None] == clusters[None, :]).astype(np.float32)

    # ── Training ──

    def fit(self, X, y, cat_indices=None, teacher_probas=None, X_unlabeled=None, verbose=False):
        cat_indices = cat_indices or []
        X = np.array(X, dtype=float)
        y = np.array(y)

        # Missingness
        X, cat_indices = self._preprocess_missing(X, cat_indices, fit=True)

        self.models = []
        self.n_classes = len(np.unique(y))
        n, d = X.shape
        boost_alpha = compute_adaptive_alpha(n, d, self.n_classes,
                                             self.base_alpha)

        # Compute block mask if needed
        block_mask = None
        if self.dag_type == 'block':
            prep = CategoricalPreprocessor()
            X_num_raw, _ = prep.fit_transform(X, cat_indices)
            scaler = StandardScaler()
            X_num_scaled = (scaler.fit_transform(X_num_raw).astype(np.float32)
                            if X_num_raw.shape[1] > 0 else X_num_raw)
            block_mask = self._compute_block_mask(X_num_scaled, prep.n_features)

        base_seed = int(self.random_state) % 100000  # fixed seed; model m uses base_seed + 17m
        acc_logits = np.zeros((len(X), self.n_classes), dtype=np.float32)
        sw = None

        for i in range(self.n_models):
            seed = base_seed + i * 17
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Scale Gumbel temperature per model for structural diversity
            # Model 0: default temp (sharp). Later models: higher temp (exploratory).
            if self.diverse_dag and self.n_models > 1:
                tau_scale = 1.0 + 2.0 * i / (self.n_models - 1)  # 1.0 → 3.0
                model_tau_edge = self.tau_edge * tau_scale
            else:
                model_tau_edge = self.tau_edge

            clf = CINNClassifier(
                hidden_dim=self.hidden_dim, emb_dim=self.emb_dim, emb_dim_formula=self.emb_dim_formula,
                nam_shape=self.nam_shape, n_slots=self.n_slots,
                sparse_dag=self.sparse_dag, attn_dag=self.attn_dag,
                qb_n_receptors=self.qb_n_receptors, qb_n_sites=self.qb_n_sites,
                qb_n_families=self.qb_n_families, qb_rho_dim=self.qb_rho_dim, qb_rank=self.qb_rank,
                head_hidden=self.head_hidden, head_dropout=self.head_dropout,
                warmup_epochs=self.warmup_epochs,
                joint_epochs=self.joint_epochs,
                finetune_epochs=self.finetune_epochs,
                lr=self.lr, w_lr=self.w_lr,
                batch_size=self.batch_size, patience=self.patience,
                lambda_dag=self.lambda_dag,
                lambda_sparse=self.lambda_sparse,
                dag_gamma_max=self.dag_gamma_max,
                lambda_intervention=self.lambda_intervention,
                lambda_contrastive=self.lambda_contrastive,
                lambda_w_l1=self.lambda_w_l1,
                lambda_denoise=self.lambda_denoise,
                lambda_decorrelation=self.lambda_decorrelation,
                lambda_rank=self.lambda_rank,
                edge_dropout=self.edge_dropout,
                label_smoothing=self.label_smoothing,
                n_rounds=self.n_rounds,
                dag_type=self.dag_type,
                n_latent=self.n_latent,
                n_groups=self.n_groups,
                recon_loss=self.recon_loss,
                lambda_block=self.lambda_block,
                lambda_recon=self.lambda_recon,
                block_mask=block_mask,
                acyclicity=self.acyclicity,
                tau_vcuda=self.tau_vcuda,
                tau_perm=self.tau_perm,
                tau_edge=model_tau_edge,
                fixed_dag=self._get_fixed_dag(i),
                n_dags=self.n_dags,
                shared_A=self.shared_A,
                garbage_alpha=self.garbage_alpha,
                moe_repulsion=self.moe_repulsion,
                moe_l1=self.moe_l1,
                moe_loadbal=self.moe_loadbal,
                sharpen_tau=self.sharpen_tau,
            )
            # Inject RCD DAG for rcd_residual mode
            if hasattr(self, '_rcd_dag') and self._rcd_dag is not None:
                clf.rcd_dag = self._rcd_dag
            clf.fit(X, y, cat_indices=cat_indices, sample_weights=sw,
                    teacher_probas=teacher_probas, X_unlabeled=X_unlabeled, verbose=False)
            self.models.append(clf)

            if self.n_models > 1:
                logits = clf.predict_logits(X)
                acc_logits += self.shrinkage * logits

                exp_a = np.exp(
                    acc_logits - acc_logits.max(axis=1, keepdims=True))
                proba = exp_a / exp_a.sum(axis=1, keepdims=True)
                p_c = proba[np.arange(len(y)), y]
                sw = 1.0 + boost_alpha * (1.0 - p_c)
                sw = np.clip(sw, self.weight_clamp[0], self.weight_clamp[1])
                sw = sw / sw.mean()

        if self.calibrate and self.n_models > 1:
            self.temperature = learn_temperature(acc_logits, y)
        return self

    # ── Prediction ──

    def predict_logits(self, X):
        X = np.array(X, dtype=float)
        X, _ = self._preprocess_missing(X, self._cat_indices_internal,
                                        fit=False)
        acc = np.zeros((len(X), self.n_classes), dtype=np.float32)
        for m in self.models:
            acc += self.shrinkage * m.predict_logits(X)
        return acc / self.temperature

    def predict(self, X):
        return np.argmax(self.predict_logits(X), axis=1)

    def predict_proba(self, X):
        logits = self.predict_logits(X)
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_l / exp_l.sum(axis=1, keepdims=True)


# ============================================================================
# Quick test
# ============================================================================



import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from tub import BaseModel


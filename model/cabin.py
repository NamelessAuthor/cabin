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
warnings.filterwarnings('ignore')


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
                 edge_dropout: float = 0.0, n_rounds: int = 1,
                 dag_type: str = 'factored', n_latent: int = 0,
                 n_groups: int = 32, recon_loss: bool = False,
                 lambda_block: float = 0.1, lambda_recon: float = 0.1,
                 acyclicity_type: str = 'vcuda',
                 tau_vcuda: float = 0.3, tau_perm: float = 1.0,
                 tau_edge: float = 0.5):
        super().__init__()
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
        self.recon_loss_enabled = recon_loss
        self.lambda_block = lambda_block
        self.lambda_recon = lambda_recon
        self.acyclicity_type = acyclicity_type
        self.tau_vcuda = tau_vcuda
        self.tau_perm = tau_perm
        self.tau_edge = tau_edge

        d = self.n_features

        # ── Feature embeddings ──
        self.num_embedders = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(n_num)
        ])
        self.cat_embeddings = nn.ModuleList()
        self.cat_projectors = nn.ModuleList()
        for card in cat_cardinalities:
            actual_emb_dim = min(emb_dim, max(2, int(np.sqrt(card))))
            self.cat_embeddings.append(nn.Embedding(card, actual_emb_dim))
            self.cat_projectors.append(nn.Linear(actual_emb_dim, hidden_dim))

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
            self.A_raw = nn.Parameter(torch.randn(d, k) * 0.01)
            self._init_S_params(k, acyclicity_type)
            # Dummy W_raw so state_dict is consistent
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

        # ── Classifier head ──
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * dag_d, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, n_classes)
        )

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

    # ── DAG accessors ──

    def _get_S(self):
        """Get group-level DAG matrix S [k, k] (factored mode only)."""
        if self.acyclicity_type in ('spectral', 'dagma'):
            return torch.sigmoid(self.S_raw) * self.S_no_self_loop
        elif self.acyclicity_type == 'vcuda':
            grad_p = self.priority.unsqueeze(0) - self.priority.unsqueeze(1)
            direction = torch.sigmoid(grad_p / self.tau_vcuda)
            if self.training:
                raw = torch.stack([self.edge_logits, -self.edge_logits])
                edges = F.gumbel_softmax(raw, tau=self.tau_edge,
                                         hard=True, dim=0)[0]
            else:
                edges = (torch.sigmoid(self.edge_logits) > 0.5).float()
                edges.diagonal().fill_(0)
            return edges * direction * self.no_self_loop_k
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
                raw = torch.stack([self.edge_logits, -self.edge_logits])
                edges = F.gumbel_softmax(raw, tau=self.tau_edge,
                                         hard=True, dim=0)[0]
            else:
                edges = (torch.sigmoid(self.edge_logits) > 0.5).float()
                edges.diagonal().fill_(0)
            order_mask = P @ self.triu_mask @ P.T
            return edges * order_mask * self.no_self_loop_k

    def set_fixed_dag(self, W_numpy):
        """Fix the DAG to a pre-computed adjacency matrix (non-learnable).

        Args:
            W_numpy: [d, d] numpy array, binary or weighted adjacency matrix.
        """
        W_t = torch.tensor(W_numpy, dtype=torch.float32)
        self.register_buffer('_fixed_W', W_t)

    def get_W(self):
        """Compute full adjacency matrix W [d, d]."""
        if hasattr(self, '_fixed_W') and self._fixed_W is not None:
            return self._fixed_W
        if self.dag_type == 'factored':
            A = torch.softmax(self.A_raw, dim=1)
            S = self._get_S()
            W = A @ S @ A.T
            return W * self.no_self_loop
        else:
            return torch.sigmoid(self.W_raw) * self.no_self_loop

    # ── Loss components ──

    def dag_loss(self):
        """Acyclicity constraint loss."""
        if hasattr(self, '_fixed_W') and self._fixed_W is not None:
            return torch.tensor(0.0)
        if self.dag_type == 'factored':
            if self.acyclicity_type == 'spectral':
                return spectral_radius_acyclicity(self._get_S())
            elif self.acyclicity_type == 'dagma':
                return dagma_constraint(self._get_S())
            else:
                # vcuda and dpdag are acyclic by construction
                return torch.tensor(0.0, device=self.A_raw.device)
        else:
            return spectral_radius_acyclicity(self.get_W())

    def sparsity_loss(self):
        """L1 sparsity on edge weights."""
        if self.dag_type == 'factored':
            if self.acyclicity_type in ('spectral', 'dagma'):
                return self._get_S().abs().mean()
            else:
                return torch.sigmoid(self.edge_logits).mean()
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

    # ── Forward pass ──

    def forward(self, X_num: torch.Tensor, X_cat: torch.Tensor):
        B = X_num.shape[0] if X_num.shape[1] > 0 else X_cat.shape[0]

        # Embed each feature
        embeddings = []
        for i in range(self.n_num):
            embeddings.append(self.num_embedders[i](X_num[:, i:i + 1]))
        for i in range(self.n_cat):
            emb_i = self.cat_embeddings[i](X_cat[:, i])
            embeddings.append(self.cat_projectors[i](emb_i))

        H = torch.stack(embeddings, dim=1)  # [B, d, hidden_dim]

        # Latent nodes
        if self.latent_embeds is not None:
            H = torch.cat([H, self.latent_embeds.expand(B, -1, -1)], dim=1)

        H_pre_dag = H if self.recon_loss_enabled else None

        # Message passing
        W = self.get_W()
        if self.training and self.edge_dropout > 0:
            mask = (torch.rand_like(W) > self.edge_dropout).float()
            W = W * mask

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

    def __init__(self, hidden_dim=32, emb_dim=8,
                 warmup_epochs=20, joint_epochs=100, finetune_epochs=30,
                 lr=0.001, w_lr=0.005, batch_size=32, patience=25,
                 lambda_dag=1.0, lambda_sparse=0.05, dag_gamma_max=50.0,
                 label_smoothing=0.0, edge_dropout=0.0, n_rounds=1,
                 dag_type='factored', n_latent=0, n_groups=32,
                 recon_loss=False, lambda_block=0.1, lambda_recon=0.1,
                 block_mask=None,
                 acyclicity='vcuda',
                 tau_vcuda=0.3, tau_perm=1.0, tau_edge=0.5,
                 fixed_dag=None):

        self.fixed_dag = fixed_dag
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
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
        return {'W_raw'}

    def fit(self, X, y, cat_indices=None, sample_weights=None, verbose=False):
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

        n_val = max(int(0.15 * len(X)), 20)
        idx = np.random.permutation(len(X))
        tr_idx, val_idx = idx[n_val:], idx[:n_val]

        if sw_t is not None:
            ds = TensorDataset(X_num_t[tr_idx], X_cat_t[tr_idx],
                               y_t[tr_idx], sw_t[tr_idx])
        else:
            ds = TensorDataset(X_num_t[tr_idx], X_cat_t[tr_idx], y_t[tr_idx])
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model = CausalBoostNet(
            n_num=X_num.shape[1], cat_cardinalities=cat_cards,
            n_classes=n_classes, hidden_dim=self.hidden_dim,
            emb_dim=self.emb_dim, edge_dropout=self.edge_dropout,
            n_rounds=self.n_rounds, dag_type=self.dag_type,
            n_latent=self.n_latent, n_groups=self.n_groups,
            recon_loss=self.recon_loss, lambda_block=self.lambda_block,
            lambda_recon=self.lambda_recon,
            acyclicity_type=self.acyclicity,
            tau_vcuda=self.tau_vcuda, tau_perm=self.tau_perm,
            tau_edge=self.tau_edge,
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

        val_crit = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing
            if self.label_smoothing > 0 else 0.0)
        d = self.model.n_features
        lambda_sparse = self.lambda_sparse + 0.001 * d

        best_val_loss = float('inf')
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

        for epoch in range(total):
            if epoch < self.warmup_epochs:
                phase = 'warmup'
                for p in dag_params:
                    p.requires_grad_(False)
                opt = opt_warmup
                gamma = 0.0
            elif epoch < self.warmup_epochs + self.joint_epochs:
                phase = 'joint'
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
                for p in dag_params:
                    p.requires_grad_(False)
                opt = opt_finetune
                gamma = 0.0

            self.model.train()
            for batch in loader:
                if use_w:
                    xn, xc, yb, wb = [b.to(self.device) for b in batch]
                else:
                    xn, xc, yb = [b.to(self.device) for b in batch]

                opt.zero_grad()
                logits = self.model(xn, xc)

                if use_w:
                    loss = (F.cross_entropy(
                        logits, yb, reduction='none',
                        label_smoothing=self.label_smoothing) * wb).mean()
                else:
                    loss = F.cross_entropy(
                        logits, yb,
                        label_smoothing=self.label_smoothing)

                if phase == 'joint':
                    loss = loss + gamma * self.model.dag_loss()
                    loss = loss + lambda_sparse * self.model.sparsity_loss()
                    if self.dag_type == 'block':
                        loss = (loss
                                + self.lambda_block * self.model.block_penalty())
                    if self.recon_loss:
                        loss = (loss
                                + self.lambda_recon * self.model._recon_loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()

            self.model.eval()
            with torch.no_grad():
                vl = val_crit(
                    self.model(X_num_t[val_idx].to(self.device),
                               X_cat_t[val_idx].to(self.device)),
                    y_t[val_idx].to(self.device)
                ).item()

            if vl < best_val_loss:
                best_val_loss = vl
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
                 hidden_dim=32, emb_dim=8,
                 warmup_epochs=20, joint_epochs=100, finetune_epochs=30,
                 lr=0.001, w_lr=0.005, batch_size=32, patience=25,
                 lambda_dag=1.0, lambda_sparse=0.05, dag_gamma_max=50.0,
                 # Fixed DAG (optional — overrides learned DAG)
                 fixed_dag=None, fixed_dags=None):

        # Fixed DAGs
        self.fixed_dag = fixed_dag    # single DAG for all models
        self.fixed_dags = fixed_dags  # list of DAGs, one per model
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

    def fit(self, X, y, cat_indices=None, verbose=False):
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

        import time as _time
        base_seed = int(_time.time() * 1000) % 100000
        acc_logits = np.zeros((len(X), self.n_classes), dtype=np.float32)
        sw = None

        for i in range(self.n_models):
            seed = base_seed + i * 17
            np.random.seed(seed)
            torch.manual_seed(seed)

            clf = CINNClassifier(
                hidden_dim=self.hidden_dim, emb_dim=self.emb_dim,
                warmup_epochs=self.warmup_epochs,
                joint_epochs=self.joint_epochs,
                finetune_epochs=self.finetune_epochs,
                lr=self.lr, w_lr=self.w_lr,
                batch_size=self.batch_size, patience=self.patience,
                lambda_dag=self.lambda_dag,
                lambda_sparse=self.lambda_sparse,
                dag_gamma_max=self.dag_gamma_max,
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
                tau_edge=self.tau_edge,
                fixed_dag=self._get_fixed_dag(i),
            )
            clf.fit(X, y, cat_indices=cat_indices, sample_weights=sw,
                    verbose=False)
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

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    print("=== Champion config (factored + VCUDA) ===")
    clf = CausalBoostClassifier(n_models=2, n_groups=8)
    clf.fit(X_train, y_train)
    print(f"  Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.4f}")

    print("\n=== Standard DAG + spectral radius ===")
    clf2 = CausalBoostClassifier(
        n_models=2, dag_type='standard', acyclicity='spectral')
    clf2.fit(X_train, y_train)
    print(f"  Accuracy: {accuracy_score(y_test, clf2.predict(X_test)):.4f}")

    print("\n=== Factored + DAGMA ===")
    clf3 = CausalBoostClassifier(
        n_models=2, n_groups=8, acyclicity='dagma')
    clf3.fit(X_train, y_train)
    print(f"  Accuracy: {accuracy_score(y_test, clf3.predict(X_test)):.4f}")

    print("\n=== Single model, no ensemble ===")
    clf4 = CausalBoostClassifier(
        n_models=1, calibrate=False, n_groups=8)
    clf4.fit(X_train, y_train)
    print(f"  Accuracy: {accuracy_score(y_test, clf4.predict(X_test)):.4f}")

    print("\n=== With missing values ===")
    X_miss = X.copy()
    rng = np.random.RandomState(42)
    X_miss[rng.rand(*X_miss.shape) < 0.1] = np.nan
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_miss, y, test_size=0.3, random_state=42, stratify=y)
    clf5 = CausalBoostClassifier(n_models=2, n_groups=8)
    clf5.fit(X_tr, y_tr)
    print(f"  Accuracy: {accuracy_score(y_te, clf5.predict(X_te)):.4f}")
    print(f"  Missingness indicators: {clf5.miss_handler.n_indicator_cols}")

    print("\n=== Block DAG ===")
    clf6 = CausalBoostClassifier(
        n_models=1, calibrate=False, dag_type='block', acyclicity='spectral')
    clf6.fit(X_train, y_train)
    print(f"  Accuracy: {accuracy_score(y_test, clf6.predict(X_test)):.4f}")

    print("\n=== DPDag acyclicity ===")
    clf7 = CausalBoostClassifier(
        n_models=2, n_groups=8, acyclicity='dpdag')
    clf7.fit(X_train, y_train)
    print(f"  Accuracy: {accuracy_score(y_test, clf7.predict(X_test)):.4f}")

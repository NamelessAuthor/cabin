"""
Self-contained model implementations for the CausalBoost paper.

All models needed to reproduce the paper experiments are defined here.
No external model imports required — only standard pip packages.

Models:
  - CausalBoostClassifier (V38BoostClassifier): Full CausalBoost pipeline
  - CINNClassifier (V38Classifier): Single SDCD+CINN model
  - CASTLEPipelineClassifier: Two-stage CASTLE+CINN (v2 baseline)
  - GOLEMPipelineClassifier: Two-stage GOLEM+SoftCINN (v15 baseline)

Dependencies:
  pip install torch numpy scikit-learn scipy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
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

    def __init__(self, cat_indices: Optional[List[int]] = None):
        self.cat_indices = cat_indices or []
        self.label_encoders: Dict[int, LabelEncoder] = {}
        self.cardinalities: Dict[int, int] = {}
        self.num_indices: List[int] = []
        self.n_features = 0
        self.fitted = False

    def fit(self, X: np.ndarray, cat_indices: Optional[List[int]] = None):
        if cat_indices is not None:
            self.cat_indices = cat_indices
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


# ============================================================================
# DAG Constraints
# ============================================================================

def dagma_constraint(S: torch.Tensor) -> torch.Tensor:
    """DAGMA log-det acyclicity constraint on S.

    h(S) = -log det(kI - S*S) + k*log(k)
    h(S) = 0 iff S is a DAG.
    Better gradients than spectral radius (no vanishing gradient).
    """
    import math
    k = S.shape[0]
    M = k * torch.eye(k, device=S.device) - S * S
    sign, logabsdet = torch.slogdet(M)
    if sign.item() <= 0:
        return torch.trace(torch.matrix_exp(S * S)) - k
    return -logabsdet + k * math.log(k)


def soft_sort(scores: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """SoftSort: differentiable relaxation of sorting to get a permutation matrix."""
    scores_3d = scores.unsqueeze(0).unsqueeze(-1)
    sorted_vals = scores_3d.sort(descending=True, dim=1)[0]
    diffs = (scores_3d.transpose(1, 2) - sorted_vals).abs().neg() / tau
    P = diffs.softmax(-1).squeeze(0)
    return P


def spectral_radius_acyclicity(W: torch.Tensor, n_power_iter: int = 15) -> torch.Tensor:
    """SDCD acyclicity constraint via spectral radius of W^2."""
    d = W.shape[0]
    W_sq = W * W
    W_sq = W_sq * (1 - torch.eye(d, device=W.device))
    v = torch.ones(d, device=W.device) / np.sqrt(d)
    for _ in range(n_power_iter):
        v_new = W_sq @ v
        norm = torch.norm(v_new)
        if norm > 1e-8:
            v = v_new / norm
        else:
            return torch.tensor(0.0, device=W.device)
    rho = (v @ W_sq @ v) / (v @ v + 1e-8)
    return rho


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
    """Scale boost alpha based on dataset characteristics (n, d, n_classes)."""
    d_scale = np.clip(np.log2(max(d, 2)) / np.log2(40), 0.3, 1.2)
    class_scale = 1.0 / np.sqrt(max(n_classes / 2, 1.0))
    n_scale = np.clip(np.sqrt(n / 500), 0.5, 1.5)
    return base_alpha * d_scale * class_scale * n_scale


# ============================================================================
# Projection Modules (for DAG parameterization ablations)
# ============================================================================

class LinearProjection(nn.Module):
    """Linear projection from d features to k in embedding space."""
    def __init__(self, d: int, k: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(d, k, bias=False)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.proj(H.transpose(1, 2)).transpose(1, 2)


class SoftGroupProjection(nn.Module):
    """Attention-based soft grouping: k learned queries attend to d features."""
    def __init__(self, d: int, k: int, hidden_dim: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(k, hidden_dim) * 0.02)
        self.scale = hidden_dim ** -0.5

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum('bdh,kh->bdk', H, self.queries) * self.scale
        weights = torch.softmax(scores, dim=1)
        return torch.einsum('bdh,bdk->bkh', H, weights)


# ============================================================================
# CausalBoost Neural Network (V38Model)
# ============================================================================

class CausalBoostNet(nn.Module):
    """
    SDCD+CINN network: learns a DAG adjacency matrix end-to-end
    alongside a causality-informed message-passing classifier.

    Supports multiple DAG parameterizations:
      - 'standard': Full W[d,d] adjacency
      - 'factored': W = A[d,k] @ S[k,k] @ A.T (groups correlated features)
      - 'block':    Full W with block-diagonal regularization
      - 'latent':   Full W with additional latent nodes
    """

    def __init__(self, n_num: int, cat_cardinalities: List[int],
                 n_classes: int, hidden_dim: int = 32, emb_dim: int = 8,
                 edge_dropout: float = 0.0, n_rounds: int = 1,
                 proj_type: Optional[str] = None, proj_k: int = 64,
                 dag_type: str = 'standard', n_latent: int = 0,
                 n_groups: int = 32, recon_loss: bool = False,
                 lambda_block: float = 0.1, lambda_recon: float = 0.1,
                 acyclicity_type: str = 'spectral',
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
        self.proj_type = proj_type
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

        # Feature embeddings
        self.num_embedders = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(n_num)
        ])
        self.cat_embeddings = nn.ModuleList()
        self.cat_projectors = nn.ModuleList()
        for card in cat_cardinalities:
            actual_emb_dim = min(emb_dim, max(2, int(np.sqrt(card))))
            self.cat_embeddings.append(nn.Embedding(card, actual_emb_dim))
            self.cat_projectors.append(nn.Linear(actual_emb_dim, hidden_dim))

        # Projection layer (reduces d -> proj_k before DAG)
        self.projection = None
        if proj_type in ('linear', 'pca_init', 'pca_fixed'):
            self.projection = LinearProjection(d, proj_k, hidden_dim)
        elif proj_type == 'soft_group':
            self.projection = SoftGroupProjection(d, proj_k, hidden_dim)

        # DAG dimension
        if proj_type is not None:
            dag_d = proj_k
        elif dag_type == 'latent':
            dag_d = d + n_latent
        else:
            dag_d = d
        self.dag_d = dag_d

        # Latent node embeddings
        self.latent_embeds = None
        if dag_type == 'latent' and n_latent > 0:
            self.latent_embeds = nn.Parameter(torch.randn(1, n_latent, hidden_dim) * 0.02)

        # DAG parameters
        k = n_groups
        if dag_type == 'factored':
            self.A_raw = nn.Parameter(torch.randn(d, k) * 0.01)
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
                self.register_buffer('triu_mask', torch.triu(torch.ones(k, k), diagonal=1))
                self.register_buffer('no_self_loop_k', 1.0 - torch.eye(k))
            self.W_raw = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.register_buffer('no_self_loop', 1.0 - torch.eye(d))
        else:
            self.W_raw = nn.Parameter(torch.randn(dag_d, dag_d) * 0.01)
            self.register_buffer('no_self_loop', 1.0 - torch.eye(dag_d))

        # Block mask buffer
        if dag_type == 'block':
            self.register_buffer('block_mask', torch.ones(dag_d, dag_d))

        # Message passing layers
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

        # Reconstruction heads (for recon loss variants)
        self.recon_head = None
        if recon_loss and proj_type is not None:
            self.recon_head = nn.Linear(hidden_dim, hidden_dim)
        self.factored_recon_head = None
        if recon_loss and dag_type == 'factored':
            self.factored_recon_head = nn.Linear(hidden_dim, hidden_dim)

        # Classifier
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * dag_d, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, n_classes)
        )

    def _get_S(self):
        """Get group-level DAG matrix S [k, k]."""
        if self.acyclicity_type in ('spectral', 'dagma'):
            return torch.sigmoid(self.S_raw) * self.S_no_self_loop
        elif self.acyclicity_type == 'vcuda':
            grad_p = self.priority.unsqueeze(0) - self.priority.unsqueeze(1)
            direction = torch.sigmoid(grad_p / self.tau_vcuda)
            if self.training:
                raw_logits = torch.stack([self.edge_logits, -self.edge_logits])
                edges = F.gumbel_softmax(raw_logits, tau=self.tau_edge, hard=True, dim=0)[0]
            else:
                edges = (torch.sigmoid(self.edge_logits) > 0.5).float()
                edges.diagonal().fill_(0)
            return edges * direction * self.no_self_loop_k
        elif self.acyclicity_type == 'dpdag':
            k = self.n_groups
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
                raw_logits = torch.stack([self.edge_logits, -self.edge_logits])
                edges = F.gumbel_softmax(raw_logits, tau=self.tau_edge, hard=True, dim=0)[0]
            else:
                edges = (torch.sigmoid(self.edge_logits) > 0.5).float()
                edges.diagonal().fill_(0)
            order_mask = P @ self.triu_mask @ P.T
            return edges * order_mask * self.no_self_loop_k

    def get_W(self):
        if self.dag_type == 'factored':
            A = torch.softmax(self.A_raw, dim=1)
            S = self._get_S()
            W = A @ S @ A.T
            return W * self.no_self_loop
        else:
            return torch.sigmoid(self.W_raw) * self.no_self_loop

    def dag_loss(self):
        if self.dag_type == 'factored':
            if self.acyclicity_type == 'spectral':
                return spectral_radius_acyclicity(self._get_S())
            elif self.acyclicity_type == 'dagma':
                return dagma_constraint(self._get_S())
            else:
                # vcuda and dpdag are DAG by construction
                return torch.tensor(0.0, device=self.A_raw.device)
        else:
            return spectral_radius_acyclicity(self.get_W())

    def sparsity_loss(self):
        if self.dag_type == 'factored':
            if self.acyclicity_type in ('spectral', 'dagma'):
                return self._get_S().abs().mean()
            else:
                return torch.sigmoid(self.edge_logits).mean()
        else:
            return self.get_W().abs().mean()

    def block_penalty(self):
        if self.dag_type != 'block':
            return torch.tensor(0.0, device=self.W_raw.device)
        W = self.get_W()
        cross_mask = 1.0 - self.block_mask
        return (W * cross_mask).abs().mean()

    def reconstruction_loss(self, H_proj: torch.Tensor) -> torch.Tensor:
        if self.recon_head is None:
            return torch.tensor(0.0, device=H_proj.device)
        W = self.get_W()
        parent_msgs = torch.einsum('ji,bjh->bih', W, H_proj)
        H_recon = self.recon_head(parent_msgs)
        return F.mse_loss(H_recon, H_proj.detach())

    def factored_reconstruction_loss(self, H: torch.Tensor) -> torch.Tensor:
        if self.factored_recon_head is None:
            return torch.tensor(0.0, device=H.device)
        A = torch.softmax(self.A_raw, dim=1)
        S = self._get_S()
        H_group = torch.einsum('bdh,dk->bkh', H, A)
        H_parents = torch.einsum('ji,bjh->bih', S, H_group)
        H_recon = self.factored_recon_head(H_parents)
        return F.mse_loss(H_recon, H_group.detach())

    def forward(self, X_num: torch.Tensor, X_cat: torch.Tensor):
        B = X_num.shape[0] if X_num.shape[1] > 0 else X_cat.shape[0]

        embeddings = []
        for i in range(self.n_num):
            embeddings.append(self.num_embedders[i](X_num[:, i:i+1]))
        for i in range(self.n_cat):
            emb_i = self.cat_embeddings[i](X_cat[:, i])
            embeddings.append(self.cat_projectors[i](emb_i))

        H = torch.stack(embeddings, dim=1)

        if self.projection is not None:
            H = self.projection(H)

        if self.latent_embeds is not None:
            H = torch.cat([H, self.latent_embeds.expand(B, -1, -1)], dim=1)

        H_pre_dag = H if self.recon_loss_enabled else None

        W = self.get_W()
        if self.training and self.edge_dropout > 0:
            mask = (torch.rand_like(W) > self.edge_dropout).float()
            W = W * mask

        for r in range(self.n_rounds):
            parent_msgs = torch.einsum('ji,bjh->bih', W, H)
            parent_msgs = self.parent_aggs[r](parent_msgs)
            H = self.updaters[r](torch.cat([H, parent_msgs], dim=-1))

        logits = self.predictor(H.view(B, -1))

        if self.recon_loss_enabled and self.training:
            if self.dag_type == 'factored' and H_pre_dag is not None:
                self._recon_loss = self.factored_reconstruction_loss(H_pre_dag)
            elif H_pre_dag is not None:
                self._recon_loss = self.reconstruction_loss(H_pre_dag)
            else:
                self._recon_loss = torch.tensor(0.0, device=logits.device)
        else:
            self._recon_loss = torch.tensor(0.0, device=logits.device)

        return logits


# ============================================================================
# CINNClassifier: Single SDCD+CINN Model (V38Classifier)
# ============================================================================

class CINNClassifier:
    """
    Single SDCD+CINN model with 3-phase training:
      1. Warmup: train CINN only, DAG frozen
      2. Joint:  train DAG + CINN jointly, DAG penalty annealed
      3. Finetune: train CINN only, DAG frozen, low LR

    This is the building block for CausalBoostClassifier.
    """

    def __init__(self, hidden_dim=32, emb_dim=8,
                 warmup_epochs=20, joint_epochs=100, finetune_epochs=30,
                 lr=0.001, w_lr=0.005, batch_size=32, patience=25,
                 lambda_dag=1.0, lambda_sparse=0.05, dag_gamma_max=50.0,
                 label_smoothing=0.0, edge_dropout=0.0, n_rounds=1,
                 proj_type=None, proj_k=64,
                 dag_type='standard', n_latent=0, n_groups=32,
                 recon_loss=False, lambda_block=0.1, lambda_recon=0.1,
                 pca_components=None, block_mask=None,
                 acyclicity_type='spectral',
                 tau_vcuda=0.3, tau_perm=1.0, tau_edge=0.5, restore_from='all'):

        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.warmup_epochs = warmup_epochs
        self.joint_epochs = joint_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.w_lr = w_lr
        self.batch_size = batch_size
        if restore_from not in ('all', 'finetune'):
            raise ValueError(f'restore_from must be all or finetune, not {restore_from!r}')
        self.restore_from = restore_from
        self.patience = patience
        self.lambda_dag = lambda_dag
        self.lambda_sparse = lambda_sparse
        self.dag_gamma_max = dag_gamma_max
        self.label_smoothing = label_smoothing
        self.edge_dropout = edge_dropout
        self.n_rounds = n_rounds
        self.proj_type = proj_type
        self.proj_k = proj_k
        self.dag_type = dag_type
        self.n_latent = n_latent
        self.n_groups = n_groups
        self.recon_loss = recon_loss
        self.lambda_block = lambda_block
        self.lambda_recon = lambda_recon
        self.pca_components = pca_components
        self.block_mask = block_mask
        self.acyclicity_type = acyclicity_type
        self.tau_vcuda = tau_vcuda
        self.tau_perm = tau_perm
        self.tau_edge = tau_edge

        self.model = None
        self.preprocessor = None
        self.num_scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_final = None

    def _get_dag_params(self):
        if self.dag_type == 'factored':
            if self.acyclicity_type in ('spectral', 'dagma'):
                return {'A_raw', 'S_raw'}
            elif self.acyclicity_type == 'vcuda':
                return {'A_raw', 'priority', 'edge_logits'}
            elif self.acyclicity_type == 'dpdag':
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
            n_rounds=self.n_rounds, proj_type=self.proj_type,
            proj_k=self.proj_k, dag_type=self.dag_type,
            n_latent=self.n_latent, n_groups=self.n_groups,
            recon_loss=self.recon_loss, lambda_block=self.lambda_block,
            lambda_recon=self.lambda_recon,
            acyclicity_type=self.acyclicity_type,
            tau_vcuda=self.tau_vcuda, tau_perm=self.tau_perm,
            tau_edge=self.tau_edge,
        ).to(self.device)

        # Initialize PCA weights if applicable
        if (self.pca_components is not None
                and self.proj_type in ('pca_init', 'pca_fixed')):
            with torch.no_grad():
                pca_t = torch.tensor(self.pca_components, dtype=torch.float32)
                self.model.projection.proj.weight.copy_(pca_t.T)
            if self.proj_type == 'pca_fixed':
                self.model.projection.proj.weight.requires_grad_(False)

        if self.block_mask is not None and self.dag_type == 'block':
            self.model.block_mask.copy_(
                torch.tensor(self.block_mask, dtype=torch.float32).to(self.device)
            )

        val_crit = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0
        )
        d = self.model.n_features
        lambda_sparse = self.lambda_sparse + 0.001 * d

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        total = self.warmup_epochs + self.joint_epochs + self.finetune_epochs

        dag_param_names = self._get_dag_params()
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
                    / (2.0 ** 8 - 1)
                )
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
                        loss = loss + self.lambda_block * self.model.block_penalty()
                    if self.recon_loss:
                        loss = loss + self.lambda_recon * self.model._recon_loss

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

            # restore_from='finetune': checkpoint only once the DAG is fully
            # trained. With 'all' (paper runs) the best epoch is tracked from
            # the start, so early stopping can restore a warmup checkpoint
            # whose DAG is still at its random init (35% of CC18 tasks).
            # With 'finetune' and finetune_epochs=0 the final weights are kept.
            if self.restore_from == 'finetune' and phase != 'finetune':
                continue
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
        with torch.no_grad():
            self.h_final = self.model.dag_loss().item()
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
# CausalBoostClassifier: Boosted Ensemble (V38BoostClassifier)
# ============================================================================

class CausalBoostClassifier:
    """
    CausalBoost: boosted ensemble of SDCD+CINN models with adaptive
    sample reweighting and post-hoc temperature calibration.

    Training:
      1. Train M=5 CINNClassifier models sequentially
      2. After each model, accumulate logits with shrinkage
      3. Reweight samples: upweight misclassified, downweight correct
      4. After all models, learn temperature T via LBFGS

    Prediction:
      logits = sum(shrinkage * model_logits) / T
    """

    def __init__(self, n_models=5, shrinkage=0.1, base_alpha=2.0,
                 weight_clamp=(0.5, 3.0), edge_dropout=0.2,
                 label_smoothing=0.1, proj_type=None, proj_k=64,
                 dag_type='standard', n_latent=0, n_groups=32,
                 recon_loss=False, lambda_block=0.1, lambda_recon=0.1,
                 calibrate=True, acyclicity_type='spectral',
                 tau_vcuda=0.3, tau_perm=1.0, tau_edge=0.5, restore_from='all', random_state=42):
        self.random_state = random_state
        self.n_models = n_models
        self.shrinkage = shrinkage
        self.base_alpha = base_alpha
        self.weight_clamp = weight_clamp
        self.edge_dropout = edge_dropout
        self.label_smoothing = label_smoothing
        self.proj_type = proj_type
        self.proj_k = proj_k
        self.dag_type = dag_type
        self.n_latent = n_latent
        self.n_groups = n_groups
        self.recon_loss = recon_loss
        self.lambda_block = lambda_block
        self.lambda_recon = lambda_recon
        self.calibrate = calibrate
        self.acyclicity_type = acyclicity_type
        self.tau_vcuda = tau_vcuda
        self.tau_perm = tau_perm
        self.tau_edge = tau_edge
        self.restore_from = restore_from
        self.models = []
        self.n_classes = None
        self.temperature = 1.0

    def _compute_pca_full(self, X_num, d_total, proj_k):
        if X_num.shape[1] == 0:
            return None
        d_num = X_num.shape[1]
        k = min(proj_k, d_num, X_num.shape[0])
        if k <= 0:
            return None
        pca = PCA(n_components=k)
        pca.fit(X_num)
        components = np.zeros((d_total, k), dtype=np.float32)
        components[:d_num, :] = pca.components_.T
        return components

    def _compute_block_mask(self, X_num, d_total):
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

    def fit(self, X, y, cat_indices=None, verbose=False):
        cat_indices = cat_indices or []
        self.models = []
        self.n_classes = len(np.unique(y))
        n, d = X.shape
        boost_alpha = compute_adaptive_alpha(n, d, self.n_classes,
                                             self.base_alpha)

        prep = CategoricalPreprocessor()
        X_num_raw, _ = prep.fit_transform(X, cat_indices)
        scaler = StandardScaler()
        X_num_scaled = (scaler.fit_transform(X_num_raw).astype(np.float32)
                        if X_num_raw.shape[1] > 0 else X_num_raw)
        d_total = prep.n_features

        pca_components = None
        if self.proj_type in ('pca_init', 'pca_fixed', 'pca_recon'):
            pca_components = self._compute_pca_full(X_num_scaled, d_total,
                                                    self.proj_k)
        block_mask = None
        if self.dag_type == 'block':
            block_mask = self._compute_block_mask(X_num_scaled, d_total)

        actual_proj_k = self.proj_k
        if pca_components is not None:
            actual_proj_k = pca_components.shape[1]

        base_seed = int(self.random_state) % 100000  # fixed seed; model m uses base_seed + 17m
        acc_logits = np.zeros((len(X), self.n_classes), dtype=np.float32)
        sw = None

        for i in range(self.n_models):
            seed = base_seed + i * 17
            np.random.seed(seed)
            torch.manual_seed(seed)

            clf = CINNClassifier(
                edge_dropout=self.edge_dropout,
                label_smoothing=self.label_smoothing,
                proj_type=self.proj_type, proj_k=actual_proj_k,
                dag_type=self.dag_type, n_latent=self.n_latent,
                n_groups=self.n_groups, recon_loss=self.recon_loss,
                lambda_block=self.lambda_block,
                lambda_recon=self.lambda_recon,
                pca_components=pca_components, block_mask=block_mask,
                acyclicity_type=self.acyclicity_type,
                tau_vcuda=self.tau_vcuda, tau_perm=self.tau_perm,
                tau_edge=self.tau_edge, restore_from=self.restore_from,
            )
            clf.fit(X, y, cat_indices=cat_indices, sample_weights=sw,
                    verbose=False)
            self.models.append(clf)

            logits = clf.predict_logits(X)
            acc_logits += self.shrinkage * logits

            exp_a = np.exp(acc_logits - acc_logits.max(axis=1, keepdims=True))
            proba = exp_a / exp_a.sum(axis=1, keepdims=True)
            p_c = proba[np.arange(len(y)), y]
            sw = 1.0 + boost_alpha * (1.0 - p_c)
            sw = np.clip(sw, self.weight_clamp[0], self.weight_clamp[1])
            sw = sw / sw.mean()

        if self.calibrate:
            self.temperature = learn_temperature(acc_logits, y)
        return self

    def predict_logits(self, X):
        acc = np.zeros((len(X), self.n_classes), dtype=np.float32)
        for m in self.models:
            acc += self.shrinkage * m.predict_logits(X)
        return acc / self.temperature

    def predict(self, X):
        return np.argmax(self.predict_logits(X), axis=1)


# ============================================================================
# CASTLE+CINN Pipeline (v2 baseline)
# ============================================================================

class _CASTLEv2Net(nn.Module):
    """CASTLE DAG learning via Augmented Lagrangian (NOTEARS-style)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 n_classes: int = 2, lambda_sparse: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.lambda_sparse = lambda_sparse
        self.A = nn.Parameter(torch.randn(input_dim, input_dim) * 0.3 - 0.85)
        self.reconstructors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(input_dim)
        ])
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def get_adjacency(self) -> torch.Tensor:
        A = torch.sigmoid(self.A)
        return A * (1 - torch.eye(self.input_dim, device=A.device))

    def dag_constraint(self) -> torch.Tensor:
        A = self.get_adjacency()
        d = self.input_dim
        A2 = A * A
        M = torch.eye(d, device=A.device) + A2 / d
        return torch.trace(torch.matrix_power(M, d)) - d

    def reconstruction_loss(self, X: torch.Tensor) -> torch.Tensor:
        A = self.get_adjacency()
        total_loss = 0.0
        for i in range(self.input_dim):
            pw = A[i, :]
            masked = X * pw.unsqueeze(0)
            x_recon = self.reconstructors[i](masked).squeeze(-1)
            if pw.sum() > 0.01:
                total_loss = total_loss + ((X[:, i] - x_recon) ** 2).mean()
        return total_loss / self.input_dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.predictor(X)

    def compute_losses(self, X, y, logits, alpha, rho):
        pred_loss = nn.CrossEntropyLoss()(logits, y)
        recon_loss = self.reconstruction_loss(X)
        sparse_loss = self.get_adjacency().abs().mean()
        h = self.dag_constraint()
        dag_pen = alpha * h + (rho / 2.0) * h * h
        total = pred_loss + recon_loss + self.lambda_sparse * sparse_loss + dag_pen
        return total, {'pred': pred_loss.item(), 'h(A)': h.item()}


class _OriginalCINNNet(nn.Module):
    """Original CINN: processes nodes in topological order of a fixed DAG."""

    def __init__(self, adjacency: np.ndarray, hidden_dim: int = 32,
                 n_classes: int = 2):
        super().__init__()
        self.n_nodes = adjacency.shape[0] - 1
        self.hidden_dim = hidden_dim
        self.register_buffer(
            'adjacency', torch.tensor(adjacency, dtype=torch.float32))
        in_degree = adjacency[:-1, :-1].sum(axis=0)
        self.topo_order = np.argsort(in_degree)
        self.node_embedders = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(self.n_nodes)])
        self.node_updaters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(self.n_nodes)])
        self.parent_aggregators = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.n_nodes)])
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * self.n_nodes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_classes))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        device = X.device
        adj = self.adjacency
        states = [self.node_embedders[i](X[:, i:i+1])
                  for i in range(self.n_nodes)]
        for idx in self.topo_order:
            pw = adj[:self.n_nodes, idx]
            if pw.sum() > 0:
                S = torch.stack(states, dim=1)
                agg = (S * pw.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
                agg = self.parent_aggregators[idx](agg)
            else:
                agg = torch.zeros(B, self.hidden_dim, device=device)
            states[idx] = self.node_updaters[idx](
                torch.cat([states[idx], agg], dim=1))
        return self.predictor(torch.cat(states, dim=1))


class CASTLEPipelineClassifier:
    """
    Two-stage CASTLE+CINN pipeline (v2 baseline for Dim 1).
      Stage 1: CASTLE learns DAG via augmented Lagrangian
      Stage 2: CINN uses learned DAG structure for prediction

    Input must be one-hot encoded (no native categorical support).
    """

    def __init__(self, castle_inner_epochs=50, castle_outer_iters=10,
                 cinn_epochs=200, adaptive_threshold=True,
                 castle_verbose=False):
        self.castle_inner_epochs = castle_inner_epochs
        self.castle_outer_iters = castle_outer_iters
        self.cinn_epochs = cinn_epochs
        self.adaptive_threshold = adaptive_threshold
        self.castle_verbose = castle_verbose
        self._castle_model = None
        self._cinn_model = None
        self._castle_scaler = StandardScaler()
        self._cinn_scaler = StandardScaler()
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y):
        d = X.shape[1]

        # Stage 1: Learn DAG with CASTLE
        X_s = self._castle_scaler.fit_transform(X).astype(np.float32)
        n_classes = len(np.unique(y))
        X_t = torch.tensor(X_s, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        n_val = max(int(0.15 * len(X)), 20)
        idx = np.random.permutation(len(X))
        tr, va = idx[n_val:], idx[:n_val]

        loader = DataLoader(TensorDataset(X_t[tr], y_t[tr]),
                            batch_size=32, shuffle=True)
        self._castle_model = _CASTLEv2Net(
            d, 64, n_classes, 0.1).to(self._device)
        opt = optim.Adam(self._castle_model.parameters(), lr=0.001)

        alpha, rho, prev_h = 0.0, 1.0, float('inf')
        best_state, best_vl = None, float('inf')

        for outer in range(self.castle_outer_iters):
            pc = 0
            for epoch in range(self.castle_inner_epochs):
                self._castle_model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(self._device), yb.to(self._device)
                    opt.zero_grad()
                    logits = self._castle_model(xb)
                    loss, _ = self._castle_model.compute_losses(
                        xb, yb, logits, alpha, rho)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self._castle_model.parameters(), 5.0)
                    opt.step()
                self._castle_model.eval()
                with torch.no_grad():
                    vl_logits = self._castle_model(X_t[va].to(self._device))
                    vl, _ = self._castle_model.compute_losses(
                        X_t[va].to(self._device), y_t[va].to(self._device),
                        vl_logits, alpha, rho)
                    vl = vl.item()
                if vl < best_vl:
                    best_vl = vl
                    best_state = {k: v.cpu().clone()
                                  for k, v in self._castle_model.state_dict().items()}
                    pc = 0
                else:
                    pc += 1
                    if pc >= 15:
                        break
            with torch.no_grad():
                h = self._castle_model.dag_constraint().item()
            if h < 1e-4:
                break
            alpha += rho * h
            if h > 0.25 * prev_h:
                rho = min(rho * 10, 1e8)
            prev_h = h

        if best_state:
            self._castle_model.load_state_dict(best_state)

        # Get DAG and threshold
        with torch.no_grad():
            dag = self._castle_model.get_adjacency().cpu().numpy()

        if self.adaptive_threshold:
            mask = ~np.eye(d, dtype=bool)
            weights = dag[mask]
            best_thresh, best_var = 0.3, 0.0
            if weights.max() - weights.min() > 0.05:
                for t in np.linspace(weights.min() + 0.01,
                                     weights.max() - 0.01, 50):
                    w0, w1 = weights[weights <= t], weights[weights > t]
                    if len(w0) == 0 or len(w1) == 0:
                        continue
                    p0 = len(w0) / len(weights)
                    var = p0 * (1 - p0) * (w0.mean() - w1.mean()) ** 2
                    if var > best_var:
                        best_var, best_thresh = var, t
            threshold = best_thresh
        else:
            threshold = 0.3

        full_adj = np.zeros((d + 1, d + 1))
        full_adj[:d, :d] = (dag > threshold).astype(float)
        full_adj[:d, d] = 1

        # Stage 2: Train CINN on learned DAG
        X_c = self._cinn_scaler.fit_transform(X).astype(np.float32)
        X_ct = torch.tensor(X_c, dtype=torch.float32)

        loader2 = DataLoader(TensorDataset(X_ct[tr], y_t[tr]),
                             batch_size=32, shuffle=True)
        self._cinn_model = _OriginalCINNNet(
            full_adj, 32, n_classes).to(self._device)
        opt2 = optim.Adam(self._cinn_model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()

        best_state2, best_vl2, pc2 = None, float('inf'), 0
        for epoch in range(self.cinn_epochs):
            self._cinn_model.train()
            for xb, yb in loader2:
                xb, yb = xb.to(self._device), yb.to(self._device)
                opt2.zero_grad()
                loss = crit(self._cinn_model(xb), yb)
                loss.backward()
                opt2.step()
            self._cinn_model.eval()
            with torch.no_grad():
                vl2 = crit(self._cinn_model(X_ct[va].to(self._device)),
                           y_t[va].to(self._device)).item()
            if vl2 < best_vl2:
                best_vl2 = vl2
                best_state2 = {k: v.cpu().clone()
                               for k, v in self._cinn_model.state_dict().items()}
                pc2 = 0
            else:
                pc2 += 1
                if pc2 >= 20:
                    break
        if best_state2:
            self._cinn_model.load_state_dict(best_state2)
        return self

    def predict(self, X):
        X_s = self._cinn_scaler.transform(X).astype(np.float32)
        X_t = torch.tensor(X_s, dtype=torch.float32).to(self._device)
        self._cinn_model.eval()
        with torch.no_grad():
            return self._cinn_model(X_t).argmax(dim=1).cpu().numpy()


# ============================================================================
# GOLEM+SoftCINN Pipeline (v15 baseline)
# ============================================================================

class _GOLEMNet(nn.Module):
    """GOLEM: single-loop DAG learning via BIC-like score."""

    def __init__(self, d: int, lambda_dag=5.0, lambda_sparse=0.1):
        super().__init__()
        self.d = d
        self.lambda_dag = lambda_dag
        self.lambda_sparse = lambda_sparse
        self.B = nn.Parameter(torch.zeros(d, d))
        self.log_sigma = nn.Parameter(torch.zeros(d))

    def forward(self, X):
        return X @ self.B

    def compute_loss(self, X):
        n, d = X.shape
        sigma = torch.exp(self.log_sigma)
        residual = X - self.forward(X)
        nll = n * self.log_sigma + 0.5 * (residual ** 2).sum(dim=0) / (sigma ** 2)
        nll = nll.sum()
        W_sq = self.B * self.B
        expm = torch.matrix_exp(W_sq)
        h = torch.trace(expm) - d
        l1 = torch.abs(self.B).sum()
        loss = nll / n + self.lambda_dag * h + self.lambda_sparse * l1
        return loss, {'h(B)': h.item(), 'nll': nll.item() / n}


class _SoftCINNNet(nn.Module):
    """SoftCINN: CINN with continuous edge weights (no hard threshold)."""

    def __init__(self, soft_adjacency: np.ndarray, hidden_dim: int = 32,
                 n_classes: int = 2):
        super().__init__()
        self.n_nodes = soft_adjacency.shape[0] - 1
        self.hidden_dim = hidden_dim
        self.register_buffer(
            'adjacency', torch.tensor(soft_adjacency, dtype=torch.float32))
        in_degree = soft_adjacency[:-1, :-1].sum(axis=0)
        self.topo_order = np.argsort(in_degree)
        self.node_embedders = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(self.n_nodes)])
        self.node_updaters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(self.n_nodes)])
        self.parent_aggregators = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.n_nodes)])
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * self.n_nodes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_classes))

    def forward(self, X):
        B = X.shape[0]
        device = X.device
        adj = self.adjacency
        states = [self.node_embedders[i](X[:, i:i+1])
                  for i in range(self.n_nodes)]
        for idx in self.topo_order:
            pw = adj[:self.n_nodes, idx]
            if pw.sum() > 1e-3:
                S = torch.stack(states, dim=1)
                agg = (S * pw.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
                agg = self.parent_aggregators[idx](agg)
            else:
                agg = torch.zeros(B, self.hidden_dim, device=device)
            states[idx] = self.node_updaters[idx](
                torch.cat([states[idx], agg], dim=1))
        return self.predictor(torch.cat(states, dim=1))


def _correlation_dag(X, threshold=0.1):
    """Build a DAG from correlations (fallback when GOLEM doesn't converge)."""
    d = X.shape[1]
    corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(corr, 0)
    order = np.argsort(corr.sum(axis=0))
    rank = np.zeros(d, dtype=int)
    for i, node in enumerate(order):
        rank[node] = i
    A = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j and corr[i, j] > threshold and rank[j] < rank[i]:
                A[i, j] = corr[i, j]
    return A


class GOLEMPipelineClassifier:
    """
    Two-stage GOLEM+SoftCINN pipeline (v15 baseline for Dim 1).
      Stage 1: GOLEM learns linear DAG (single-loop, no aug. Lagrangian)
      Stage 2: SoftCINN uses soft adjacency for prediction

    Falls back to correlation-based DAG if GOLEM doesn't converge.
    Input must be one-hot encoded (no native categorical support).
    """

    def __init__(self, golem_epochs=200, cinn_epochs=200,
                 h_threshold=0.1, corr_threshold=0.15,
                 lambda_dag=5.0, lambda_sparse=0.5):
        self.golem_epochs = golem_epochs
        self.cinn_epochs = cinn_epochs
        self.h_threshold = h_threshold
        self.corr_threshold = corr_threshold
        self.lambda_dag = lambda_dag
        self.lambda_sparse = lambda_sparse
        self._cinn_model = None
        self._cinn_scaler = StandardScaler()
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y):
        d = X.shape[1]
        n_classes = len(np.unique(y))

        # Stage 1: GOLEM DAG learning
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        X_t = torch.tensor(X_s, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        lambda_sparse = self.lambda_sparse + 0.02 * d
        golem = _GOLEMNet(d, self.lambda_dag, lambda_sparse)

        pred_head = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, n_classes))
        params = list(golem.parameters()) + list(pred_head.parameters())
        opt = optim.Adam(params, lr=0.01)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.golem_epochs)
        ce = nn.CrossEntropyLoss()

        best_loss, best_B, pc = float('inf'), None, 0
        for epoch in range(self.golem_epochs):
            opt.zero_grad()
            gloss, metrics = golem.compute_loss(X_t)
            ploss = ce(pred_head(X_t), y_t)
            total = gloss + ploss
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                golem.B.fill_diagonal_(0)
            if metrics['h(B)'] < 1e-8:
                break
            if total.item() < best_loss:
                best_loss = total.item()
                best_B = golem.B.detach().clone()
                pc = 0
            else:
                pc += 1
                if pc > 30:
                    break

        if best_B is not None:
            golem.B.data = best_B

        h_final = metrics['h(B)']
        dag = np.abs(golem.B.detach().cpu().numpy())

        # Fallback if not converged
        if h_final > self.h_threshold:
            dag = _correlation_dag(X, self.corr_threshold)

        # Build soft adjacency (d+1 x d+1, last node = Y)
        full_adj = np.zeros((d + 1, d + 1))
        full_adj[:d, :d] = dag
        full_adj[:d, d] = 1.0

        # Stage 2: SoftCINN
        X_c = self._cinn_scaler.fit_transform(X).astype(np.float32)
        X_ct = torch.tensor(X_c, dtype=torch.float32)

        n_val = max(int(0.15 * len(X)), 20)
        idx = np.random.permutation(len(X))
        tr, va = idx[n_val:], idx[:n_val]

        loader = DataLoader(TensorDataset(X_ct[tr], y_t[tr]),
                            batch_size=32, shuffle=True)
        self._cinn_model = _SoftCINNNet(
            full_adj, 32, n_classes).to(self._device)
        opt2 = optim.Adam(self._cinn_model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()

        best_state, best_vl, pc2 = None, float('inf'), 0
        for epoch in range(self.cinn_epochs):
            self._cinn_model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                opt2.zero_grad()
                loss = crit(self._cinn_model(xb), yb)
                loss.backward()
                opt2.step()
            self._cinn_model.eval()
            with torch.no_grad():
                vl = crit(self._cinn_model(X_ct[va].to(self._device)),
                          y_t[va].to(self._device)).item()
            if vl < best_vl:
                best_vl = vl
                best_state = {k: v.cpu().clone()
                              for k, v in self._cinn_model.state_dict().items()}
                pc2 = 0
            else:
                pc2 += 1
                if pc2 >= 20:
                    break
        if best_state:
            self._cinn_model.load_state_dict(best_state)
        return self

    def predict(self, X):
        X_s = self._cinn_scaler.transform(X).astype(np.float32)
        X_t = torch.tensor(X_s, dtype=torch.float32).to(self._device)
        self._cinn_model.eval()
        with torch.no_grad():
            return self._cinn_model(X_t).argmax(dim=1).cpu().numpy()

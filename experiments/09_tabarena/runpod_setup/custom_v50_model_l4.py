"""V50 Missingness Indicator CINN model wrapped as AutoGluon AbstractModel for TabArena.

Self-contained: inlines needed utilities from pipe_cinn_v28 and benchmark_v50.
Supports classification (binary/multiclass) and regression.

Lessons applied from RealMLP / TabM:
- Adaptive batch size based on n_train
- QuantileTransformer for numerical preprocessing
- Y normalization for regression
- Early stopping with patience across all training phases
- Label smoothing disabled for log_loss/AUC metrics
- SimpleImputer with add_indicator for missingness (via AG preprocess)
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from autogluon.core.models import AbstractModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# Utilities
# ============================================================================

def get_auto_batch_size(n_train: int) -> int:
    """Adaptive batch size, following TabM/RealMLP conventions."""
    if n_train < 2_800:
        return 32
    if n_train < 4_500:
        return 64
    if n_train < 6_400:
        return 128
    if n_train < 32_000:
        return 256
    if n_train < 108_000:
        return 512
    return 1024


def learn_temperature(logits: np.ndarray, labels: np.ndarray,
                      lr: float = 0.01, max_iter: int = 200) -> float:
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


def compute_adaptive_alpha(n: int, d: int, n_classes: int,
                           base_alpha: float = 2.0) -> float:
    d_scale = np.clip(np.log2(max(d, 2)) / np.log2(40), 0.3, 1.2)
    class_scale = 1.0 / np.sqrt(max(n_classes / 2, 1.0))
    n_scale = np.clip(np.sqrt(n / 500), 0.5, 1.5)
    return base_alpha * d_scale * class_scale * n_scale


class RTDLQuantileTransformer:
    """QuantileTransformer with noise, adapted from TabM/pytabkit."""

    def __init__(self, noise=1e-5, random_state=None, n_quantiles=1000):
        self.noise = noise
        self.random_state = random_state
        self.n_quantiles = n_quantiles
        self._qt = None

    def fit(self, X):
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)
        self._qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            subsample=1_000_000_000,
            random_state=self.random_state,
        )
        X_noisy = X + np.random.default_rng(self.random_state).normal(0, 1e-5, X.shape).astype(X.dtype) if self.noise > 0 else X
        self._qt.fit(X_noisy)
        return self

    def transform(self, X):
        return self._qt.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================================
# V50 Model (supports classification and regression)
# ============================================================================

class V50Model(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: List[int],
                 n_classes: int, hidden_dim: int = 32, emb_dim: int = 8,
                 edge_dropout: float = 0.0, n_rounds: int = 1,
                 n_groups: int = 32,
                 tau_a: float = 1.0, tau_vcuda: float = 0.3, tau_edge: float = 0.5,
                 variational: bool = True, n_latent: int = 16,
                 is_regression: bool = False):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        self.n_features = n_num + self.n_cat
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.edge_dropout = edge_dropout
        self.n_rounds = n_rounds
        self.n_groups = n_groups
        self.tau_a = tau_a
        self.tau_vcuda = tau_vcuda
        self.tau_edge = tau_edge
        self.variational = variational
        self.n_latent = n_latent
        self.is_regression = is_regression
        self.force_gumbel = False

        d = self.n_features
        d_dag = d + n_latent
        self.d_dag = d_dag
        k = n_groups

        self.num_embedders = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(n_num)
        ])
        self.cat_embeddings = nn.ModuleList()
        self.cat_projectors = nn.ModuleList()
        for card in cat_cardinalities:
            actual_emb_dim = min(emb_dim, max(2, int(np.sqrt(card))))
            self.cat_embeddings.append(nn.Embedding(card, actual_emb_dim))
            self.cat_projectors.append(nn.Linear(actual_emb_dim, hidden_dim))

        self.latent_embeds = nn.Parameter(torch.randn(1, n_latent, hidden_dim) * 0.02)
        self.A_raw = nn.Parameter(torch.randn(d_dag, k) * 0.01)
        self.priority = nn.Parameter(torch.randn(k))
        self.edge_logits = nn.Parameter(torch.zeros(k, k))
        nn.init.uniform_(self.edge_logits, -1, 1)
        with torch.no_grad():
            self.edge_logits.diagonal().fill_(-300)
        self.register_buffer('no_self_loop_k', 1.0 - torch.eye(k))

        if variational:
            self.edge_logits_logvar = nn.Parameter(torch.zeros(k, k) - 2.0)
            self.priority_logvar = nn.Parameter(torch.zeros(k) - 2.0)

        self.W_raw = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.register_buffer('no_self_loop', 1.0 - torch.eye(d_dag))

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

        pred_input = hidden_dim * d_dag
        output_dim = 1 if is_regression else n_classes
        self.predictor = nn.Sequential(
            nn.Linear(pred_input, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def get_A(self):
        if self.training or self.force_gumbel:
            return F.gumbel_softmax(self.A_raw, tau=self.tau_a, hard=True, dim=1)
        idx = self.A_raw.argmax(dim=1)
        return F.one_hot(idx, self.n_groups).float()

    def get_S(self):
        if self.variational:
            p_std = (0.5 * self.priority_logvar).exp()
            priority = self.priority + p_std * torch.randn_like(self.priority)
            e_std = (0.5 * self.edge_logits_logvar).exp()
            edge_logits = self.edge_logits + e_std * torch.randn_like(self.edge_logits)
        else:
            priority = self.priority
            edge_logits = self.edge_logits

        grad_p = priority.unsqueeze(0) - priority.unsqueeze(1)
        direction = torch.sigmoid(grad_p / self.tau_vcuda)

        if self.training or self.force_gumbel:
            raw_logits = torch.stack([edge_logits, -edge_logits])
            edges = F.gumbel_softmax(raw_logits, tau=self.tau_edge, hard=True, dim=0)[0]
        else:
            edges = (torch.sigmoid(edge_logits) > 0.5).float()
            edges.diagonal().fill_(0)

        return edges * direction * self.no_self_loop_k

    def get_W(self):
        A = self.get_A()
        S = self.get_S()
        return (A @ S @ A.T) * self.no_self_loop

    def dag_loss(self):
        return torch.tensor(0.0, device=self.A_raw.device)

    def sparsity_loss(self):
        return torch.sigmoid(self.edge_logits).mean()

    def kl_loss(self):
        if not self.variational:
            return torch.tensor(0.0, device=self.A_raw.device)
        kl = 0.0
        for mu, logvar in [(self.priority, self.priority_logvar),
                           (self.edge_logits, self.edge_logits_logvar)]:
            kl = kl + 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum()
        return kl

    def forward(self, X_num: torch.Tensor, X_cat: torch.Tensor):
        B = X_num.shape[0] if X_num.shape[1] > 0 else X_cat.shape[0]
        embeddings = []
        for i in range(self.n_num):
            embeddings.append(self.num_embedders[i](X_num[:, i:i + 1]))
        for i in range(self.n_cat):
            emb_i = self.cat_embeddings[i](X_cat[:, i])
            embeddings.append(self.cat_projectors[i](emb_i))
        H = torch.stack(embeddings, dim=1)
        H = torch.cat([H, self.latent_embeds.expand(B, -1, -1)], dim=1)

        W = self.get_W()
        if self.training and self.edge_dropout > 0:
            W = W * (torch.rand_like(W) > self.edge_dropout).float()
        for r in range(self.n_rounds):
            parent_msgs = self.parent_aggs[r](torch.einsum('ji,bjh->bih', W, H))
            H = self.updaters[r](torch.cat([H, parent_msgs], dim=-1))

        logits = self.predictor(H.reshape(B, -1))
        return logits.squeeze(-1) if self.is_regression else logits


def predict_mc_v50(model, X_num, X_cat, n_samples=8):
    model.eval()
    model.force_gumbel = True
    all_outputs = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(X_num, X_cat)
            if model.is_regression:
                all_outputs.append(out)
            else:
                all_outputs.append(F.softmax(out, dim=-1))
    model.force_gumbel = False
    return torch.stack(all_outputs).mean(0)


# ============================================================================
# V50 Single Model (classification + regression)
# ============================================================================

class V50SingleModel:
    """Trains a single V50 CINN model. Uses external val set for early stopping."""

    def __init__(self, hidden_dim=32, emb_dim=8,
                 warmup_epochs=20, joint_epochs=100, finetune_epochs=30,
                 lr=0.001, w_lr=0.005, batch_size="auto", patience=16,
                 lambda_dag=1.0, lambda_sparse=0.05, dag_gamma_max=50.0,
                 label_smoothing=0.0, edge_dropout=0.0, n_rounds=1,
                 n_groups=32,
                 tau_a_init=1.0, tau_a_min=0.1,
                 tau_vcuda=0.3, tau_edge=0.5,
                 mc_samples=8,
                 variational=True, beta_kl=0.001,
                 n_latent=16,
                 is_regression=False):
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
        self.n_groups = n_groups
        self.tau_a_init = tau_a_init
        self.tau_a_min = tau_a_min
        self.tau_vcuda = tau_vcuda
        self.tau_edge = tau_edge
        self.mc_samples = mc_samples
        self.variational = variational
        self.beta_kl = beta_kl
        self.n_latent = n_latent
        self.is_regression = is_regression

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_dag_params(self):
        names = {'A_raw', 'priority', 'edge_logits'}
        if self.variational:
            names.update({'edge_logits_logvar', 'priority_logvar'})
        return names

    def fit(self, X_num_t, X_cat_t, y_t,
            X_num_val_t, X_cat_val_t, y_val_t,
            n_classes, cat_cards,
            sample_weights=None):
        """Fit on pre-processed tensors. Val set used for early stopping."""

        n_train = X_num_t.shape[0]
        batch_size = get_auto_batch_size(n_train) if self.batch_size == "auto" else self.batch_size

        sw_t = (torch.tensor(sample_weights, dtype=torch.float32)
                if sample_weights is not None else None)

        if sw_t is not None:
            ds = TensorDataset(X_num_t, X_cat_t, y_t, sw_t)
        else:
            ds = TensorDataset(X_num_t, X_cat_t, y_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        self.model = V50Model(
            n_num=X_num_t.shape[1], cat_cardinalities=cat_cards,
            n_classes=n_classes, hidden_dim=self.hidden_dim, emb_dim=self.emb_dim,
            edge_dropout=self.edge_dropout, n_rounds=self.n_rounds,
            n_groups=self.n_groups, tau_a=self.tau_a_init,
            tau_vcuda=self.tau_vcuda, tau_edge=self.tau_edge,
            variational=self.variational, n_latent=self.n_latent,
            is_regression=self.is_regression,
        ).to(self.device)

        if self.is_regression:
            val_crit = nn.MSELoss()
        else:
            val_crit = nn.CrossEntropyLoss()

        d = self.model.n_features
        lambda_sparse = self.lambda_sparse + 0.001 * d

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        total = self.warmup_epochs + self.joint_epochs + self.finetune_epochs

        dag_param_names = self._get_dag_params()
        dag_params, cinn_params = [], []
        for n, p in self.model.named_parameters():
            (dag_params if n in dag_param_names else cinn_params).append(p)
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
        opt_finetune = optim.AdamW(cinn_params, lr=self.lr * 0.1, weight_decay=0.01)

        use_w = sw_t is not None

        for epoch in range(total):
            if epoch < self.warmup_epochs:
                phase = 'warmup'
                for p in dag_params:
                    p.requires_grad_(False)
                opt, gamma = opt_warmup, 0.0
            elif epoch < self.warmup_epochs + self.joint_epochs:
                phase = 'joint'
                for p in dag_params:
                    p.requires_grad_(True)
                opt = opt_joint
                jp = (epoch - self.warmup_epochs) / max(self.joint_epochs, 1)
                gamma = self.lambda_dag * min(
                    self.dag_gamma_max,
                    self.dag_gamma_max * (2.0 ** (jp * 8) - 1) / (2.0 ** 8 - 1)
                )
            else:
                phase = 'finetune'
                for p in dag_params:
                    p.requires_grad_(False)
                opt, gamma = opt_finetune, 0.0

            if phase == 'joint':
                jp = (epoch - self.warmup_epochs) / max(self.joint_epochs, 1)
                self.model.tau_a = max(self.tau_a_min,
                                       self.tau_a_init * (1 - jp) + self.tau_a_min * jp)

            self.model.train()
            for batch in loader:
                if use_w:
                    xn, xc, yb, wb = [b.to(self.device) for b in batch]
                else:
                    xn, xc, yb = [b.to(self.device) for b in batch]

                opt.zero_grad()
                output = self.model(xn, xc)

                if self.is_regression:
                    base_loss = F.mse_loss(output, yb, reduction='none') if use_w else F.mse_loss(output, yb)
                else:
                    base_loss = F.cross_entropy(output, yb, reduction='none',
                                                label_smoothing=self.label_smoothing) if use_w else \
                        F.cross_entropy(output, yb, label_smoothing=self.label_smoothing)
                loss = (base_loss * wb).mean() if use_w else base_loss

                if phase == 'joint':
                    dag_l = self.model.dag_loss()
                    if dag_l.item() > 0:
                        loss = loss + gamma * dag_l
                    loss = loss + lambda_sparse * self.model.sparsity_loss()
                    if self.variational:
                        loss = loss + self.beta_kl * self.model.kl_loss()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()

            # Early stopping on val — active in ALL phases
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_num_val_t.to(self.device),
                                     X_cat_val_t.to(self.device))
                vl = val_crit(val_out, y_val_t.to(self.device)).item()

            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def predict_raw(self, X_num_t, X_cat_t):
        X_num_t = X_num_t.to(self.device)
        X_cat_t = X_cat_t.to(self.device)
        if self.mc_samples > 0:
            return predict_mc_v50(self.model, X_num_t, X_cat_t, n_samples=self.mc_samples)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_num_t, X_cat_t)

    def predict_logits(self, X_num_t, X_cat_t):
        out = self.predict_raw(X_num_t, X_cat_t)
        if self.is_regression:
            return out.cpu().numpy()
        return out.log().cpu().numpy()


# ============================================================================
# V50 Boosted Ensemble (classification + regression)
# ============================================================================

class V50BoostEnsemble:
    def __init__(self, n_models=5, shrinkage=0.1, base_alpha=2.0,
                 weight_clamp=(0.5, 3.0), edge_dropout=0.2, label_smoothing=0.1,
                 n_groups=32, n_rounds=1,
                 tau_a_init=1.0, tau_a_min=0.1,
                 tau_vcuda=0.3, tau_edge=0.5,
                 mc_samples=8,
                 variational=True, beta_kl=0.001,
                 n_latent=16,
                 calibrate=True,
                 is_regression=False,
                 patience=16, random_state=42):
        self.random_state = random_state
        self.n_models = n_models
        self.shrinkage = shrinkage
        self.base_alpha = base_alpha
        self.weight_clamp = weight_clamp
        self.edge_dropout = edge_dropout
        self.label_smoothing = label_smoothing
        self.n_groups = n_groups
        self.n_rounds = n_rounds
        self.tau_a_init = tau_a_init
        self.tau_a_min = tau_a_min
        self.tau_vcuda = tau_vcuda
        self.tau_edge = tau_edge
        self.mc_samples = mc_samples
        self.variational = variational
        self.beta_kl = beta_kl
        self.n_latent = n_latent
        self.calibrate = calibrate
        self.is_regression = is_regression
        self.patience = patience
        self.models = []
        self.n_classes = None
        self.temperature = 1.0

    def fit(self, X_num_t, X_cat_t, y_t, X_num_val_t, X_cat_val_t, y_val_t,
            n_classes, cat_cards):
        import time as _time
        self.models = []
        self.n_classes = n_classes
        n = X_num_t.shape[0]
        d = X_num_t.shape[1] + X_cat_t.shape[1]

        base_seed = int(self.random_state) % 100000  # fixed seed; model m uses base_seed + 17m
        sw = None

        if self.is_regression:
            acc_preds = np.zeros(n, dtype=np.float32)
        else:
            acc_logits = np.zeros((n, n_classes), dtype=np.float32)
            boost_alpha = compute_adaptive_alpha(n, d, n_classes, self.base_alpha)

        for i in range(self.n_models):
            seed = base_seed + i * 17
            np.random.seed(seed)
            torch.manual_seed(seed)

            clf = V50SingleModel(
                edge_dropout=self.edge_dropout,
                label_smoothing=self.label_smoothing if not self.is_regression else 0.0,
                n_groups=self.n_groups, n_rounds=self.n_rounds,
                tau_a_init=self.tau_a_init, tau_a_min=self.tau_a_min,
                tau_vcuda=self.tau_vcuda, tau_edge=self.tau_edge,
                mc_samples=self.mc_samples,
                variational=self.variational, beta_kl=self.beta_kl,
                n_latent=self.n_latent, is_regression=self.is_regression,
                patience=self.patience,
            )
            clf.fit(X_num_t, X_cat_t, y_t,
                    X_num_val_t, X_cat_val_t, y_val_t,
                    n_classes=n_classes, cat_cards=cat_cards,
                    sample_weights=sw)
            self.models.append(clf)

            if self.is_regression:
                preds = clf.predict_raw(X_num_t, X_cat_t).cpu().numpy()
                acc_preds += self.shrinkage * preds
                residuals = np.abs(y_t.numpy() - acc_preds)
                sw = 1.0 + self.base_alpha * (residuals / (residuals.mean() + 1e-8))
                sw = np.clip(sw, self.weight_clamp[0], self.weight_clamp[1])
                sw = (sw / sw.mean()).astype(np.float32)
            else:
                logits = clf.predict_logits(X_num_t, X_cat_t)
                acc_logits += self.shrinkage * logits
                exp_a = np.exp(acc_logits - acc_logits.max(axis=1, keepdims=True))
                proba = exp_a / exp_a.sum(axis=1, keepdims=True)
                p_c = proba[np.arange(n), y_t.numpy()]
                sw = 1.0 + boost_alpha * (1.0 - p_c)
                sw = np.clip(sw, self.weight_clamp[0], self.weight_clamp[1])
                sw = (sw / sw.mean()).astype(np.float32)

        if self.calibrate and not self.is_regression:
            # Calibrate on validation set (not training) to avoid overfitting
            acc_val_logits = np.zeros((X_num_val_t.shape[0], n_classes), dtype=np.float32)
            for m in self.models:
                acc_val_logits += self.shrinkage * m.predict_logits(X_num_val_t, X_cat_val_t)
            self.temperature = learn_temperature(acc_val_logits, y_val_t.numpy())
        return self

    def predict_logits(self, X_num_t, X_cat_t):
        if self.is_regression:
            acc = np.zeros(X_num_t.shape[0], dtype=np.float32)
            for m in self.models:
                acc += self.shrinkage * m.predict_raw(X_num_t, X_cat_t).cpu().numpy()
            return acc
        acc = np.zeros((X_num_t.shape[0], self.n_classes), dtype=np.float32)
        for m in self.models:
            acc += self.shrinkage * m.predict_logits(X_num_t, X_cat_t)
        return acc / self.temperature

    def predict_proba(self, X_num_t, X_cat_t):
        logits = self.predict_logits(X_num_t, X_cat_t)
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_l / exp_l.sum(axis=1, keepdims=True)

    def predict(self, X_num_t, X_cat_t):
        if self.is_regression:
            return self.predict_logits(X_num_t, X_cat_t)
        return np.argmax(self.predict_logits(X_num_t, X_cat_t), axis=1)


# ============================================================================
# AutoGluon AbstractModel Wrapper
# ============================================================================

class CustomV50Model(AbstractModel):
    """V50 Missingness Indicator CINN model for TabArena.

    Follows RealMLP/TabM patterns:
    - Uses X_val/y_val from AG bagging for early stopping
    - Uses SimpleImputer with add_indicator for missingness (via _preprocess)
    - QuantileTransformer for numerical preprocessing
    - Y normalization for regression
    - Label smoothing disabled for log_loss/AUC metrics
    """

    ag_key = "V50_MISS"
    ag_name = "V50MissInd"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None
        self._indicator_columns = None
        self._cat_col_names = None
        self._num_transformer = None
        self._y_mean = None
        self._y_std = None

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        """Impute missing values with median + add binary indicator columns."""
        X = super()._preprocess(X, **kwargs)
        X = X.copy(deep=True)

        if is_train:
            self._features_to_impute = self._feature_metadata.get_features(
                valid_raw_types=["int", "float"]
            )
            self._features_to_keep = self._feature_metadata.get_features(
                invalid_raw_types=["int", "float"]
            )
            if self._features_to_impute:
                self._imputer = SimpleImputer(strategy="median", add_indicator=True)
                self._imputer.fit(X=X[self._features_to_impute])
                self._indicator_columns = [
                    c for c in self._imputer.get_feature_names_out()
                    if c not in self._features_to_impute
                ]

        if self._imputer is not None:
            X_impute = self._imputer.transform(X=X[self._features_to_impute])
            X_impute = pd.DataFrame(
                X_impute, index=X.index,
                columns=self._imputer.get_feature_names_out()
            )
            if self._indicator_columns:
                X_impute[self._indicator_columns] = X_impute[self._indicator_columns].astype("category")
            X = pd.concat([X[self._features_to_keep], X_impute], axis=1)

        return X

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        **kwargs,
    ):
        X = self.preprocess(X, y=y, is_train=True)

        if X_val is None:
            from autogluon.core.utils import generate_train_test_split
            X, X_val, y, y_val = generate_train_test_split(
                X=X, y=y, problem_type=self.problem_type, test_size=0.2, random_state=0,
            )
        else:
            X_val = self.preprocess(X_val)

        is_regression = self.problem_type == "regression"
        params = self._get_model_params()
        n_models = params.get("n_models", 5)

        # Identify cat/num columns
        cat_cols = X.select_dtypes(include="category").columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]
        self._cat_col_names = cat_cols

        # Cat cardinalities — use max across BOTH train and val to avoid OOB embedding
        cat_cards = []
        for c in cat_cols:
            max_code = max(int(X[c].cat.codes.max()), int(X_val[c].cat.codes.max()))
            cat_cards.append(max_code + 2)

        # QuantileTransformer for numerical features (like TabM)
        if num_cols:
            self._num_transformer = RTDLQuantileTransformer(random_state=0)
            X_num = self._num_transformer.fit_transform(
                X[num_cols].values.astype(np.float32)
            ).astype(np.float32)
            X_val_num = self._num_transformer.transform(
                X_val[num_cols].values.astype(np.float32)
            ).astype(np.float32)
            # Replace any remaining NaN from QT
            X_num = np.nan_to_num(X_num, nan=0.0)
            X_val_num = np.nan_to_num(X_val_num, nan=0.0)
        else:
            X_num = np.zeros((len(X), 0), dtype=np.float32)
            X_val_num = np.zeros((len(X_val), 0), dtype=np.float32)

        # Encode categoricals
        X_cat = np.zeros((len(X), len(cat_cols)), dtype=np.int64)
        X_val_cat = np.zeros((len(X_val), len(cat_cols)), dtype=np.int64)
        for i, c in enumerate(cat_cols):
            X_cat[:, i] = X[c].cat.codes.values.clip(min=0)
            X_val_cat[:, i] = X_val[c].cat.codes.values.clip(min=0)

        # Build tensors
        X_num_t = torch.tensor(X_num, dtype=torch.float32)
        X_cat_t = torch.tensor(X_cat, dtype=torch.long)
        X_num_val_t = torch.tensor(X_val_num, dtype=torch.float32)
        X_cat_val_t = torch.tensor(X_val_cat, dtype=torch.long)

        if is_regression:
            y_np = y.values.astype(np.float32)
            y_val_np = y_val.values.astype(np.float32)
            # Y normalization (like TabM)
            self._y_mean = float(np.mean(y_np))
            self._y_std = float(np.std(y_np)) + 1e-8
            y_t = torch.tensor((y_np - self._y_mean) / self._y_std, dtype=torch.float32)
            y_val_t = torch.tensor((y_val_np - self._y_mean) / self._y_std, dtype=torch.float32)
            n_classes = 1
        else:
            # Use LabelEncoder to ensure contiguous 0..C-1 labels across train+val
            from sklearn.preprocessing import LabelEncoder as _LE
            _le = _LE()
            import pandas as pd
            all_labels = pd.concat([y, y_val])
            _le.fit(all_labels)
            y_t = torch.tensor(_le.transform(y), dtype=torch.long)
            y_val_t = torch.tensor(_le.transform(y_val), dtype=torch.long)
            n_classes = len(_le.classes_)

        self.model = V50BoostEnsemble(
            n_models=n_models,
            shrinkage=0.1,
            base_alpha=2.0,
            edge_dropout=0.2,
            label_smoothing=0.1,
            n_groups=32, n_rounds=1,
            tau_a_init=1.0, tau_a_min=0.1,
            tau_vcuda=0.3, tau_edge=0.5,
            mc_samples=8,
            variational=True, beta_kl=0.001,
            n_latent=16,
            calibrate=not is_regression,
            is_regression=is_regression,
            patience=16,
        )
        self.model.fit(X_num_t, X_cat_t, y_t,
                       X_num_val_t, X_cat_val_t, y_val_t,
                       n_classes=n_classes, cat_cards=cat_cards)

    def _to_tensors(self, X: pd.DataFrame):
        cat_cols = self._cat_col_names or []
        num_cols = [c for c in X.columns if c not in cat_cols]

        if num_cols and self._num_transformer is not None:
            X_num = self._num_transformer.transform(
                X[num_cols].values.astype(np.float32)
            ).astype(np.float32)
            X_num = np.nan_to_num(X_num, nan=0.0)
        else:
            X_num = np.zeros((len(X), 0), dtype=np.float32)

        X_cat = np.zeros((len(X), len(cat_cols)), dtype=np.int64)
        for i, c in enumerate(cat_cols):
            X_cat[:, i] = X[c].cat.codes.values.clip(min=0)

        return (torch.tensor(X_num, dtype=torch.float32),
                torch.tensor(X_cat, dtype=torch.long))

    def _predict_proba(self, X: pd.DataFrame, **kwargs):
        X = self.preprocess(X)
        X_num_t, X_cat_t = self._to_tensors(X)
        if self.problem_type == "regression":
            preds = self.model.predict(X_num_t, X_cat_t)
            return preds * self._y_std + self._y_mean
        proba = self.model.predict_proba(X_num_t, X_cat_t)
        if self.problem_type == "binary":
            return proba[:, 1]
        return proba

    def _predict(self, X: pd.DataFrame, **kwargs):
        X = self.preprocess(X)
        X_num_t, X_cat_t = self._to_tensors(X)
        if self.problem_type == "regression":
            preds = self.model.predict(X_num_t, X_cat_t)
            return preds * self._y_std + self._y_mean
        return self.model.predict(X_num_t, X_cat_t)

    def _set_default_params(self):
        default_params = {"n_models": 5}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update({
            "valid_raw_types": ["int", "float", "category"],
        })
        return default_auxiliary_params

    def _get_default_stopping_metric(self):
        return self.eval_metric

    def _get_default_resources(self) -> tuple[int, int]:
        # Request 4 CPUs + 0.125 GPU per fold → allows 8 parallel folds on 1 GPU
        num_gpus = 0.125
        return 4, num_gpus

"""
Cumulative ablation configs (Paper Table 5).

Starting from the simplest baseline (CASTLE+CINN two-stage), each step adds
one improvement. This traces the full design evolution of CausalBoost.

Steps:
  0. CASTLE+CINN two-stage (augmented Lagrangian DAG, hard threshold)
  1. SDCD+CINN end-to-end (spectral radius DAG constraint, no two-stage)
  2. + Factored DAG W = A @ S @ A.T (k=32 groups)
  3. + Label smoothing 0.1 + edge dropout 0.2
  4. + Boosted ensemble M=5 with shrinkage=0.1
  5. + Adaptive sample reweighting (base_alpha=2.0)
  6. + Post-hoc temperature calibration (= full CausalBoost)
"""

from collections import OrderedDict

SEED = 42

# Shared kwargs for single-model configs (no ensemble, no reweighting)
_SINGLE = dict(n_models=1, shrinkage=0.1, base_alpha=0.0, calibrate=False)

CONFIGS = OrderedDict()

CONFIGS['cum_castle'] = {
    'group': 'cumulative',
    'description': 'CASTLE+CINN two-stage baseline (v2)',
    'model': 'castle',
    'kwargs': dict(castle_inner_epochs=50, castle_outer_iters=10,
                   cinn_epochs=200, adaptive_threshold=True),
}

CONFIGS['cum0_base'] = {
    'group': 'cumulative',
    'description': 'SDCD+CINN, full W[d,d], single model, no regularization',
    'model': 'boost',
    'kwargs': {**_SINGLE,
               'dag_type': 'standard',
               'edge_dropout': 0.0, 'label_smoothing': 0.0},
}

CONFIGS['cum1_factored'] = {
    'group': 'cumulative',
    'description': '+ Factored DAG W=ASA^T (k=32 groups)',
    'model': 'boost',
    'kwargs': {**_SINGLE,
               'dag_type': 'factored', 'n_groups': 32,
               'edge_dropout': 0.0, 'label_smoothing': 0.0},
}

CONFIGS['cum2_reg'] = {
    'group': 'cumulative',
    'description': '+ Label smoothing 0.1 + edge dropout 0.2',
    'model': 'boost',
    'kwargs': {**_SINGLE,
               'dag_type': 'factored', 'n_groups': 32,
               'edge_dropout': 0.2, 'label_smoothing': 0.1},
}

CONFIGS['cum3_boost'] = {
    'group': 'cumulative',
    'description': '+ Boosted ensemble M=5, shrinkage=0.1 (no reweighting)',
    'model': 'boost',
    'kwargs': dict(n_models=5, shrinkage=0.1, base_alpha=0.0,
                   calibrate=False,
                   dag_type='factored', n_groups=32,
                   edge_dropout=0.2, label_smoothing=0.1),
}

CONFIGS['cum4_reweight'] = {
    'group': 'cumulative',
    'description': '+ Adaptive sample reweighting (base_alpha=2.0)',
    'model': 'boost',
    'kwargs': dict(n_models=5, shrinkage=0.1, base_alpha=2.0,
                   calibrate=False,
                   dag_type='factored', n_groups=32,
                   edge_dropout=0.2, label_smoothing=0.1),
}

CONFIGS['cum5_causalboost'] = {
    'group': 'cumulative',
    'description': '+ Temperature calibration (= full CausalBoost)',
    'model': 'boost',
    'kwargs': dict(n_models=5, shrinkage=0.1, base_alpha=2.0,
                   calibrate=True,
                   dag_type='factored', n_groups=32,
                   edge_dropout=0.2, label_smoothing=0.1),
}

GROUP = ('cumulative', [
    'cum_castle', 'cum0_base', 'cum1_factored', 'cum2_reg',
    'cum3_boost', 'cum4_reweight', 'cum5_causalboost',
])

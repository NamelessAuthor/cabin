"""
Baseline configs for comparison with external methods (Paper Table 8).

All baselines use default or commonly-used hyperparameters from their
respective papers/libraries. CausalBoost (cum5_causalboost) is included
as a cross-reference for direct comparison.

Dependencies:
  pip install xgboost catboost lightgbm tabicl tabpfn
"""

from collections import OrderedDict

SEED = 42

CONFIGS = OrderedDict()

CONFIGS['base_xgboost'] = {
    'group': 'baselines',
    'description': 'XGBoost (n_estimators=500, max_depth=6)',
    'model': 'xgboost',
    'kwargs': dict(n_estimators=500, max_depth=6, tree_method='hist',
                   device='cpu', random_state=SEED, n_jobs=1),
}

CONFIGS['base_catboost'] = {
    'group': 'baselines',
    'description': 'CatBoost (iterations=500)',
    'model': 'catboost',
    'kwargs': dict(iterations=500, random_seed=SEED, thread_count=1),
}

CONFIGS['base_lightgbm'] = {
    'group': 'baselines',
    'description': 'LightGBM (n_estimators=500)',
    'model': 'lightgbm',
    'kwargs': dict(n_estimators=500, random_state=SEED, n_jobs=1),
}

CONFIGS['base_tabicl'] = {
    'group': 'baselines',
    'description': 'TabICL v2 (default)',
    'model': 'tabicl',
    'kwargs': dict(checkpoint_version='tabicl-classifier-v2-20260212.ckpt'),
}

CONFIGS['base_tabicl_v1.1'] = {
    'group': 'baselines',
    'description': 'TabICL v1.1',
    'model': 'tabicl',
    'kwargs': dict(checkpoint_version='tabicl-classifier-v1.1-20250506.ckpt'),
}

CONFIGS['base_tabpfn'] = {
    'group': 'baselines',
    'description': 'TabPFN (default, n<10K only)',
    'model': 'tabpfn',
    'kwargs': {},
}

GROUP = ('baselines', [
    'base_xgboost', 'base_catboost', 'base_lightgbm',
    'base_tabicl', 'base_tabicl_v1.1', 'base_tabpfn',
    'cum5_causalboost',  # Our method for comparison
])

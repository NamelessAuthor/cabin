"""
Dimension 2: DAG Parameterization (Paper Table 2).

How is the DAG adjacency matrix W structured?

All configs use SDCD end-to-end, single model, no regularization.
Only the DAG parameterization changes.

Variants:
  - Standard:    Full W[d,d] parameter matrix.
  - Factored:    W = A[d,k] @ S[k,k] @ A[d,k].T where k=32 groups.
                 Correlated features naturally merge into same group.
                 Acyclicity constraint on S (32x32) instead of W (dxd).
  - Block:       Full W[d,d] with block-diagonal regularization.
                 Encourages sparse blocks via correlation-based clusters.
  - Latent(16):  Augments DAG with 16 latent nodes (d+16 total).
  - Latent(32):  Augments DAG with 32 latent nodes.

Baselines cum0_base and cum1_factored are defined in cumulative.py.
"""

from collections import OrderedDict

_SINGLE = dict(n_models=1, shrinkage=0.1, base_alpha=0.0, calibrate=False)

CONFIGS = OrderedDict()

CONFIGS['dim2_block'] = {
    'group': 'dim2',
    'description': 'Block-diagonal DAG regularization (v38_block)',
    'model': 'boost',
    'kwargs': {**_SINGLE,
               'dag_type': 'block', 'lambda_block': 0.1,
               'edge_dropout': 0.0, 'label_smoothing': 0.0},
}

CONFIGS['dim2_latent16'] = {
    'group': 'dim2',
    'description': '+ 16 latent nodes in DAG (v38/v45)',
    'model': 'boost',
    'kwargs': {**_SINGLE,
               'dag_type': 'latent', 'n_latent': 16,
               'edge_dropout': 0.0, 'label_smoothing': 0.0},
}

CONFIGS['dim2_latent32'] = {
    'group': 'dim2',
    'description': '+ 32 latent nodes in DAG (v38/v45)',
    'model': 'boost',
    'kwargs': {**_SINGLE,
               'dag_type': 'latent', 'n_latent': 32,
               'edge_dropout': 0.0, 'label_smoothing': 0.0},
}

GROUP = ('dim2_dag_parameterization', [
    'cum0_base',       # Full W[d,d] (from cumulative)
    'cum1_factored',   # Factored W=ASA^T (from cumulative, winner)
    'dim2_block',
    'dim2_latent16',
    'dim2_latent32',
])

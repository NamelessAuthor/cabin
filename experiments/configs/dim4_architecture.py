"""
Dimension 4: CINN Architecture — Message Passing Rounds (Paper Table 4).

How many rounds of message passing through the DAG?

The CINN processes features in topological order. Multi-round message passing
(K>1) allows information to propagate further through the graph, analogous
to multiple GNN layers. However, prior experiments (v10, v16) showed this
hurts performance, likely due to oversmoothing.

All configs use SDCD end-to-end, factored DAG, single model, no regularization.
Only n_rounds changes.

Uses CINNClassifier (single model) since CausalBoostClassifier doesn't
expose the n_rounds parameter.

Baseline cum1_factored (K=1) is defined in cumulative.py.
"""

from collections import OrderedDict

CONFIGS = OrderedDict()

CONFIGS['dim4_multi2'] = {
    'group': 'dim4',
    'description': 'Multi-round message passing K=2 (tested in v10/v16)',
    'model': 'single',
    'kwargs': dict(dag_type='factored', n_groups=32, n_rounds=2,
                   edge_dropout=0.0, label_smoothing=0.0),
}

CONFIGS['dim4_multi3'] = {
    'group': 'dim4',
    'description': 'Multi-round message passing K=3',
    'model': 'single',
    'kwargs': dict(dag_type='factored', n_groups=32, n_rounds=3,
                   edge_dropout=0.0, label_smoothing=0.0),
}

GROUP = ('dim4_architecture', [
    'cum1_factored',   # Single pass K=1 (from cumulative, winner)
    'dim4_multi2',
    'dim4_multi3',
])

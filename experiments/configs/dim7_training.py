"""
Dimension 7: Training Procedure — Phase Schedule (Paper Table 7).

Does the 3-phase training schedule (warmup → joint → finetune) matter?

CausalBoost uses a 3-phase training schedule:
  1. Warmup (20 epochs):   Train CINN only, DAG frozen at initialization.
  2. Joint (100 epochs):   Train DAG and CINN together end-to-end.
  3. Finetune (30 epochs): Freeze DAG, fine-tune CINN on the learned structure.

This dimension tests whether each phase contributes by removing them.

All configs use SDCD, factored DAG, single model, no regularization.
Uses CINNClassifier (single model) to expose phase parameters.

Baseline cum1_factored (default 20/100/30) is defined in cumulative.py.
"""

from collections import OrderedDict

CONFIGS = OrderedDict()

CONFIGS['dim7_no_warmup'] = {
    'group': 'dim7',
    'description': 'No warmup phase (0/100/30)',
    'model': 'single',
    'kwargs': dict(dag_type='factored', n_groups=32,
                   warmup_epochs=0, joint_epochs=100, finetune_epochs=30,
                   edge_dropout=0.0, label_smoothing=0.0),
}

CONFIGS['dim7_no_finetune'] = {
    'group': 'dim7',
    'description': 'No finetune phase (20/100/0)',
    'model': 'single',
    'kwargs': dict(dag_type='factored', n_groups=32,
                   warmup_epochs=20, joint_epochs=100, finetune_epochs=0,
                   edge_dropout=0.0, label_smoothing=0.0),
}

CONFIGS['dim7_joint_only'] = {
    'group': 'dim7',
    'description': 'Joint phase only (0/150/0)',
    'model': 'single',
    'kwargs': dict(dag_type='factored', n_groups=32,
                   warmup_epochs=0, joint_epochs=150, finetune_epochs=0,
                   edge_dropout=0.0, label_smoothing=0.0),
}

GROUP = ('dim7_training', [
    'cum1_factored',     # Default 3-phase 20/100/30 (from cumulative, winner)
    'dim7_no_warmup',
    'dim7_no_finetune',
    'dim7_joint_only',
])

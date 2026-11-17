"""
Dimension 1: DAG Learning Method (Paper Table 1).

How is the causal graph discovered from data?

Compares three approaches:
  - CASTLE: Augmented Lagrangian with h(W) = tr(e^W) - d constraint.
            Two-stage: learn DAG first, then train CINN separately.
  - GOLEM:  Single-loop optimization with log-det score function.
            Two-stage: learn DAG first, then train CINN separately.
  - SDCD:   Spectral radius constraint rho(|W|) < 1.
            End-to-end: DAG and CINN trained jointly.

Baselines cum_castle and cum0_base are defined in cumulative.py.
Only dim1_golem is unique to this dimension.
"""

from collections import OrderedDict

CONFIGS = OrderedDict()

CONFIGS['dim1_golem'] = {
    'group': 'dim1',
    'description': 'GOLEM+CINN two-stage (v15, single-loop DAG)',
    'model': 'golem',
    'kwargs': dict(golem_epochs=200, cinn_epochs=200,
                   lambda_dag=5.0, lambda_sparse=0.5),
}

GROUP = ('dim1_dag_learning', [
    'cum_castle',    # CASTLE two-stage (from cumulative)
    'dim1_golem',    # GOLEM two-stage
    'cum0_base',     # SDCD end-to-end (from cumulative, winner)
])

# Experiment 4: Acyclicity Constraint (Table 3, Dimension 3)

Compares four methods for enforcing the DAG acyclicity constraint on the group-level matrix S.

## Configs

| Config | Description |
|---|---|
| `cum1_factored` | Spectral radius: power iteration on S^2, explicit penalty |
| `dim3_dagma` | DAGMA: log-determinant constraint log det(sI - S*S) |
| `dim3_vcuda` | VCUDA: priority scores + edge logits, acyclic by construction |
| `dim3_dpdag` | DP-DAG: SoftSort permutation + edge logits, acyclic by construction |

## Key Finding

All four methods achieve comparable accuracy. Construction-based methods (VCUDA, DP-DAG) guarantee acyclicity without an explicit penalty term but don't improve accuracy over the simpler spectral radius approach.

## Results

72 OpenML-CC18 datasets, 5-fold stratified CV. See `results.csv`.

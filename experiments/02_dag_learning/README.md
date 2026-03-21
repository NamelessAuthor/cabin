# Experiment 2: DAG Learning Method (Table 1, Dimension 1)

Compares three approaches to learning the DAG structure:
two-stage methods (learn DAG first, then train CINN) vs end-to-end joint optimization.

## Configs

| Config | Description |
|---|---|
| `cum_castle` | Two-stage: CASTLE (augmented Lagrangian) then CINN |
| `dim1_golem` | Two-stage: GOLEM (single-loop) then CINN |
| `cum0_base` | End-to-end: SDCD jointly learns DAG + CINN |

## Key Finding

End-to-end SDCD outperforms both two-stage baselines. Joint optimization allows the DAG structure to adapt to the classification objective.

## Results

72 OpenML-CC18 datasets, 5-fold stratified CV. See `results.csv`.

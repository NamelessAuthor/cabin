# Experiment 2: DAG Learning Method (paper Appendix B, Table 8)

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

All 72 OpenML-CC18 datasets were attempted, 5-fold stratified CV, 1 hour per fit. The paper
reports means on the datasets where every compared configuration finished all 5 folds
(`10_routing_and_structure/scripts/ablations.py`). See `results.csv`.

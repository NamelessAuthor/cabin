# Experiment 7: Training Procedure (Table 7, Dimension 7)

Tests the contribution of the 3-phase training schedule: warmup (DAG frozen), joint (DAG + CINN), finetune (DAG frozen, low LR).

## Configs

| Config | Description |
|---|---|
| `cum1_factored` | Default 3-phase: 20 warmup + 100 joint + 30 finetune epochs |
| `dim7_no_warmup` | Skip warmup: 0 + 100 + 30 |
| `dim7_no_finetune` | Skip finetune: 20 + 100 + 0 |
| `dim7_joint_only` | Joint only: 0 + 150 + 0 |

## Key Finding

The full 3-phase schedule is optimal. Warmup lets the CINN adapt before DAG gradients flow, and finetuning with frozen DAG stabilizes the final model. Removing either phase hurts accuracy.

## Results

72 OpenML-CC18 datasets, 5-fold stratified CV. See `results.csv`.

# Experiment 1: Cumulative Construction (paper Table 3)

Progressive ablation showing the contribution of each design dimension.
Starting from a CASTLE+CINN two-stage baseline, each row adds one improvement.

## Configs

| Config | Description |
|---|---|
| `cum_castle` | CASTLE+CINN two-stage baseline (augmented Lagrangian DAG) |
| `cum0_base` | SDCD+CINN end-to-end, full W[d,d], single model |
| `cum1_factored` | + Factored DAG W=ASA^T (k=32 groups) |
| `cum2_reg` | + Label smoothing 0.1 + edge dropout 0.2 |
| `cum3_boost` | + Boosted ensemble M=5, shrinkage=0.1 |
| `cum4_reweight` | + Adaptive sample reweighting (base_alpha=2.0) |
| `cum5_causalboost` | + Temperature calibration = full Cabin |

## Results

All 72 OpenML-CC18 datasets were attempted, 5-fold stratified CV, 1 hour per fit. The paper
reports means on the datasets where every compared configuration finished all 5 folds
(`10_routing_and_structure/scripts/ablations.py`). See `results.csv`.

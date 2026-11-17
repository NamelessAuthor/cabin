# Experiment 8: Baselines in the design-study harness (paper Appendix B)

Compares Cabin against external baselines on OpenML-CC18.

## Configs

| Config | Description |
|---|---|
| `cum5_causalboost` | Cabin (full method) |
| `base_xgboost` | XGBoost (n_estimators=500, max_depth=6) |
| `base_catboost` | CatBoost (iterations=500) |
| `base_lightgbm` | LightGBM (n_estimators=500) |
| `base_tabicl` | TabICL (default) |
| `base_tabicl_v1.1` | TabICL v1.1 (default) |

## Results

All 72 OpenML-CC18 datasets were attempted, 5-fold stratified CV, 1 hour per fit. The paper
reports means on the datasets where every compared configuration finished all 5 folds
(`10_routing_and_structure/scripts/ablations.py`). See `results.csv`.

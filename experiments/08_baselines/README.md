# Experiment 8: Baseline Comparisons (Table 8)

Compares Cabin against external baselines on the same 72 OpenML-CC18 datasets.

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

72 OpenML-CC18 datasets, 5-fold stratified CV. See `results.csv`.

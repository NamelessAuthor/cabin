# Experiment 9: TabArena Living Benchmark

Evaluation on 28 classification datasets from TabArena v0.1, a living benchmark for tabular ML.

TabArena contains 51 datasets (38 classification + 13 regression). We evaluate on 28 classification tasks; 10 were excluded due to training time, 13 are regression (Cabin is classification-only).

## Files

- `results.csv` — Per-dataset, per-fold results for all methods (Cabin + 20 baselines across default/tuned/ensembled configs)
- `leaderboard.csv` — Aggregated leaderboard with ELO ratings, ranks, win rates, and timing

## Cabin Performance

- **ELO**: 1333
- **Rank**: 7th out of 21 default-configuration methods
- **Win rate**: 54.3%
- **Avg training time**: 372s

Cabin places between TabPFNv2 (default) and TabM (default), outperforming EBM, ModernNCA, RealMLP, XGBoost, and LightGBM defaults — all without hyperparameter tuning.

## 28 Evaluated Datasets

blood-transfusion, diabetes, credit-g, maternal_health_risk, qsar-biodeg, website_phishing, Fitness_Club, MIC, Is-this-a-good-customer, Marketing_Campaign, hazelnut-spread, seismic-bumps, splice, hiva_agnostic, students_dropout, churn, polish_companies_bankruptcy, taiwanese_bankruptcy, NATICUSdroid, heloc, jm1, E-CommereShippingData, online_shoppers_intention, in_vehicle_coupon, HR_Analytics, Diabetes130US, SDSS17, GiveMeSomeCredit.

## Evaluation Protocol

Following TabArena: 10x3-fold CV for small datasets (n < 2500), 3x3-fold CV for larger ones. Metrics: ROC-AUC (binary), log-loss (multiclass).

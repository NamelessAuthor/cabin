# Experiment 9: TabArena Living Benchmark

Evaluation on 28 classification datasets from TabArena v0.1, a living benchmark for tabular ML.

TabArena contains 51 datasets (38 classification + 13 regression). We evaluate on 28 classification tasks; 10 were excluded due to training time, 13 are regression (Cabin is classification-only).

## Files

- `results.csv`: per-dataset, per-fold results for all methods (Cabin and the TabArena baselines, default/tuned/ensembled configurations).
- `leaderboard.csv`: the aggregated leaderboard of the full TabArena pool (ELO, ranks, win rates, timing).
- `runpod_setup/`: scripts that ran Cabin on TabArena on RunPod GPUs (see `runpod_setup/README.txt`).

## In the paper

Table 1 and Table 10 rank Cabin among 14 methods with default configurations: Cabin reaches
ELO 1203 (rank 7 of 14), above every method trained from scratch, including CatBoost (1197);
the six methods above it are pre-trained in-context learners. The full leaderboard here
contains more methods and configurations, so its ELO values differ.

## 28 Evaluated Datasets

blood-transfusion, diabetes, credit-g, maternal_health_risk, qsar-biodeg, website_phishing, Fitness_Club, MIC, Is-this-a-good-customer, Marketing_Campaign, hazelnut-spread, seismic-bumps, splice, hiva_agnostic, students_dropout, churn, polish_companies_bankruptcy, taiwanese_bankruptcy, NATICUSdroid, heloc, jm1, E-CommereShippingData, online_shoppers_intention, in_vehicle_coupon, HR_Analytics, Diabetes130US, SDSS17, GiveMeSomeCredit.

## Evaluation Protocol

Following TabArena: 10x3-fold CV for small datasets (n < 2500), 3x3-fold CV for larger ones. Metrics: ROC-AUC (binary), log-loss (multiclass).

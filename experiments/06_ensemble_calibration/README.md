# Experiment 6: Ensemble & Calibration (Table 6, Dimension 6)

Progressively adds ensemble components: boosting, adaptive sample reweighting, and temperature calibration.

## Configs

| Config | Description |
|---|---|
| `cum2_reg` | Single model with label smoothing + edge dropout (no ensemble) |
| `cum3_boost` | Boosted ensemble M=5, shrinkage=0.1, no sample reweighting |
| `cum4_reweight` | + Adaptive sample reweighting: weight = 1 + alpha * (1 - p_correct) |
| `cum5_causalboost` | + Post-hoc temperature calibration via LBFGS (= full Cabin) |

## Key Finding

All three components contribute complementarily. Boosting provides the largest gain. Adaptive alpha scales reweighting strength by dataset characteristics (n, d, n_classes). Temperature calibration adds a small but consistent improvement.

## Results

72 OpenML-CC18 datasets, 5-fold stratified CV. See `results.csv`.

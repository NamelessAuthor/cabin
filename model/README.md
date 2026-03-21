# Cabin Model

Self-contained single-file implementation of Cabin (CausalBoost) as described in the paper
*"How Far Can DAG-Informed Neural Networks Go? A Systematic Design Study for Tabular Classification"*.

## Usage

```python
from cabin import CausalBoostClassifier

# Champion config (default)
clf = CausalBoostClassifier()
clf.fit(X_train, y_train, cat_indices=[2, 5])
preds = clf.predict(X_test)
proba = clf.predict_proba(X_test)
```

## Configurable Design Dimensions

All seven design dimensions from the paper are exposed as constructor arguments:

| Dimension | Parameter | Options | Default (champion) |
|---|---|---|---|
| Dim 1 - DAG learning | (always end-to-end SDCD) | -- | -- |
| Dim 2 - DAG parameterization | `dag_type` | `factored`, `standard`, `block`, `latent` | `factored` |
| Dim 3 - Acyclicity constraint | `acyclicity` | `vcuda`, `spectral`, `dagma`, `dpdag` | `vcuda` |
| Dim 4 - Message passing | `n_rounds` | 1, 2, 3 | 1 |
| Dim 5 - Feature representation | `add_missing_indicators` | True, False | True |
| Dim 6 - Ensemble & calibration | `n_models`, `calibrate` | 1-10, True/False | 5, True |
| Dim 7 - Training procedure | `label_smoothing`, `edge_dropout` | floats | 0.1, 0.2 |

## Ablation Examples

```python
# Single model, no ensemble
clf = CausalBoostClassifier(n_models=1, calibrate=False)

# Standard DAG with spectral radius
clf = CausalBoostClassifier(dag_type='standard', acyclicity='spectral')

# Multi-round message passing
clf = CausalBoostClassifier(n_rounds=3)
```

## Dependencies

```bash
pip install torch numpy scikit-learn scipy
```

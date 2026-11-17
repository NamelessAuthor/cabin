# Cabin model

Single-file implementation of Cabin (`CausalBoostClassifier`, scikit-learn style).

```python
from cabin import CausalBoostClassifier, PAPER_CONFIG

clf = CausalBoostClassifier(**PAPER_CONFIG, random_state=42)   # configuration of the paper
clf.fit(X_train, y_train, cat_indices=[2, 5])
clf.predict(X_test); clf.predict_proba(X_test)
```

`PAPER_CONFIG = dict(nam_shape=True, n_slots=2, n_groups=32)`: two slots per feature, each
embedded by a small MLP, k = 32 groups. The other defaults are those of the paper: factored
W = A S Aᵀ, VCUDA ordering (`tau_vcuda=0.3`, `tau_edge=0.5`), one round of message passing,
missingness indicators, edge dropout 0.2, label smoothing 0.1, three-phase training
(20 warmup, 100 joint, 30 finetune epochs), M = 5 boosted networks with shrinkage 0.1 and
adaptive reweighting, temperature calibration.

`random_state` (default 42) seeds every network: model m of the chain uses `random_state + 17 m`.

## Design dimensions

| Dimension | Parameter | Options | Default |
|---|---|---|---|
| DAG parameterization | `dag_type` | `factored`, `standard`, `block`, `latent` | `factored` |
| Acyclicity | `acyclicity` | `vcuda`, `spectral`, `dagma`, `dpdag` | `vcuda` |
| Message passing | `n_rounds` | 1, 2, 3 | 1 |
| Feature representation | `add_missing_indicators`, `n_slots` | bool, int | True, 1 (2 in `PAPER_CONFIG`) |
| Ensemble & calibration | `n_models`, `calibrate` | int, bool | 5, True |
| Training | `label_smoothing`, `edge_dropout` | floats | 0.1, 0.2 |
| Fixed routing | `fixed_dag`, `fixed_dags` | d×d array(s) | None (learned) |

Other arguments (MoE-DAG, contrastive and auxiliary losses, alternative heads) are research
options, off by default and not used in the paper.

`python cabin.py` runs a smoke test on Iris.

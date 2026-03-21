# Cabin

Code and experiments for *"How Far Can DAG-Informed Neural Networks Go? A Systematic Design Study for Tabular Classification"*.

Cabin is a DAG-informed neural network for tabular classification. It learns a directed acyclic graph (DAG) over input features end-to-end and processes nodes in topological order. The paper systematically evaluates seven design dimensions — DAG learning, parameterization, acyclicity enforcement, message passing, feature representation, ensemble strategy, and training procedure — across 72 OpenML-CC18 datasets.

## Quick Start

```python
from model.cabin import CausalBoostClassifier

clf = CausalBoostClassifier()
clf.fit(X_train, y_train, cat_indices=[2, 5])
preds = clf.predict(X_test)
```

## Repository Structure

```
cabin/
  model/
    cabin.py          # Self-contained single-file model with all design dimensions configurable
  experiments/
    01_cumulative_ablation/    # Table 5: progressive ablation from baseline to full Cabin
    02_dag_learning/           # Table 1: CASTLE vs GOLEM vs SDCD (end-to-end)
    03_dag_parameterization/   # Table 2: full W vs factored vs block vs latent
    04_acyclicity_constraint/  # Table 3: spectral radius vs DAGMA vs VCUDA vs DP-DAG
    05_message_passing/        # Table 4: K=1 vs K=2 vs K=3 topological passes
    06_ensemble_calibration/   # Table 6: boosting + reweighting + temperature calibration
    07_training_procedure/     # Table 7: 3-phase training schedule ablation
    08_baselines/              # Table 8: Cabin vs XGBoost, CatBoost, LightGBM, TabICL
    09_tabarena/               # TabArena living benchmark (28 classification datasets)
```

Each experiment folder contains:
- `results.csv` — per-dataset, per-fold results
- `README.md` — experiment description, configs, and key findings

## Design Dimensions

| Dim | Question | Winner | Options Tested |
|-----|----------|--------|----------------|
| 1 | How to learn the DAG? | End-to-end SDCD | CASTLE, GOLEM, SDCD |
| 2 | How to parameterize the adjacency matrix? | Factored W=ASA^T | Full, factored, block, latent |
| 3 | How to enforce acyclicity? | All comparable | Spectral radius, DAGMA, VCUDA, DP-DAG |
| 4 | How many message-passing rounds? | K=1 | K=1, 2, 3 |
| 5 | Feature representation? | Entity embeddings + missingness | -- |
| 6 | Ensemble strategy? | Boosted M=5 + calibration | Single, boosted, +reweighting, +calibration |
| 7 | Training schedule? | 3-phase (warmup/joint/finetune) | 3-phase, no warmup, no finetune, joint only |

## Dependencies

```bash
pip install torch numpy scikit-learn scipy
```

## Citation

```
@article{cabin2026,
  title={How Far Can DAG-Informed Neural Networks Go? A Systematic Design Study for Tabular Classification},
  year={2026}
}
```

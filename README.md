# Cabin

Code and results for *"Causal Inductive Biases Help Tabular Models Only When Learned Right"*.

Cabin is a DAG-informed neural network for tabular classification. Its message weights
W = A S Aᵀ route information between features through an ordered graph S between groups of
features (VCUDA ordering, k = 32 groups), learned end-to-end with the predictor, and five
networks are combined by boosting with temperature calibration. The paper builds Cabin one
step at a time, compares it with tree-based and neural methods on OpenML-CC18 and TabArena,
then substitutes its learned routing with controls (no graph, dense random, sparse random,
true and estimated graphs) on synthetic SCMs and on real data to test when a graph helps.

## Quick start

```python
import sys; sys.path.insert(0, 'model')
from cabin import CausalBoostClassifier, PAPER_CONFIG

clf = CausalBoostClassifier(**PAPER_CONFIG, random_state=42)
clf.fit(X_train, y_train, cat_indices=[2, 5])
clf.predict(X_test)
```

```bash
pip install torch numpy pandas scipy scikit-learn openml filelock lightgbm
```

## Seeds

Every run uses fixed seeds. Model m of the boosted chain uses seed `random_state + 17 m`
(default `random_state=42`). The run scripts of `experiments/10_routing_and_structure` set
`clf.random_state = legacy_seed(seed)`, which reproduces the seeds of the paper runs.

## Repository structure

```
paper/
  main.tex, references.bib     # LaTeX sources (CLeaR 2027 class); paper.pdf, supplementary.pdf
  generated/                   # numbers.tex, tables and figures written by make_numbers.py
model/
  cabin.py                     # Cabin (CausalBoostClassifier); PAPER_CONFIG = paper configuration
experiments/
  run.py, common.py, models.py, configs/   # design-study runner: python -m experiments.run --list
  01_cumulative_ablation/      # Table 3: cumulative construction of Cabin
  02_dag_learning/             # Appendix B: CASTLE vs GOLEM vs end-to-end SDCD
  03_dag_parameterization/     # Appendix B: full W vs factored vs block vs latent
  04_acyclicity_constraint/    # Appendix B: spectral radius vs DAGMA vs VCUDA vs DP-DAG
  05_message_passing/          # Appendix B: 1, 2 or 3 rounds of message passing
  06_ensemble_calibration/     # Table 3 rows 4-6: boosting, reweighting, calibration
  07_training_procedure/       # Appendix B: three-phase training schedule
  08_baselines/                # Appendix B: baselines in the design-study harness
  09_tabarena/                 # Table 1 (TabArena columns), Table 10; runpod_setup/ runs Cabin on TabArena
  10_routing_and_structure/    # Tables 1-2 (CC18), Sections 6-7, Appendices D-H
```

Each numbered folder has a `README.md` and its per-dataset, per-fold results.

## Datasets

- **OpenML-CC18** (72 datasets). With a limit of one hour per fit, the main comparison
  (Table 1) uses the 52 datasets on which every method finished, 5-fold stratified CV,
  default hyperparameters.
- **TabArena** (28 classification datasets, TabArena protocol).
- **Dev8**: the eight development datasets on which design choices and defaults were set
  (sonar, cylinder-bands, spambase, steel-plates-fault, mfeat-zernike, segment, semeion, cnae-9).

## License

See `LICENSE`.

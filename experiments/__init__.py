"""
CausalBoost Paper Experiments
=============================

Self-contained experiment code for reproducing all results in:
  "Closing the Gap: How Causal Inductive Biases Can Approach
   State-of-the-Art Tabular Classification"

Structure:
  models.py    — All model implementations (CausalBoost, CASTLE, GOLEM, CINN)
  common.py    — Dataset loading, result management, job runner
  run.py       — CLI entry point
  configs/     — One file per design dimension

Usage:
  cd paper/
  python -m experiments.run --list              # show all configs
  python -m experiments.run --workers 4         # run everything
  python -m experiments.run --summary           # print results
"""

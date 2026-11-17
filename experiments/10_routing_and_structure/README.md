# Experiment 10: CC18 comparison, routing substitutions and structure recovery

Code and per-fold outputs for Tables 1-2 (CC18 columns) and Sections 6-7 of the paper.
The `rerun_*` folders reference each other by relative path; keep them side by side.
Library files `cabin_v33_lib*.py` and `rerun_synthetic/cabin.py` are pinned copies of
`model/cabin.py` used by each experiment, some with switches for the diagnostics (search `DIAG`).

| Folder | Paper |
|---|---|
| `results/tub/` | Table 1 (CC18, 52 datasets) and Table 2 (bins vs LightGBM): per-fold JSONs of every method, `table2.py` rebuilds the table |
| `rerun_interact/` | Section 6.1: synthetic SCMs with interactions, λ sweep, routings of Table 4 (Table 5, Figure 2) |
| `rerun_headopt/`, `rerun_dagopt/` | Section 6.1 and Appendix H: additive head, estimated and sparse random graphs on SCMs (Tables 14-15) |
| `rerun_diag/` | Section 6.2: additive head and test-time W = 0 on CC18 |
| `rerun_real_estgraph/`, `rerun_highd_real/`, `rerun_real_stg/` | Section 6.2: routing substitutions on CC18 (Table 6) |
| `rerun_sensitivity/` | Appendix D: sensitivity to k, M and the restore rule (Table 11) |
| `rerun_synthetic/` | Section 7 and Appendix E: structure recovery on synthetic DAGs (Table 12) |
| `rerun_dag/` | Section 7 and Appendix F: learned routing on real data (Figure 3) |
| `scripts/` | `make_numbers.py` computes every number, table and figure of the paper from the result files and writes them to `paper/generated/` |

Each run script has a docstring with its variants and usage; `analyze*.py` prints the tables.
Folds 0-2 and seed 42 are used for the real-data substitutions.

`scripts/make_numbers.py` reads the design-study CSVs of `../0*_*/`, the outputs here and
`../09_tabarena/paper_results.csv` (the 14-method TabArena pool of Table 1).
It also reads `results/tub/tasks/<task>/task.json` (n and d of each CC18 task) for Table 2.

# Experiment 3: DAG Parameterization (paper Appendix B, Table 8)

Compares different ways to parameterize the DAG adjacency matrix W.

## Configs

| Config | Description |
|---|---|
| `cum0_base` | Full W[d,d] — standard dense adjacency matrix |
| `cum1_factored` | Factored W = A @ S @ A^T — A[d,32] assigns features to groups, S[32,32] is group-level DAG |
| `dim2_block` | Full W with block-diagonal regularization via correlation-based clusters |
| `dim2_latent16` | Full W with 16 additional latent nodes appended |
| `dim2_latent32` | Full W with 32 additional latent nodes appended |

## Key Finding

Factored DAG wins. Correlated features naturally merge into the same group via A, eliminating redundant edges. Acyclicity constraint on small S[32x32] is cheaper and better-conditioned than on full W[d,d].

## Results

All 72 OpenML-CC18 datasets were attempted, 5-fold stratified CV, 1 hour per fit. The paper
reports means on the datasets where every compared configuration finished all 5 folds
(`10_routing_and_structure/scripts/ablations.py`). See `results.csv`.

# Experiment 5: Message Passing (paper Appendix B, Table 8)

Tests whether several rounds of message passing along W improve over a single round of parent aggregation.

## Configs

| Config | Description |
|---|---|
| `cum1_factored` | K=1: one round of parent aggregation along W |
| `dim4_multi2` | K=2: two rounds of message passing |
| `dim4_multi3` | K=3: three rounds of message passing |

## Key Finding

A single pass (K=1) is optimal. Additional rounds add computation without improving accuracy, suggesting that one round already captures the relevant causal structure.

## Results

All 72 OpenML-CC18 datasets were attempted, 5-fold stratified CV, 1 hour per fit. The paper
reports means on the datasets where every compared configuration finished all 5 folds
(`10_routing_and_structure/scripts/ablations.py`). See `results.csv`.

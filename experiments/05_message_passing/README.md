# Experiment 5: Message Passing Architecture (Table 4, Dimension 4)

Tests whether multiple rounds of message passing through the DAG improve over a single topological pass.

## Configs

| Config | Description |
|---|---|
| `cum1_factored` | K=1: single forward pass through DAG in topological order |
| `dim4_multi2` | K=2: two rounds of message passing |
| `dim4_multi3` | K=3: three rounds of message passing |

## Key Finding

A single pass (K=1) is optimal. Additional rounds add computation without improving accuracy, suggesting that one topological sweep already captures the relevant causal structure.

## Results

72 OpenML-CC18 datasets, 5-fold stratified CV. See `results.csv`.

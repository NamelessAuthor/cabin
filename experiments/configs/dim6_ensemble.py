"""
Dimension 6: Ensemble Strategy & Calibration (Paper Table 6).

How to combine multiple CINN models?

The cumulative ablation already traces this dimension step-by-step:
  - cum2_reg:          Single model (regularized) — baseline
  - cum3_boost:        Boosted ensemble M=5, shrinkage=0.1, no reweighting
  - cum4_reweight:     + Adaptive sample reweighting (alpha=2.0)
  - cum5_causalboost:  + Post-hoc temperature calibration

All configs are cross-referenced from cumulative.py.
No unique configs in this dimension — the cumulative steps already cover it.
"""

from collections import OrderedDict

CONFIGS = OrderedDict()

# No unique configs: all are cross-references from cumulative

GROUP = ('dim6_ensemble', [
    'cum2_reg',          # Single model baseline (from cumulative)
    'cum3_boost',        # Boosted M=5 (from cumulative)
    'cum4_reweight',     # + Adaptive reweighting (from cumulative)
    'cum5_causalboost',  # + Temperature calibration (from cumulative, winner)
])

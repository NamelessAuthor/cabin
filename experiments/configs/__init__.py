"""
Experiment configurations for the CausalBoost ablation study.

Each module defines configs for one design dimension (or group).
Every module exports:
  - CONFIGS: OrderedDict of config_name -> config_dict
  - GROUP:   (group_name, [config_names]) for the summary table
"""

from collections import OrderedDict

from .cumulative import CONFIGS as CUM_CONFIGS, GROUP as CUM_GROUP
from .dim1_dag_learning import CONFIGS as D1_CONFIGS, GROUP as D1_GROUP
from .dim2_dag_parameterization import CONFIGS as D2_CONFIGS, GROUP as D2_GROUP
from .dim4_architecture import CONFIGS as D4_CONFIGS, GROUP as D4_GROUP
from .dim6_ensemble import CONFIGS as D6_CONFIGS, GROUP as D6_GROUP
from .dim7_training import CONFIGS as D7_CONFIGS, GROUP as D7_GROUP
from .baselines import CONFIGS as BL_CONFIGS, GROUP as BL_GROUP

# Merge all configs (order matters: earlier keys win on collision)
ALL_CONFIGS = OrderedDict()
for module_configs in [
    CUM_CONFIGS, D1_CONFIGS, D2_CONFIGS, D4_CONFIGS,
    D6_CONFIGS, D7_CONFIGS, BL_CONFIGS,
]:
    for k, v in module_configs.items():
        if k not in ALL_CONFIGS:
            ALL_CONFIGS[k] = v

ALL_GROUPS = OrderedDict([
    CUM_GROUP, BL_GROUP, D1_GROUP, D2_GROUP,
    D4_GROUP, D6_GROUP, D7_GROUP,
])

SEED = 42

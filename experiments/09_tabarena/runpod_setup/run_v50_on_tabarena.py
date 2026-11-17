"""Run V50 Missingness Indicator CINN model on full TabArena benchmark."""

from __future__ import annotations

from pathlib import Path

import openml

from custom_v50_model import CustomV50Model

from tabarena.benchmark.experiment import AGModelBagExperiment, run_experiments_new

TABARENA_DIR = str(Path(__file__).parent / "tabarena_out" / "v50_miss_ind")


if __name__ == "__main__":
    task_ids = openml.study.get_suite("tabarena-v0.1").tasks

    model_experiments = [
        AGModelBagExperiment(
            name="V50MissInd_BAG_L1",
            model_cls=CustomV50Model,
            model_hyperparameters={"n_models": 5},
            num_bag_folds=8,
            time_limit=3600,
        ),
    ]

    run_experiments_new(
        output_dir=TABARENA_DIR,
        model_experiments=model_experiments,
        tasks=task_ids,
        repetitions_mode="TabArena",
    )

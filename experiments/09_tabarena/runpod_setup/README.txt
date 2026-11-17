RunPod TabArena Setup — Cabin v50
==================================

Files:
  custom_v50_model_l4.py     — Cabin model adapted for L4 GPUs (smaller batch)
  custom_v50_model_full.py   — Full Cabin model (for larger GPUs)
  run_v50_on_tabarena.py     — TabArena runner script
  setup_redo_pods.py         — Automated parallel pod setup (upload, install, launch)
  generate_tabarena_plot.py  — Convert pkl results to TabArena format + generate comparison
  dump_tabarena.py           — Dump all TabArena baseline results to CSV

To reproduce on a new RunPod:
  1. Create pod with PyTorch template, L4 or better GPU
  2. Upload custom_v50_model_l4.py as /workspace/custom_v50_model.py
  3. Run on pod:
       pip install 'ray>=2.43.0,<2.53.0' autogluon.tabular openml huggingface_hub pandas
       git clone https://github.com/autogluon/tabarena.git /workspace/tabarena_src
  4. Fix tabarena bug (may be fixed upstream by now):
       In tabarena_src/tabarena/tabarena/benchmark/experiment/experiment_runner_api.py:
       - Change: metadata_task_ids = tasks_metadata["task_id"].astype(str).tolist()
       - To:     metadata_task_ids = tasks_metadata["task_id"].astype(str).reset_index(drop=True)
       - Change: assert t_id in metadata_task_ids
       - To:     assert t_id in metadata_task_ids.values
  5. Create runner script (see setup_redo_pods.py RUNNER_TEMPLATE for format)
  6. Launch: nohup python3 run_tasks.py > /tmp/tabarena_run.log 2>&1 &

Results are saved as pkl files in /workspace/tabarena_out/v50_clf/data/CausalBoost_BAG_L1/{task_id}/{fold}_{repeat}/results.pkl

RunPod API key: set the RUNPOD_API_KEY environment variable
SSH key: ~/.ssh/id_ed25519

Results backup: /tmp/tabarena_results/ (664 pkl files, 28 datasets, 27 complete)

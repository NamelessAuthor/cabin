#!/usr/bin/env python3
"""Set up all 12 redo pods in parallel: upload model, install deps, fix bug, launch."""
import subprocess
import threading
import time
import os

SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
MODEL_FILE = "/tmp/custom_v50_model_l4.py"

# (name, host, ssh port, TabArena task ids) of the pods you created
REDO_PODS = [
]

RUNNER_TEMPLATE = '''#!/usr/bin/env python3
from __future__ import annotations
import sys, os
sys.path.insert(0, "/workspace/tabarena_src/tabarena")
sys.path.insert(0, "/workspace")
from custom_v50_model import CustomV50Model
from tabarena.benchmark.experiment import AGModelBagExperiment, run_experiments_new
TABARENA_DIR = "/workspace/tabarena_out/v50_clf"
TASKS = {task_list}
if __name__ == "__main__":
    print(f"TabArena pod: {{len(TASKS)}} tasks: {{TASKS}}")
    model_experiments = [
        AGModelBagExperiment(
            name="CausalBoost_BAG_L1",
            model_cls=CustomV50Model,
            model_hyperparameters={{"n_models": 5}},
            num_bag_folds=8,
            time_limit=3600,
        ),
    ]
    run_experiments_new(
        output_dir=TABARENA_DIR,
        model_experiments=model_experiments,
        tasks=TASKS,
        repetitions_mode="TabArena",
    )
    print("DONE!")
'''

SETUP_SCRIPT = r'''#!/bin/bash
set -e
pip install -q 'ray>=2.43.0,<2.53.0' autogluon.tabular openml huggingface_hub pandas 2>&1 | tail -3
if [ ! -d /workspace/tabarena_src ]; then
    git clone https://github.com/autogluon/tabarena.git /workspace/tabarena_src 2>&1 | tail -2
fi
echo "SETUP_DONE"
'''

FIX_SCRIPT = '''#!/usr/bin/env python3
filepath = "/workspace/tabarena_src/tabarena/tabarena/benchmark/experiment/experiment_runner_api.py"
with open(filepath) as f:
    content = f.read()
content = content.replace(
    'metadata_task_ids = tasks_metadata["task_id"].astype(str).tolist()',
    'metadata_task_ids = tasks_metadata["task_id"].astype(str).reset_index(drop=True)'
)
content = content.replace(
    'assert t_id in metadata_task_ids',
    'assert t_id in metadata_task_ids.values'
)
with open(filepath, "w") as f:
    f.write(content)
print("FIXED")
'''

LAUNCH_SCRIPT = '''#!/bin/bash
kill $(pgrep -f run_tasks) 2>/dev/null
sleep 1
cd /workspace
nohup python3 run_tasks.py > /tmp/tabarena_run.log 2>&1 &
echo "PID=$!"
'''

def ssh_cmd(host, port, cmd, timeout=600):
    try:
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-p", str(port), "-i", SSH_KEY, f"root@{host}", cmd],
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout + result.stderr, result.returncode
    except Exception as e:
        return str(e), -1

def scp_to(host, port, local_path, remote_path):
    try:
        result = subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no", "-P", str(port),
             "-i", SSH_KEY, local_path, f"root@{host}:{remote_path}"],
            capture_output=True, text=True, timeout=120
        )
        return result.returncode == 0
    except:
        return False

def setup_pod(name, host, port, tasks, results):
    print(f"[{name}] Starting setup for tasks {tasks}...")

    # Wait for SSH
    for attempt in range(30):
        out, rc = ssh_cmd(host, port, "echo ok", timeout=10)
        if "ok" in out:
            break
        time.sleep(10)
        if attempt % 5 == 4:
            print(f"[{name}] Still waiting for SSH (attempt {attempt+1})...")
    else:
        results[name] = "SSH TIMEOUT"
        print(f"[{name}] SSH TIMEOUT")
        return

    print(f"[{name}] SSH connected, uploading files...")

    # Upload model
    if not scp_to(host, port, MODEL_FILE, "/workspace/custom_v50_model.py"):
        results[name] = "SCP MODEL FAILED"
        print(f"[{name}] SCP MODEL FAILED")
        return

    # Create and upload runner
    runner_code = RUNNER_TEMPLATE.format(task_list=tasks)
    local_runner = f"/tmp/runner_{name}.py"
    with open(local_runner, "w") as f:
        f.write(runner_code)
    scp_to(host, port, local_runner, "/workspace/run_tasks.py")

    # Create and upload scripts
    for content, fname in [(SETUP_SCRIPT, "setup.sh"), (FIX_SCRIPT, "fix_tabarena.py"), (LAUNCH_SCRIPT, "launch.sh")]:
        local = f"/tmp/{fname}_{name}"
        with open(local, "w") as f:
            f.write(content)
        scp_to(host, port, local, f"/workspace/{fname}")

    # Install deps
    print(f"[{name}] Installing dependencies...")
    out, rc = ssh_cmd(host, port, "bash /workspace/setup.sh", timeout=600)
    if "SETUP_DONE" not in out:
        results[name] = f"SETUP FAILED: {out[-200:]}"
        print(f"[{name}] SETUP FAILED: {out[-200:]}")
        return

    # Apply fix
    out, rc = ssh_cmd(host, port, "python3 /workspace/fix_tabarena.py", timeout=30)
    if "FIXED" not in out:
        print(f"[{name}] Fix warning: {out[:100]}")

    # Launch
    out, rc = ssh_cmd(host, port, "bash /workspace/launch.sh", timeout=30)
    results[name] = f"LAUNCHED: {out.strip()}"
    print(f"[{name}] LAUNCHED: {out.strip()}")

def main():
    print(f"Setting up {len(REDO_PODS)} redo pods in parallel...")
    results = {}
    threads = []

    for name, host, port, tasks in REDO_PODS:
        t = threading.Thread(target=setup_pod, args=(name, host, port, tasks, results))
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=900)

    print(f"\n{'='*60}")
    launched = sum(1 for v in results.values() if "LAUNCHED" in v)
    print(f"{launched}/{len(REDO_PODS)} pods launched successfully")
    for name, host, port, tasks in REDO_PODS:
        status = results.get(name, "UNKNOWN")
        print(f"  {name} [{tasks}]: {status}")

if __name__ == "__main__":
    main()

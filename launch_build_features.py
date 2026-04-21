"""Run locally to enqueue a feature extraction task on the remote ClearML agent."""

import subprocess
from clearml import Task

branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

task = Task.create(
    project_name="diagfhfl",
    task_name="Build WSI features (resnet50)",
    repo="https://github.com/psandersi/diagfhfl",
    branch=branch,
    commit=commit,
    script="wrapper_build_features.py",
    docker="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
    docker_args="--shm-size=8g",
    docker_bash_setup_script=(
        "apt-get update -qq && apt-get install -y -qq openslide-tools libgl1 && "
        "pip install -r requirements.txt torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    ),
)

task.set_parameters({
    "General/encoder": "resnet50",   # resnet50 | uni | conch
    "General/batch_size": 64,
    "General/hf_token": "",          # нужен для uni/conch, или передать через HF_TOKEN env
})

Task.enqueue(task, queue_name="default")
print(f"Task enqueued! ID: {task.id}")
print(f"Branch: {branch}, Commit: {commit}")

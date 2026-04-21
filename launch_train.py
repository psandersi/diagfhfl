"""Run locally to enqueue a training task on the remote ClearML agent."""

import subprocess
from clearml import Task

branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

task = Task.create(
    project_name="diagfhfl",
    task_name="Train LeNet WSI classifier",
    repo="https://github.com/psandersi/diagfhfl",
    branch=branch,
    commit=commit,
    script="wrapper_train.py",
    docker="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
    docker_args="--shm-size=16g",
    docker_bash_setup_script="pip install -r requirements.txt",
)

task.set_parameters({
    "General/level": 3,
    "General/imsize": 299,
    "General/maxpatches": 100,
    "General/train_ratio": 0.7,
    "General/val_ratio": 0.15,
    "General/epochs": 30,
    "General/batchsize": 128,
    "General/seed": 42,
})

Task.enqueue(task, queue_name="default")
print(f"Task enqueued! ID: {task.id}")
print(f"Branch: {branch}, Commit: {commit}")

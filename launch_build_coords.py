"""Run locally to enqueue a CLAM patching task on the remote ClearML agent."""

import os
import subprocess
from clearml import Task

branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
minio_access_key = os.environ.get("MINIO_ACCESS_KEY", "")
minio_secret_key = os.environ.get("MINIO_SECRET_KEY", "")

if not minio_access_key or not minio_secret_key:
    print("WARNING: local MINIO_ACCESS_KEY/MINIO_SECRET_KEY are empty.")
    print("The remote task will fail unless the ClearML agent already has these env variables.")

task = Task.create(
    project_name="diagfhfl",
    task_name="Build WSI coords (CLAM patching)",
    repo="https://github.com/psandersi/diagfhfl",
    branch=branch,
    commit=commit,
    script="wrapper_build_coords.py",
    docker="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
    docker_args=(
        "--shm-size=8g "
        "-e MINIO_ENDPOINT -e MINIO_BUCKET -e MINIO_PREFIX "
        "-e MINIO_ACCESS_KEY -e MINIO_SECRET_KEY"
    ),
    docker_bash_setup_script=(
        "apt-get update -qq && apt-get install -y -qq openslide-tools libgl1 git && "
        "pip install -r requirements.txt"
    ),
)

task.set_parameters({
    "General/patch_size": 256,
    "General/step_size": 256,
    "General/preset": "tcga.csv",
    "General/minio_endpoint": "https://api.blackhole2.ai.innopolis.university:443",
    "General/minio_bucket": "pershin-medailab",
    "General/minio_prefix": "Pathomorphology/CAMELYON/16/training",
    "General/minio_access_key": minio_access_key,
    "General/minio_secret_key": minio_secret_key,
    "General/verify_ssl": True,
    "General/max_slides_per_class": 0,
    "General/coords_output_prefix": "Pathomorphology/CAMELYON/processed/clam_coords",
    "General/overwrite_outputs": False,
})

Task.enqueue(task, queue_name="default")
print(f"Task enqueued! ID: {task.id}")
print(f"Branch: {branch}, Commit: {commit}")

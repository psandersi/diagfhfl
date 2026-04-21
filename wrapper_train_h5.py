"""Runs on the remote ClearML agent: downloads patch H5 files from MinIO and trains LeNet."""

import subprocess
import sys
from pathlib import Path

from minio_utils import download_key, get_minio_client, join_prefix, list_keys, minio_config


def download_patch_h5_dataset(client, bucket: str, patch_h5_prefix: str, local_root: Path) -> Path:
    total = 0
    for class_name in ("normal", "tumor"):
        class_prefix = join_prefix(patch_h5_prefix, class_name)
        keys = list_keys(client, bucket, class_prefix, suffix=".h5")
        print(f"Found {len(keys)} {class_name} patch H5 files in MinIO")
        for key in keys:
            download_key(client, bucket, key, local_root / class_name / Path(key).name)
            total += 1

    if total == 0:
        raise RuntimeError(f"No patch H5 files found under s3://{bucket}/{patch_h5_prefix}/")

    return local_root


def main() -> None:
    from clearml import Task

    task = Task.current_task()
    params = task.get_parameters_as_dict().get("General", {})
    epochs = int(params.get("epochs", 30))
    batchsize = int(params.get("batchsize", 128))
    imsize = int(params.get("imsize", 256))
    train_ratio = float(params.get("train_ratio", 0.7))
    val_ratio = float(params.get("val_ratio", 0.15))
    seed = int(params.get("seed", 42))
    patch_h5_input_prefix = str(
        params.get("patch_h5_input_prefix")
        or f"Pathomorphology/CAMELYON/processed/patch_h5_{imsize}"
    ).strip("/")

    config = minio_config(params)
    client = get_minio_client(config)
    print(f"MinIO endpoint: {config['endpoint']}")
    print(f"MinIO bucket: {config['bucket']}")
    print(f"Patch H5 input prefix: s3://{config['bucket']}/{patch_h5_input_prefix}/")

    patches_path = download_patch_h5_dataset(
        client=client,
        bucket=config["bucket"],
        patch_h5_prefix=patch_h5_input_prefix,
        local_root=Path("/tmp/train_patch_h5"),
    )

    output_dir = Path("/tmp/model_output")

    subprocess.run(
        [
            sys.executable,
            "train_lenet_h5.py",
            "--patches-dir",
            str(patches_path),
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(epochs),
            "--batchsize",
            str(batchsize),
            "--imsize",
            str(imsize),
            "--train-ratio",
            str(train_ratio),
            "--val-ratio",
            str(val_ratio),
            "--seed",
            str(seed),
        ],
        check=True,
    )

    task.upload_artifact("patch_h5_minio_prefix", f"s3://{config['bucket']}/{patch_h5_input_prefix}/")
    for artifact_file in sorted(output_dir.iterdir()):
        task.upload_artifact(artifact_file.stem, str(artifact_file))
        print(f"Uploaded artifact: {artifact_file.name}")


if __name__ == "__main__":
    main()

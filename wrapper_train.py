"""Runs on the remote ClearML agent: downloads TIF dataset, builds PNG patches, trains."""

import sys
import subprocess
from pathlib import Path


def main():
    from clearml import Dataset, Task

    task = Task.current_task()

    params = task.get_parameters_as_dict().get("General", {})
    level = int(params.get("level", 3))
    imsize = int(params.get("imsize", 299))
    maxpatches = int(params.get("maxpatches", 100))
    train_ratio = float(params.get("train_ratio", 0.7))
    val_ratio = float(params.get("val_ratio", 0.15))
    epochs = int(params.get("epochs", 30))
    batchsize = int(params.get("batchsize", 128))
    seed = int(params.get("seed", 42))

    ds = Dataset.get(dataset_project="diagfhfl", dataset_name="WSI_TIF_dataset")
    tif_path = ds.get_local_copy()
    print(f"TIF dataset downloaded to: {tif_path}")

    png_output = Path("/tmp/png_dataset")
    subprocess.run(
        [
            sys.executable, "build_png_dataset.py",
            "--input-dir", tif_path,
            "--output-dir", str(png_output),
            "--level", str(level),
            "--imsize", str(imsize),
            "--maxpatches", str(maxpatches),
            "--train-ratio", str(train_ratio),
            "--val-ratio", str(val_ratio),
            "--seed", str(seed),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable, "train_lenet_local.py",
            "--dataset", str(png_output),
            "--epochs", str(epochs),
            "--batchsize", str(batchsize),
            "--imsize", str(imsize),
        ],
        check=True,
    )

    model_dir = png_output / "Model" / "LenetLocal"
    for artifact_file in sorted(model_dir.iterdir()):
        task.upload_artifact(artifact_file.stem, str(artifact_file))
        print(f"Uploaded artifact: {artifact_file.name}")


if __name__ == "__main__":
    main()

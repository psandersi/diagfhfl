"""Runs on the remote ClearML agent: reads WSI from MinIO, runs CLAM, uploads coords."""

import sys
import subprocess
from pathlib import Path

from minio_utils import (
    bool_param,
    download_slide,
    get_minio_client,
    join_prefix,
    list_slide_keys,
    minio_config,
    object_exists,
    upload_file,
)


def clone_clam(target: Path) -> None:
    if not target.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/mahmoodlab/CLAM", str(target)],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(target / "requirements.txt")],
        check=True,
    )


def run_clam_patching(clam_dir: Path, source_dir: Path, save_dir: Path, patch_size: int, step_size: int, preset: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, str(clam_dir / "create_patches_fp.py"),
            "--source", str(source_dir),
            "--save_dir", str(save_dir),
            "--patch_size", str(patch_size),
            "--step_size", str(step_size),
            "--preset", preset,
            "--seg", "--patch", "--stitch",
        ],
        check=True,
    )


def main() -> None:
    from clearml import Task

    task = Task.current_task()
    params = task.get_parameters_as_dict().get("General", {})
    patch_size = int(params.get("patch_size", 256))
    step_size = int(params.get("step_size", 256))
    preset = str(params.get("preset", "tcga.csv"))
    max_slides_per_class = int(params.get("max_slides_per_class", 0) or 0)
    coords_output_prefix = str(
        params.get("coords_output_prefix") or "Pathomorphology/CAMELYON/processed/clam_coords"
    ).strip("/")
    overwrite_outputs = bool_param(params.get("overwrite_outputs"), False)

    config = minio_config(params)
    client = get_minio_client(config)
    print(f"MinIO endpoint: {config['endpoint']}")
    print(f"MinIO bucket: {config['bucket']}")
    print(f"MinIO training prefix: {config['prefix']}")
    print(f"Coords output prefix: s3://{config['bucket']}/{coords_output_prefix}/")

    clam_dir = Path("/tmp/CLAM")
    clone_clam(clam_dir)

    coords_root = Path("/tmp/coords")
    slides_root = Path("/tmp/wsi_slides")

    # CLAM expects a flat directory of slides; process each class separately
    for class_name in ("normal", "tumor"):
        class_prefix = join_prefix(config["prefix"], class_name)
        slide_keys = list_slide_keys(client, config["bucket"], class_prefix)
        if max_slides_per_class > 0:
            slide_keys = slide_keys[:max_slides_per_class]

        if not slide_keys:
            print(f"No WSI files found for {class_name} under s3://{config['bucket']}/{class_prefix}/")
            continue

        class_dir = slides_root / class_name
        print(f"Processing {len(slide_keys)} {class_name} slides from MinIO...")
        for index, key in enumerate(slide_keys, start=1):
            slide_name = Path(key).stem
            output_key = join_prefix(coords_output_prefix, class_name, f"{slide_name}.h5")
            if not overwrite_outputs and object_exists(client, config["bucket"], output_key):
                print(f"[SKIP] Existing coords: s3://{config['bucket']}/{output_key}")
                continue

            print(f"[{class_name}] slide {index}/{len(slide_keys)}: {key}")
            local_slide = download_slide(client, config["bucket"], key, class_dir)
            try:
                run_clam_patching(
                    clam_dir=clam_dir,
                    source_dir=class_dir,
                    save_dir=coords_root / class_name,
                    patch_size=patch_size,
                    step_size=step_size,
                    preset=preset,
                )
                local_coords = coords_root / class_name / "patches" / f"{slide_name}.h5"
                if not local_coords.exists():
                    raise FileNotFoundError(f"CLAM did not create expected coords file: {local_coords}")
                upload_file(client, config["bucket"], local_coords, output_key, overwrite=overwrite_outputs)
            finally:
                if local_slide.exists():
                    local_slide.unlink()
                    print(f"[CLEANUP] Deleted cached WSI: {local_slide}")

    task.upload_artifact("coords_minio_prefix", f"s3://{config['bucket']}/{coords_output_prefix}/")
    print(f"Coords saved to: s3://{config['bucket']}/{coords_output_prefix}/")


if __name__ == "__main__":
    main()

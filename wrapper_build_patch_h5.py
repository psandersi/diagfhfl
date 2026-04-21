"""Runs on the remote ClearML agent: downloads WSI from MinIO and writes patch H5 files."""

from pathlib import Path

import h5py
import numpy as np
from openslide import OpenSlide

from minio_utils import (
    bool_param,
    download_key,
    download_slide,
    get_minio_client,
    join_prefix,
    list_keys,
    list_slide_keys,
    minio_config,
    object_exists,
    slide_key_by_stem,
    upload_file,
)


def extract_patches_to_h5(wsi_path: Path, coords_h5_path: Path, output_h5_path: Path, patch_size: int) -> int:
    with h5py.File(coords_h5_path, "r") as f:
        coords = f["coords"][:]
        patch_level = int(f["coords"].attrs.get("patch_level", 0))

    slide = OpenSlide(str(wsi_path))
    n = len(coords)

    with h5py.File(output_h5_path, "w") as out:
        imgs = out.create_dataset(
            "imgs",
            shape=(n, patch_size, patch_size, 3),
            dtype="uint8",
            chunks=(1, patch_size, patch_size, 3),
            compression="lzf",
        )
        out.create_dataset("coords", data=coords)
        out["imgs"].attrs["patch_level"] = patch_level
        out["imgs"].attrs["patch_size"] = patch_size

        for i, (x, y) in enumerate(coords):
            region = slide.read_region((int(x), int(y)), patch_level, (patch_size, patch_size))
            imgs[i] = np.array(region)[:, :, :3]

    slide.close()
    return n


def main() -> None:
    from clearml import Task

    task = Task.current_task()
    params = task.get_parameters_as_dict().get("General", {})
    patch_size = int(params.get("patch_size", 256))
    coords_input_prefix = str(
        params.get("coords_input_prefix") or "Pathomorphology/CAMELYON/processed/clam_coords"
    ).strip("/")
    patch_h5_output_prefix = str(
        params.get("patch_h5_output_prefix")
        or f"Pathomorphology/CAMELYON/processed/patch_h5_{patch_size}"
    ).strip("/")
    overwrite_outputs = bool_param(params.get("overwrite_outputs"), False)

    config = minio_config(params)
    client = get_minio_client(config)
    print(f"MinIO endpoint: {config['endpoint']}")
    print(f"MinIO bucket: {config['bucket']}")
    print(f"MinIO training prefix: {config['prefix']}")
    print(f"Coords input prefix: s3://{config['bucket']}/{coords_input_prefix}/")
    print(f"Patch H5 output prefix: s3://{config['bucket']}/{patch_h5_output_prefix}/")

    output_root = Path("/tmp/patch_h5")
    coords_root = Path("/tmp/coords_from_minio")
    slide_cache_root = Path("/tmp/wsi_h5_cache")

    for class_name in ("normal", "tumor"):
        out_dir = output_root / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        coords_prefix = join_prefix(coords_input_prefix, class_name)
        coords_keys = list_keys(client, config["bucket"], coords_prefix, suffix=".h5")
        if not coords_keys:
            print(f"No coords found for class {class_name} under s3://{config['bucket']}/{coords_prefix}/")
            continue

        class_prefix = join_prefix(config["prefix"], class_name)
        keys_by_stem = slide_key_by_stem(list_slide_keys(client, config["bucket"], class_prefix))
        print(f"Found {len(keys_by_stem)} {class_name} WSI files in MinIO")

        for coords_key in coords_keys:
            slide_name = Path(coords_key).stem
            output_key = join_prefix(patch_h5_output_prefix, class_name, f"{slide_name}.h5")
            if not overwrite_outputs and object_exists(client, config["bucket"], output_key):
                print(f"[SKIP] Existing patch H5: s3://{config['bucket']}/{output_key}")
                continue

            slide_key = keys_by_stem.get(slide_name)
            if slide_key is None:
                print(f"WSI not found in MinIO for {class_name}/{slide_name}, skipping")
                continue

            coords_h5 = download_key(
                client,
                config["bucket"],
                coords_key,
                coords_root / class_name / f"{slide_name}.h5",
            )
            out_h5 = out_dir / f"{slide_name}.h5"
            if out_h5.exists() and out_h5.stat().st_size > 0:
                out_h5.unlink()

            local_slide = download_slide(client, config["bucket"], slide_key, slide_cache_root / class_name)
            try:
                print(f"Extracting {class_name}/{slide_name}...")
                n = extract_patches_to_h5(local_slide, coords_h5, out_h5, patch_size)
                print(f"  -> {n} patches saved")
                upload_file(client, config["bucket"], out_h5, output_key, overwrite=overwrite_outputs)
            finally:
                if local_slide.exists():
                    local_slide.unlink()
                    print(f"[CLEANUP] Deleted cached WSI: {local_slide}")

    task.upload_artifact("patch_h5_minio_prefix", f"s3://{config['bucket']}/{patch_h5_output_prefix}/")
    print(f"Patch H5 files saved to: s3://{config['bucket']}/{patch_h5_output_prefix}/")


if __name__ == "__main__":
    main()

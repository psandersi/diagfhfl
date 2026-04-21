"""Runs on the remote ClearML agent: extracts patch features via pretrained encoder, saves H5 per slide."""

import sys
import subprocess
from pathlib import Path

import h5py
import numpy as np
from openslide import OpenSlide


ENCODERS = ("resnet50", "uni", "conch")


def load_encoder(encoder_name: str, hf_token: str | None):
    import torch
    import torchvision.transforms as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if encoder_name == "resnet50":
        import torchvision.models as models
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        feature_dim = 2048

    elif encoder_name == "uni":
        import timm
        from huggingface_hub import login
        if hf_token:
            login(hf_token)
        model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        feature_dim = 1024

    elif encoder_name == "conch":
        from conch.open_clip_custom import create_model_from_pretrained
        from huggingface_hub import login
        if hf_token:
            login(hf_token)
        model, transform = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
        model = model.visual
        feature_dim = 512

    else:
        raise ValueError(f"Unknown encoder: {encoder_name}. Choose from {ENCODERS}")

    model.eval().to(device)
    return model, transform, device, feature_dim


def extract_features_to_h5(
    wsi_path: Path,
    coords_h5_path: Path,
    output_h5_path: Path,
    model,
    transform,
    device,
    feature_dim: int,
    batch_size: int,
) -> int:
    import torch
    from PIL import Image

    with h5py.File(coords_h5_path, "r") as f:
        coords = f["coords"][:]
        patch_level = int(f["coords"].attrs.get("patch_level", 0))
        patch_size = int(f["coords"].attrs.get("patch_size", 256))

    slide = OpenSlide(str(wsi_path))
    n = len(coords)

    with h5py.File(output_h5_path, "w") as out:
        feats_ds = out.create_dataset("features", shape=(n, feature_dim), dtype="float32")
        out.create_dataset("coords", data=coords)
        out["features"].attrs["encoder"] = str(model.__class__.__name__)
        out["features"].attrs["feature_dim"] = feature_dim

        for start in range(0, n, batch_size):
            batch_coords = coords[start: start + batch_size]
            imgs = []
            for x, y in batch_coords:
                region = slide.read_region((int(x), int(y)), patch_level, (patch_size, patch_size))
                imgs.append(transform(region.convert("RGB")))

            batch = torch.stack(imgs).to(device)
            with torch.no_grad():
                features = model(batch).cpu().numpy()
            feats_ds[start: start + len(batch_coords)] = features

    slide.close()
    return n


def main() -> None:
    from clearml import Dataset, Task

    task = Task.current_task()
    params = task.get_parameters_as_dict().get("General", {})
    encoder_name = str(params.get("encoder", "resnet50"))
    batch_size = int(params.get("batch_size", 64))
    hf_token = str(params.get("hf_token", "")) or None

    import os
    hf_token = hf_token or os.environ.get("HF_TOKEN")

    if encoder_name in ("uni", "conch"):
        subprocess.run([sys.executable, "-m", "pip", "install", "timm", "huggingface_hub"], check=True)
    if encoder_name == "conch":
        subprocess.run([sys.executable, "-m", "pip", "install", "conch"], check=True)

    wsi_ds = Dataset.get(dataset_project="diagfhfl", dataset_name="WSI_TIF_dataset")
    wsi_path = Path(wsi_ds.get_local_copy())

    coords_ds = Dataset.get(dataset_project="diagfhfl", dataset_name="WSI_coords")
    coords_path = Path(coords_ds.get_local_copy())

    model, transform, device, feature_dim = load_encoder(encoder_name, hf_token)
    print(f"Encoder: {encoder_name}, feature_dim: {feature_dim}, device: {device}")

    output_root = Path("/tmp/features")
    dataset_name = f"WSI_features_{encoder_name}"

    for class_name in ("normal", "tumor"):
        coords_dir = coords_path / class_name / "patches"
        wsi_dir = wsi_path / class_name
        out_dir = output_root / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if not coords_dir.exists():
            print(f"No coords for class {class_name}, skipping")
            continue

        for coords_h5 in sorted(coords_dir.glob("*.h5")):
            slide_name = coords_h5.stem
            wsi_file = next(
                (wsi_dir / f"{slide_name}{ext}" for ext in (".tif", ".tiff", ".svs", ".ndpi")
                 if (wsi_dir / f"{slide_name}{ext}").exists()),
                None,
            )
            if wsi_file is None:
                print(f"WSI not found for {slide_name}, skipping")
                continue

            out_h5 = out_dir / f"{slide_name}.h5"
            print(f"Extracting features: {class_name}/{slide_name}...")
            n = extract_features_to_h5(wsi_file, coords_h5, out_h5, model, transform, device, feature_dim, batch_size)
            print(f"  → {n} feature vectors saved")

    feat_ds = Dataset.create(
        dataset_project="diagfhfl",
        dataset_name=dataset_name,
        parent_datasets=[wsi_ds.id, coords_ds.id],
    )
    feat_ds.add_files(str(output_root))
    feat_ds.upload(show_progress=True)
    feat_ds.finalize()
    print(f"{dataset_name} dataset ID: {feat_ds.id}")
    task.upload_artifact("features_dataset_id", feat_ds.id)


if __name__ == "__main__":
    main()

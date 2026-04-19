# Local WSI PNG Pipeline

This folder contains a fully local training pipeline for a dataset with the following layout:

```text
dataset/
├── normal/
│   ├── slide_001.tif
│   └── ...
└── tumor/
    ├── slide_001.tif
    └── ...
```

The pipeline does two steps:

1. split whole-slide TIFF files into `Training / Validation / Test`
2. extract PNG patches and train a local LeNet-style classifier

## Files

- `build_png_dataset.py`: split local WSI and create PNG patches
- `train_lenet_local.py`: train locally on PNG patches
- `requirements.txt`: local dependencies

## Install

```bash
pip install -r local_wsi_png_pipeline/requirements.txt
```

## Step 1: Build PNG dataset

```bash
python local_wsi_png_pipeline/build_png_dataset.py \
  --input-dir "C:/path/to/dataset" \
  --output-dir "C:/path/to/output_png_dataset" \
  --level 3 \
  --imsize 299 \
  --maxpatches 100 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --seed 42
```

This creates:

```text
output_png_dataset/
├── Training/
│   ├── normal/
│   └── tumor/
├── Validation/
│   ├── normal/
│   └── tumor/
├── Test/
│   ├── normal/
│   └── tumor/
└── Model/
```

## Step 2: Train locally

```bash
python local_wsi_png_pipeline/train_lenet_local.py \
  --dataset "C:/path/to/output_png_dataset" \
  --epochs 1 \
  --batchsize 4 \
  --imsize 299
```

The model output is saved to:

```text
output_png_dataset/Model/LenetLocal/
```

## Smoke test

For a quick smoke test, start with:

```bash
python local_wsi_png_pipeline/build_png_dataset.py \
  --input-dir "C:/path/to/dataset" \
  --output-dir "C:/path/to/output_png_dataset_smoke" \
  --level 3 \
  --imsize 299 \
  --maxpatches 20 \
  --max-slides-per-class 2

python local_wsi_png_pipeline/train_lenet_local.py \
  --dataset "C:/path/to/output_png_dataset_smoke" \
  --epochs 1 \
  --batchsize 4 \
  --imsize 299
```

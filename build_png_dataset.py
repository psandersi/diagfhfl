"""Build a local PNG patch dataset from dataset/{normal,tumor} whole-slide files."""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import numpy as np
from openslide import OpenSlide
from skimage.exposure import is_low_contrast
from skimage.io import imsave
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".svs", ".ndpi"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split local WSI files into Training/Validation/Test and generate PNG patches.",
    )
    parser.add_argument("--input-dir", required=True, help="Dataset root containing normal/ and tumor/ slide folders.")
    parser.add_argument("--output-dir", required=True, help="Output root for Training/Validation/Test patch dataset.")
    parser.add_argument("--level", type=int, default=3, help="OpenSlide pyramid level. Default: 3")
    parser.add_argument("--imsize", type=int, default=299, help="Patch size in pixels. Default: 299")
    parser.add_argument("--maxpatches", type=int, default=100, help="Maximum patches per slide. Default: 100")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio. Default: 0.7")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio. Default: 0.15")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument(
        "--max-slides-per-class",
        type=int,
        default=0,
        help="Limit slides per class per run. Default: 0 means no limit.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip patch files that already exist.",
    )
    return parser


def validate_ratios(train_ratio: float, val_ratio: float) -> None:
    test_ratio = 1.0 - train_ratio - val_ratio
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be > 0, and train_ratio + val_ratio must be < 1.0")


def get_tissue(image: np.ndarray, blacktol: int = 0, whitetol: int = 230) -> np.ndarray:
    binarymask = np.ones_like(image[:, :, 0], dtype=bool)
    for color in range(3):
        binarymask = np.logical_and(binarymask, image[:, :, color] < whitetol)
        binarymask = np.logical_and(binarymask, image[:, :, color] > blacktol)
    return binarymask


def randomized_regular_seed(shape: tuple[int, int], width: int, randomizer: random.Random):
    maxi = width * int(shape[1] / width)
    maxj = width * int(shape[0] / width)
    cols = list(range(0, maxj, width))
    rows = list(range(0, maxi, width))
    randomizer.shuffle(cols)
    randomizer.shuffle(rows)
    for point in itertools.product(rows, cols):
        yield point


def randomized_patches(slide: OpenSlide, level: int, patchsize: int, randomizer: random.Random, n_max: int):
    counter = 0
    width, height = slide.level_dimensions[level]
    for i, j in randomized_regular_seed((width, height), patchsize, randomizer):
        if counter >= n_max:
            break
        x = j * (2 ** level)
        y = i * (2 ** level)
        image = np.array(slide.read_region((x, y), level, (patchsize, patchsize)))[:, :, 0:3]
        if (not is_low_contrast(image)) and (get_tissue(image).sum() > 0.5 * patchsize * patchsize):
            counter += 1
            yield x, y, level, image


def list_slides(class_dir: Path) -> list[Path]:
    if not class_dir.exists():
        raise FileNotFoundError(f"Class directory not found: {class_dir}")
    slides = [path for path in sorted(class_dir.iterdir()) if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not slides:
        raise ValueError(f"No supported slide files found in {class_dir}")
    return slides


def split_slides(slides: list[Path], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    if len(slides) < 3:
        raise ValueError(f"Need at least 3 slides for split, got {len(slides)}")
    shuffled = list(slides)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    if train_count <= 0 or val_count <= 0 or test_count <= 0:
        raise ValueError("Current split settings produce an empty subset.")
    return {
        "Training": shuffled[:train_count],
        "Validation": shuffled[train_count:train_count + val_count],
        "Test": shuffled[train_count + val_count:],
    }


def ensure_output_layout(output_dir: Path) -> None:
    for subset in ("Training", "Validation", "Test"):
        for class_name in ("normal", "tumor"):
            (output_dir / subset / class_name).mkdir(parents=True, exist_ok=True)
    (output_dir / "Model").mkdir(parents=True, exist_ok=True)


def build_slide_plan(input_dir: Path, train_ratio: float, val_ratio: float, seed: int, max_slides_per_class: int) -> dict[str, dict[str, list[Path]]]:
    plan: dict[str, dict[str, list[Path]]] = {}
    for class_name in ("normal", "tumor"):
        slides = list_slides(input_dir / class_name)
        splits = split_slides(slides, train_ratio, val_ratio, seed)
        if max_slides_per_class > 0:
            splits = {subset: items[:max_slides_per_class] for subset, items in splits.items()}
        plan[class_name] = splits
    return plan


def save_slide_patches(slide_path: Path, subset_dir: Path, class_name: str, level: int, imsize: int, maxpatches: int, seed: int, skip_existing: bool) -> int:
    randomizer = random.Random(seed)
    slide = OpenSlide(str(slide_path))
    slide_name = slide_path.stem
    patch_count = 0
    try:
        for x, y, patch_level, image in randomized_patches(slide, level, imsize, randomizer, maxpatches):
            output_path = subset_dir / class_name / f"{slide_name}_{x}_{y}_{patch_level}.png"
            if skip_existing and output_path.exists():
                continue
            imsave(str(output_path), image)
            patch_count += 1
    finally:
        slide.close()
    return patch_count


def main() -> int:
    args = build_parser().parse_args()
    validate_ratios(args.train_ratio, args.val_ratio)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_layout(output_dir)

    plan = build_slide_plan(input_dir, args.train_ratio, args.val_ratio, args.seed, args.max_slides_per_class)

    for class_name, class_splits in plan.items():
        for subset_name, slides in class_splits.items():
            print(f"Generating {subset_name} patches for {class_name}: {len(slides)} slides")
            for slide_path in tqdm(slides):
                patch_count = save_slide_patches(
                    slide_path=slide_path,
                    subset_dir=output_dir / subset_name,
                    class_name=class_name,
                    level=args.level,
                    imsize=args.imsize,
                    maxpatches=args.maxpatches,
                    seed=args.seed,
                    skip_existing=args.skip_existing,
                )
                print(f"Processed {slide_path.name} -> {patch_count} patches")

    print(f"PNG dataset created at: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# coding: utf8

import argparse
import json
import math
import pickle
import random
from pathlib import Path

import h5py
import numpy as np
from tensorflow.keras.applications.xception import preprocess_input as preproc_xce
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import Sequence, to_categorical


CLASS_NAMES = ["normal", "tumor"]


# ── model ─────────────────────────────────────────────────────────────────────

class Brick:
    def __init__(self, brickname):
        self.name = brickname
        self.ops = []
        self.trainable_weights = []

    def weights(self):
        w = []
        for op in self.ops:
            w += op.trainable_weights
        return w

    def __call__(self, arg_tensor):
        y = self.ops[0](arg_tensor)
        for op in self.ops[1:]:
            y = op(y)
        if not self.trainable_weights:
            self.trainable_weights = self.weights()
        return y


class Classifier(Brick):
    def __init__(
        self,
        brickname="classifier",
        filters=[32, 64, 128],
        kernels=[4, 5, 6],
        strides=[1, 1, 1],
        dropouts=[0.0, 0.0, 0.0],
        fc=[1024, 1024],
        fcdropouts=[0.5, 0.5],
        conv_activations=["relu", "relu", "relu"],
        fc_activations=["relu", "relu"],
        end_activation="softmax",
        output_channels=2,
    ):
        super().__init__(brickname)
        for depth in range(len(filters)):
            self.ops.append(Conv2D(filters[depth], kernels[depth], strides=(strides[depth], strides[depth]),
                                   activation=conv_activations[depth], padding="same", name=f"convolution_{depth}"))
            self.ops.append(MaxPooling2D(pool_size=(2, 2), name=f"pool_{depth}"))
            self.ops.append(Dropout(rate=dropouts[depth], name=f"dropout_{depth}"))
        self.ops.append(Flatten())
        for depth in range(len(fc)):
            self.ops.append(Dense(fc[depth], activation=fc_activations[depth], name=f"fc_{depth}"))
            self.ops.append(Dropout(fcdropouts[depth], name=f"fc_dropout_{depth}"))
        self.ops.append(Dense(output_channels, activation=end_activation, name="final_fc"))


# ── data loading ──────────────────────────────────────────────────────────────

class PatchH5Sequence(Sequence):
    """Keras Sequence that reads patches on-the-fly from per-slide H5 files."""

    def __init__(self, slides: list[tuple[Path, int]], batch_size: int, img_size: int, augment: bool = False, shuffle: bool = True):
        # slides: list of (h5_path, class_idx)
        # Build flat index: [(h5_path, patch_idx_within_file, class_idx), ...]
        self.index: list[tuple[Path, int, int]] = []
        for h5_path, class_idx in slides:
            with h5py.File(h5_path, "r") as f:
                n = f["imgs"].shape[0]
            self.index.extend((h5_path, i, class_idx) for i in range(n))

        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.num_classes = len(CLASS_NAMES)

        if shuffle:
            random.shuffle(self.index)

    def __len__(self) -> int:
        return math.ceil(len(self.index) / self.batch_size)

    def __getitem__(self, idx: int):
        batch = self.index[idx * self.batch_size: (idx + 1) * self.batch_size]

        # Group by file — minimise open/close calls; sort indices for H5 fancy indexing
        by_file: dict[Path, list[tuple[int, int]]] = {}
        for h5_path, patch_idx, class_idx in batch:
            by_file.setdefault(h5_path, []).append((patch_idx, class_idx))

        imgs_out, labels_out = [], []
        for h5_path, items in by_file.items():
            sorted_items = sorted(items, key=lambda t: t[0])
            sorted_patch_indices = [t[0] for t in sorted_items]
            sorted_class_indices = [t[1] for t in sorted_items]

            with h5py.File(h5_path, "r") as f:
                patches = f["imgs"][sorted_patch_indices]  # (k, H, W, 3) uint8

            for patch, class_idx in zip(patches, sorted_class_indices):
                if patch.shape[:2] != (self.img_size, self.img_size):
                    import cv2
                    patch = cv2.resize(patch, (self.img_size, self.img_size))
                if self.augment:
                    if random.random() > 0.5:
                        patch = np.fliplr(patch)
                    if random.random() > 0.5:
                        patch = np.flipud(patch)
                imgs_out.append(patch)
                labels_out.append(class_idx)

        X = preproc_xce(np.array(imgs_out, dtype=np.float32))
        y = to_categorical(labels_out, num_classes=self.num_classes)
        return X, y

    def on_epoch_end(self) -> None:
        if self.shuffle:
            random.shuffle(self.index)


# ── dataset split ─────────────────────────────────────────────────────────────

def build_slide_splits(patches_dir: Path, train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[tuple[Path, int]]]:
    """Split slides into train/val/test at slide level to avoid data leakage."""
    splits: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = patches_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        slides = sorted(class_dir.glob("*.h5"))
        if len(slides) < 3:
            raise ValueError(f"Need at least 3 slides per class for split, got {len(slides)} in {class_dir}")

        shuffled = list(slides)
        random.Random(seed).shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits["train"].extend((p, class_idx) for p in shuffled[:n_train])
        splits["val"].extend((p, class_idx) for p in shuffled[n_train: n_train + n_val])
        splits["test"].extend((p, class_idx) for p in shuffled[n_train + n_val:])

    return splits


# ── metrics ───────────────────────────────────────────────────────────────────

def save_metrics(output_dir: Path, history_dict: dict, test_loss: float, test_accuracy: float) -> None:
    metrics = {"history": history_dict, "test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches-dir", required=True, help="Path to WSI_patches_h5 root (contains normal/ and tumor/).")
    parser.add_argument("--output-dir", required=True, help="Where to save model weights and metrics.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--imsize", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    patches_dir = Path(args.patches_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = build_slide_splits(patches_dir, args.train_ratio, args.val_ratio, args.seed)

    train_seq = PatchH5Sequence(splits["train"], args.batchsize, args.imsize, augment=True, shuffle=True)
    val_seq = PatchH5Sequence(splits["val"], args.batchsize, args.imsize, augment=False, shuffle=False)
    test_seq = PatchH5Sequence(splits["test"], args.batchsize, args.imsize, augment=False, shuffle=False)

    print(f"Train patches: {len(train_seq.index)}, Val: {len(val_seq.index)}, Test: {len(test_seq.index)}")

    arch = Classifier(output_channels=len(CLASS_NAMES))
    input_tensor = Input(shape=(args.imsize, args.imsize, 3))
    model = Model(inputs=input_tensor, outputs=arch(input_tensor))
    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])

    print("Training model")
    history = model.fit(train_seq, epochs=args.epochs, validation_data=val_seq, verbose=1)
    test_loss, test_accuracy = model.evaluate(test_seq, verbose=1)

    with open(output_dir / "lenet_model.json", "w", encoding="utf-8") as f:
        f.write(model.to_json())
    with open(output_dir / "lenet_history.p", "wb") as f:
        pickle.dump(history.history, f)
    model.save_weights(output_dir / "lenet.weights.h5")
    save_metrics(output_dir, history.history, test_loss, test_accuracy)

    print(f"Saved model to: {output_dir}")
    print(f"Test loss: {test_loss:.6f}  Test accuracy: {test_accuracy:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

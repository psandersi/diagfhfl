# coding: utf8

import argparse
import errno
import json
import os
import pickle
from pathlib import Path

from tensorflow.keras.applications.xception import preprocess_input as preproc_xce
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Brick:
    def __init__(self, brickname):
        self.name = brickname
        self.ops = []
        self.trainable_weights = []

    def weights(self):
        weights = []
        for op in self.ops:
            weights += op.trainable_weights
        return weights

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
            self.ops.append(
                Conv2D(
                    filters=filters[depth],
                    kernel_size=kernels[depth],
                    strides=(strides[depth], strides[depth]),
                    activation=conv_activations[depth],
                    padding="same",
                    name=f"convolution_{depth}",
                )
            )
            self.ops.append(MaxPooling2D(pool_size=(2, 2), name=f"pool_{depth}"))
            self.ops.append(Dropout(rate=dropouts[depth], name=f"dropout_{depth}"))

        self.ops.append(Flatten())

        for depth in range(len(fc)):
            self.ops.append(Dense(fc[depth], activation=fc_activations[depth], name=f"fc_{depth}"))
            self.ops.append(Dropout(fcdropouts[depth], name=f"fc_dropout_{depth}"))

        self.ops.append(Dense(output_channels, activation=end_activation, name="final_fc"))


def validate_dataset_layout(dataset_dir: Path) -> tuple[Path, Path, Path, Path]:
    train_dir = dataset_dir / "Training"
    validation_dir = dataset_dir / "Validation"
    test_dir = dataset_dir / "Test"
    model_dir = dataset_dir / "Model"

    for path in (dataset_dir, train_dir, validation_dir, test_dir):
        if not path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

    if not any(train_dir.rglob("*.*")):
        raise ValueError(f"Training directory is empty: {train_dir}")
    if not any(validation_dir.rglob("*.*")):
        raise ValueError(f"Validation directory is empty: {validation_dir}")
    if not any(test_dir.rglob("*.*")):
        raise ValueError(f"Test directory is empty: {test_dir}")

    return train_dir, validation_dir, test_dir, model_dir


def save_metrics(output_dir: Path, history_dict: dict, test_loss: float, test_accuracy: float) -> None:
    metrics = {
        "history": history_dict,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to PNG dataset directory.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training the model.")
    parser.add_argument("--batchsize", type=int, default=128, help="Number of samples in one batch for fitting.")
    parser.add_argument("--imsize", type=int, default=299, help="Side size of an image in pixels.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    train_dir, validation_dir, test_dir, model_dir = validate_dataset_layout(dataset_dir)

    model_output_dir = model_dir / "LenetLocal"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    train_datagen = ImageDataGenerator(preprocessing_function=preproc_xce, horizontal_flip=True, vertical_flip=True)
    valid_datagen = ImageDataGenerator(preprocessing_function=preproc_xce)
    test_datagen = ImageDataGenerator(preprocessing_function=preproc_xce)

    train_generator = train_datagen.flow_from_directory(str(train_dir), target_size=(args.imsize, args.imsize), batch_size=args.batchsize)
    valid_generator = valid_datagen.flow_from_directory(str(validation_dir), target_size=(args.imsize, args.imsize), batch_size=args.batchsize)
    test_generator = test_datagen.flow_from_directory(str(test_dir), target_size=(args.imsize, args.imsize), batch_size=args.batchsize, shuffle=False)

    if train_generator.samples == 0 or valid_generator.samples == 0 or test_generator.samples == 0:
        raise ValueError("Training, Validation, and Test generators must all contain at least one image.")

    base_archi = Classifier(output_channels=train_generator.num_classes)
    input_tensor = Input(shape=(args.imsize, args.imsize, 3))
    predictions = base_archi(input_tensor)
    model = Model(inputs=input_tensor, outputs=predictions)
    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])

    print("Training model")
    history = model.fit(train_generator, epochs=args.epochs, validation_data=valid_generator, verbose=1)
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

    with open(model_output_dir / "lenet_model.json", "w", encoding="utf-8") as json_file:
        json_file.write(model.to_json())
    with open(model_output_dir / "lenet_history.p", "wb") as pickle_file:
        pickle.dump(history.history, pickle_file)
    model.save_weights(model_output_dir / "lenet.weights.h5")
    save_metrics(model_output_dir, history.history, test_loss, test_accuracy)

    print(f"Saved model to disk: {model_output_dir}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_accuracy:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

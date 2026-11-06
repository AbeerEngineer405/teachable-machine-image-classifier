"""
Classify an image with the Teachable Machine Keras model.

Run this script from the project folder:
    python predict.py path-to-image.jpg
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tf_keras.models import load_model


# Use the files beside this script, regardless of the current working folder.
PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_FOLDER / "keras_model.h5"
LABELS_PATH = PROJECT_FOLDER / "labels.txt"


def load_class_names():
    """Read labels and remove the numeric class index at the start of each line."""
    class_names = []

    with LABELS_PATH.open("r", encoding="utf-8") as labels_file:
        for line in labels_file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                class_names.append(parts[1].strip())

    if not class_names:
        raise ValueError(f"No class names were found in {LABELS_PATH}.")

    return class_names


def friendly_class_name(label):
    """Convert labels such as 'Cat1' and 'dog 2' to the requested display names."""
    normalized_label = label.casefold()
    if normalized_label.startswith("cat"):
        return "Cat"
    if normalized_label.startswith("dog"):
        return "Dog"
    return label


def main():
    # Read the image path from the command line.
    parser = argparse.ArgumentParser(
        description="Predict whether an image contains a cat or a dog."
    )
    parser.add_argument("image", help="Path to the image file to classify")
    args = parser.parse_args()
    image_path = Path(args.image)

    if not image_path.is_file():
        parser.error(f"Image file not found: {image_path}")
    if not MODEL_PATH.is_file():
        parser.error(f"Keras model not found: {MODEL_PATH}")
    if not LABELS_PATH.is_file():
        parser.error(f"Labels file not found: {LABELS_PATH}")

    # Legacy Keras supports the layer settings saved by Teachable Machine H5 models.
    # Compiling is unnecessary because this script only runs predictions.
    model = load_model(MODEL_PATH, compile=False)
    class_names = load_class_names()

    # Convert the image to RGB, resize to the model's expected size, and make an array.
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((224, 224))
        image_array = np.asarray(image, dtype=np.float32)

    # Teachable Machine image models expect pixel values normalized from [0, 255]
    # to [-1, 1].
    image_array = (image_array / 127.5) - 1.0
    image_array = np.expand_dims(image_array, axis=0)

    # Run the model and find the class with the highest score.
    predictions = model.predict(image_array, verbose=0)[0]
    if len(predictions) != len(class_names):
        raise ValueError(
            "The number of model outputs does not match the number of labels "
            f"({len(predictions)} outputs, {len(class_names)} labels)."
        )

    predicted_index = int(np.argmax(predictions))
    predicted_label = class_names[predicted_index]
    predicted_class = friendly_class_name(predicted_label)
    confidence = float(predictions[predicted_index]) * 100

    # Display the friendly class, original label, and confidence percentage.
    print(f"Predicted class: {predicted_class} (model label: {predicted_label})")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()

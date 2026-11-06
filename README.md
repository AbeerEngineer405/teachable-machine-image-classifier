# Cat and Dog Image Classifier

This project uses an image classification model trained with Google Teachable Machine. The model was exported in TensorFlow Keras format and predicts whether an input image belongs to one of two classes: **Cat** or **Dog**.

## Project files

| File | Description |
| --- | --- |
| `keras_model.h5` | Trained Keras model exported from Teachable Machine |
| `labels.txt` | Class labels used by the model |
| `predict.py` | Python script for classifying an input image |
| `test_cat.jpg` | Example image for testing the prediction script |
| `output_screenshot.png` | Screenshot showing example prediction output |

## Training and exporting the model

1. Open [Google Teachable Machine](https://teachablemachine.withgoogle.com/) and create an **Image Project**.
2. Create two classes named **Cat** and **Dog**.
3. Add representative cat images to the Cat class and dog images to the Dog class. Use a range of examples for more reliable predictions.
4. Train the model and check its performance with images that were not used for training.
5. Select **Export Model**, choose the **TensorFlow** tab and **Keras** format, then download the model.
6. Place the exported `keras_model.h5` and `labels.txt` in the same folder as `predict.py`.

## Requirements

Use Python and install the required packages:

```bash
python -m pip install tensorflow==2.21.0 tf-keras==2.21.0 pillow numpy
```

`tf-keras` is used to load the Teachable Machine H5 model with its saved layer configuration. The prediction script uses it alongside TensorFlow.

## Run a prediction

From the project folder, run the script with the path to an image:

```bash
python predict.py test_cat.jpg
```

You can replace `test_cat.jpg` with another image path. The script resizes the image to 224 × 224 pixels, applies the expected normalization, and prints the predicted class and confidence percentage.

Example output:

```text
Predicted class: Dog (model label: dog 2)
Confidence: 88.69%
```

The model was tested successfully. The example output is also shown in [`output_screenshot.png`](output_screenshot.png).

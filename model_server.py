import tensorflow as tf
import numpy as np
import flask
import io
from skimage.io import imread
from skimage.filters import threshold_yen
from skimage.morphology import closing, square
from skimage.measure import label as sk_label
from skimage.measure import regionprops
from skimage.transform import resize
from PIL import Image

app = flask.Flask(__name__)
model = None


def load_model(model_path):
    global model
    model = tf.keras.models.load_model(model_path)
    global graph
    graph = tf.get_default_graph()


def skimage_cropping(img, target_height, target_width):
    # First crop
    image_crop = img[160:160 + 880, 580:580 + 820]

    # Masking/closing pixel regions and labeling pixels
    thresh = threshold_yen(image_crop)
    img_closing = closing(image_crop > thresh, square(3))
    img_label = sk_label(img_closing)

    # Search for biggest area and extract centroid
    max_area = 0
    for region in regionprops(img_label):
        if region.area > max_area:
            max_area = region.area
            biggest = region
    center = biggest.centroid

    # Draw square bounding box around centroid
    square_side = 300
    step = square_side / 2
    min_row, max_row, min_col, max_col = max([0, int(center[0] - step)]), \
                                         int(center[0] + step), \
                                         max([0, int(center[1] - step)]), \
                                         int(center[1] + step)

    # Crop and resize image to square bounding box
    image_square = image_crop[min_row:max_row, min_col:max_col]
    image_resize = resize(image_square, [target_height, target_width], preserve_range=True).astype(np.uint8)

    return image_resize.reshape((1, 200, 200, 1))


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = imread(io.BytesIO(image))
            # image = Image.open(io.BytesIO(image))

            image = skimage_cropping(image, 200, 200)
            with graph.as_default():
                prediction_proba = float(model.predict(image))
                if prediction_proba > 0.5:
                    prediction = 'bad'
                else:
                    prediction = 'good'
                    prediction_proba = 1.0 - prediction_proba  # Invert to get positive prediction probability

                data["predictions"] = {"label": prediction,
                                       "probability": prediction_proba}

                data["success"] = True

    return flask.jsonify(data)


if __name__ == '__main__':
    print("--> Loading Keras Model and starting server")
    load_model('keras_model.h5')
    app.run()



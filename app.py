from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template
import io
import pickle
import tensorflow as tf

app = Flask(__name__)
model = None

def load_model():
    global model
    model = tf.keras.models.load_model('varroa_model.pb')

def prepare_image(image, target):
	if image.mode != "RGB":
		image = image.convert("RGB")
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)
	return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    image = flask.request.files["image"].read()
    image = Image.open(io.BytesIO(image))
    image = prepare_image(image, target=(224, 224))

    return jsonify(model.predict([image]))


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	load_model()
	app.run()
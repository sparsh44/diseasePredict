from flask import Flask, request, json,jsonify
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras import models, layers
import numpy as np
from flask_cors import CORS,cross_origin
app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

IMAGE_SIZE = 256
MODEL = tf.keras.models.load_model("../model/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

headers = {}
def get_class(image):
    new_image = image.resize((256, 256))
    img_arr = tf.keras.preprocessing.image.img_to_array(new_image)
    img_arr = tf.expand_dims(img_arr, 0)
    predictions = MODEL.predict(img_arr)
    print(CLASS_NAMES[np.argmax(predictions[0])])
    predicted_class =CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])*100
    print("Confidence" , np.max(predictions[0])*100)
    return predicted_class, confidence


@app.route('/')
def main():
    return "hi"

@app.route('/predict', methods = ['POST'])
def predict():
    file = request.files['file']
    print("IN GET CLASS")
    img = Image.open(file)
    print("............................")
    if file.filename == '':
            print('No selected file')
    predicted_class, confidence = get_class(img)
    response = jsonify({
        'predicted_class': predicted_class,
        'confidence' : confidence
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return json.loads(response.get_data().decode("utf-8"))

if __name__=='__main__':
    app.run(debug=True, port ='5000')
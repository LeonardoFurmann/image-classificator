from flask import Flask, render_template, request

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tf_explain.core.grad_cam import GradCAM

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():

    model = load_model('./train_model/temp_result.h5')
    img_height, img_width = 150, 150

    imageFile = request.files['imagefile']
    image_path = "./images/" + imageFile.filename
    imageFile.save(image_path)

    img = image.load_img(image_path, target_size=(img_height, img_width))

    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  

    predictions = model.predict(img_array)  
    predicted_class = np.argmax(predictions[0])  

    class_labels = {0: 'Normal', 1: 'Pneumonia'}
    predicted_label = class_labels[predicted_class]

    return render_template('index.html', prediction= predicted_label)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
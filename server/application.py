
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('../models/modelo_multiclase.h5')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = image.load_img(file, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    return jsonify({'class_index': int(class_idx)})

if __name__ == '__main__':
    app.run(debug=True)

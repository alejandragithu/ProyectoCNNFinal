from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO

app = Flask(__name__)
model = load_model('../models/modelo_multiclase.h5')

class_labels = ['monitor', 'teclado', 'raton', 'taza', 'libro', 'movil']

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = image.load_img(BytesIO(file.read()), target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar predicción
    prediction = model.predict(img_array)[0]

    # Obtener las dos clases más probables
    sorted_indices = np.argsort(prediction)[::-1]
    top_1_idx = sorted_indices[0]
    top_2_idx = sorted_indices[1]

    top_1_confidence = float(prediction[top_1_idx])
    top_2_confidence = float(prediction[top_2_idx])

    response = {
        'class_index': int(top_1_idx),
        'confidence': top_1_confidence,
        'second_class_index': int(top_2_idx),
        'second_confidence': top_2_confidence
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
    
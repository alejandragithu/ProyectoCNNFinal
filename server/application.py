from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO

app = Flask(__name__)

# Cargar el modelo
model = load_model('../models/modelo_multiclase.h5')

# Definir las etiquetas de clase
class_labels = ['monitor', 'teclado', 'raton', 'taza', 'libro', 'movil']

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    try:
        # Convertir a BytesIO para que pueda ser leído por load_img
        img = image.load_img(BytesIO(file.read()), target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Realizar predicción
        prediction = model.predict(img_array)
        class_idx = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        # Obtener el nombre de la clase
        predicted_class = class_labels[class_idx]

        return jsonify({
            'class_index': class_idx,
            'class_name': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

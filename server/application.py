from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

# Cargar el modelo
MODEL_PATH = '../models/modelo_multiclase.h5'

try:
    model = load_model(MODEL_PATH)
    print(f"Modelo cargado correctamente desde {MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# Definir las etiquetas de clase
class_labels = ['monitor', 'teclado', 'ratón', 'taza', 'libro', 'movil']

# Umbral de confianza
CONFIDENCE_THRESHOLD = 0.6

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó ningún archivo'}), 400

    file = request.files['file']

    # Verificar el tipo de archivo
    try:
        img = Image.open(BytesIO(file.read()))
        img.verify()  # Verifica que el archivo sea una imagen válida
        file.seek(0)  # Restablecer el puntero del archivo para poder leerlo de nuevo
    except (UnidentifiedImageError, Exception) as e:
        return jsonify({'error': 'Formato de archivo no soportado o archivo dañado. Solo se permiten JPG, JPEG y PNG'}), 400

    try:
        # Leer y procesar la imagen
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

        # Clasificar como "No es un objeto de escritorio" si la confianza es menor al umbral
        top_1_label = class_labels[top_1_idx] if top_1_confidence >= CONFIDENCE_THRESHOLD else "No es un objeto de escritorio"
        top_2_label = class_labels[top_2_idx]

        response = {
            'class_index': int(top_1_idx),
            'class_label': top_1_label,
            'confidence': round(top_1_confidence, 4),
            'second_class_index': int(top_2_idx),
            'second_class_label': top_2_label,
            'second_confidence': round(top_2_confidence, 4)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f"Error en la predicción: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

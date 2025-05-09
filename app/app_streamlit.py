### Proyecto Final: App Web de Clasificación de Objetos de Escritorio
### Alumna: Alejandra Pérez Quintana

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Configuración inicial
st.title("Clasificador de algunos Objetos de Escritorio 🖥️ 📚 ☕ 📱 ⌨️ 🖱️")

st.write("""
Sube una imagen de un objeto y el modelo intentará predecir de qué objeto de escritorio se trata.
Si el objeto no pertenece a un escritorio, te lo indicaremos.
""")

# Cargar modelo
model_path = '../models/modelo_multiclase.h5'
assert os.path.exists(model_path), "Modelo no encontrado en models/modelo_multiclase.h5"
model = load_model(model_path)

# Definición de etiquetas
class_labels = ['monitor', 'teclado', 'raton', 'taza', 'libro', 'movil']

# Umbral de confianza para decidir si es un objeto de escritorio
confidence_threshold = 0.6

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption='Imagen subida', use_column_width=True)

    # Preprocesar
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class = class_labels[predicted_class_idx]
    confidence = np.max(prediction)

    st.write(f"**Confianza del modelo:** {confidence:.2f}")

    # Decidir si es de escritorio o no
    if confidence >= confidence_threshold:
        st.success(f"Objeto detectado: **{predicted_class}**")
    else:
        st.error("Este objeto **no parece ser de escritorio**.")


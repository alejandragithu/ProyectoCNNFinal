import streamlit as st
import requests
from PIL import Image, ExifTags
from io import BytesIO

# URL del servidor Flask
SERVER_URL = "http://127.0.0.1:5000/predict"

# Etiquetas de clases
class_labels = ['monitor', 'teclado', 'raton', 'taza', 'libro', 'movil']

st.title("Clasificador de algunos Objetos de Escritorio 🖥️ 📚 ☕ 📱 ⌨️ 🖱️")
st.write("""
Sube una imagen de un objeto y el modelo intentará predecir de qué objeto de escritorio se trata.
Si el objeto no pertenece a un escritorio, te lo indicaremos.
""")

# Umbral de confianza ajustable
confidence_threshold = st.slider("Umbral de confianza (entre 0 y 1)", 0.0, 1.0, 0.6, 0.05)

def correct_orientation(img):
    """ Corrige la orientación de una imagen usando los metadatos EXIF. """
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = img._getexif()

        if exif is not None:
            orientation = exif.get(orientation)

            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        # Si no hay metadatos EXIF o no se puede acceder a ellos
        pass

    return img

# Cargar imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Cargar y corregir la orientación
    img = Image.open(uploaded_file)
    img = correct_orientation(img)

    # Mostrar la imagen corregida
    st.image(img, caption="Imagen cargada (orientación corregida)", use_container_width=True)

    try:
        # Convertir la imagen a bytes para enviarla al servidor
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()

        # Enviar la imagen al servidor Flask
        response = requests.post(SERVER_URL, files={'file': img_bytes})

        if response.status_code == 200:
            result = response.json()

            # Extraer los valores del JSON
            class_name = result.get('class_label', "Ninguno")
            confidence = result.get('confidence', 0.0)

            second_class_name = result.get('second_class_label', "Ninguno")
            second_confidence = result.get('second_confidence', 0.0)

            # Mostrar resultados
            st.write(f"**1. {class_name}** - Confianza: {confidence:.2f}")
            st.write(f"**2. {second_class_name}** - Confianza: {second_confidence:.2f}")

            # Verificar si la confianza es baja
            if class_name == "No es un objeto de escritorio":
                st.warning("La imagen no ha sido reconocida como un objeto de escritorio.")

        else:
            st.error(f"Error en la predicción: {response.json().get('error')}")

    except Exception as e:
        st.error(f"Error al comunicarse con el servidor: {str(e)}")

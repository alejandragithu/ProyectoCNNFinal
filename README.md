
# Proyecto Final CNN - Clasificación de Imágenes

Este proyecto consiste en el desarrollo de un modelo de red neuronal convolucional (CNN) basado en VGG16 preentrenada para clasificar imágenes de objetos de escritorio en seis categorías: Monitor, Teclado, Ratón, Taza, Libro y Móvil, y además, objetos de no escritorio. He implementado una API Flask y una interfaz Streamlit para realizar predicciones y evaluar el modelo.

## Instrucciones rápidas
1. Añadir las imágenes en `data/train`, `data/validation` y `data/test`.
2. Correr el servidor Flask: `cd server`, `python application.py`
3. Correr la app web: `cd app`, `streamlit run app_streamlit.py`


##  Estructura del Proyecto

app/: Contiene los archivos app_streamlit.py y application.py para la implementación de Streamlit y Flask.

data/: Carpeta que contiene las imágenes distribuidas en subcarpetas para train, validation y test.

models/: Carpeta donde se almacena el modelo entrenado (modelo_multiclase.h5).

Entrenar_CNN.ipynb: Notebook donde se entrena el modelo y se realiza el análisis del dataset.

Inferencia_CNN.ipynb: Notebook donde se prueban predicciones del modelo usando imágenes de test.


## Requisitos
Python 3.10
TensorFlow
Streamlit
Flask
PIL (Pillow)
Requests
Matplotlib

## Estructura del Dataset
Monitor: 70 imágenes
Teclado: 70 imágenes
Ratón: 70 imágenes
Taza: 70 imágenes
Libro: 70 imágenes
Móvil: 70 imágenes

### Distribución:
Train: 52 imágenes por clase
Validation: 10 imágenes por clase
Test: 8 imágenes por clase

## Optimización del Modelo
Data Augmentation: Rotación, zoom, ajuste de brillo y desplazamiento.
Early Stopping con paciencia de 3 épocas.
Reducción dinámica del learning rate (ReduceLROnPlateau).

## Mejora:
Aumentar el dataset a más de 200 imágenes por clase para mejorar la generalización.
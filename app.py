# Importamos las librerías necesarias para crear la UI y soportar los modelos de deep learning del proyecto
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image



st.title("Universidad Autónoma de Bucaramanga - UNAB")
st.header('Aplicación para la detección de objetos con YOLOv8')

with st.container():
    st.subheader("Detección de Equipos de Protección Personal (EPP)")
    st.write("Desarrollado por Ivan Tarazona Rios. Esta aplicación permite detectar botas, guantes, cascos, chalecos y personas utilizando un modelo YOLOv8 entrenado para estas clases.")


# Subtítulo visual centralizado
st.markdown("<h2 style='text-align: center;'>Sube una imagen o captura una foto para detectar objetos.</h2>", unsafe_allow_html=True)

# Cargar el modelo YOLOv8 previamente entrenado
model = YOLO("best.pt")

# Opciones de entrada (cámara o subir imagen)
opcion = st.radio("Selecciona el método de entrada", ("Cámara", "Subir imagen"))

# Función para mostrar los resultados de detección
def mostrar_resultado(imagen):
    imagen = imagen.convert("RGB")  # Convertir la imagen a RGB
    results = model.predict(imagen, conf=0.25)
    pred = results[0].plot()  # Procesamos la predicción
    pred = Image.fromarray(pred[:, :, ::-1])  # Convertimos de BGR a RGB
    st.image(pred, caption="Resultado de la Detección", use_container_width=True)

# Opción para capturar imagen desde la cámara
if opcion == "Cámara":
    img_file_buffer = st.camera_input("Captura una foto para detectar objetos")
    if img_file_buffer is None:
        st.info("Por favor, captura una foto.")
    else:
        image = Image.open(img_file_buffer)
        st.subheader("Imagen capturada")
        st.image(image, caption="Imagen capturada", use_container_width=True)
        st.subheader("Resultado de la detección")
        mostrar_resultado(image)

# Opción para subir una imagen
elif opcion == "Subir imagen":
    uploaded_file = st.file_uploader("Sube una imagen para detectar objetos", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.subheader("🖼️ Imagen seleccionada")
        st.image(image, caption="Imagen cargada", use_container_width=True)
        st.subheader("🔍 Resultado de la detección")
        mostrar_resultado(image)

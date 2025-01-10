import streamlit as st
from PIL import Image
import os

# Título de la aplicación
st.title("🧠 MRI Stroke Image Uploader")

# Descripción de la aplicación
st.write(
    "Sube imágenes de resonancia magnética (MRI) relacionadas con el accidente cerebrovascular. "
    "Puedes previsualizar las imágenes y procesarlas para análisis futuros."
)

# Sección de subida de archivos
uploaded_files = st.file_uploader(
    "Sube tus imágenes (formatos permitidos: JPEG, PNG, DICOM)",
    type=["jpg", "jpeg", "png", "dcm"],
    accept_multiple_files=True,
)

# Mostrar imágenes subidas
if uploaded_files:
    st.write("### Imágenes cargadas:")
    for uploaded_file in uploaded_files:
        try:
            # Intentar abrir como imagen
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)
        except Exception:
            st.warning(f"El archivo {uploaded_file.name} no es una imagen válida o no pudo abrirse.")
else:
    st.write("No se han subido imágenes aún.")

# Botón para análisis futuro (placeholder)
if st.button("Iniciar análisis"):
    st.write("🔄 Funcionalidad de análisis aún no implementada.")

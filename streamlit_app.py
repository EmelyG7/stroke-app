import streamlit as st
from PIL import Image
import numpy as np
from fpdf import FPDF

# Función simulada para predicción
def fake_model_prediction(image_array):
    return "Stroke Detectado"

# Función para generar un PDF con la imagen
def generate_pdf_with_image(diagnosis, age, gender, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Título
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(200, 10, txt="Reporte de Diagnóstico de Stroke", ln=True, align='C')
    pdf.ln(10)

    # Información general
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Diagnóstico: {diagnosis}", ln=True)
    pdf.cell(200, 10, txt=f"Edad: {age} años", ln=True)
    pdf.cell(200, 10, txt=f"Sexo: {gender}", ln=True)
    pdf.ln(10)

    # Insertar la imagen
    pdf.cell(200, 10, txt="Imagen analizada:", ln=True)
    pdf.image(image_path, x=10, y=50, w=180)

    pdf.ln(100)
    pdf.cell(200, 10, txt="Gracias por usar nuestro sistema de diagnóstico.", ln=True, align='C')

    return pdf

# Inicializar estados de sesión
if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = None
if "image_array" not in st.session_state:
    st.session_state.image_array = None
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None
if "age" not in st.session_state:
    st.session_state.age = 30  # Valor por defecto
if "gender" not in st.session_state:
    st.session_state.gender = "Masculino"

# Función para reiniciar el estado
def reset_session_state():
    st.session_state.diagnosis = None
    st.session_state.image_array = None
    st.session_state.uploaded_file_path = None
    st.session_state.age = 30
    st.session_state.gender = "Masculino"

# Interfaz de usuario
st.title("🧠 Diagnóstico de Stroke basado en MRI")
st.write("Sube una imagen de resonancia magnética para obtener un diagnóstico detallado.")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen (formatos: JPEG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    try:
        # Mostrar la imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)

        # Guardar la imagen en un archivo temporal
        st.session_state.uploaded_file_path = f"temp_{uploaded_file.name}"
        image.save(st.session_state.uploaded_file_path)

        # Convertir la imagen a numpy array para procesarla
        st.session_state.image_array = np.array(image)

        # Botón para obtener diagnóstico
        if st.button("Obtener Diagnóstico"):
            st.session_state.diagnosis = fake_model_prediction(st.session_state.image_array)
            st.success(f"Diagnóstico obtenido: {st.session_state.diagnosis}")
    except Exception as e:
        st.error(f"Ocurrió un error al procesar la imagen: {e}")
else:
    # Reiniciar estado si no hay imagen
    reset_session_state()

# Mostrar resultados después del diagnóstico
if st.session_state.diagnosis:
    st.info(f"Diagnóstico: {st.session_state.diagnosis}")

    # Solicitar edad y sexo
    st.session_state.age = st.number_input("Ingresa la edad", min_value=0, max_value=120, value=st.session_state.age)
    st.session_state.gender = st.selectbox("Selecciona el sexo", options=["Masculino", "Femenino", "Otro"], index=["Masculino", "Femenino", "Otro"].index(st.session_state.gender))

    # Mostrar opción de generar informe
    if st.button("Generar Informe"):
        pdf = generate_pdf_with_image(
            st.session_state.diagnosis,
            st.session_state.age,
            st.session_state.gender,
            st.session_state.uploaded_file_path
        )
        pdf_output = f"reporte_diagnostico.pdf"
        pdf.output(pdf_output)

        # Descargar PDF
        with open(pdf_output, "rb") as pdf_file:
            st.download_button(
                label="📄 Descargar Informe en PDF",
                data=pdf_file,
                file_name=pdf_output,
                mime="application/pdf"
            )

# import streamlit as st
# from PIL import Image
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from fpdf import FPDF
# import os
#
# # Cargar el modelo entrenado
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "modelo_resnet50_brain_data.h5")
# model = load_model(MODEL_PATH)
#
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"El archivo del modelo no se encuentra en la ruta: {MODEL_PATH}")
# model = load_model(MODEL_PATH)
#
# # Clase predicha por el modelo
# class_labels = ["Normal", "Stroke"]
#
#
# # Función para predecir la clase utilizando el modelo entrenado
# def predict_image_with_model(image_array):
#     # Preprocesar la imagen
#     IMG_SIZE = (224, 224)  # Tamaño esperado por el modelo
#     image = img_to_array(image_array.resize(IMG_SIZE)) / 255.0
#     # Si la imagen tiene un solo canal (escala de grises), conviértela a 3 canales (color)
#     if image.shape[-1] == 1:
#         image = np.concatenate([image] * 3, axis=-1)  # Duplica el canal 1 tres veces
#
#     image = np.expand_dims(image, axis=0)
#
#     # Predicción
#     prediction = model.predict(image)
#     predicted_class = class_labels[np.argmax(prediction)]
#     confidencee = np.max(prediction)
#
#     return predicted_class, confidencee
#
#
# # Función para generar un PDF con la imagen cargada y el diagnóstico
# def generate_pdf_with_image(diagnostico, age, gender, image_path):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#
#     # Título
#     pdf.set_font("Arial", 'B', size=16)
#     pdf.cell(200, 10, txt="Reporte de Diagnóstico de Stroke", ln=True, align='C')
#     pdf.ln(10)
#
#     # Información general
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt=f"Diagnóstico: {diagnostico}", ln=True)
#     pdf.cell(200, 10, txt=f"Edad: {age} años", ln=True)
#     pdf.cell(200, 10, txt=f"Sexo: {gender}", ln=True)
#     pdf.ln(30)
#
#     # Insertar la imagen
#     pdf.cell(200, 10, txt="Imagen analizada:", ln=True)
#     pdf.image(image_path, x=10, y=50, w=180)
#
#     pdf.ln(100)
#     return pdf
#
#
# # Inicializar estados de sesión
# if "diagnosis" not in st.session_state:
#     st.session_state.diagnosis = None
# if "image_array" not in st.session_state:
#     st.session_state.image_array = None
# if "uploaded_file_path" not in st.session_state:
#     st.session_state.uploaded_file_path = None
# if "age" not in st.session_state:
#     st.session_state.age = 30  # Valor por defecto
# if "gender" not in st.session_state:
#     st.session_state.gender = "Masculino"
#
#
# # Función para reiniciar el estado
# def reset_session_state():
#     st.session_state.diagnosis = None
#     st.session_state.image_array = None
#     st.session_state.uploaded_file_path = None
#     st.session_state.age = 1
#     st.session_state.gender = "Masculino"
#
#
# # Interfaz de usuario
# st.title("🧠 Diagnóstico de Stroke basado en MRI")
# st.write("Sube una imagen de resonancia magnética para obtener un diagnóstico detallado.")
#
# # Subir imagen
# uploaded_file = st.file_uploader("Sube una imagen (formatos: JPEG, PNG)", type=["jpg", "jpeg", "png"])
# if uploaded_file:
#     try:
#         # Mostrar la imagen
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Imagen cargada", use_container_width=True)
#
#         # Guardar la imagen en un archivo temporal
#         st.session_state.uploaded_file_path = f"temp_{uploaded_file.name}"
#         image.save(st.session_state.uploaded_file_path)
#
#         # Asignar la imagen al estado de sesión
#         st.session_state.image_array = image
#
#         # Botón para obtener diagnóstico
#         if st.button("Obtener Diagnóstico"):
#             diagnosis, confidence = predict_image_with_model(image)
#             st.session_state.diagnosis = f"{diagnosis} (Confianza: {confidence * 100:.2f}%)"
#             st.success(f"Diagnóstico obtenido: {st.session_state.diagnosis}")
#     except Exception as e:
#         st.error(f"Ocurrió un error al procesar la imagen: {e}")
# else:
#     # Reiniciar estado si no hay imagen
#     reset_session_state()
#
# # Mostrar resultados después del diagnóstico
# if st.session_state.diagnosis:
#     st.info(f"Diagnóstico: {st.session_state.diagnosis}")
#
#     # Solicitar edad y sexo
#     st.session_state.age = st.number_input("Ingresa la edad", min_value=0, max_value=120, value=st.session_state.age)
#     st.session_state.gender = st.selectbox("Selecciona el sexo", options=["Masculino", "Femenino", "Otro"],
#                                            index=["Masculino", "Femenino", "Otro"].index(st.session_state.gender))
#
#     # Mostrar opción de generar informe
#     if st.button("Generar Informe"):
#         pdf = generate_pdf_with_image(
#             st.session_state.diagnosis,
#             st.session_state.age,
#             st.session_state.gender,
#             st.session_state.uploaded_file_path
#         )
#         pdf_output = f"reporte_diagnostico.pdf"
#         pdf.output(pdf_output)
#
#         # Descargar PDF
#         with open(pdf_output, "rb") as pdf_file:
#             st.download_button(
#                 label="📄 Descargar Informe en PDF",
#                 data=pdf_file,
#                 file_name=pdf_output,
#                 mime="application/pdf"
#             )
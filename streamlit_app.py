import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from fpdf import FPDF
import tempfile
import datetime
import cv2

# Configuración de la página
st.set_page_config(
    page_title="Stroke Detection App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Definir la métrica F1Score personalizada
@tf.keras.utils.register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


# Cargar el modelo
@st.cache_resource
def load_custom_model():
    try:
        model_path = "C:/Users/Coshita/PycharmProjects/stroke-app/best_densenet121_fold_5.keras"  # Asegúrate de tener este archivo
        model = load_model(model_path, custom_objects={'F1Score': F1Score})
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None


model = load_custom_model()

# Clases del modelo
class_labels = ["Normal", "Stroke"]


# Preprocesamiento de imágenes médicas
def preprocess_medical_image(image):
    image = np.array(image)
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    if len(image.shape) == 3 and image.shape[-1] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    bilateral = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
    unsharp_image = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    normalized = cv2.normalize(unsharp_image, None, 0, 255, cv2.NORM_MINMAX)
    processed = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    return processed.astype(np.float32) / 255.0


# Función para predecir
def predict_image(image):
    try:
        # Preprocesamiento
        img = image.resize((240, 240))
        img_array = img_to_array(img)

        # Convertir a RGB si es necesario
        if img_array.shape[-1] == 1:  # Si es escala de grises
            img_array = np.concatenate([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:  # Si tiene canal alpha
            img_array = img_array[..., :3]

        # Aplicar preprocesamiento médico
        img_array = preprocess_medical_image(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción
        prediction = model.predict(img_array)
        predicted_class = class_labels[int(prediction[0][0] >= 0.5)]
        confidence = float(prediction[0][0]) if predicted_class == "Stroke" else 1 - float(prediction[0][0])
        probability = float(prediction[0][0])  # Convertir a float directamente

        return predicted_class, confidence, probability
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return "Error", 0.0, 0.0


# Generar PDF con los resultados
def create_pdf_report(image_path, prediction, confidence, probability):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Encabezado
    pdf.cell(200, 10, txt="Informe de Diagnóstico por Imagen", ln=1, align='C')
    pdf.cell(200, 10, txt="Stroke Detection System", ln=1, align='C')
    pdf.ln(10)

    # Fecha y hora
    now = datetime.datetime.now()
    pdf.cell(200, 10, txt=f"Fecha del análisis: {now.strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.ln(5)

    # Resultados
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt="Resultados del análisis:", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Diagnóstico: {prediction}", ln=1)
    pdf.cell(200, 10, txt=f"Confianza: {confidence * 100:.2f}%", ln=1)
    pdf.cell(200, 10, txt=f"Probabilidad de stroke: {probability * 100:.2f}%", ln=1)
    pdf.ln(10)

    # Imagen
    pdf.cell(200, 10, txt="Imagen analizada:", ln=1)
    pdf.image(image_path, x=50, w=100)

    # Pie de página
    pdf.ln(20)
    pdf.set_font("Arial", 'I', size=8)
    pdf.cell(200, 10, txt="Este informe ha sido generado automáticamente por el sistema de diagnóstico por imagen.",
             ln=1, align='C')

    return pdf


# Interfaz de usuario
def main():
    st.title("🏥 Sistema de Detección de Stroke por Imagen DWI")
    st.markdown("""
    Esta aplicación utiliza un modelo de deep learning para analizar imágenes de resonancia magnética 
    (DWI - Diffusion Weighted Imaging) y detectar posibles casos de stroke.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Cargar Imagen")
        uploaded_file = st.file_uploader("Seleccione una imagen médica (DWI)",
                                         type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_column_width=True)

            # Guardar temporalmente para el PDF
            temp_dir = tempfile.mkdtemp()
            temp_image_path = os.path.join(temp_dir, "temp_image.jpg")
            image.save(temp_image_path)

            if st.button("Analizar Imagen"):
                with st.spinner("Procesando imagen..."):
                    predicted_class, confidence, probability = predict_image(image)

                    st.session_state['prediction'] = predicted_class
                    st.session_state['confidence'] = confidence
                    st.session_state['probability'] = probability
                    st.session_state['image_path'] = temp_image_path

    with col2:
        st.header("Resultados del Análisis")

        if 'prediction' in st.session_state:
            st.subheader("Diagnóstico")

            if st.session_state['prediction'] == "Stroke":
                st.error(f"🚨 Resultado: {st.session_state['prediction']}")
                st.warning(f"Confianza: {float(st.session_state['confidence']) * 100:.2f}%")
            else:
                st.success(f"✅ Resultado: {st.session_state['prediction']}")
                st.info(f"Confianza: {float(st.session_state['confidence']) * 100:.2f}%")

            # Barra de probabilidad
            st.progress(float(st.session_state['probability']))
            st.caption(f"Probabilidad de stroke: {float(st.session_state['probability']) * 100:.2f}%")

            # Explicación
            st.subheader("Interpretación")
            if st.session_state['prediction'] == "Stroke":
                st.markdown("""
                **El modelo ha detectado signos de stroke en la imagen.**
                - Recomendamos una revisión inmediata por un especialista.
                - Los falsos positivos son posibles, pero no se debe ignorar este resultado.
                """)
            else:
                st.markdown("""
                **No se han detectado signos evidentes de stroke.**
                - El modelo no encontró patrones asociados a stroke en la imagen.
                - Para un diagnóstico completo, consulte siempre con un especialista.
                """)

            # Generar PDF
            st.subheader("Generar Informe")
            pdf = create_pdf_report(
                st.session_state['image_path'],
                st.session_state['prediction'],
                st.session_state['confidence'],
                st.session_state['probability']
            )

            pdf_output = os.path.join(temp_dir, "reporte.pdf")
            pdf.output(pdf_output)

            with open(pdf_output, "rb") as f:
                st.download_button(
                    label="Descargar Informe en PDF",
                    data=f,
                    file_name="reporte_diagnostico.pdf",
                    mime="application/pdf"
                )


if __name__ == "__main__":
    main()
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import pandas as pd
from fpdf import FPDF
import tempfile
import datetime
import os
import matplotlib.pyplot as plt
from io import BytesIO
import pymongo
from gridfs import GridFS
from pymongo import MongoClient
from bson.objectid import ObjectId
import tempfile

# Configuración de Streamlit
st.set_page_config(page_title="Stroke Management App", page_icon="🏥", layout="wide")


# --------------------------
# Configuración de MongoDB
# --------------------------
@st.cache_resource
def init_mongodb():
    try:
        client = MongoClient("mongodb+srv://emelygomez:BpGHOqzYhF9lNm9O@cluster0.mronmnn.mongodb.net/")
        db = client['stroke_database']

        if 'patients' not in db.list_collection_names():
            db.create_collection('patients')
        if 'consultations' not in db.list_collection_names():
            db.create_collection('consultations')

        fs = GridFS(db)
        return db, fs
    except Exception as e:
        st.error(f"Error al conectar con MongoDB: {e}")
        return None, None


db, fs = init_mongodb()

if db is None or fs is None:
    st.error("No se pudo conectar a MongoDB. Verifique la cadena de conexión.")
    st.stop()


# --------------------------
# Modelo y Procesamiento de Imágenes
# --------------------------
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


@st.cache_resource
def load_custom_model():
    try:
        model_path = "/best_densenet121_fold_5.keras"
        model = load_model(model_path, custom_objects={'F1Score': F1Score})
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None


model = load_custom_model()
class_labels = ["Normal", "Stroke"]


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


def predict_image(image):
    try:
        img = image.resize((240, 240))
        img_array = img_to_array(img)

        if img_array.shape[-1] == 1:
            img_array = np.concatenate([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[..., :3]

        processed_img = preprocess_medical_image(img_array)
        prediction = model.predict(np.expand_dims(processed_img, axis=0))
        predicted_class = class_labels[int(prediction[0][0] >= 0.5)]
        confidence = float(prediction[0][0]) if predicted_class == "Stroke" else 1 - float(prediction[0][0])
        probability = float(prediction[0][0])

        display_img = (processed_img * 255).astype(np.uint8)
        return predicted_class, confidence, probability, display_img
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return "Error", 0.0, 0.0, None


# --------------------------
# Funciones de Base de Datos
# --------------------------
def save_image_to_mongodb(image_file, filename):
    try:
        image_id = fs.put(image_file.read(), filename=filename)
        return image_id
    except Exception as e:
        st.error(f"Error al guardar imagen en MongoDB: {e}")
        return None


def get_image_from_mongodb(image_id):
    try:
        image_data = fs.get(image_id).read()
        return image_data
    except Exception as e:
        st.error(f"Error al recuperar imagen de MongoDB: {e}")
        return None


def add_patient(name, age, gender):
    try:
        patient = {
            "name": name,
            "age": age,
            "gender": gender,
            "created_at": datetime.datetime.now()
        }
        result = db.patients.insert_one(patient)
        return result.inserted_id
    except Exception as e:
        st.error(f"Error al agregar paciente: {e}")
        return None


def get_patients():
    try:
        patients = list(db.patients.find().sort("created_at", -1))
        return patients
    except Exception as e:
        st.error(f"Error al obtener pacientes: {e}")
        return []


def add_consultation(patient_id, date, notes):
    try:
        if isinstance(date, datetime.date):
            date = datetime.datetime.combine(date, datetime.datetime.min.time())

        consultation = {
            "patient_id": patient_id,
            "date": date,
            "notes": notes,
            "created_at": datetime.datetime.now()
        }
        result = db.consultations.insert_one(consultation)
        return result.inserted_id
    except Exception as e:
        st.error(f"Error al agregar consulta: {e}")
        return None


def get_consultations():
    try:
        pipeline = [
            {
                "$lookup": {
                    "from": "patients",
                    "localField": "patient_id",
                    "foreignField": "_id",
                    "as": "patient"
                }
            },
            {
                "$unwind": "$patient"
            },
            {
                "$project": {
                    "consultation_id": "$_id",
                    "patient_id": 1,
                    "patient_name": "$patient.name",
                    "date": 1,
                    "notes": 1,
                    "created_at": 1
                }
            }
        ]
        consultations = list(db.consultations.aggregate(pipeline))

        # Agregar número de consulta por paciente
        patient_consultation_counts = {}
        for consult in consultations:
            patient_id = str(consult["patient_id"])
            if patient_id not in patient_consultation_counts:
                patient_consultation_counts[patient_id] = 0
            patient_consultation_counts[patient_id] += 1
            consult["consultation_number"] = patient_consultation_counts[patient_id]

        return consultations
    except Exception as e:
        st.error(f"Error al obtener consultas: {e}")
        return []


def add_image_analysis(consultation_id, image_id, diagnosis, confidence, probability):
    try:
        analysis = {
            "consultation_id": consultation_id,
            "image_id": image_id,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "probability": probability,
            "created_at": datetime.datetime.now()
        }
        result = db.image_analyses.insert_one(analysis)
        return result.inserted_id
    except Exception as e:
        st.error(f"Error al agregar análisis de imagen: {e}")
        return None


def get_images_by_consultation(consultation_id):
    try:
        pipeline = [
            {
                "$match": {"consultation_id": consultation_id}
            },
            {
                "$lookup": {
                    "from": "fs.files",
                    "localField": "image_id",
                    "foreignField": "_id",
                    "as": "image_info"
                }
            },
            {
                "$unwind": "$image_info"
            },
            {
                "$project": {
                    "image_id": 1,
                    "diagnosis": 1,
                    "confidence": 1,
                    "probability": 1,
                    "filename": "$image_info.filename",
                    "created_at": 1
                }
            }
        ]
        images = list(db.image_analyses.aggregate(pipeline))
        return images
    except Exception as e:
        st.error(f"Error al obtener imágenes de consulta: {e}")
        return []


# --------------------------
# Generación de Reportes
# --------------------------
def create_pdf_report(consultation_id, patient_name, date, notes, images_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Informe de Consulta - Stroke Detection System", ln=1, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Paciente: {patient_name}", ln=1)
    pdf.cell(200, 10, txt=f"Fecha: {date.strftime('%Y-%m-%d')}", ln=1)
    pdf.cell(200, 10, txt=f"ID de Consulta: {consultation_id}", ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Notas Médicas:", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(200, 10, txt=notes if notes else "Sin notas adicionales.")
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Resumen de Resultados:", ln=1)
    pdf.set_font("Arial", size=12)

    probabilities = [data['probability'] for data in images_data]
    avg_probability = np.mean(probabilities) if probabilities else 0
    final_diagnosis = "Stroke" if avg_probability >= 0.5 else "Normal"
    min_prob = min(probabilities) * 100 if probabilities else 0
    max_prob = max(probabilities) * 100 if probabilities else 0

    pdf.cell(200, 10, txt=f"Diagnóstico Promedio: {final_diagnosis} (Probabilidad: {avg_probability * 100:.2f}%)", ln=1)
    pdf.cell(200, 10, txt=f"Rango de Probabilidades: {min_prob:.2f}% - {max_prob:.2f}%", ln=1)
    pdf.cell(200, 10, txt=f"Número de Imágenes Analizadas: {len(images_data)}", ln=1)

    if images_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(images_data)), [p * 100 for p in probabilities])
        ax.set_ylim(0, 100)
        ax.set_ylabel('Probabilidad de Stroke (%)')
        ax.set_title('Probabilidades por Imagen')

        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        plt.close()

        temp_dir = tempfile.mkdtemp()
        chart_path = os.path.join(temp_dir, "chart.png")
        with open(chart_path, "wb") as f:
            f.write(img_buf.getvalue())

        pdf.ln(10)
        pdf.cell(200, 10, txt="Distribución de Probabilidades:", ln=1)
        pdf.image(chart_path, x=10, w=180)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Detalles de Imágenes:", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.cell(40, 8, txt="Imagen", border=1)
    pdf.cell(40, 8, txt="Diagnóstico", border=1)
    pdf.cell(40, 8, txt="Confianza (%)", border=1)
    pdf.cell(40, 8, txt="Probabilidad (%)", border=1)
    pdf.ln(8)

    for i, data in enumerate(images_data, 1):
        pdf.cell(40, 8, txt=f"Imagen {i}", border=1)
        pdf.cell(40, 8, txt=data['diagnosis'], border=1)
        pdf.cell(40, 8, txt=f"{data['confidence'] * 100:.2f}", border=1)
        pdf.cell(40, 8, txt=f"{data['probability'] * 100:.2f}", border=1)
        pdf.ln(8)

    pdf.ln(10)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(200, 10, txt="Generado por Stroke Management System - No sustituye diagnóstico médico profesional.",
             ln=1, align="C")

    return pdf


# --------------------------
# Interfaz de Usuario
# --------------------------
def patient_management():
    st.header("Gestión de Pacientes")

    with st.form("patient_form"):
        name = st.text_input("Nombre del Paciente")
        age = st.number_input("Edad", min_value=0, max_value=120, step=1)
        gender = st.selectbox("Género", ["Masculino", "Femenino", "Otro"])
        submitted = st.form_submit_button("Agregar Paciente")

        if submitted and name and age and gender:
            patient_id = add_patient(name, age, gender)
            if patient_id:
                st.success("Paciente agregado exitosamente")

    st.subheader("Lista de Pacientes")
    patients = get_patients()
    if patients:
        df = pd.DataFrame([{
            "ID": str(p["_id"]),
            "Nombre": p["name"],
            "Edad": p["age"],
            "Género": p["gender"],
            "Fecha Registro": p["created_at"].strftime("%Y-%m-%d %H:%M")
        } for p in patients])
        st.dataframe(df, use_container_width=True)


def consultation_management():
    st.header("Gestión de Consultas")

    # Sección para registrar nueva consulta
    st.subheader("Registrar Nueva Consulta")
    patients = get_patients()
    patient_options = {p["name"]: p["_id"] for p in patients}

    with st.form("consultation_form"):
        patient_name = st.selectbox("Paciente", list(patient_options.keys()))
        date = st.date_input("Fecha")
        notes = st.text_area("Notas Médicas")
        uploaded_files = st.file_uploader("Subir Imágenes DWI", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submitted = st.form_submit_button("Registrar Consulta")

        if submitted and patient_name and date and uploaded_files:
            patient_id = patient_options[patient_name]
            consultation_id = add_consultation(patient_id, date, notes)

            if consultation_id:
                st.success("Consulta registrada exitosamente")

                images_data = []
                for uploaded_file in uploaded_files:
                    image_id = save_image_to_mongodb(uploaded_file, uploaded_file.name)

                    if image_id:
                        image = Image.open(uploaded_file)
                        predicted_class, confidence, probability, display_img = predict_image(image)

                        analysis_id = add_image_analysis(
                            consultation_id,
                            image_id,
                            predicted_class,
                            confidence,
                            probability
                        )

                        if analysis_id:
                            images_data.append({
                                "image_id": image_id,
                                "diagnosis": predicted_class,
                                "confidence": confidence,
                                "probability": probability,
                                "filename": uploaded_file.name,
                                "display_img": display_img
                            })

                if images_data:
                    st.subheader("Resultados de la Consulta")

                    probabilities = [data['probability'] for data in images_data]
                    avg_probability = np.mean(probabilities)
                    final_diagnosis = "Stroke" if avg_probability >= 0.5 else "Normal"

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Diagnóstico Promedio", final_diagnosis)
                        st.metric("Probabilidad Promedio", f"{avg_probability * 100:.2f}%")
                    with col2:
                        st.metric("Mínima Probabilidad", f"{min(probabilities) * 100:.2f}%")
                        st.metric("Máxima Probabilidad", f"{max(probabilities) * 100:.2f}%")

                    fig, ax = plt.subplots()
                    ax.bar(range(len(probabilities)), [p * 100 for p in probabilities])
                    ax.set_ylim(0, 100)
                    ax.set_ylabel('Probabilidad de Stroke (%)')
                    ax.set_title('Probabilidades por Imagen')
                    st.pyplot(fig)

                    st.subheader("Análisis Individual por Imagen")
                    for i, data in enumerate(images_data, 1):
                        st.markdown(f"**Imagen {i}: {data['filename']}**")

                        col1, col2 = st.columns(2)
                        with col1:
                            if data['display_img'] is not None:
                                st.image(data['display_img'], caption=f"Diagnóstico: {data['diagnosis']}")

                        with col2:
                            st.metric("Resultado", data['diagnosis'])
                            st.metric("Confianza", f"{data['confidence'] * 100:.2f}%")
                            st.metric("Probabilidad Stroke", f"{data['probability'] * 100:.2f}%")
                            st.progress(data['probability'])

                        st.markdown("---")

                    # Generar PDF inmediatamente después de registrar la consulta
                    temp_dir = tempfile.mkdtemp()
                    pdf = create_pdf_report(str(consultation_id), patient_name, date, notes, images_data)
                    pdf_output = os.path.join(temp_dir, f"consulta_{consultation_id}.pdf")
                    pdf.output(pdf_output)

                    with open(pdf_output, "rb") as f:
                        st.download_button(
                            label="Descargar Informe de Consulta",
                            data=f.read(),
                            file_name=f"consulta_{patient_name.replace(' ', '_')}_{date.strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )

    # Sección para ver consultas anteriores
    st.subheader("Consultas Registradas")
    consultations = get_consultations()

    if consultations:
        # Crear lista de consultas para el selectbox con formato: "Paciente - Consulta #N"
        consultation_options = {
            f"{c['patient_name']} - Consulta #{c['consultation_number']}": str(c["consultation_id"])
            for c in consultations
        }

        selected_consultation_label = st.selectbox(
            "Seleccionar Consulta",
            options=list(consultation_options.keys())
        )

        if selected_consultation_label:
            selected_consultation_id = consultation_options[selected_consultation_label]
            consultation_data = next(
                (c for c in consultations if str(c["consultation_id"]) == selected_consultation_id),
                None
            )

            if consultation_data:
                st.markdown(f"**Paciente:** {consultation_data['patient_name']}")
                st.markdown(f"**Fecha:** {consultation_data['date'].strftime('%Y-%m-%d')}")
                st.markdown(f"**Notas:** {consultation_data['notes']}")

                images = get_images_by_consultation(ObjectId(selected_consultation_id))

                if images:
                    st.subheader("Imágenes de la Consulta")

                    probabilities = [img['probability'] for img in images]
                    avg_probability = np.mean(probabilities)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Diagnóstico Promedio", "Stroke" if avg_probability >= 0.5 else "Normal")
                        st.metric("Probabilidad Promedio", f"{avg_probability * 100:.2f}%")
                    with col2:
                        st.metric("Número de Imágenes", len(images))
                        st.metric("Rango de Probabilidades",
                                  f"{min(probabilities) * 100:.2f}% - {max(probabilities) * 100:.2f}%")

                    for img in images:
                        st.markdown(f"**{img['filename']}**")

                        col1, col2 = st.columns(2)
                        with col1:
                            image_data = get_image_from_mongodb(img['image_id'])
                            if image_data:
                                st.image(image_data, caption=f"Diagnóstico: {img['diagnosis']}")

                        with col2:
                            st.metric("Resultado", img['diagnosis'])
                            st.metric("Confianza", f"{img['confidence'] * 100:.2f}%")
                            st.metric("Probabilidad Stroke", f"{img['probability'] * 100:.2f}%")
                            st.progress(img['probability'])

                        st.markdown("---")

                    # Botón para generar PDF de la consulta seleccionada
                    images_data_for_pdf = [{
                        "image_path": "",
                        "diagnosis": img['diagnosis'],
                        "confidence": img['confidence'],
                        "probability": img['probability']
                    } for img in images]

                    temp_dir = tempfile.mkdtemp()
                    pdf = create_pdf_report(
                        selected_consultation_id,
                        consultation_data['patient_name'],
                        consultation_data['date'],
                        consultation_data['notes'],
                        images_data_for_pdf
                    )
                    pdf_output = os.path.join(temp_dir, f"consulta_{selected_consultation_id}.pdf")
                    pdf.output(pdf_output)

                    with open(pdf_output, "rb") as f:
                        st.download_button(
                            label="Generar Informe PDF de esta Consulta",
                            data=f.read(),
                            file_name=f"consulta_{consultation_data['patient_name'].replace(' ', '_')}_{consultation_data['date'].strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )


# --------------------------
# Aplicación Principal
# --------------------------
def main():
    st.title("🏥 Sistema de Gestión para Detección de Stroke")
    st.markdown("Gestión de pacientes y consultas con análisis de imágenes DWI")

    menu = st.sidebar.selectbox("Menú", ["Gestión de Pacientes", "Gestión de Consultas"])

    if menu == "Gestión de Pacientes":
        patient_management()
    elif menu == "Gestión de Consultas":
        consultation_management()


if __name__ == "__main__":
    main()
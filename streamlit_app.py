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
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash
import pymongo
from gridfs import GridFS
from pymongo import MongoClient
from bson.objectid import ObjectId

import tempfile
import calendar
from datetime import datetime as dt
from streamlit.runtime.scriptrunner import RerunException


# Configuración de Streamlit
st.set_page_config(page_title="Stroke Management App", page_icon="🏥", layout="wide")

st.write("Streamlit version:", st.__version__)

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
        if 'stats' not in db.list_collection_names():
            db.create_collection('stats')

        fs = GridFS(db)
        return db, fs
    except Exception as e:
        st.error(f"Error al conectar con MongoDB: {e}")
        return None, None


db, fs = init_mongodb()

if db is None or fs is None:
    st.error("No se pudo conectar a MongoDB. Verifique la cadena de conexión.")
    st.stop()

#Funciones de acceso a doctores
def get_doctor(username: str):
    return db.doctors.find_one({"username": username})

def authenticate_doctor(username: str, password: str):
    doc = get_doctor(username)
    if doc and check_password_hash(doc["password_hash"], password):
        return doc
    return None

def login_page():
    st.title("🔐 Login")
    with st.form("login_form"):
        user = st.text_input("Usuario")
        pwd  = st.text_input("Contraseña", type="password")
        ok   = st.form_submit_button("Ingresar")

    if ok:
        doctor = authenticate_doctor(user, pwd)
        if doctor:
            st.session_state["user"] = {
                "username":  doctor["username"],
                "full_name": doctor["full_name"],
                "role":      doctor["role"]
            }
            st.success("¡Bienvenido, " + doctor["full_name"] + "!")
           #st.experimental_rerun()   
        else:
            st.error("Usuario o contraseña inválidos")



def logout():
    if "user" in st.session_state:
        del st.session_state["user"]
    #st.experimental_rerun()




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
        model_path = "best_densenet121_fold_5.keras"
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


def add_patient(name, age, gender, smoker=False, alcoholic=False, hypertension=False, diabetes=False,
                heart_disease=False):
    try:
        patient = {
            "name": name,
            "age": age,
            "gender": gender,
            "smoker": smoker,
            "alcoholic": alcoholic,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "heart_disease": heart_disease,
            "created_at": dt.now()
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


def add_consultation(patient_id, date, notes, diagnosis=None, probability=None):
    try:
        if isinstance(date, datetime.date):
            date = dt.combine(date, dt.min.time())

        consultation = {
            "patient_id": patient_id,
            "date": date,
            "notes": notes,
            "diagnosis": diagnosis,
            "probability": probability,
            "created_at": dt.now()
        }
        result = db.consultations.insert_one(consultation)

        # Actualizar estadísticas si es un stroke
        if diagnosis == "Stroke":
            update_monthly_stats(date, probability)

        return result.inserted_id
    except Exception as e:
        st.error(f"Error al agregar consulta: {e}")
        return None


def update_monthly_stats(date, probability):
    try:
        year_month = date.strftime("%Y-%m")

        db.stats.update_one(
            {"year_month": year_month},
            {"$inc": {"stroke_count": 1}, "$push": {"probabilities": probability}},
            upsert=True
        )
    except Exception as e:
        st.error(f"Error al actualizar estadísticas: {e}")


def get_monthly_stats():
    try:
        pipeline = [
            {"$sort": {"year_month": 1}},
            {"$project": {
                "year_month": 1,
                "stroke_count": 1,
                "avg_probability": {"$avg": "$probabilities"}
            }}
        ]
        stats = list(db.stats.aggregate(pipeline))
        return stats
    except Exception as e:
        st.error(f"Error al obtener estadísticas: {e}")
        return []


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
                    "diagnosis": 1,
                    "probability": 1,
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
            "created_at": dt.now()
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
def create_pdf_report(consultation_id, patient_name, date, notes, images_data, patient_data=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Informe de Consulta - Stroke Detection System", ln=1, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Paciente: {patient_name}", ln=1)
    pdf.cell(200, 10, txt=f"Fecha: {date.strftime('%Y-%m-%d')}", ln=1)
    pdf.cell(200, 10, txt=f"ID de Consulta: {consultation_id}", ln=1)

    # Add patient medical history if available
    if patient_data:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, txt="Antecedentes Médicos:", ln=1)
        pdf.set_font("Arial", size=12)

        # Create medical history string
        medical_history = []
        if patient_data.get("hypertension", False):
            medical_history.append("Hipertensión")
        if patient_data.get("diabetes", False):
            medical_history.append("Diabetes")
        if patient_data.get("heart_disease", False):
            medical_history.append("Enfermedad cardíaca")
        if patient_data.get("smoker", False):
            medical_history.append("Fumador")
        if patient_data.get("alcoholic", False):
            medical_history.append("Consumo de alcohol")

        history_text = ", ".join(medical_history) if medical_history else "Ninguno registrado"
        pdf.cell(200, 10, txt=history_text, ln=1)

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
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
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
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Nombre del Paciente*")
            age = st.number_input("Edad*", min_value=0, max_value=120, step=1)
            gender = st.selectbox("Género*", ["Masculino", "Femenino", "Otro"])

        with col2:
            st.markdown("**Antecedentes Médicos**")
            smoker = st.checkbox("Fumador")
            alcoholic = st.checkbox("Consume alcohol")
            hypertension = st.checkbox("Hipertensión")
            diabetes = st.checkbox("Diabetes")
            heart_disease = st.checkbox("Enfermedad cardíaca")

        submitted = st.form_submit_button("Agregar Paciente")

        if submitted:
            if not name or not age or not gender:
                st.error("Por favor complete los campos obligatorios (*)")
            else:
                patient_id = add_patient(
                    name, age, gender,
                    smoker, alcoholic,
                    hypertension, diabetes,
                    heart_disease
                )
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
            "Fumador": "Sí" if p.get("smoker", False) else "No",
            "Alcohol": "Sí" if p.get("alcoholic", False) else "No",
            "Hipertensión": "Sí" if p.get("hypertension", False) else "No",
            "Diabetes": "Sí" if p.get("diabetes", False) else "No",
            "Cardíaco": "Sí" if p.get("heart_disease", False) else "No",
            "Fecha Registro": p["created_at"].strftime("%Y-%m-%d %H:%M")
        } for p in patients])
        st.dataframe(df, use_container_width=True)


def register_consultation():
    st.header("Registrar Nueva Consulta")

    patients = get_patients()
    patient_options = {p["name"]: p["_id"] for p in patients}

    with st.form("consultation_form"):
        patient_name = st.selectbox("Paciente*", list(patient_options.keys()))
        date = st.date_input("Fecha*", value=dt.now())
        notes = st.text_area("Notas Médicas")
        uploaded_files = st.file_uploader("Subir Imágenes DWI*", type=["jpg", "jpeg", "png"],
                                          accept_multiple_files=True)

        submitted = st.form_submit_button("Registrar Consulta")

        if submitted:
            if not patient_name or not date or not uploaded_files:
                st.error("Por favor complete los campos obligatorios (*)")
            else:
                patient_id = patient_options[patient_name]
                patient_data = next((p for p in patients if p["_id"] == patient_id), None)

                # Procesar imágenes primero para obtener diagnóstico
                images_data = []
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    predicted_class, confidence, probability, display_img = predict_image(image)

                    images_data.append({
                        "predicted_class": predicted_class,
                        "probability": probability,
                        "confidence": confidence
                    })

                # Calcular diagnóstico general
                probabilities = [data['probability'] for data in images_data]
                avg_probability = np.mean(probabilities)
                final_diagnosis = "Stroke" if avg_probability >= 0.5 else "Normal"

                # Registrar consulta
                consultation_id = add_consultation(
                    patient_id,
                    date,
                    notes,
                    final_diagnosis,
                    avg_probability
                )

                if consultation_id:
                    st.success("Consulta registrada exitosamente")

                    # Guardar imágenes y análisis
                    for i, (uploaded_file, data) in enumerate(zip(uploaded_files, images_data)):
                        uploaded_file.seek(0)  # Rewind the file
                        image_id = save_image_to_mongodb(uploaded_file, uploaded_file.name)

                        if image_id:
                            analysis_id = add_image_analysis(
                                consultation_id,
                                image_id,
                                data["predicted_class"],
                                data["confidence"],
                                data["probability"]
                            )

                    # Mostrar resumen
                    st.subheader("Resumen de la Consulta")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Diagnóstico Final", final_diagnosis)
                        st.metric("Probabilidad Promedio", f"{avg_probability * 100:.2f}%")
                    with col2:
                        st.metric("Número de Imágenes", len(uploaded_files))
                        st.metric("Rango de Probabilidades",
                                  f"{min(probabilities) * 100:.2f}% - {max(probabilities) * 100:.2f}%")

                    # Generate PDF in memory
                    pdf = create_pdf_report(
                        str(consultation_id),
                        patient_name,
                        date,
                        notes,
                        [{
                            "image_path": "",
                            "diagnosis": data["predicted_class"],
                            "confidence": data["confidence"],
                            "probability": data["probability"]
                        } for data in images_data],
                        patient_data  # Pass patient data to include medical history
                    )

                    # Create a temporary file for the PDF
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        pdf_output = tmp.name
                        pdf.output(pdf_output)

                    # Create download button
                    with open(pdf_output, "rb") as f:
                        pdf_bytes = f.read()

                    # Remove temporary file
                    try:
                        os.unlink(pdf_output)
                    except:
                        pass

                    st.download_button(
                        label="Descargar Informe de Consulta",
                        data=pdf_bytes,
                        file_name=f"consulta_{patient_name.replace(' ', '_')}_{date.strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )


def view_consultations():
    st.header("Consultas Registradas")

    # Estadísticas mensuales
    st.subheader("Estadísticas de Strokes por Mes")
    monthly_stats = get_monthly_stats()

    if monthly_stats:
        stats_df = pd.DataFrame([{
            "Mes/Año": s["year_month"],
            "N° Strokes": s["stroke_count"],
            "Probabilidad Promedio": f"{s.get('avg_probability', 0) * 100:.2f}%"
        } for s in monthly_stats])

        st.dataframe(stats_df, use_container_width=True)

        # Gráfico de barras
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(stats_df["Mes/Año"], stats_df["N° Strokes"])
        ax.set_title("Casos de Stroke por Mes")
        ax.set_ylabel("Número de Casos")
        ax.set_xlabel("Mes/Año")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("No hay estadísticas disponibles aún")

    # Lista de consultas
    st.subheader("Historial de Consultas")
    consultations = get_consultations()

    if consultations:
        # Crear lista de consultas para el selectbox con formato: "Paciente - Consulta #N - Fecha"
        consultation_options = {
            f"{c['patient_name']} - Consulta #{c['consultation_number']} - {c['date'].strftime('%Y-%m-%d')}": str(
                c["consultation_id"])
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
                st.markdown("---")
                st.subheader("Detalles de la Consulta")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Paciente:** {consultation_data['patient_name']}")
                    st.markdown(f"**Fecha:** {consultation_data['date'].strftime('%Y-%m-%d')}")
                with col2:
                    st.markdown(f"**Diagnóstico:** {consultation_data.get('diagnosis', 'No registrado')}")
                    if consultation_data.get('probability'):
                        st.markdown(f"**Probabilidad:** {consultation_data['probability'] * 100:.2f}%")

                st.markdown(f"**Notas:** {consultation_data['notes']}")

                # Obtener datos del paciente para antecedentes médicos
                patient_data = db.patients.find_one({"_id": consultation_data["patient_id"]})

                # Mostrar antecedentes médicos
                if patient_data:
                    st.subheader("Antecedentes Médicos del Paciente")
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Hipertensión", "Sí" if patient_data.get("hypertension", False) else "No")
                    with cols[1]:
                        st.metric("Diabetes", "Sí" if patient_data.get("diabetes", False) else "No")
                    with cols[2]:
                        st.metric("Enf. Cardíaca", "Sí" if patient_data.get("heart_disease", False) else "No")
                    with cols[3]:
                        st.metric("Fumador", "Sí" if patient_data.get("smoker", False) else "No")

                # Mostrar imágenes de la consulta
                images = get_images_by_consultation(ObjectId(selected_consultation_id))

                if images:
                    st.subheader("Imágenes de la Consulta")

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

                    # Botón para generar PDF
                    images_data_for_pdf = [{
                        "image_path": "",
                        "diagnosis": img['diagnosis'],
                        "confidence": img['confidence'],
                        "probability": img['probability']
                    } for img in images]

                    # Generate PDF in memory
                    pdf = create_pdf_report(
                        selected_consultation_id,
                        consultation_data['patient_name'],
                        consultation_data['date'],
                        consultation_data['notes'],
                        images_data_for_pdf,
                        patient_data  # Include patient medical history
                    )

                    # Create a temporary file for the PDF
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        pdf_output = tmp.name
                        pdf.output(pdf_output)

                    # Create download button
                    with open(pdf_output, "rb") as f:
                        pdf_bytes = f.read()

                    # Remove temporary file
                    try:
                        os.unlink(pdf_output)
                    except:
                        pass

                    st.download_button(
                        label="Generar Informe PDF de esta Consulta",
                        data=pdf_bytes,
                        file_name=f"consulta_{consultation_data['patient_name'].replace(' ', '_')}_{consultation_data['date'].strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.info("Esta consulta no tiene imágenes asociadas")
    else:
        st.info("No hay consultas registradas aún")


# --------------------------
# Aplicación Principal
# --------------------------
# --------------------------
# Aplicación Principal
# --------------------------

def manage_users():
    st.header("🔧 Gestión de Usuarios")
    # 1) Listado de usuarios existentes
    docs = list(db.doctors.find().sort("username", 1))
    if docs:
        df = pd.DataFrame([{
            "ID":        str(d["_id"]),
            "Usuario":   d["username"],
            "Nombre":    d["full_name"],
            "Rol":       d["role"]
        } for d in docs])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No hay usuarios registrados aún.")

    st.markdown("---")

    # 2) Formulario para crear un nuevo usuario
    with st.expander("➕ Crear nuevo usuario"):
        with st.form("form_create_user"):
            new_user   = st.text_input("Usuario")
            new_name   = st.text_input("Nombre completo")
            new_role   = st.selectbox("Rol", ["admin", "doctor"])
            new_pass   = st.text_input("Contraseña", type="password")
            submit_new = st.form_submit_button("Crear usuario")

        if submit_new:
            if not new_user or not new_name or not new_pass:
                st.error("Todos los campos son obligatorios.")
            elif db.doctors.find_one({"username": new_user}):
                st.error("Ese nombre de usuario ya existe.")
            else:
                pw_hash = generate_password_hash(new_pass, method="pbkdf2:sha256")
                db.doctors.insert_one({
                    "username":      new_user,
                    "password_hash": pw_hash,
                    "full_name":     new_name,
                    "role":          new_role
                })
                st.success(f"Usuario **{new_user}** creado.")
                st.rerun()

    st.markdown("---")

    # 3) Seleccionar un usuario para editar o borrar
    user_options = {f"{d['username']} ({d['full_name']})": str(d["_id"]) for d in docs}
    display_names = [""] + list(user_options.keys())
    
    sel = st.selectbox("Seleccionar usuario para editar / borrar", display_names)
    if sel and sel != "":
        selected_id = user_options[sel]
        doc = db.doctors.find_one({"_id": ObjectId(selected_id)})
        if doc:
            # Usar session_state para controlar el estado del checkbox
            if "change_pw" not in st.session_state:
                st.session_state.change_pw = False

            # Checkbox fuera del formulario para actualizar el estado dinámicamente
            st.session_state.change_pw = st.checkbox("Cambiar contraseña", value=st.session_state.change_pw)

            with st.form("form_edit_user"):
                e_name = st.text_input("Nombre completo", value=doc["full_name"])
                e_role = st.selectbox("Rol", ["admin", "doctor"], index=0 if doc["role"]=="admin" else 1)
                e_pass = None
                e_pass_confirm = None
                if st.session_state.change_pw:
                    e_pass = st.text_input("Nueva contraseña", type="password")
                    e_pass_confirm = st.text_input("Confirmar nueva contraseña", type="password")
                save = st.form_submit_button("Guardar cambios")

            if save:
                update = {"full_name": e_name, "role": e_role}
                if st.session_state.change_pw:
                    if not e_pass or not e_pass_confirm:
                        st.error("Debes ingresar la nueva contraseña y su confirmación.")
                    elif e_pass != e_pass_confirm:
                        st.error("Las contraseñas no coinciden.")
                    elif len(e_pass) < 8:
                        st.error("La contraseña debe tener al menos 8 caracteres.")
                    else:
                        update["password_hash"] = generate_password_hash(e_pass, method="pbkdf2:sha256")
                try:
                    db.doctors.update_one({"_id": doc["_id"]}, {"$set": update})
                    st.success("Usuario actualizado.")
                    st.session_state.change_pw = False  # Resetear el checkbox
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al actualizar el usuario: {e}")

            if st.button("🗑️ Eliminar usuario"):
                try:
                    db.doctors.delete_one({"_id": doc["_id"]})
                    st.success("Usuario eliminado.")
                    st.session_state.change_pw = False  # Resetear el checkbox
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al eliminar el usuario: {e}")

def main():
    # 1) Si no hay usuario en sesión, mostramos login y salimos
    if "user" not in st.session_state:
        login_page()
        return

    # 2) Ya autenticado: saludo y logout
    user = st.session_state["user"]
    st.sidebar.markdown(f"👋 Hola, **{user['full_name']}**")
    if st.sidebar.button("Cerrar sesión"):
        logout()

    # 3) Título y descripción
    st.title("🏥 Sistema de Gestión para Detección de Stroke")
    st.markdown("Gestión de pacientes, consultas y análisis de imágenes DWI")

    # 4) Menú dinámico según rol
    if user["role"] == "admin":
        opciones = [
            "Gestión de Usuarios",
            "Gestión de Pacientes",
            "Registrar Consulta",
            "Ver Consultas y Estadísticas"
        ]
    else:  # doctor
        opciones = [
            "Gestión de Pacientes",
            "Registrar Consulta",
            "Ver Consultas y Estadísticas"
        ]

    menu = st.sidebar.selectbox("Menú", opciones)

    # 5) Ruteo de cada opción
    if menu == "Gestión de Usuarios":
        manage_users()             # Sólo disponible para admin
    elif menu == "Gestión de Pacientes":
        patient_management()
    elif menu == "Registrar Consulta":
        register_consultation()
    elif menu == "Ver Consultas y Estadísticas":
        view_consultations()


if __name__ == "__main__":
    main()



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Cargar el modelo
model = load_model("modelo_resnet50_brain_data.h5")

# Preprocesar una nueva imagen
image_path = "ruta_a_tu_imagen.jpg"
image = load_img(image_path, target_size=(224, 224))  # Redimensionar al tamaño esperado
image_array = img_to_array(image) / 255.0  # Normalizar valores (0-1)
image_array = np.expand_dims(image_array, axis=0)  # Añadir dimensión batch

# Realizar predicción
predictions = model.predict(image_array)
predicted_class = np.argmax(predictions)  # Clase con mayor probabilidad
class_labels = train_generator.class_indices  # Mapeo de etiquetas
class_labels = {v: k for k, v in class_labels.items()}  # Invertir mapeo
print(f"Clase predicha: {class_labels[predicted_class]}")
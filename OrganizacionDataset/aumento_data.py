import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras import datasets, layers, models, regularizers, optimizers
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import os
import shutil
import random
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import numpy as np  # Asegurarse de importar numpy
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import image_dataset_from_directory
import glob

#Aumento de datos

# Configuración
IMG_SIZE = (160, 160)  # Reducir de 224x224 a 160x160
BATCH_SIZE = 8  # Reducir de 16 a 8
TARGET_SIZE = (160, 160, 3)  # Forma de las imágenes
EPOCHS = 50
NUM_CLASSES = 2  # Dos clases: Normal y Stroke

# Definir las capas de preprocesamiento
resize_and_rescale = tf.keras.Sequential([
        layers.Rescaling(1./255),
        layers.Resizing(IMG_SIZE[0], IMG_SIZE[1]),

])

data_augmentation = tf.keras.Sequential([
      layers.RandomFlip("horizontal_and_vertical"),
      layers.RandomRotation(0.8),
      layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
      layers.RandomZoom(0.3),
])

# Aumento de datos de la clase normal
def create_augmented_data_normal(base_dir, img_size, augment_factor):
    """
    Crea datos aumentados para la clase Normal y los fusiona en la carpeta Normal original usando
    capas de preprocesamiento de Keras
    """
    # Definir directorios
    normal_dir = os.path.join(base_dir, "Normal")
    aumentado_dir = os.path.join(base_dir, "Normal_Augmented")

    # Crear directorio temporal si no existe
    if not os.path.exists(aumentado_dir):
        os.makedirs(aumentado_dir)

    print("Generando imágenes aumentadas...")
    # Generar aumentos para cada imagen
    for file_name in os.listdir(normal_dir):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(normal_dir, file_name)

           # Cargar y preparar la imagen
            img = tf.keras.utils.load_img(file_path)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # Aplicar resize y rescaling
            img_array = resize_and_rescale(img_array)

            # Generar imágenes aumentadas
            for i in range(augment_factor):
                # Aplicar data augmentation
                augmented_img = data_augmentation(img_array)

                # Guardar la imagen aumentada
                output_path = os.path.join(aumentado_dir, f"Normal_Aug_{file_name.split('.')[0]}_{i}.jpeg")
                tf.keras.utils.save_img(
                    output_path,
                    tf.cast(augmented_img[0] * 255, tf.uint8)
                )

    print(f"Imágenes aumentadas generadas: {len(os.listdir(aumentado_dir))}")

    # Mover las imágenes aumentadas a la carpeta Normal original
    print("Moviendo imágenes aumentadas a la carpeta Normal...")
    for img_file in glob.glob(os.path.join(aumentado_dir, "*")):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy(img_file, normal_dir)

    # Eliminar el directorio temporal de imágenes aumentadas
    shutil.rmtree(aumentado_dir)

    total_imagenes = len(os.listdir(normal_dir))
    print(f"Fusión completada. Total imágenes en {normal_dir}: {total_imagenes}")

    return normal_dir


# Directorio base
base_dir = "../Dataset"
classes = ["Normal", "Stroke"]


# Crear datos aumentados
combinado_dir = create_augmented_data_normal(
    base_dir=base_dir,
    img_size=IMG_SIZE,
    augment_factor= 10  # Número de aumentos por imagen
)

# Aumento de datos de la clase stroke

def create_augmented_data_stroke(base_dir, img_size, augment_factor):
    """
    Crea datos aumentados para la clase Stroke y los fusiona en la carpeta Stroke original
    """
    # Definir directorios
    stroke_dir = os.path.join(base_dir, "Stroke")
    aumentado_dir = os.path.join(base_dir, "Stroke_Augmented")

    # Crear directorio temporal si no existe
    if not os.path.exists(aumentado_dir):
        os.makedirs(aumentado_dir)

    print("Generando imágenes aumentadas...")
    # Generar aumentos para cada imagen
    for file_name in os.listdir(stroke_dir):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(stroke_dir, file_name)

           # Cargar y preparar la imagen
            img = tf.keras.utils.load_img(file_path)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # Aplicar resize y rescaling
            img_array = resize_and_rescale(img_array)

            # Generar imágenes aumentadas
            for i in range(augment_factor):
                # Aplicar data augmentation
                augmented_img = data_augmentation(img_array)

                # Guardar la imagen aumentada
                output_path = os.path.join(aumentado_dir, f"Stroke_Aug_{file_name.split('.')[0]}_{i}.jpeg")
                tf.keras.utils.save_img(
                    output_path,
                    tf.cast(augmented_img[0] * 255, tf.uint8)
                )

    print(f"Imágenes aumentadas generadas: {len(os.listdir(aumentado_dir))}")

    # Mover las imágenes aumentadas a la carpeta Stroke original
    print("Moviendo imágenes aumentadas a la carpeta Stroke...")
    for img_file in glob.glob(os.path.join(aumentado_dir, "*")):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy(img_file, stroke_dir)

    # Eliminar el directorio temporal de imágenes aumentadas
    shutil.rmtree(aumentado_dir)

    total_imagenes = len(os.listdir(stroke_dir))
    print(f"Fusión completada. Total imágenes en {stroke_dir}: {total_imagenes}")

    return stroke_dir

# Directorio base
base_dir = "../Dataset"
classes = ["Normal", "Stroke"]

# Configuración
IMG_SIZE = (160, 160)  # Reducir de 224x224 a 160x160
BATCH_SIZE = 8  # Reducir de 16 a 8
TARGET_SIZE = (160, 160, 3)  # Forma de las imágenes
EPOCHS = 50
NUM_CLASSES = 2  # Dos clases: Normal y Stroke

# Crear datos aumentados
combinado_dir = create_augmented_data_stroke(
    base_dir=base_dir,
    img_size=IMG_SIZE,
    augment_factor= 6  # Número de aumentos por imagen
)

# Imprimir información sobre el balance de clases
print("\nDistribución final de clases:")
print(f"Stroke (con aumento): {len(os.listdir(combinado_dir))}")
print(f"Stroke: {len(os.listdir(os.path.join(base_dir, 'Stroke')))}")
print(f"Normal: {len(os.listdir(os.path.join(base_dir, 'Normal')))}")


if not os.path.exists(base_dir):
    raise FileNotFoundError(f"La ruta {base_dir} no existe.")

normal = len(os.listdir("../Dataset/Normal"))
stroke = len(os.listdir("../Dataset/Stroke"))

# Calcular totales
total_images = normal + stroke

print(f"Total Normal: {normal}")
print(f"Total Stroke: {stroke}")
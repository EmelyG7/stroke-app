from pathlib import Path
import shutil
import logging

# # Configuración del registro de actividad
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
# # Ruta al directorio principal del conjunto de datos
# dataset_dir = Path('C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/only-slices-with-stroke')
#
# # Ruta al nuevo directorio donde se copiarán las imágenes
# destination_dir = Path('C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Stroke Imagenes/stroke')
#
# # Crear el nuevo directorio si no existe
# destination_dir.mkdir(parents=True, exist_ok=True)
#
# # Función personalizada para ordenar las subcarpetas correctamente
# def folder_sort_key(folder_path):
#     # Extraemos el número de la subcarpeta y lo convertimos en entero
#     return int(folder_path.name.split('sub-strokecase')[1])
#
# # Obtener y ordenar las subcarpetas correctamente
# subfolders = sorted(dataset_dir.glob('sub-strokecase*'), key=folder_sort_key)
#
# # Recorrer las carpetas sub-strokecaseXXXX en orden
# for subfolder_path in subfolders:
#     if subfolder_path.is_dir():
#         for file_path in sorted(subfolder_path.iterdir()):
#             if 'dwi_slice' in file_path.name:
#                 try:
#                     # Nuevo nombre: sub-strokecaseXXXX_dwi_slice_XXX.png
#                     new_filename = f"{subfolder_path.name}_{file_path.name}"
#                     destination_file = destination_dir / new_filename
#
#                     # Copiar el archivo al directorio de destino con el nuevo nombre
#                     shutil.copy2(file_path, destination_file)
#                     logging.info(f'Copiado: {file_path} a {destination_file}')
#                 except Exception as e:
#                     logging.error(f'Error al copiar {file_path}: {e}')
#     else:
#         logging.warning(f'La carpeta {subfolder_path.name} no existe')

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ========================
# Funciones para leer imágenes
# ========================

def get_stroke_images(stroke_folder):
    """
    Recorre todas las subcarpetas de stroke y recolecta las rutas de las imágenes que comienzan con 'dwi_slice_'.
    """
    stroke_images = []
    for root, _, files in os.walk(stroke_folder):
        for file in files:
            if file.startswith('dwi_slice_') and file.endswith(('.png', '.jpg', '.jpeg')):
                stroke_images.append(os.path.join(root, file))
    return stroke_images

def get_all_substrokes_images(stroke_folder):
    """
    Recorre todas las subcarpetas de stroke y recolecta las rutas de las imágenes por subcarpeta.
    """
    substrokes_images = {}
    for substroke in os.listdir(stroke_folder):
        substroke_folder = os.path.join(stroke_folder, substroke)
        if os.path.isdir(substroke_folder):
            substrokes_images[substroke] = get_stroke_images(substroke_folder)
    return substrokes_images

# ========================
# Función para verificar fuga de datos a nivel de píxeles
# ========================

def check_pixel_level_leakage(train_images, val_images, test_images, threshold=0.98):
    """
    Verifica si hay imágenes casi idénticas entre los conjuntos de entrenamiento, validación y prueba.
    """
    def calculate_similarity(img1, img2):
        # Redimensionar las imágenes al mismo tamaño si es necesario
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        # Calcular la diferencia absoluta entre las imágenes
        diff = cv2.absdiff(img1, img2)
        # Calcular la similitud como el porcentaje de píxeles que no difieren
        similarity = 1.0 - (np.mean(diff) / 255.0)
        return similarity

    def find_similar_images(set1, set2, threshold):
        similar_pairs = []
        for i, img_path1 in enumerate(set1):
            img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
            for j, img_path2 in enumerate(set2):
                img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
                if img1 is not None and img2 is not None:
                    similarity = calculate_similarity(img1, img2)
                    if similarity > threshold:
                        similar_pairs.append((img_path1, img_path2, similarity))
        return similar_pairs

    # Verificar similitud entre train y val
    print("Checking for similar images between train and val sets...")
    train_val_similar = find_similar_images(train_images, val_images, threshold)
    if train_val_similar:
        print(f"Pixel-level data leakage detected: {len(train_val_similar)} similar images between train and val sets.")
    else:
        print("No similar images found between train and val sets.")

    # Verificar similitud entre train y test
    print("Checking for similar images between train and test sets...")
    train_test_similar = find_similar_images(train_images, test_images, threshold)
    if train_test_similar:
        print(f"Pixel-level data leakage detected: {len(train_test_similar)} similar images between train and test sets.")
    else:
        print("No similar images found between train and test sets.")

    # Verificar similitud entre val y test
    print("Checking for similar images between val and test sets...")
    val_test_similar = find_similar_images(val_images, test_images, threshold)
    if val_test_similar:
        print(f"Pixel-level data leakage detected: {len(val_test_similar)} similar images between val and test sets.")
    else:
        print("No similar images found between val and test sets.")

    if not train_val_similar and not train_test_similar and not val_test_similar:
        print("No pixel-level data leakage detected.")

# ========================
# División de datos por subcarpeta de stroke
# ========================

def split_data_by_substroke(substrokes_images, test_size=0.2, val_size=0.2):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba por subcarpeta de stroke.
    """
    # Obtén la lista de subcarpetas de stroke
    substrokes = list(substrokes_images.keys())

    # Divide las subcarpetas en conjuntos de entrenamiento, validación y prueba
    train_substrokes, test_substrokes = train_test_split(substrokes, test_size=test_size, random_state=42)
    train_substrokes, val_substrokes = train_test_split(train_substrokes, test_size=val_size, random_state=42)

    # Asigna las imágenes a los conjuntos según la subcarpeta
    train_images = []
    val_images = []
    test_images = []

    for substroke, images in substrokes_images.items():
        if substroke in train_substrokes:
            train_images.extend(images)
        elif substroke in val_substrokes:
            val_images.extend(images)
        elif substroke in test_substrokes:
            test_images.extend(images)

    print(f"Train set: {len(train_images)} imágenes")
    print(f"Val set: {len(val_images)} imágenes")
    print(f"Test set: {len(test_images)} imágenes")

    return train_images, val_images, test_images

# ========================
# Ejecución principal
# ========================

if __name__ == "__main__":
    # Ruta a la carpeta 'stroke'
    stroke_folder = "C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Stroke Imagenes/only-slices-with-stroke"

    # Obtén las imágenes de todas las subcarpetas de stroke
    substrokes_images = get_all_substrokes_images(stroke_folder)

    # Imprime el número de imágenes por subcarpeta
    for substroke, images in substrokes_images.items():
        print(f"Substroke {substroke}: {len(images)} imágenes")

    # Divide los datos por subcarpeta de stroke
    train_images, val_images, test_images = split_data_by_substroke(substrokes_images)

    # Verifica la fuga de datos a nivel de píxeles
    check_pixel_level_leakage(train_images, val_images, test_images, threshold=0.98)
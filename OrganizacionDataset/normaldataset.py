from pathlib import Path
import shutil

# Ruta a la carpeta donde están las imágenes
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/9/dDWI_og_SENSE_803")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/9/DWI_og_801")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/9/eReg__DWI_og_SENSE_804")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/9/Reg__DWI_og_SENSE_802")

#10
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/10/dDWI_og_SENSE_703")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/10/DWI_og_701")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/10/eReg__DWI_og_SENSE_704")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/10/Reg__DWI_og_SENSE_702")

#11
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/11/dDWI_og_SENSE_803")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/11/DWI_og_801")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/11/eReg__DWI_og_SENSE_804")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/11/Reg__DWI_og_SENSE_802")


#12
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/12/dDWI_og_SENSE_703")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/12/DWI_og_701")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/12/eReg__DWI_og_SENSE_704")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/12/Reg__DWI_og_SENSE_702")


#13
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/13/dDWI_og_SENSE_703")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/13/DWI_og_701")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/13/eReg__DWI_og_SENSE_704")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/13/Reg__DWI_og_SENSE_702")


#14
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/14/dDWI_og_SENSE_803")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/14/DWI_og_801")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/14/eReg__DWI_og_SENSE_804")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/14/Reg__DWI_og_SENSE_802")

#15
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/15/dDWI_og_SENSE_903")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/15/DWI_og_901")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/15/eReg__DWI_og_SENSE_904")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/15/Reg__DWI_og_SENSE_902")

####
#16
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/16/dDWI_og_SENSE_703")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/16/DWI_og_701")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/16/eReg__DWI_og_SENSE_704")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/16/Reg__DWI_og_SENSE_702")

#17
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/17/dDWI_og_SENSE_903")
#image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/17/eReg__DWI_og_SENSE_904")
# #image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/17/Reg__DWI_og_SENSE_902")
# image_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/17/DWI_og_901")
#
#
#
# # Ruta a la carpeta de destino donde se copiarán las imágenes renombradas
# destination_folder = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes/normal")

# # Verificar que la carpeta existe
# if not image_folder.exists():
#     print(f"La carpeta {image_folder} no existe.")
# else:
#     # Recorrer los archivos en la carpeta
#     for file_path in image_folder.iterdir():
#         if file_path.is_file() and file_path.name.startswith("IM"):
#             # Generar el nuevo nombre reemplazando "IM" por "normalcase0001_dwi_slice"
#             new_name = file_path.name.replace("IM", "normalcase0017_dwi_slice", 1)  # Solo reemplaza la primera ocurrencia
#             new_path = destination_folder / new_name
#
#             # Copiar el archivo con el nuevo nombre
#             shutil.copy2(file_path, new_path)
#             print(f'Copiado y renombrado: {file_path.name} -> {new_name}')
#
#             # # Renombrar el archivo
#             # file_path.rename(new_path)
#             # print(f'Renombrado: {file_path.name} -> {new_name}')

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ========================
# Funciones para leer imágenes
# ========================

def get_patient_images(patient_folder):
    """
    Recorre todas las subcarpetas de un paciente y recolecta las rutas de las imágenes.
    """
    patient_images = []
    for root, _, files in os.walk(patient_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                patient_images.append(os.path.join(root, file))
    return patient_images

def get_all_patients_images(normal_folder):
    """
    Recorre todas las carpetas de pacientes y recolecta las rutas de las imágenes.
    """
    patients_images = {}
    for patient in os.listdir(normal_folder):
        patient_folder = os.path.join(normal_folder, patient)
        if os.path.isdir(patient_folder):
            patients_images[patient] = get_patient_images(patient_folder)
    return patients_images

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
# División de datos por paciente
# ========================

def split_data_by_patient(patients_images, test_size=0.2, val_size=0.2):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba por paciente.
    """
    # Obtén la lista de pacientes
    patients = list(patients_images.keys())

    # Divide los pacientes en conjuntos de entrenamiento, validación y prueba
    train_patients, test_patients = train_test_split(patients, test_size=test_size, random_state=42)
    train_patients, val_patients = train_test_split(train_patients, test_size=val_size, random_state=42)

    # Asigna las imágenes a los conjuntos según el paciente
    train_images = []
    val_images = []
    test_images = []

    for patient, images in patients_images.items():
        if patient in train_patients:
            train_images.extend(images)
        elif patient in val_patients:
            val_images.extend(images)
        elif patient in test_patients:
            test_images.extend(images)

    print(f"Train set: {len(train_images)} imágenes")
    print(f"Val set: {len(val_images)} imágenes")
    print(f"Test set: {len(test_images)} imágenes")

    return train_images, val_images, test_images

# ========================
# Ejecución principal
# ========================

if __name__ == "__main__":
    # Ruta a la carpeta 'normal'
    normal_folder = "C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/DWI Clase Normal Imagenes"

    # Obtén las imágenes de todos los pacientes
    patients_images = get_all_patients_images(normal_folder)

    # Imprime el número de imágenes por paciente
    for patient, images in patients_images.items():
        print(f"Paciente {patient}: {len(images)} imágenes")

    # Divide los datos por paciente
    train_images, val_images, test_images = split_data_by_patient(patients_images)

    # Verifica la fuga de datos a nivel de píxeles
    check_pixel_level_leakage(train_images, val_images, test_images, threshold=0.98)
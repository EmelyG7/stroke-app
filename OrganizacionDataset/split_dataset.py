import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Define las rutas originales y la ruta de salida
dataset1_path = "../Brain_Stroke_CT-SCAN_image"
dataset2_path = "../Brain_Data_Organised"
merged_dataset_path = "../Dataset"

# Crear carpetas principales de salida
categories = ['Normal', 'Stroke']
for category in categories:
    os.makedirs(os.path.join(merged_dataset_path, category), exist_ok=True)

# Dividir en conjuntos de entrenamiento y validación
def split_dataset_three_sets(base_dir, categories, val_ratio=0.2, test_ratio=0.1):
    """
    Divide el dataset entre train, val y test según las proporciones indicadas.

    Args:
    - base_dir: Ruta base donde se encuentran las carpetas de las categorías.
    - categories: Lista de categorías (por ejemplo, ['Normal', 'Stroke']).
    - val_ratio: Proporción del conjunto de validación.
    - test_ratio: Proporción del conjunto de prueba.
    """
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    # Crear directorios train, val y test
    for category in categories:
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

    for category in categories:
        category_path = os.path.join(base_dir, category)
        images = [img for img in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, img))]

        # Dividir imágenes en train, val y test
        train_imgs, temp_imgs = train_test_split(images, test_size=(val_ratio + test_ratio), random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio / (val_ratio + test_ratio),
                                               random_state=42)

        # Mover imágenes a las carpetas correspondientes
        for img_name in train_imgs:
            src_path = os.path.join(category_path, img_name)
            dst_path = os.path.join(train_dir, category, img_name)
            shutil.move(src_path, dst_path)

        for img_name in val_imgs:
            src_path = os.path.join(category_path, img_name)
            dst_path = os.path.join(val_dir, category, img_name)
            shutil.move(src_path, dst_path)

        for img_name in test_imgs:
            src_path = os.path.join(category_path, img_name)
            dst_path = os.path.join(test_dir, category, img_name)
            shutil.move(src_path, dst_path)

    print("¡Dataset dividido en train, val y test!")


# Dividir el conjunto fusionado en train, val y test
split_dataset_three_sets(merged_dataset_path, categories, val_ratio=0.2, test_ratio=0.1)

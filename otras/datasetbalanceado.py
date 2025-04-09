from pathlib import Path
import shutil
import random

# Directorios de origen
source_normal = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/dataset_dwi_stroke_sin_balancear/normal")
source_stroke = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/dataset_dwi_stroke_sin_balancear/stroke")

# Directorios de destino (nuevo dataset balanceado)
dest_dataset = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/dataset_dwi_stroke")
dest_normal = dest_dataset / "normal"
dest_stroke = dest_dataset / "stroke"

# Crear carpetas destino si no existen
dest_normal.mkdir(parents=True, exist_ok=True)
dest_stroke.mkdir(parents=True, exist_ok=True)

# Copiar todas las imágenes de la carpeta "normal" (3264 imágenes)
for file in source_normal.iterdir():
    if file.is_file():
        shutil.copy2(file, dest_normal / file.name)

print(f"Copiadas {len(list(source_normal.iterdir()))} imágenes de 'normal'.")

# Seleccionar aleatoriamente 3264 imágenes de "stroke"
stroke_files = list(source_stroke.iterdir())
selected_stroke_files = random.sample(stroke_files, 3264)  # Selección aleatoria

# Copiar las imágenes seleccionadas de "stroke"
for file in selected_stroke_files:
    shutil.copy2(file, dest_stroke / file.name)

print(f"Copiadas {len(selected_stroke_files)} imágenes de 'stroke'.")
print("✅ Dataset balanceado creado exitosamente.")


#para mover las imagenes que no se usaron en el dataset usado actualmente, estqas imagenes se copiaron en otra carpeta
#para realizar las pruebas con imagenes que no tenga el dataset, es decir imagenes nuevas

# from pathlib import Path
# import shutil
#
# # Directorios a comparar
# source_stroke = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/dataset_dwi_stroke_sin_balancear/stroke")
# dest_stroke = Path("C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/dataset_dwi_stroke/stroke")
#
# # Directorio destino para las imágenes no seleccionadas
# dest_missing = Path("C:/Users/Coshita/Downloads/tesis presentacion graficas/streamlit imagenes para prueba/stroke")
# dest_missing.mkdir(parents=True, exist_ok=True)  # Crear carpeta si no existe
#
# # Obtener los nombres de los archivos en ambos directorios
# source_files = {file.name for file in source_stroke.iterdir() if file.is_file()}
# dest_files = {file.name for file in dest_stroke.iterdir() if file.is_file()}
#
# # Identificar archivos que están en source_stroke pero no en dest_stroke
# missing_files = source_files - dest_files
#
# # Copiar las imágenes faltantes al nuevo directorio
# copied_count = 0
# if missing_files:
#     print(f"Se encontraron {len(missing_files)} imágenes en '{source_stroke}' que no están en '{dest_stroke}':")
#     for file_name in sorted(missing_files):  # Ordenar para mejor legibilidad
#         source_path = source_stroke / file_name
#         dest_path = dest_missing / file_name
#         shutil.copy2(source_path, dest_path)  # Copiar archivo preservando metadatos
#         copied_count += 1
#         print(f" - Copiada: {file_name}")
# else:
#     print(f"Todas las imágenes de '{source_stroke}' están presentes en '{dest_stroke}'.")
#
# # Mostrar resumen
# print(f"\nTotal de imágenes en '{source_stroke}': {len(source_files)}")
# print(f"Total de imágenes en '{dest_stroke}': {len(dest_files)}")
# print(f"Total de imágenes copiadas a '{dest_missing}': {copied_count}")
# print("✅ Proceso completado exitosamente.")
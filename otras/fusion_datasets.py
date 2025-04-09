# import os
# import shutil
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
#
# # Define las rutas originales y la ruta de salida
# dataset1_path = "../Brain_Stroke_CT-SCAN_image"
# dataset2_path = "../Brain_Data_Organised"
# merged_dataset_path = "../Dataset"
#
# # Función para copiar imágenes
# def copy_images(src, dst):
#     for img_name in tqdm(os.listdir(src), desc=f"Copiando desde {src}"):
#         src_path = os.path.join(src, img_name)
#         dst_path = os.path.join(dst, img_name)
#
#         # Crear directorio de destino si no existe
#         os.makedirs(os.path.dirname(dst_path), exist_ok=True)
#
#         try:
#             shutil.copy(src_path, dst_path)
#         except Exception as e:
#             print(f"Error al copiar {src_path} a {dst_path}: {e}")
#
#
# #Procesar Dataset 1 (Stroke)
# copy_images(os.path.join(dataset1_path, "Train", "hemorrhagic"),
#             os.path.join(merged_dataset_path, "Stroke"))
# copy_images(os.path.join(dataset1_path, "Train", "ischaemic"),
#             os.path.join(merged_dataset_path, "Stroke"))
# copy_images(os.path.join(dataset1_path, "Validation", "hemorrhagic"),
#             os.path.join(merged_dataset_path, "Stroke"))
# copy_images(os.path.join(dataset1_path, "Validation", "ischaemic"),
#             os.path.join(merged_dataset_path, "Stroke"))
# copy_images(os.path.join(dataset1_path, "Test", "hemorrhagic"),
#             os.path.join(merged_dataset_path, "Stroke"))
# copy_images(os.path.join(dataset1_path, "Test", "ischaemic"),
#             os.path.join(merged_dataset_path, "Stroke"))
#
# # Procesar Dataset 2
# copy_images(os.path.join(dataset2_path, "Normal"),
#             os.path.join(merged_dataset_path, "Normal"))
# copy_images(os.path.join(dataset2_path, "Stroke"),
#             os.path.join(merged_dataset_path, "Stroke"))
#
# # Verificar el número de imágenes copiadas
# normal_count = len(os.listdir(os.path.join(merged_dataset_path, "Normal")))
# stroke_count = len(os.listdir(os.path.join(merged_dataset_path, "Stroke")))
#
# print("\nResumen de la fusión:")
# print(f"Imágenes en carpeta Normal: {normal_count}")
# print(f"Imágenes en carpeta Stroke: {stroke_count}")
# print("¡Fusión completada!")
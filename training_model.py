import os
import shutil
import random
import logging
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
import PIL.Image

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import datasets, layers, models, regularizers, optimizers
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.utils import to_categorical, image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4, ResNet50
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation,
                                     BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras.metrics import Precision, Recall, RootMeanSquaredError

from sklearn.model_selection import (train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, KFold)
from sklearn.metrics import (classification_report, f1_score, confusion_matrix,
                             roc_auc_score, accuracy_score)
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler
from scikeras.wrappers import KerasClassifier

from tqdm import tqdm

import keras_tuner as kt

import gc

# def clean_memory():
#     gc.collect()
#     K.clear_session()
#
# # Llamar antes de entrenar
# clean_memory()

# Disable TensorFlow warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# Configuraciones globales
IMG_SIZE = (160, 224)
TARGET_SIZE = (160, 224, 3)
BATCH_SIZE = 32
EPOCHS = 60
NUM_CLASSES = 2  # Normal y Stroke
AUTOTUNE = tf.data.AUTOTUNE

# Constantes de evaluación y regularización
LEARNING_RATE = 0.0003
W_REGULARIZER = 1e-5
PATIENCE = 7
FACTOR = 0.3

# Definir optimizadores
optimizers = [
    tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=0.9, epsilon=1e-08),
    tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
    tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE)
]

# Definir métricas
METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

# Callbacks
reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=FACTOR,
    patience=PATIENCE,
    min_lr=W_REGULARIZER,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    patience=10,
    restore_best_weights=True
)

# Capas de preprocesamiento
resize_and_rescale = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.Resizing(IMG_SIZE[0], IMG_SIZE[1])
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.8),
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    layers.RandomZoom(0.3)
])

base_dir = "Dataset/"
batch_size = 32
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_dir, "train"),
    labels='inferred',
    label_mode='binary',
    shuffle=True,
    seed=123,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_dir, "val"),
    labels='inferred',
    label_mode='binary',
    shuffle=False,
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_dir, "test"),
    labels='inferred',
    label_mode='binary',
    shuffle=False,
    batch_size=batch_size
)

print(len(train_ds))

class_names = train_ds.class_names
print(class_names)

# Opcional: Verificación de los tamaños de los subconjuntos
# Conteo de imágenes
train_normal = len(os.listdir(os.path.join(base_dir, "train", "Normal")))
train_stroke = len(os.listdir(os.path.join(base_dir, "train", "Stroke")))
val_normal = len(os.listdir(os.path.join(base_dir, "val", "Normal")))
val_stroke = len(os.listdir(os.path.join(base_dir, "val", "Stroke")))
test_normal = len(os.listdir(os.path.join(base_dir, "test", "Normal")))
test_stroke = len(os.listdir(os.path.join(base_dir, "test", "Stroke")))


print(f"Tamaño del conjunto de entrenamiento 'train/Normal': {train_normal}")
print(f"Tamaño del conjunto de entrenamiento 'train/Stroke': {train_stroke}")
print(f"Tamaño del conjunto de validación 'val/Normal': {val_normal}")
print(f"Tamaño del conjunto de validación 'val/Stroke': {val_stroke}")
print(f"Tamaño del conjunto de prueba 'test/Normal': {test_normal}")
print(f"Tamaño del conjunto de prueba 'test/Stroke': {test_stroke}")

train_dataset = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

images_total, labels_total = next(iter(train_ds))
print('Batch shape: ', images_total.shape)
print('Label shape: ', labels_total.shape)

images, labels = next(iter(train_dataset))
print('Batch shape: ', images.shape)
print('Label shape: ', labels.shape)

image_val,labels_val = next(iter(val_dataset))
print('Batch shape: ', image_val.shape)
print('Label shape: ', labels_val.shape)

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(5):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.axis("off")
#
# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#   for i in range(9):
#     resized = resize_and_rescale(images)
#     augmented_images = data_augmentation(resized)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(augmented_images[0])
#     plt.axis("off")


model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=TARGET_SIZE),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

#build and summary layers
# model.build(input_shape=input_shape)
model.summary()



model.compile(
    loss='binary_crossentropy',  # Cambiado a binary_crossentropy
    optimizer=optimizers[0],  # Usando Adam
    metrics=METRICS
)

# Entrenar el modelo
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[reduce_lr_on_plateau, early_stopping],
    verbose=1
)

metrics = model.evaluate(test_dataset, verbose=2)

precision = metrics[2]
recall = metrics[3]
accuracy = metrics[1]
loss = metrics[0]
RMSE = metrics[4]
f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

print(f'Exactitud: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Loss: {loss}')
print(f'f1_score: {f1_score}')

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label = 'Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label = 'Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

y_pred = model.predict(test_dataset)
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
print(len(y_true))
print(len(y_pred))
rounded_pred = np.round(y_pred).astype(int)  # Para clasificación binaria
print(y_pred)

print(classification_report(y_true, rounded_pred, target_names=class_names))

# Generar la matriz de confusión
cf_matrix = confusion_matrix(y_true, rounded_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cf_matrix, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Anotar los valores en la matriz de confusión
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, str(cf_matrix[i, j]), horizontalalignment='center', verticalalignment='center', color='black')

plt.show()

# Liberar memoria GPU si estás usando una
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    )



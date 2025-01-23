import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
# Añadir EfficientNet como modelo alternativo para pruebas
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf

def focal_loss(alpha, gamma=2):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        weight = alpha * y_true * (1 - y_pred)**gamma + (1 - alpha) * (1 - y_true) * y_pred**gamma
        focal_loss = tf.reduce_mean(weight * cross_entropy)
        return focal_loss
    return focal_loss_fixed
# Configuración
IMG_SIZE = (224, 224)  # Tamaño de las imágenes que usa ResNet-50
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 2  # Dos clases: Normal y Stroke
class_weights = {
    0: 1.0,  # Normal
    1: 10.0   # Stroke
}

# Directorio base
base_dir = "Brain_Data_Organised/"
classes = ["Normal", "Stroke"]

# Crear carpetas destino para train y val
for split in ["train", "val"]:
    for cls in classes:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# Por clase, dividir imágenes en train/val
#Probando el GIT1
#Probando el GIT222222
for cls in classes:
    class_dir = os.path.join(base_dir, cls)
    images = os.listdir(class_dir)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)  # 80% train, 20% val

    # Copiar imágenes a las respectivas carpetas
    for img in train_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(base_dir, "train", cls, img))
    for img in val_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(base_dir, "val", cls, img))

# Nota: Tu directorio original queda vacío tras este proceso

if not os.path.exists(base_dir):
    raise FileNotFoundError(f"La ruta {base_dir} no existe.")

train_normal = len(os.listdir("Brain_Data_Organised/train/Normal"))
train_stroke = len(os.listdir("Brain_Data_Organised/train/Stroke"))

val_normal = len(os.listdir("Brain_Data_Organised/val/Normal"))
val_stroke = len(os.listdir("Brain_Data_Organised/val/Stroke"))

print(f"Imágenes en 'train/Normal': {train_normal}")
print(f"Imágenes en 'train/Stroke': {train_stroke}")
print(f"Imágenes en 'val/Normal': {val_normal}")
print(f"Imágenes en 'val/Stroke': {val_stroke}")

# Dividir automáticamente los datos de entrenamiento y validación (entrenamiento: 80%, validación: 20%)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normaliza los datos entre 0 y 1
    rotation_range=45,  # Incrementa la rotación
    width_shift_range=0.4,  # Permite mayores desplazamientos horizontales
    height_shift_range=0.4,  # Permite mayores desplazamientos verticales
    shear_range=0.4,  # Permite un mayor rango de inclinación
    zoom_range=0.4,  # Aumenta el rango del zoom
    horizontal_flip=True,
    fill_mode="nearest"  # Rellena píxeles que falten tras transformaciones
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalizar valores entre 0 y 1
)

# Generadores de datos para entrenamiento y validación
train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",  # Para clasificación binaria
    shuffle=True  # Mezclar datos
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(base_dir, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",  # Para clasificación binaria
    shuffle=False  # Sin mezclar para que las predicciones sean consistentes
)

print("Distribución de entrenamiento:")
for cls in classes:
    print(f"{cls}: {len(os.listdir(os.path.join(base_dir, 'train', cls)))} imágenes")

print("\nDistribución de validación:")
for cls in classes:
    print(f"{cls}: {len(os.listdir(os.path.join(base_dir, 'val', cls)))} imágenes")

# Cargar el modelo base ResNet-50 y EfficientNetB0 preentrenado en ImageNet
use_efficientnet = True  # Cambiar a True para probar EfficientNetB0
base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Congelar las capas base (para transfer learning)
# Ajustar capas base para fine-tuning avanzado
base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.7)  # Descongelar parcialmente el 30%
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
# Añadir capas personalizadas en la parte superior
x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)  # Incrementa la capacidad de esta capa
x = BatchNormalization()(x)  # Capa de normalización
x = Dropout(0.5)(x)  # Dropout para evitar sobreajuste
x = Dense(128, activation="relu")(x)  # Añade otra capa completamente conectada
x = BatchNormalization()(x)  # Capa de normalización
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

# Crear el modelo completo
model = Model(inputs=base_model.input, outputs=output)

# Compilar el modelo
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=focal_loss(alpha=0.75, gamma=2.0),
    metrics=["accuracy", "Precision", "Recall"]
)
# Preparar ajuste fino después de algunas épocas (descomentar cuando sea necesario)
def fine_tune_model():
    for layer in base_model.layers[-20:]:  # Ajustar solo últimas 20 capas
        layer.trainable = True


# Callback para reducir la tasa de aprendizaje si no mejora
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,  # Número de épocas sin mejora antes de detener
    restore_best_weights=True  # Restaurar el modelo con los mejores pesos
)

# Actualizar los callbacks
callbacks = [reduce_lr, early_stopping]

# Entrenar con los nuevos callbacks
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks,
    steps_per_epoch=train_generator.samples // BATCH_SIZE
)

# Evaluar el modelo
results = model.evaluate(val_generator)
loss, accuracy, precision, recall = results  # Ajustar para capturar todas las métricas devueltas
print(f"Pérdida en validación: {loss}")
print(f"Precisión en validación: {accuracy}")
print(f"Exactitud en validación (precision): {precision}")
print(f"Sensibilidad en validación (recall): {recall}")

# Predicciones
y_true = val_generator.classes
y_pred = model.predict(val_generator)
y_pred_classes = y_pred > 0.5

print(classification_report(y_true, y_pred_classes, target_names=list(val_generator.class_indices.keys())))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy entrenamiento')
plt.plot(history.history['val_accuracy'], label='Accuracy validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss entrenamiento')
plt.plot(history.history['val_loss'], label='Loss validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()



# Guardar el modelo entrenado
model.save("modelo_resnet50_brain_data.h5")
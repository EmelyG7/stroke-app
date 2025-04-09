# Importando librerias
import warnings

from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.legacy import backend

warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from PIL import Image
import tensorflow as tf
from keras import Sequential
from keras.src.optimizers import Adam
from keras.src.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Data Collection
normal_path = "Brain_Data_Organised/Normal"
stroke_path = "Brain_Data_Organised/Stroke"

normal_folder = os.listdir(normal_path)
stroke_folder = os.listdir(stroke_path)

# IMAGE PROCESSING
# Resize the Images and Convert to Numpy Arrays
data = []
for img_file in normal_folder:
    image = Image.open(os.path.join(normal_path, img_file))
    image = image.resize((224, 224))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

for img_file in stroke_folder:
    image = Image.open(os.path.join(stroke_path, img_file))
    image = image.resize((224, 224))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

# Class Labels
normal_label = [0] * len(normal_folder)
stroke_label = [1] * len(stroke_folder)
Target_label = normal_label + stroke_label

# Convert Image data and target labels into array
x = np.array(data)
y = np.array(Target_label)

# Split The Data for training and testing purpose
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=True)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Definir las capas de preprocesamiento y aumento de datos
resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),  # Normalización de píxeles
    tf.keras.layers.Resizing(224, 224)  # Redimensionamiento
])

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),  # Volteo aleatorio
    tf.keras.layers.RandomRotation(0.2),  # Rotación aleatoria (reducida para evitar distorsiones)
    tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),  # Traslación aleatoria
    tf.keras.layers.RandomZoom(0.1)  # Zoom aleatorio (reducido para evitar distorsiones)
])

# Aplicar preprocesamiento y aumento de datos al dataset de entrenamiento
x_train_augmented = data_augmentation(resize_and_rescale(x_train))

# Aplicar solo preprocesamiento al dataset de prueba
x_test_rescaled = resize_and_rescale(x_test)

# Image Data Visualization
class_labels = ["Normal", "Stroke"]
plt.figure(figsize=(16, 24))
for i in range(24):
    plt.subplot(6, 4, i + 1)
    plt.imshow(x_train_augmented[i].numpy())
    plt.title(f"Actual Label: {class_labels[y_train[i]]}")
    plt.axis("off")
plt.show()

# Definir el optimizador con 'learning_rate'
optimizer = Adam(learning_rate=0.001)

# Create Model Using CNN
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu", input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))  # Añadir Dropout para regularización

model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))  # Añadir Dropout para regularización

model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))  # Añadir Dropout para regularización

model.add(Flatten())
model.add(Dense(units=256, activation="relu"))  # Reducir unidades
model.add(Dropout(0.5))  # Añadir Dropout para regularización
model.add(Dense(units=128, activation="relu"))  # Reducir unidades
model.add(Dropout(0.5))  # Añadir Dropout para regularización
model.add(Dense(units=1, activation="sigmoid"))

# Compilar el modelo con pesos de clase para manejar el desbalance
class_weights = {0: 1.0, 1: 2.0}  # Dar más peso a la clase "Stroke"
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

# Callbacks para evitar sobreajuste
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)


# Corregir acceso al learning rate, si es necesario un registro o ajuste adicional
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        current_lr = backend.get_value(self.model.optimizer.learning_rate)
        print(f"Epoch {epoch + 1}: Learning rate is {current_lr}")


# Entrenar el modelo con los callbacks corregidos
history = model.fit(
    x_train_augmented, y_train,
    batch_size=32,
    epochs=30,  # Aumentar el número de épocas
    validation_data=(x_test_rescaled, y_test),
    callbacks=[early_stopping, reduce_lr, CustomCallback()],
    class_weight=class_weights
)

# Model Evaluation on Test and Train Data
loss, acc = model.evaluate(x_test_rescaled, y_test)
print("Loss on Test Data:", loss)
print("Accuracy on Test Data:", acc)

loss, acc = model.evaluate(x_train_augmented, y_train)
print("Loss on Train Data:", loss)
print("Accuracy on Train Data:", acc)

# Predictions on Test Image Data
y_pred_test = model.predict(x_test_rescaled)
y_pred_test_label = [1 if i >= 0.5 else 0 for i in y_pred_test]

print("Actual Label:", y_test[:10])
print("Predicted Label:", y_pred_test_label[:10])

# Metrics Evaluation On Test Data
print("-----Metrics Evaluation On Test Data-----")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_label))
print("Classification Report:\n", classification_report(y_test, y_pred_test_label))

# ROC Curve
from sklearn.metrics import roc_curve, auc

y_prob = model.predict(x_test_rescaled)
fpr, tpr, threshold = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"ROC - Area: {roc_auc}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# PR Curve
from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
print(f"PR - Area: {pr_auc}")

plt.figure(figsize=(8, 6))
plt.step(recall, precision, color='b', where='post', label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PR) Curve')
plt.legend(loc='upper right')
plt.show()

# Image Predictions on Test Data
plt.figure(figsize=(16, 32))
for i in range(30):
    plt.subplot(8, 4, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Actual: {class_labels[y_test[i]]}\nPredicted: {class_labels[y_pred_test_label[i]]}")
    plt.axis("off")
plt.show()
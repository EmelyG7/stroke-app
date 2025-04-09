import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, GlobalAveragePooling2D, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Configuración inicial
img_size = (240, 240)
batch_size = 16
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# Métrica F1 personalizada
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)


# Preprocesamiento de imágenes médicas
def preprocess_medical_image(img):
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[-1] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    bilateral = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
    unsharp_image = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    normalized = cv2.normalize(unsharp_image, None, 0, 255, cv2.NORM_MINMAX)
    processed = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    return processed.astype(np.float32) / 255.0


# Generadores de datos
def get_train_datagen():
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='reflect',
        preprocessing_function=preprocess_medical_image
    )


def get_val_datagen():
    return ImageDataGenerator(
        preprocessing_function=preprocess_medical_image
    )


# Dataset con pacientes sintéticos y restricción a 13 imágenes por paciente en "normal"
def prepare_balanced_dataset_with_synthetics(data_dir, target_patients_per_class=246, target_images_per_patient=13):
    normal_dir = os.path.join(data_dir, "normal")
    stroke_dir = os.path.join(data_dir, "stroke")
    normal_patients = defaultdict(list)
    stroke_patients = defaultdict(list)

    for img_file in os.listdir(normal_dir):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            patient_id = img_file.split('_')[0]
            normal_patients[patient_id].append(os.path.join(normal_dir, img_file))

    for img_file in os.listdir(stroke_dir):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            patient_id = img_file.split('_')[0]
            stroke_patients[patient_id].append(os.path.join(stroke_dir, img_file))

    all_paths = []
    all_labels = []
    patient_groups = []

    for patient_id in normal_patients:
        images = normal_patients[patient_id]
        selected_images = random.sample(images, min(target_images_per_patient, len(images)))
        all_paths.extend(selected_images)
        all_labels.extend([0] * len(selected_images))
        patient_groups.extend([patient_id] * len(selected_images))

    num_synthetic_patients = target_patients_per_class - len(normal_patients)
    synthetic_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.9, 1.1),
        fill_mode='reflect'
    )
    normal_images = []
    for patient_id in normal_patients:
        for img_path in normal_patients[patient_id][:target_images_per_patient]:
            img = cv2.imread(img_path)
            if img is not None:
                normal_images.append(img)
    normal_images = np.array(normal_images)

    os.makedirs("synthetic", exist_ok=True)
    for i in range(num_synthetic_patients):
        synthetic_patient_id = f"normal_synth_{i + 1}"
        synthetic_images = []
        for j in range(target_images_per_patient):
            base_idx = np.random.randint(0, len(normal_images))
            base_img = normal_images[base_idx].copy()
            transformed = synthetic_datagen.random_transform(base_img)
            gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
            kernel_size = np.random.choice([3, 5])
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            morph = cv2.dilate(gray, kernel, iterations=1) if np.random.rand() < 0.5 else cv2.erode(gray, kernel,
                                                                                                    iterations=1)
            blend_factor = np.random.uniform(0.7, 0.9)
            blended = cv2.addWeighted(gray, blend_factor, morph, 1 - blend_factor, 0)
            synthetic_img = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)
            synthetic_path = os.path.join("synthetic", f"{synthetic_patient_id}_{j + 1}.jpg")
            cv2.imwrite(synthetic_path, synthetic_img)
            synthetic_images.append(synthetic_path)
        all_paths.extend(synthetic_images)
        all_labels.extend([0] * len(synthetic_images))
        patient_groups.extend([synthetic_patient_id] * len(synthetic_images))

    for patient_id in stroke_patients:
        images = stroke_patients[patient_id]
        all_paths.extend(images)
        all_labels.extend([1] * len(images))
        patient_groups.extend([patient_id] * len(images))

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)
    patient_groups = np.array(patient_groups)

    print(f"Dataset balanceado - Normal: {len(np.unique(patient_groups[all_labels == 0]))} pacientes, "
          f"{np.sum(all_labels == 0)} imágenes")
    print(f"Dataset balanceado - Stroke: {len(np.unique(patient_groups[all_labels == 1]))} pacientes, "
          f"{np.sum(all_labels == 1)} imágenes")
    return all_paths, all_labels, patient_groups


# Modelo DenseNet121 con mecanismo de atención
def create_densenet_model(input_shape=(240, 240, 3)):
    base_model = DenseNet121(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    attention = Dense(1, activation='sigmoid')(x)
    attention = tf.keras.layers.Reshape((1,))(attention)
    x = Multiply()([x, attention])

    x = Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model


# Crear generadores de datos
def create_data_generators(x_train_paths, y_train, x_val_paths, y_val):
    train_df = pd.DataFrame({'filename': x_train_paths, 'class': y_train.astype(str)})
    val_df = pd.DataFrame({'filename': x_val_paths, 'class': y_val.astype(str)})

    train_gen = get_train_datagen().flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    val_gen = get_val_datagen().flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return train_gen, val_gen


# Entrenamiento del modelo
def train_densenet_model(x_train_paths, y_train, x_val_paths, y_val, fold):
    model = create_densenet_model()
    train_gen, val_gen = create_data_generators(x_train_paths, y_train, x_val_paths, y_val)

    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', AUC(name='auc'), F1Score()]
    )

    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(f"best_densenet121_fold_{fold}.keras", monitor="val_auc", mode='max', save_best_only=True,
                        verbose=1)
    ]

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    history = model.fit(
        train_gen,
        epochs=40,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    return model, history


# Evaluación del modelo
def evaluate_model_with_optimal_threshold(model, x_val_paths, y_val):
    val_df = pd.DataFrame({'filename': x_val_paths, 'class': y_val.astype(str)})
    val_gen = get_val_datagen().flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    predictions = model.predict(val_gen).flatten()
    fpr, tpr, thresholds = roc_curve(y_val, predictions)
    j_scores = tpr - fpr
    optimal_threshold = thresholds[np.argmax(j_scores)]
    y_pred_binary = (predictions >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_val, y_pred_binary)
    precision = np.sum(y_pred_binary * y_val) / (np.sum(y_pred_binary) + 1e-10)
    recall = np.sum(y_pred_binary * y_val) / (np.sum(y_val) + 1e-10)
    specificity = np.sum((1 - y_pred_binary) * (1 - y_val)) / (np.sum(1 - y_val) + 1e-10)
    f1 = f1_score(y_val, y_pred_binary)
    auc_score = roc_auc_score(y_val, predictions)
    cm = confusion_matrix(y_val, y_pred_binary)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'auc': auc_score,
        'confusion_matrix': cm,
        'optimal_threshold': optimal_threshold
    }, predictions


# Visualización de resultados
def visualize_results(history, results, y_val, y_pred_prob, fold, model_name="DenseNet121"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy
    axs[0, 0].plot(history.history['accuracy'])
    axs[0, 0].plot(history.history['val_accuracy'])
    axs[0, 0].set_title(f'{model_name} - Fold {fold} Accuracy')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].legend(['Train', 'Validation'], loc='upper left')

    # Loss
    axs[0, 1].plot(history.history['loss'])
    axs[0, 1].plot(history.history['val_loss'])
    axs[0, 1].set_title(f'{model_name} - Fold {fold} Loss')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].legend(['Train', 'Validation'], loc='upper left')

    # Confusion Matrix
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axs[1, 0])
    axs[1, 0].set_title(f'{model_name} - Fold {fold} Confusion Matrix')
    axs[1, 0].set_ylabel('True Label')
    axs[1, 0].set_xlabel('Predicted Label')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axs[1, 1].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    axs[1, 1].plot([0, 1], [0, 1], 'k--')
    axs[1, 1].set_title(f'{model_name} - Fold {fold} ROC Curve')
    axs[1, 1].set_xlabel('False Positive Rate')
    axs[1, 1].set_ylabel('True Positive Rate')
    axs[1, 1].legend(loc='lower right')

    plt.tight_layout()
    plt.show()


# Evaluación en el conjunto de test
def evaluate_on_test_set(best_model_path, test_paths, test_labels):
    best_model = tf.keras.models.load_model(best_model_path, custom_objects={"F1Score": F1Score})
    test_df = pd.DataFrame({'filename': test_paths, 'class': test_labels.astype(str)})
    test_gen = get_val_datagen().flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    predictions = best_model.predict(test_gen).flatten()
    fpr, tpr, thresholds = roc_curve(test_labels, predictions)
    j_scores = tpr - fpr
    optimal_threshold = thresholds[np.argmax(j_scores)]
    y_pred_binary = (predictions >= optimal_threshold).astype(int)
    accuracy = accuracy_score(test_labels, y_pred_binary)
    precision = np.sum(y_pred_binary * test_labels) / (np.sum(y_pred_binary) + 1e-10)
    recall = np.sum(y_pred_binary * test_labels) / (np.sum(test_labels) + 1e-10)
    specificity = np.sum((1 - y_pred_binary) * (1 - test_labels)) / (np.sum(1 - test_labels) + 1e-10)
    f1 = f1_score(test_labels, y_pred_binary)
    auc_score = roc_auc_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, y_pred_binary)

    print("\n=== Evaluación en el Conjunto de Test ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test Specificity: {specificity:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test AUC: {auc_score:.4f}")
    print(f"Test Optimal Threshold: {optimal_threshold:.4f}")

    # Visualización
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0])
    axs[0].set_title('Test Confusion Matrix')
    axs[0].set_ylabel('True Label')
    axs[0].set_xlabel('Predicted Label')
    axs[1].plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set_title('Test ROC Curve')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].legend(loc='lower right')
    plt.tight_layout()
    plt.show()


# Ejecución principal
if __name__ == "__main__":
    # Cambia la ruta según tu dataset en Kaggle
    data_dir = "C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/dataset_dwi_stroke"
    all_paths, all_labels, patient_groups = prepare_balanced_dataset_with_synthetics(data_dir)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    results = []
    best_fold = None
    best_auc = -1

    # Cross-validation
    for train_idx, val_idx in sgkf.split(all_paths, all_labels, patient_groups):
        print(f"\n== Fold {fold}/5 ==")

        x_train_paths = all_paths[train_idx]
        y_train = all_labels[train_idx]
        x_val_paths = all_paths[val_idx]
        y_val = all_labels[val_idx]

        print(f"Imágenes en entrenamiento: {len(x_train_paths)} "
              f"(Normal: {np.sum(y_train == 0)}, Stroke: {np.sum(y_train == 1)}, "
              f"Ratio: {np.sum(y_train == 0) / np.sum(y_train == 1):.2f})")
        print(f"Imágenes en validación: {len(x_val_paths)} "
              f"(Normal: {np.sum(y_val == 0)}, Stroke: {np.sum(y_val == 1)}, "
              f"Ratio: {np.sum(y_val == 0) / np.sum(y_val == 1):.2f})")

        model, history = train_densenet_model(x_train_paths, y_train, x_val_paths, y_val, fold)
        metrics, predictions = evaluate_model_with_optimal_threshold(model, x_val_paths, y_val)

        print(f"Fold {fold} - DenseNet121:")
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                print(f"{key.capitalize()}: {value:.4f}")

        visualize_results(history, metrics, y_val, predictions, fold)
        results.append(metrics)

        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_fold = fold

        fold += 1

    # Resultados promedio
    avg_metrics = {key: np.mean([r[key] for r in results]) for key in results[0].keys() if key != 'confusion_matrix'}
    print("\nResultados Promedio DenseNet121:")
    for key, value in avg_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

    # Evaluación en conjunto de test (simulamos que usamos el 20% del dataset como test)
    test_size = int(0.2 * len(all_paths))
    test_idx = np.random.choice(len(all_paths), test_size, replace=False)
    test_paths = all_paths[test_idx]
    test_labels = all_labels[test_idx]
    best_model_path = f"best_densenet121_fold_{best_fold}.keras"
    evaluate_on_test_set(best_model_path, test_paths, test_labels)
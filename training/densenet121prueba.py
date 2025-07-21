import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from tensorflow.keras.mixed_precision import Policy, set_global_policy
from tensorflow.keras import regularizers
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
import gc
import psutil

# Configuración inicial
img_size = (128, 128)
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Limpiar sesión al inicio
tf.keras.backend.clear_session()

# Directorio de salida: mismo directorio que el archivo .py
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Diagnósticos del entorno
print("TensorFlow version:", tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU detected:", physical_devices)
else:
    print("No GPU detected, using CPU")
print(f"Available memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")

# Configuración de mixed precision
policy = Policy('mixed_float16')
set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)


# Ajuste dinámico del batch size según memoria disponible
def get_optimal_batch_size():
    try:
        mem = psutil.virtual_memory().available / (1024 ** 3)  # GB disponibles
        if mem > 12:
            return 32
        elif mem > 8:
            return 24
        elif mem > 4:
            return 16
        else:
            return 8  # Reducido para PCs con poca memoria
    except:
        return 8  # Valor por defecto seguro


batch_size = get_optimal_batch_size()
print(f"Batch size seleccionado: {batch_size}")


# Métrica F1 personalizada
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision(thresholds=threshold, dtype=tf.float32)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold, dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = tf.cast(self.precision.result(), tf.float32)
        r = tf.cast(self.recall.result(), tf.float32)
        f1 = 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
        return tf.cast(f1, tf.float32)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


# Preprocesamiento optimizado con depuración
def preprocess_medical_image(img, debug=False):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    normalized = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, img_size)
    rgb = rgb.astype(np.float32) / 255.0
    if debug:
        print(f"Processed image shape: {rgb.shape}, min: {rgb.min():.4f}, max: {rgb.max():.4f}")
    return rgb


# Generador de datos
class MedicalImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, is_training, debug=False):
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size
        self.is_training = is_training
        self.debug = debug
        self.indices = np.arange(len(paths))
        self.img_size = img_size
        if is_training:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = self.paths[batch_indices]
        batch_labels = self.labels[batch_indices]

        batch_images = []
        valid_labels = []
        for path, label in zip(batch_paths, batch_labels):
            try:
                img = cv2.imread(path)
                if img is None:
                    print(f"Warning: Failed to load image {path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                if self.is_training:
                    if random.random() > 0.5:
                        img = cv2.flip(img, 1)
                img = preprocess_medical_image(img, debug=self.debug and idx == 0)
                if img.shape != (self.img_size[0], self.img_size[1], 3):
                    print(f"Warning: Image {path} has incorrect shape {img.shape}")
                    continue
                batch_images.append(img)
                valid_labels.append(label)
            except Exception as e:
                print(f"Error processing image {path}: {str(e)}")
                continue

        if not batch_images:
            raise ValueError("No valid images in batch")

        return np.array(batch_images), np.array(valid_labels)

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)


# Dataset balanceado
def create_balanced_dataset(paths, labels, oversample_ratio=1.8):
    class0_paths = paths[labels == 0]
    class1_paths = paths[labels == 1]
    class0_labels = labels[labels == 0]
    class1_labels = labels[labels == 1]

    n_samples = int(len(class1_paths) * oversample_ratio)
    indices = np.random.choice(len(class0_paths), size=n_samples, replace=True)
    oversampled_paths = class0_paths[indices]
    oversampled_labels = class0_labels[indices]

    balanced_paths = np.concatenate([oversampled_paths, class1_paths])
    balanced_labels = np.concatenate([oversampled_labels, class1_labels])

    shuffle_idx = np.random.permutation(len(balanced_paths))
    balanced_paths = balanced_paths[shuffle_idx]
    balanced_labels = balanced_labels[shuffle_idx]

    return MedicalImageGenerator(
        balanced_paths,
        balanced_labels,
        batch_size,
        is_training=True,
        debug=False
    )


# Carga de datos
def prepare_dataset(data_dir):
    normal_dir = os.path.join(data_dir, "normal")
    stroke_dir = os.path.join(data_dir, "stroke")
    normal_patients = defaultdict(list)
    stroke_patients = defaultdict(list)
    stroke_types = {}

    stroke_case_pattern1 = "strokecase"
    stroke_case_pattern2 = "sub-strokecase"

    for img_file in os.listdir(stroke_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(stroke_dir, img_file)
            if stroke_case_pattern1 in img_file and "_dwi_slice-" in img_file:
                match = re.match(r'^(strokecase\d{3})_dwi_slice-\d{4}-\d{4}\.(jpg|png|jpeg)$', img_file)
                if match:
                    patient_id = match.group(1)
                    stroke_patients[patient_id].append(img_path)
                    stroke_types[patient_id] = 1
            elif stroke_case_pattern2 in img_file and "_dwi_slice_" in img_file:
                match = re.match(r'^(sub-strokecase\d{4})_dwi_slice_\d{3}\.(jpg|png|jpeg)$', img_file)
                if match:
                    patient_id = match.group(1)
                    stroke_patients[patient_id].append(img_path)
                    stroke_types[patient_id] = 2

    for img_file in os.listdir(normal_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(normal_dir, img_file)
            patient_id = img_file.split('_')[0]
            normal_patients[patient_id].append(img_path)

    all_paths = []
    all_labels = []
    patient_groups = []
    stroke_type_labels = []

    for patient_id in normal_patients:
        images = normal_patients[patient_id]
        all_paths.extend(images)
        all_labels.extend([0] * len(images))
        patient_groups.extend([patient_id] * len(images))
        stroke_type_labels.extend([0] * len(images))

    for patient_id in stroke_patients:
        images = stroke_patients[patient_id]
        all_paths.extend(images)
        all_labels.extend([1] * len(images))
        patient_groups.extend([patient_id] * len(images))
        stroke_type_labels.extend([stroke_types[patient_id]] * len(images))

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)
    patient_groups = np.array(patient_groups)
    stroke_type_labels = np.array(stroke_type_labels)

    print(f"Dataset - Normal: {len(np.unique(patient_groups[all_labels == 0]))} pacientes, "
          f"{np.sum(all_labels == 0)} imágenes")
    print(f"Dataset - ACV: {len(np.unique(patient_groups[all_labels == 1]))} pacientes, "
          f"{np.sum(all_labels == 1)} imágenes")
    print(f"Distribución ACV: Type1 (strokecase)={sum(1 for p in stroke_patients if p.startswith('strokecase'))}, "
          f"Type2 (sub-strokecase)={sum(1 for p in stroke_patients if p.startswith('sub-strokecase'))}")

    dataset_info = pd.DataFrame({
        'path': all_paths,
        'label': all_labels,
        'patient': patient_groups,
        'stroke_type': stroke_type_labels,
        'class': ['normal' if x == 0 else 'acv' for x in all_labels]
    })
    dataset_info.to_csv(os.path.join(output_dir, 'dataset_composition.csv'), index=False)

    return all_paths, all_labels, patient_groups, stroke_type_labels


# Modelo optimizado con más regularización
def create_densenet_model(input_shape=(128, 128, 3), fold=1):
    base_model = DenseNet121(
        include_top=False,
        input_shape=input_shape,
        weights='imagenet',
        name=f'densenet121_fold_{fold}'
    )

    for layer in base_model.layers[:250]:
        layer.trainable = False
    for layer in base_model.layers[250:]:
        layer.trainable = True
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = regularizers.l2(1e-3)

    inputs = Input(shape=input_shape, dtype=tf.float32, name=f'input_fold_{fold}')
    x = base_model(inputs)
    x = GlobalAveragePooling2D(name=f'gap_fold_{fold}')(x)
    x = Dense(128, activation='swish', kernel_regularizer=regularizers.l2(1e-3), dtype=tf.float32,
              name=f'dense_fold_{fold}')(x)
    x = Dropout(0.6, name=f'dropout_fold_{fold}')(x)
    outputs = Dense(1, activation='sigmoid', dtype=tf.float32, name=f'output_fold_{fold}')(x)

    model = Model(inputs, outputs, name=f'model_fold_{fold}')
    return model


# Focal Loss
def focal_loss(gamma=3.0, alpha=0.6):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_factor = y_true * tf.cast(alpha, tf.float32) + (1 - y_true) * tf.cast(1 - alpha, tf.float32)
        loss = alpha_factor * modulating_factor * cross_entropy
        return tf.reduce_mean(loss)

    return focal_loss_fn


# Callbacks
def get_callbacks(fold):
    return [
        EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(output_dir, f'best_fold_{fold}.keras'), monitor='val_auc', save_best_only=True,
                        mode='max', verbose=1),  # Modificado a .keras
        ReduceLROnPlateau(monitor='val_auc', factor=0.3, patience=5, min_lr=1e-7, mode='max', verbose=1),
        CSVLogger(os.path.join(output_dir, f'training_log_fold_{fold}.csv')),
        TerminateOnNaN()
    ]


# Entrenamiento
def train_densenet_model(x_train_paths, y_train, x_val_paths, y_val, fold):
    model = create_densenet_model(fold=fold)

    class_ratio = np.sum(y_train) / len(y_train)
    gamma = 2.0 + (1 - class_ratio) * 2.0
    alpha = 0.5 + (1 - class_ratio) * 0.3

    optimizer = Adam(learning_rate=3e-5, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=gamma, alpha=alpha),
        metrics=[
            BinaryAccuracy(name='accuracy', threshold=0.5, dtype=tf.float32),
            AUC(name='auc', curve='PR', dtype=tf.float32),
            F1Score(threshold=0.5, dtype=tf.float32)
        ]
    )

    train_gen = create_balanced_dataset(x_train_paths, y_train, oversample_ratio=1.8)
    val_gen = MedicalImageGenerator(x_val_paths, y_val, batch_size, is_training=False, debug=False)

    history = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=get_callbacks(fold),
        class_weight={0: 1.0, 1: 2.5},
        verbose=1
    )

    return model, history


# Evaluación
def evaluate_model_with_optimal_threshold(model, x_val_paths, y_val):
    val_gen = MedicalImageGenerator(x_val_paths, y_val, batch_size, is_training=False, debug=False)

    predictions = model.predict(val_gen, verbose=0).flatten()

    thresholds = np.linspace(0.3, 0.7, 50)
    best_score = -1
    optimal_threshold = 0.5
    best_confusion_matrix = None

    for th in thresholds:
        y_pred = (predictions >= th).astype(int)
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        score = 0.6 * recall + 0.4 * specificity

        if score > best_score:
            best_score = score
            optimal_threshold = th
            best_confusion_matrix = cm

    y_pred_binary = (predictions >= optimal_threshold).astype(int)
    tn, fp, fn, tp = best_confusion_matrix.ravel()

    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp + 1e-10),
        'recall': tp / (tp + fn + 1e-10),
        'specificity': tn / (tn + fp + 1e-10),
        'f1_score': f1_score(y_val, y_pred_binary),
        'auc': roc_auc_score(y_val, predictions),
        'optimal_threshold': optimal_threshold,
        'fnr': fn / (fn + tp + 1e-10),
        'custom_score': best_score,
        'confusion_matrix': best_confusion_matrix
    }

    return metrics, predictions


# Visualización
def visualize_results(history, results, y_val, y_pred_prob, fold, model_name="DenseNet121"):
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))

    axs[0, 0].plot(history.history['accuracy'])
    axs[0, 0].plot(history.history['val_accuracy'])
    axs[0, 0].set_title(f'{model_name} - Fold {fold} Accuracy')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].legend(['Train', 'Validation'], loc='upper left')

    axs[0, 1].plot(history.history['loss'])
    axs[0, 1].plot(history.history['val_loss'])
    axs[0, 1].set_title(f'{model_name} - Fold {fold} Loss')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].legend(['Train', 'Validation'], loc='upper left')

    axs[1, 0].plot(history.history['auc'])
    axs[1, 0].plot(history.history['val_auc'])
    axs[1, 0].set_title(f'{model_name} - Fold {fold} AUC')
    axs[1, 0].set_ylabel('AUC')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].legend(['Train', 'Validation'], loc='upper left')

    axs[1, 1].plot(history.history['f1_score'])
    axs[1, 1].plot(history.history['val_f1_score'])
    axs[1, 1].set_title(f'{model_name} - Fold {fold} F1 Score')
    axs[1, 1].set_ylabel('F1 Score')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].legend(['Train', 'Validation'], loc='upper left')

    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axs[2, 0])
    axs[2, 0].set_title(f'{model_name} - Fold {fold} Confusion Matrix')
    axs[2, 0].set_ylabel('True Label')
    axs[2, 0].set_xlabel('Predicted Label')

    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axs[2, 1].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    axs[2, 1].plot([0, 1], [0, 1], 'k--')
    axs[2, 1].set_title(f'{model_name} - Fold {fold} ROC Curve')
    axs[2, 1].set_xlabel('False Positive Rate')
    axs[2, 1].set_ylabel('True Positive Rate')
    axs[2, 1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fold_{fold}_results.png'))
    plt.close()


# Visualización de predicciones en test
def visualize_test_predictions(paths, true_labels, pred_probs, threshold, n_samples=10):
    indices = np.random.choice(len(paths), n_samples, replace=False)
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    for i, idx in enumerate(indices):
        img = cv2.imread(paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = pred_probs[idx]
        label = true_labels[idx]
        pred_label = int(pred >= threshold)
        ax = axs[i // 5, i % 5]
        ax.imshow(img)
        ax.set_title(f"True: {label}, Pred: {pred_label}\nProb: {pred:.4f}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_predictions.png'))
    plt.close()


# Test-Time Augmentation
def predict_with_tta(model, paths, n_augments=1):
    predictions = []
    for _ in range(n_augments):
        gen = MedicalImageGenerator(paths, np.zeros(len(paths)), batch_size, is_training=True, debug=False)
        preds = model.predict(gen, verbose=0).flatten()
        predictions.append(preds[:len(paths)])
    return np.mean(predictions, axis=0)


# Predicción sin TTA
def predict_without_tta(model, paths):
    gen = MedicalImageGenerator(paths, np.zeros(len(paths)), batch_size, is_training=False, debug=False)
    return model.predict(gen, verbose=0).flatten()


# Ensemble
def create_ensemble(models, method='average'):
    inputs = Input(shape=(128, 128, 3), dtype=tf.float32, name='ensemble_input')
    outputs = [model(inputs) for model in models]
    outputs = tf.keras.layers.Average(name='average_ensemble')(outputs)
    ensemble_model = Model(inputs, outputs, name='ensemble_model')
    return ensemble_model


# Evaluación en test
def evaluate_on_test_set(best_models, test_paths, test_labels, model_weights=None):
    ensemble_model = create_ensemble(best_models, method='average')

    predictions = predict_without_tta(ensemble_model, test_paths)

    fpr, tpr, thresholds = roc_curve(test_labels, predictions)
    j_scores = 0.6 * tpr + 0.4 * (1 - fpr)
    optimal_threshold = thresholds[np.argmax(j_scores)]
    y_pred_binary = (predictions >= optimal_threshold).astype(int)

    cm = confusion_matrix(test_labels, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'accuracy': accuracy_score(test_labels, y_pred_binary),
        'precision': tp / (tp + fp + 1e-10),
        'recall': tp / (tp + fn + 1e-10),
        'specificity': tn / (tn + fp + 1e-10),
        'f1_score': f1_score(test_labels, y_pred_binary),
        'auc': roc_auc_score(test_labels, predictions),
        'confusion_matrix': cm,
        'optimal_threshold': optimal_threshold,
        'npv': tn / (tn + fn + 1e-10),
        'fnr': fn / (fn + tp + 1e-10),
        'fpr': fp / (fp + tn + 1e-10),
        'tpr': tp / (tp + fn + 1e-10),
        'tnr': tn / (tn + fp + 1e-10),
        'ppv': tp / (tp + fp + 1e-10),
        'predictions': predictions
    }

    print("\n=== Evaluación en el Conjunto de Test ===")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test Specificity: {metrics['specificity']:.4f}")
    print(f"Test F1-Score: {metrics['f1_score']:.4f}")
    print(f"Test AUC: {metrics['auc']:.4f}")
    print(f"Test Optimal Threshold: {metrics['optimal_threshold']:.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axs[0])
    axs[0].set_title('Test Confusion Matrix')
    axs[0].set_ylabel('True Label')
    axs[0].set_xlabel('Predicted Label')
    axs[1].plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.3f}')
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set_title('Test ROC Curve')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_results.png'))
    plt.close()

    visualize_test_predictions(test_paths, test_labels, predictions, optimal_threshold)

    return metrics


# Main
def main():
    tf.keras.backend.set_floatx('float32')
    tf.config.optimizer.set_jit(True)
    print("TensorFlow version:", tf.__version__)

    data_dir = "C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/dataset_prueba_final"
    all_paths, all_labels, patient_groups, stroke_types = prepare_dataset(data_dir)

    # Dividir en train+val y test con StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_val_idx, test_idx in sgkf.split(all_paths, all_labels, patient_groups):
        train_val_paths = all_paths[train_val_idx]
        train_val_labels = all_labels[train_val_idx]
        train_val_groups = patient_groups[train_val_idx]
        test_paths = all_paths[test_idx]
        test_labels = all_labels[test_idx]
        test_groups = patient_groups[test_idx]
        break

    print(f"Test set - Normal: {np.sum(test_labels == 0)}, ACV: {np.sum(test_labels == 1)}, "
          f"Ratio: {np.sum(test_labels == 0) / np.sum(test_labels == 1):.2f}")

    test_patients = np.unique(test_groups)
    train_val_patients = np.unique(train_val_groups)
    overlap = np.intersect1d(test_patients, train_val_patients)
    print(f"Patient overlap between test and train/val: {len(overlap)} patients")

    results = []
    best_models = []
    best_aucs = []
    fold = 1

    for train_idx, val_idx in sgkf.split(train_val_paths, train_val_labels, train_val_groups):
        print(f"\n== Fold {fold}/5 ==")

        x_train_paths = train_val_paths[train_idx]
        y_train = train_val_labels[train_idx]
        x_val_paths = train_val_paths[val_idx]
        y_val = train_val_labels[val_idx]

        print(f"Imágenes en entrenamiento: {len(x_train_paths)} "
              f"(Normal: {np.sum(y_train == 0)}, ACV: {np.sum(y_train == 1)}, "
              f"Ratio: {np.sum(y_train == 0) / np.sum(y_train == 1):.2f})")
        print(f"Imágenes en validación: {len(x_val_paths)} "
              f"(Normal: {np.sum(y_val == 0)}, ACV: {np.sum(y_val == 1)}, "
              f"Ratio: {np.sum(y_val == 0) / np.sum(y_val == 1):.2f})")

        print("\nDistribución de tipos de ACV en entrenamiento:")
        train_stroke_types = stroke_types[train_idx]
        print(f"Type1: {np.sum(train_stroke_types == 1)}, Type2: {np.sum(train_stroke_types == 2)}")

        print("Distribución de tipos de ACV en validación:")
        val_stroke_types = stroke_types[val_idx]
        print(f"Type1: {np.sum(val_stroke_types == 1)}, Type2: {np.sum(val_stroke_types == 2)}")

        model, history = train_densenet_model(x_train_paths, y_train, x_val_paths, y_val, fold)
        metrics, predictions = evaluate_model_with_optimal_threshold(model, x_val_paths, y_val)

        print(f"\nFold {fold} - DenseNet121:")
        for key, value in metrics.items():
            if key not in ['confusion_matrix', 'predictions']:
                print(f"{key.capitalize()}: {value:.4f}")

        visualize_results(history, metrics, y_val, predictions, fold)
        results.append(metrics)
        best_models.append(model)
        best_aucs.append(metrics['auc'])

        del model, history
        tf.keras.backend.clear_session()
        gc.collect()

        fold += 1

    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in results]),
        'precision': np.mean([r['precision'] for r in results]),
        'recall': np.mean([r['recall'] for r in results]),
        'specificity': np.mean([r['specificity'] for r in results]),
        'f1_score': np.mean([r['f1_score'] for r in results]),
        'auc': np.mean([r['auc'] for r in results]),
        'fnr': np.mean([r['fnr'] for r in results]),
        'custom_score': np.mean([r['custom_score'] for r in results]),
        'optimal_threshold': np.mean([r['optimal_threshold'] for r in results])
    }

    print("\nResultados Promedio DenseNet121 (5-Fold CV):")
    for key, value in avg_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

    for i, model in enumerate(best_models):
        metrics, _ = evaluate_model_with_optimal_threshold(model, test_paths, test_labels)
        print(f"\nModel {i + 1} Test Metrics:")
        for key, value in metrics.items():
            if key not in ['confusion_matrix', 'predictions']:
                print(f"{key.capitalize()}: {value:.4f}")

    model_weights = [1.0 / len(best_models)] * len(best_models)
    test_metrics = evaluate_on_test_set(best_models, test_paths, test_labels, model_weights)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), index=False)

    avg_results_df = pd.DataFrame([avg_metrics])
    avg_results_df.to_csv(os.path.join(output_dir, 'average_metrics.csv'), index=False)

    test_results_df = pd.DataFrame([test_metrics])
    test_results_df.to_csv(os.path.join(output_dir, 'test_set_results.csv'), index=False)


if __name__ == "__main__":
    main()
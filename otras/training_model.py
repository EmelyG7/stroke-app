# Importaciones esenciales
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
import numpy as np

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clase para métricas personalizadas (movida antes de su uso)
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convertir predicciones a binario explícitamente
        y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return tf.cond(
            tf.math.greater(p + r, 0),
            lambda: 2 * ((p * r) / (p + r + tf.keras.backend.epsilon())),
            lambda: tf.cast(0.0, tf.float32)
        )

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

class Config:
    IMG_SIZE = (128, 128)
    TARGET_SIZE = (128, 128, 3)
    BATCH_SIZE = 32
    EPOCHS = 60
    NUM_CLASSES = 2
    LEARNING_RATE = 0.0001
    W_REGULARIZER = 1e-5
    PATIENCE = 7
    FACTOR = 0.3
    AUTOTUNE = tf.data.AUTOTUNE
    CACHE_DIR = './dataset_cache'  # Directorio para cache en disco

def setup_gpu():
    """Configuración mejorada de GPU con validación"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
                )
            # Verificar configuración
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(f"GPUs físicas: {len(gpus)}, GPUs lógicas: {len(logical_gpus)}")
        else:
            logger.warning("No se encontró GPU, usando CPU")
    except Exception as e:
        logger.error(f"Error al configurar GPU: {e}")
        logger.info("Continuando con CPU")

def validate_dataset(ds_path, name):
    """Validación mejorada de datasets"""
    if not os.path.exists(ds_path):
        raise ValueError(f"El directorio {name} no existe: {ds_path}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    files = os.listdir(ds_path)
    
    if not files:
        raise ValueError(f"El directorio {name} está vacío: {ds_path}")
    
    invalid_files = [f for f in files if not any(f.lower().endswith(ext) for ext in valid_extensions)]
    if invalid_files:
        logger.warning(f"Archivos no válidos en {name}: {invalid_files}")

def create_data_augmentation():
    """Data augmentation con parámetros ajustables"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomTranslation(0.15, 0.15),
        layers.RandomZoom(0.2),
    ])

def preprocess_dataset(ds, config):
    """Preprocesamiento con cache en disco opcional"""
    return (ds
        .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), 
             num_parallel_calls=config.AUTOTUNE)
        .cache(config.CACHE_DIR)  # Cache en disco
        .batch(config.BATCH_SIZE)
        .prefetch(buffer_size=config.AUTOTUNE)
    )

def calculate_class_weights(train_ds):
    """Cálculo optimizado de pesos de clase"""
    total_samples = 0
    label_counts = {}
    
    for _, labels in train_ds:
        for label in labels:
            label_val = int(label.numpy())
            label_counts[label_val] = label_counts.get(label_val, 0) + 1
            total_samples += 1
    
    class_weights = {
        label: total_samples / (len(label_counts) * count)
        for label, count in label_counts.items()
    }
    
    logger.info(f"Pesos calculados por clase: {class_weights}")
    return class_weights

def create_model(config):
    """Modelo con validaciones adicionales"""
    model = models.Sequential([
        layers.Input(shape=config.TARGET_SIZE),
        layers.Rescaling(1./255),
        create_data_augmentation(),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                F1Score(name='f1')]
    )
    
    return model

class MemoryCallback(tf.keras.callbacks.Callback):
    """Callback mejorado para monitoreo de memoria"""
    def on_epoch_end(self, epoch, logs=None):
        try:
            import psutil
            process = psutil.Process()
            memory = process.memory_info().rss / 1024 / 1024
            logger.info(f"Uso de memoria: {memory:.2f} MB")
        except ImportError:
            logger.warning("psutil no está instalado. Para monitorear memoria: pip install psutil")
        except Exception as e:
            logger.error(f"Error al monitorear memoria: {e}")

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Visualización mejorada de matriz de confusión"""
    if len(class_names) != len(set(y_true) | set(y_pred)):
        logger.warning("Número de clases no coincide con los datos")
        
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

def load_datasets(config):
    """Carga los datasets de entrenamiento, validación y prueba"""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "Dataset/train",
        labels='inferred',
        label_mode='binary',
        shuffle=True,
        seed=123,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "Dataset/val",
        labels='inferred',
        label_mode='binary',
        shuffle=False,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        "Dataset/test",
        labels='inferred',
        label_mode='binary',
        shuffle=False,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE
    )

    return train_ds, val_ds, test_ds

def evaluate_and_visualize(model, test_ds, history):
    """Evalúa el modelo y visualiza los resultados"""
    # Evaluación
    metrics = model.evaluate(test_ds, verbose=1)
    logger.info("Métricas de prueba:")
    for name, value in zip(model.metrics_names, metrics):
        logger.info(f"{name}: {value:.4f}")

    # Predicciones
    y_pred = []
    y_true = []
    
    # Recolectar predicciones y etiquetas verdaderas
    for images, labels in test_ds:
        pred_batch = model.predict(images, verbose=0)
        y_pred.extend(pred_batch.flatten())  # Aplanar las predicciones
        y_true.extend(labels.numpy().flatten())  # Aplanar las etiquetas
    
    # Convertir a arrays de numpy y asegurar tipos
    y_pred = np.array(y_pred, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)
    
    # Convertir predicciones a clases
    y_pred_classes = (y_pred > 0.5).astype(np.int32)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred_classes)
    class_names = ['Normal', 'Stroke']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

    # Gráficas de entrenamiento
    metrics_to_plot = ['accuracy', 'loss', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics_to_plot):
        if metric in history.history:
            axes[idx].plot(history.history[metric], label='Train')
            axes[idx].plot(history.history[f'val_{metric}'], label='Val')
            axes[idx].set_title(f'Model {metric}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric)
            axes[idx].legend()

    plt.tight_layout()
    plt.show()

def main():
    """Función principal con mejor manejo de errores"""
    setup_gpu()
    config = Config()
    
    try:
        # Crear directorio de cache si no existe
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        
        # Cargar y validar datasets
        train_ds, val_ds, test_ds = load_datasets(config)
        
        # Crear y entrenar modelo
        model = create_model(config)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.PATIENCE,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=config.FACTOR,
                patience=config.PATIENCE//2,
                min_lr=config.W_REGULARIZER
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            MemoryCallback()
        ]
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.EPOCHS,
            callbacks=callbacks
        )
        
        # Evaluar y visualizar resultados
        evaluate_and_visualize(model, test_ds, history)
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
    finally:
        # Limpiar memoria y cache
        tf.keras.backend.clear_session()
        if os.path.exists(config.CACHE_DIR):
            import shutil
            shutil.rmtree(config.CACHE_DIR)

if __name__ == "__main__":
    main()



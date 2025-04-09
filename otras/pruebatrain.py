import warnings

warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ========================
# Data Preparation
# ========================

# Define paths - update these to match your directory structure
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")


# Function to split data into train/val/test directories
def split_data_into_train_val_test(normal_path, stroke_path, output_dir, test_size=0.2, val_size=0.2):
    """
    Split data from source directories into train/val/test directories
    Returns lists of files for train, val, and test sets.
    """
    # Create directory structure if it doesn't exist
    os.makedirs(os.path.join(output_dir, "train", "normal"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "stroke"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "normal"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "stroke"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "normal"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "stroke"), exist_ok=True)

    # Get file lists
    normal_files = os.listdir(normal_path)
    stroke_files = os.listdir(stroke_path)

    # Split normal files
    normal_train, normal_test = train_test_split(normal_files, test_size=test_size, random_state=42)
    normal_train, normal_val = train_test_split(normal_train, test_size=val_size, random_state=42)

    # Split stroke files
    stroke_train, stroke_test = train_test_split(stroke_files, test_size=test_size, random_state=42)
    stroke_train, stroke_val = train_test_split(stroke_train, test_size=val_size, random_state=42)

    # Copy files to their respective directories
    for file in normal_train:
        src = os.path.join(normal_path, file)
        dst = os.path.join(output_dir, "train", "normal", file)
        tf.io.gfile.copy(src, dst, overwrite=True)

    for file in normal_val:
        src = os.path.join(normal_path, file)
        dst = os.path.join(output_dir, "val", "normal", file)
        tf.io.gfile.copy(src, dst, overwrite=True)

    for file in normal_test:
        src = os.path.join(normal_path, file)
        dst = os.path.join(output_dir, "test", "normal", file)
        tf.io.gfile.copy(src, dst, overwrite=True)

    for file in stroke_train:
        src = os.path.join(stroke_path, file)
        dst = os.path.join(output_dir, "train", "stroke", file)
        tf.io.gfile.copy(src, dst, overwrite=True)

    for file in stroke_val:
        src = os.path.join(stroke_path, file)
        dst = os.path.join(output_dir, "val", "stroke", file)
        tf.io.gfile.copy(src, dst, overwrite=True)

    for file in stroke_test:
        src = os.path.join(stroke_path, file)
        dst = os.path.join(output_dir, "test", "stroke", file)
        tf.io.gfile.copy(src, dst, overwrite=True)

    print(f"Train set: {len(normal_train)} normal, {len(stroke_train)} stroke")
    print(f"Val set: {len(normal_val)} normal, {len(stroke_val)} stroke")
    print(f"Test set: {len(normal_test)} normal, {len(stroke_test)} stroke")

    # Return the lists of files
    return normal_train, normal_val, normal_test, stroke_train, stroke_val, stroke_test


def check_for_data_leakage(train_files, val_files, test_files):
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)

    # Check for overlap between train and val
    train_val_overlap = train_set.intersection(val_set)
    if train_val_overlap:
        print(f"Data leakage detected: {len(train_val_overlap)} files overlap between train and val sets.")

    # Check for overlap between train and test
    train_test_overlap = train_set.intersection(test_set)
    if train_test_overlap:
        print(f"Data leakage detected: {len(train_test_overlap)} files overlap between train and test sets.")

    # Check for overlap between val and test
    val_test_overlap = val_set.intersection(test_set)
    if val_test_overlap:
        print(f"Data leakage detected: {len(val_test_overlap)} files overlap between val and test sets.")

    if not train_val_overlap and not train_test_overlap and not val_test_overlap:
        print("No data leakage detected.")


# def check_pixel_level_leakage(train_dir, val_dir, test_dir, threshold=0.95):
#     import cv2
#     import numpy as np
#
#     def load_images_from_dir(directory):
#         images = []
#         for root, _, files in os.walk(directory):
#             for file in files:
#                 if file.endswith(('.png', '.jpg', '.jpeg')):
#                     img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
#                     if img is not None:
#                         images.append(img)
#         return images
#
#     def calculate_similarity(img1, img2):
#         # Redimensionar las imágenes al mismo tamaño si es necesario
#         if img1.shape != img2.shape:
#             img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
#         # Calcular la diferencia absoluta entre las imágenes
#         diff = cv2.absdiff(img1, img2)
#         # Calcular la similitud como el porcentaje de píxeles que no difieren
#         similarity = 1.0 - (np.mean(diff) / 255.0)
#         return similarity
#
#     train_images = load_images_from_dir(train_dir)
#     val_images = load_images_from_dir(val_dir)
#     test_images = load_images_from_dir(test_dir)
#
#     def find_similar_images(set1, set2, threshold):
#         similar_pairs = []
#         for i, img1 in enumerate(set1):
#             for j, img2 in enumerate(set2):
#                 if img1.shape == img2.shape:
#                     similarity = calculate_similarity(img1, img2)
#                     if similarity > threshold:
#                         similar_pairs.append((i, j, similarity))
#         return similar_pairs
#
#     # Check for similar images between train and val
#     train_val_similar = find_similar_images(train_images, val_images, threshold)
#     if train_val_similar:
#         print(f"Pixel-level data leakage detected: {len(train_val_similar)} similar images between train and val sets.")
#
#     # Check for similar images between train and test
#     train_test_similar = find_similar_images(train_images, test_images, threshold)
#     if train_test_similar:
#         print(f"Pixel-level data leakage detected: {len(train_test_similar)} similar images between train and test sets.")
#
#     # Check for similar images between val and test
#     val_test_similar = find_similar_images(val_images, test_images, threshold)
#     if val_test_similar:
#         print(f"Pixel-level data leakage detected: {len(val_test_similar)} similar images between val and test sets.")
#
#     if not train_val_similar and not train_test_similar and not val_test_similar:
#         print("No pixel-level data leakage detected.")


# Uncomment to run the split (first time only)
normal_path = "C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/dataset_dwi_stroke/normal"
stroke_path = "C:/Users/Coshita/Downloads/stroke datos de entrenamiento/DWI/dataset_dwi_stroke/stroke"
normal_train, normal_val, normal_test, stroke_train, stroke_val, stroke_test = split_data_into_train_val_test(
    normal_path, stroke_path, "data"
)
# split_data_into_train_val_test(normal_path, stroke_path, "data")
normal_files = os.listdir(normal_path)
stroke_files = os.listdir(stroke_path)
check_for_data_leakage(normal_train + stroke_train, normal_val + stroke_val, normal_test + stroke_test)
# check_pixel_level_leakage(train_dir, val_dir, test_dir)


# Enhanced preprocessing with additional randomization
def preprocess_medical_image(img, add_noise=False, random_contrast=False):
    """
    Preprocess medical images with various enhancements.
    Returns a float32 image in the range [0, 1] to be compatible with rescaling operations.
    """
    # First convert to float to avoid any uint8 overflow issues
    img_float = img.astype(np.float32)

    # Convert to grayscale if it's a color image
    if len(img_float.shape) == 3 and img_float.shape[2] == 3:
        # OpenCV expects values in [0, 255] for color conversion
        if img_float.max() <= 1.0:
            img_float = img_float * 255.0
        gray = cv2.cvtColor(img_float.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        # If already grayscale but has additional dimensions, squeeze them
        if len(img_float.shape) > 2:
            gray = np.squeeze(img_float)
        else:
            gray = img_float.copy()

    # Ensure the gray image is in uint8 for CLAHE
    if gray.max() <= 1.0:
        gray = (gray * 255.0).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)

    # Apply CLAHE with random parameters
    clip_limit = np.random.uniform(1.0, 3.0) if random_contrast else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)

    # Add random noise (for data augmentation)
    if add_noise and np.random.random() > 0.5:
        noise = np.random.normal(0, 5, cl1.shape).astype(np.uint8)
        cl1 = cv2.add(cl1, noise)

    # Edge enhancement with random thresholds
    if np.random.random() > 0.5:
        low_threshold = np.random.randint(30, 70)
        high_threshold = np.random.randint(100, 200)
        edges = cv2.Canny(cl1, low_threshold, high_threshold)
        enhanced = cv2.addWeighted(cl1, 1.5, edges, 0.5, 0)
    else:
        enhanced = cl1

    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    # Convert back to float32 in range [0, 1] for the ImageDataGenerator's rescaling
    enhanced_rgb = enhanced_rgb.astype(np.float32) / 255.0

    return enhanced_rgb


# ========================
# Data Augmentation and Loading
# ========================

# Define image size
img_size = (224, 224)
batch_size = 16

# Create data generators with stronger augmentation
train_datagen = ImageDataGenerator(
    rotation_range=50,  # Increased rotation
    width_shift_range=0.5,  # Increased shift
    height_shift_range=0.5,  # Increased shift
    shear_range=0.5,  # Increased shear
    zoom_range=0.5,  # Increased zoom
    horizontal_flip=True,
    vertical_flip=True,  # Added vertical flip
    fill_mode='reflect',
    brightness_range=[0.5, 1.5],  # Wider brightness range
    preprocessing_function=lambda img: preprocess_medical_image(img, add_noise=True, random_contrast=True)
)

# Only rescale validation and test data without augmentation
val_datagen = ImageDataGenerator(
    preprocessing_function=lambda img: preprocess_medical_image(img, add_noise=False, random_contrast=False))

test_datagen = ImageDataGenerator(
    preprocessing_function=lambda img: preprocess_medical_image(img, add_noise=False, random_contrast=False))

# Load data using flow_from_directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Calculate class weights for imbalanced dataset
labels = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class weights: {class_weight_dict}")


# ========================
# Model Definition with Higher Regularization
# ========================

def create_regularized_model(input_shape=(224, 224, 3), dropout_rate=0.7, l2_reg=0.05):
    """
    Create DenseNet169 model with stronger regularization
    """
    # Load base model with pre-trained weights
    base_model = DenseNet121(
        include_top=False,
        input_shape=input_shape,
        weights='imagenet'
    )

    # Freeze fewer layers - allow more fine-tuning
    for layer in base_model.layers[:-100]:  # Freeze all but last 50 layers
        layer.trainable = False

    # Build model architecture with stronger regularization
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)  # Increased dropout
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)  # Smaller layer
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)  # Increased dropout
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="stroke_detection_model")

    return model, base_model  # Return both model and base_model for Grad-CAM


# ========================
# Training Function with Visualization and Metrics
# ========================

def train_and_evaluate_model(fold=0, visualize_samples=5):
    """
    Train model and visualize Grad-CAM results
    """
    # Create and compile model
    model, base_model = create_regularized_model(dropout_rate=0.6, l2_reg=0.02)

    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # Reduced learning rate
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # Define callbacks with more patience
    callbacks = [
        EarlyStopping(
            monitor='val_loss',  # Changed to val_loss instead of val_auc
            patience=15,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',  # Changed to val_loss
            factor=0.5,  # Less aggressive reduction
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f"best_model_fold_{fold}.keras",
            monitor="val_loss",  # Changed to val_loss
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Evaluate model on validation set
    val_loss, val_accuracy, val_auc, val_precision, val_recall = model.evaluate(val_generator)

    # Evaluate model on test set
    test_loss, test_accuracy, test_auc, test_precision, test_recall = model.evaluate(test_generator)

    # Print evaluation metrics
    print(f"\nValidation Metrics:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")

    print(f"\nTest Metrics:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_history_fold_{fold}.png')
    plt.close()

    # Create confusion matrix for test set
    y_true = test_generator.classes
    y_pred_prob = model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Stroke'],
                yticklabels=['Normal', 'Stroke'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test Set)')
    plt.savefig(f'confusion_matrix_fold_{fold}.png')
    plt.close()

    # Create classification report for test set
    cr = classification_report(y_true, y_pred, target_names=['Normal', 'Stroke'])
    print("\nClassification Report (Test Set):")
    print(cr)

    # Generate ROC curve for test set
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Test Set)')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_fold_{fold}.png')
    plt.close()

    # Visualize Grad-CAM for some examples
    last_conv_layer_name = "conv5_block32_concat"  # Last conv layer in DenseNet169

    # Create figure for Grad-CAM visualizations
    plt.figure(figsize=(15, 12))

    # Get some test samples
    test_batches = list(test_generator)

    # Sample images from test set
    for i in range(min(visualize_samples, len(test_batches))):
        try:
            # Get a batch
            batch_x, batch_y = test_batches[i]

            # Get first image from batch
            img_array = batch_x[0:1]
            img = (img_array[0] * 255).astype(np.uint8)

            # Generate heatmap
            heatmap = make_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name)

            # Display Grad-CAM
            superimposed_img = display_gradcam(img, heatmap)

            # Plot side by side
            plt.subplot(visualize_samples, 2, i * 2 + 1)
            plt.imshow(img)
            plt.title(f"Original - Class: {'Stroke' if batch_y[0] == 1 else 'Normal'}")
            plt.axis('off')

            plt.subplot(visualize_samples, 2, i * 2 + 2)
            plt.imshow(superimposed_img)
            plt.title(f"Grad-CAM - Pred: {'Stroke' if y_pred_prob[i][0] > 0.5 else 'Normal'} ({y_pred_prob[i][0]:.2f})")
            plt.axis('off')
        except Exception as e:
            print(f"Error generating Grad-CAM for sample {i}: {e}")

    plt.tight_layout()
    plt.savefig(f'gradcam_visualization_fold_{fold}.png')
    plt.close()

    return model, history, {'accuracy': test_accuracy, 'auc': test_auc, 'precision': test_precision,
                            'recall': test_recall}


# ========================
# Main Execution
# ========================

if __name__ == "__main__":
    # Run single model training and evaluation with Grad-CAM
    print("Training and evaluating model with Grad-CAM visualization...")
    model, history, metrics = train_and_evaluate_model(fold=0, visualize_samples=5)
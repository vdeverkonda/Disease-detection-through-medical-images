import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU is available and will be used:", physical_devices[0])
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found. Running on CPU.")

# Paths to data directories
train_dir = '/Users/karthikdeverkonda/Desktop/amat 554 project datasets/untitled folder/chest_xray/train'
val_dir = '/Users/karthikdeverkonda/Desktop/amat 554 project datasets/untitled folder/chest_xray/val'
test_dir = '/Users/karthikdeverkonda/Desktop/amat 554 project datasets/untitled folder/chest_xray/test'

# Create output directory
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# Image data generators with more aggressive augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load a pre-trained MobileNetV2 model as base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

# Build the model with added regularization
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),  # L2 Regularization
    Dropout(0.5),  # Increased dropout rate
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train the model with modified parameters
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model on the test set
test_generator = val_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Get predictions on the test set
predictions = model.predict(test_generator)
predictions = np.round(predictions).astype(int).flatten()

# Generate and save the classification report
report = classification_report(test_generator.classes, predictions, target_names=test_generator.class_indices.keys())
report_path = os.path.join(output_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)

# Generate and save the confusion matrix
conf_matrix = confusion_matrix(test_generator.classes, predictions)
conf_matrix_path = os.path.join(output_dir, 'confusion_matrix.txt')
with open(conf_matrix_path, 'w') as f:
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix))

print("Classification Report and Confusion Matrix saved to the output folder.")

# Plot training & validation accuracy values and save the plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the plot to the output directory
plot_path = os.path.join(output_dir, 'accuracy_loss_plot.png')
plt.savefig(plot_path)
print("Accuracy and Loss plots saved to", plot_path)

# Display a message to confirm that all files have been saved
print("All outputs have been saved in the output directory.")
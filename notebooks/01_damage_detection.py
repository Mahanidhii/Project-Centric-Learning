import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# CONFIGURATION - Update these paths to match your setup
DATASET_DIR = os.path.join("..", "data", "car_damage")
MODEL_SAVE_PATH = os.path.join("..", "models", "damage_model.h5")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Data Loading & Augmentation
print("=" * 50)
print("Loading and Augmenting Data")
print("=" * 50)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,  
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Try to load training and validation separately
training_dir = os.path.join(DATASET_DIR, "training")
validation_dir = os.path.join(DATASET_DIR, "validation")

if os.path.exists(training_dir) and os.path.exists(validation_dir):
    print("Found separate training and validation folders.")
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    val_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
else:
    print("Using single directory with 80/20 split.")
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )
    val_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

NUM_CLASSES = train_generator.num_classes
CLASS_NAMES = list(train_generator.class_indices.keys())
print(f"\nClasses found: {CLASS_NAMES}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Model Architecture
print("\n" + "=" * 50)
print("Building Model (MobileNetV2 + Custom Head)")
print("=" * 50)

base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


#  MODEL TRAINING
print("\n" + "=" * 50)
print("STEP 3: Training the Model")
print("=" * 50)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True),
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
)

# Fine-tuning (Unfreeze last 30 layers)
print("\n" + "=" * 50)
print("Fine-tuning (unfreezing last 30 layers)")
print("=" * 50)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks,
)

# RESULTS:
print("\n" + "=" * 50)
print("STEP 5: Results")
print("=" * 50)

val_loss, val_acc = model.evaluate(val_generator)
print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Final Validation Loss: {val_loss:.4f}")
print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# Save class names for later use
import json

class_names_path = os.path.join("..", "models", "damage_classes.json")
with open(class_names_path, "w") as f:
    json.dump(CLASS_NAMES, f)
print(f"Class names saved to: {class_names_path}")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history["accuracy"] + history_fine.history["accuracy"])
ax1.plot(history.history["val_accuracy"] + history_fine.history["val_accuracy"])
ax1.set_title("Model Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend(["Train", "Validation"])

ax2.plot(history.history["loss"] + history_fine.history["loss"])
ax2.plot(history.history["val_loss"] + history_fine.history["val_loss"])
ax2.set_title("Model Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend(["Train", "Validation"])

plt.tight_layout()
plt.savefig(os.path.join("..", "models", "training_history.png"))
plt.show()
print("Training plot saved.")
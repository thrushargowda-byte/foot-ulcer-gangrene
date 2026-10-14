import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, Reshape, multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- CONFIG ----------------
IMG_SIZE = 224
NUM_CLASSES = 3
BATCH_SIZE = 15  # must be divisible by NUM_CLASSES for balanced sampling (5 per class)
PER_CLASS = BATCH_SIZE // NUM_CLASSES
CLASS_NAMES = ["normal", "DFU", "gangrene"]

# ---------------- LOAD DATA (raw 0-255 pixel values) ----------------
print("Loading preprocessed data...")
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_val = np.load("data/X_val.npy")
y_val = np.load("data/y_val.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Train class counts: {np.bincount(y_train)}")
print(f"Pixel value range check -- min: {X_train.min()}, max: {X_train.max()}")

# Preprocess validation/test sets ONCE upfront (ResNet50-specific normalization)
X_val_proc = preprocess_input(X_val.copy())
X_test_proc = preprocess_input(X_test.copy())

# ---------------- AUGMENTATION (applied on raw 0-255 images) ----------------
augmentor = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.75, 1.25],
    zoom_range=0.15,
    fill_mode="nearest"
)

# ---------------- BALANCED BATCH GENERATOR ----------------
class BalancedGenerator(Sequence):
    """
    Every batch contains an EQUAL number of images from each class.
    Augmentation is applied on raw 0-255 images, THEN preprocess_input
    is applied right before feeding into ResNet50.
    """
    def __init__(self, X, y, per_class=PER_CLASS, steps_per_epoch=40):
        self.X = X
        self.y = y
        self.per_class = per_class
        self.steps_per_epoch = steps_per_epoch
        self.class_indices = {
            c: np.where(y == c)[0] for c in range(NUM_CLASSES)
        }

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        batch_X = []
        batch_y = []
        for c in range(NUM_CLASSES):
            chosen = np.random.choice(self.class_indices[c], self.per_class, replace=True)
            for i in chosen:
                img = self.X[i]
                img = augmentor.random_transform(img)  # still 0-255 range
                batch_X.append(img)
                batch_y.append(c)
        batch_X = np.array(batch_X, dtype="float32")
        batch_y = np.array(batch_y)
        perm = np.random.permutation(len(batch_y))
        batch_X = batch_X[perm]
        batch_y = batch_y[perm]

        # Apply ResNet50-specific preprocessing right before returning
        batch_X = preprocess_input(batch_X)
        return batch_X, batch_y

train_gen = BalancedGenerator(X_train, y_train, per_class=PER_CLASS, steps_per_epoch=40)

# ---------------- SQUEEZE-AND-EXCITATION BLOCK ----------------
def se_block(input_tensor, ratio=16):
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se)
    se = Dense(channels // ratio, activation="relu", use_bias=False)(se)
    se = Dense(channels, activation="sigmoid", use_bias=False)(se)
    return multiply([input_tensor, se])

# ---------------- BUILD SE-RESNET50 MODEL ----------------
def build_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = se_block(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model

model, base_model = build_model()
model.summary()

# ---------------- CALLBACKS ----------------
callbacks = [
    ModelCheckpoint(
        "model/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ---------------- PHASE 1: TRAIN TOP LAYERS ONLY ----------------
print("\n===== PHASE 1: Training top layers (base frozen) =====")
model.compile(
    optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history1 = model.fit(
    train_gen,
    validation_data=(X_val_proc, y_val),
    epochs=12,
    callbacks=callbacks
)

# ---------------- SANITY CHECK ----------------
print("\n===== SANITY CHECK after Phase 1 =====")
val_preds = np.argmax(model.predict(X_val_proc), axis=1)
unique_preds, pred_counts = np.unique(val_preds, return_counts=True)
print(f"Validation prediction distribution: {dict(zip(unique_preds, pred_counts))}")
if len(unique_preds) == 1:
    print("WARNING: Model is only predicting ONE class after Phase 1.")
else:
    print("Model is predicting multiple classes -- proceeding to Phase 2.")

# ---------------- PHASE 2: FINE-TUNE ----------------
print("\n===== PHASE 2: Fine-tuning (unfreezing last layers of ResNet50) =====")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5, clipnorm=1.0),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_gen,
    validation_data=(X_val_proc, y_val),
    epochs=15,
    callbacks=callbacks
)

# ---------------- EVALUATE ON TEST SET ----------------
print("\n===== FINAL EVALUATION ON TEST SET =====")
test_loss, test_acc = model.evaluate(X_test_proc, y_test, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

y_pred_probs = model.predict(X_test_proc)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------- SAVE FINAL MODEL ----------------
model.save("model/final_model.keras")
print("\nModel saved to model/final_model.keras")
print("Best checkpoint saved to model/best_model.keras")
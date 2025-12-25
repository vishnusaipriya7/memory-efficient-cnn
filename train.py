"""
train.py
--------
Trains a memory-efficient CNN using CIFAR-10 dataset.
Includes best practices like callbacks, validation split,
and model checkpointing.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from model import build_model


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 20
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
MODEL_SAVE_PATH = "cnn_fp32.keras"



# ---------------------------------------------------------
# Load and Preprocess Dataset
# ---------------------------------------------------------
def load_data():
    """
    Loads CIFAR-10 dataset and applies normalization.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize images to [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------
def train():
    # Load data
    x_train, y_train, x_test, y_test = load_data()

    # Build model
    model = build_model(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        width_multiplier=1.0
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # -----------------------------------------------------
    # Callbacks
    # -----------------------------------------------------
    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            verbose=1
        )
    ]

    # -----------------------------------------------------
    # Train Model
    # -----------------------------------------------------
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=callbacks
    )

    # -----------------------------------------------------
    # Evaluate on Test Data
    # -----------------------------------------------------
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # -----------------------------------------------------
    # Save Final FP32 Model
    # -----------------------------------------------------
    # Save Keras format (for reuse / resume training)
    model.save("cnn_fp32.keras")

# Save TensorFlow SavedModel (for TFLite conversion)
    model.export("cnn_fp32")

    print(f"\nModel saved at: {MODEL_SAVE_PATH}")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    train()

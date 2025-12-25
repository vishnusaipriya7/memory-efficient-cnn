"""
model.py
---------
Memory-efficient CNN architecture designed for low-resource devices.

Key features:
- Depthwise separable convolutions
- Batch normalization for stability
- ReLU activations (quantization-friendly)
- Modular design (clean + extensible)
"""

import tensorflow as tf
from tensorflow.keras import layers, models

# Depthwise Separable Convolution Block

def depthwise_separable_block(
    x,
    filters,
    kernel_size=3,
    strides=1,
    padding="same",
    use_batchnorm=True,
    activation="relu"
):
    """
    A single depthwise separable convolution block.

    Steps:
    1. Depthwise Convolution (spatial filtering per channel)
    2. Pointwise Convolution (1x1 to mix channels)
    3. Batch Normalization
    4. Activation

    This block dramatically reduces parameters and computation.
    """

    # Depthwise convolution
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False
    )(x)

    if use_batchnorm:
        x = layers.BatchNormalization()(x)

    if activation:
        x = layers.Activation(activation)(x)

    # Pointwise convolution (1x1)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False
    )(x)

    if use_batchnorm:
        x = layers.BatchNormalization()(x)

    if activation:
        x = layers.Activation(activation)(x)

    return x

# Build Complete CNN Model
def build_model(
    input_shape=(32, 32, 3),
    num_classes=10,
    width_multiplier=1.0
):
    """
    Builds a lightweight CNN optimized for edge devices.

    Args:
        input_shape: Shape of input image (H, W, C)
        num_classes: Number of output classes
        width_multiplier: Scales number of filters (MobileNet trick)

    Returns:
        tf.keras.Model
    """

    inputs = layers.Input(shape=input_shape)

    # Initial Standard Convolution (Stem Layer)
    x = layers.Conv2D(
        filters=int(32 * width_multiplier),
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Depthwise Separable Blocks
    x = depthwise_separable_block(
        x,
        filters=int(64 * width_multiplier),
        strides=1
    )

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = depthwise_separable_block(
        x,
        filters=int(128 * width_multiplier),
        strides=1
    )

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = depthwise_separable_block(
        x,
        filters=int(256 * width_multiplier),
        strides=1
    )
    # Feature Aggregation

    x = layers.GlobalAveragePooling2D()(x)

    # Classification Head

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MemoryEfficientCNN")

    return model


# Model Summary (Debug / Sanity Check)
if __name__ == "__main__":
    model = build_model()
    model.summary()

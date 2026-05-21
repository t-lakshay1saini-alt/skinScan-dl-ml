"""
Model architectures for SkinScan AI.

Provides:
- EfficientNetV2-M backbone with multi-scale pooling head
- Multimodal model (image + patient metadata fusion)
- Squeeze-and-Excitation and CBAM attention blocks
- Multi-class focal loss with label smoothing
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, GlobalMaxPooling2D,
    Dropout, BatchNormalization, Concatenate, Input, Multiply,
    Flatten, Conv2D,
)
from tensorflow.keras.models import Model

from src.config import IMG_SIZE, NUM_CLASSES, DROPOUT_RATE


# =============================================================================
# Loss Functions
# =============================================================================

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Multi-class focal loss. Focuses on hard examples and down-weights easy ones.
    Critical for imbalanced datasets like HAM10000.
    """
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss


def focal_label_smooth_loss(gamma=2.0, smoothing=0.1):
    """Focal loss combined with label smoothing — best of both worlds."""
    def loss_fn(y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_smooth = y_true * (1 - smoothing) + smoothing / num_classes
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        ce = -y_smooth * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(focal_weight * ce, axis=-1))
    return loss_fn


# =============================================================================
# Attention Blocks
# =============================================================================

def squeeze_excitation_block(x, ratio=16):
    """SE block: learns to re-weight feature channels by importance."""
    filters = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // ratio, activation="relu")(se)
    se = Dense(filters, activation="sigmoid")(se)
    se = tf.reshape(se, [-1, 1, 1, filters])
    return Multiply()([x, se])


def cbam_attention(x, ratio=8):
    """CBAM: Channel + Spatial attention combined."""
    filters = x.shape[-1]
    # Channel attention
    avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    avg_out = Dense(filters // ratio, activation="relu")(Flatten()(avg_pool))
    avg_out = Dense(filters)(avg_out)
    max_out = Dense(filters // ratio, activation="relu")(Flatten()(max_pool))
    max_out = Dense(filters)(max_out)
    channel_att = tf.sigmoid(avg_out + max_out)
    x = x * tf.reshape(channel_att, [-1, 1, 1, filters])
    # Spatial attention
    avg_s = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_s = tf.reduce_max(x, axis=-1, keepdims=True)
    spatial = tf.concat([avg_s, max_s], axis=-1)
    spatial_att = Conv2D(1, 7, padding="same", activation="sigmoid")(spatial)
    return x * spatial_att


# =============================================================================
# EfficientNetV2-M Architecture
# =============================================================================

def build_efficientnetv2m(num_classes=NUM_CLASSES, img_size=IMG_SIZE,
                          dropout_rate=DROPOUT_RATE):
    """
    EfficientNetV2-M with multi-scale pooling (GAP + GMP) head.
    Returns (model, base_model) for fine-tuning control.
    """
    base = tf.keras.applications.EfficientNetV2M(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
        include_preprocessing=True,
    )
    base.trainable = False

    inputs = Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)

    # Multi-scale pooling
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)
    x = Concatenate()([gap, gmp])

    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(dropout_rate * 0.75)(x)
    outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, base


# =============================================================================
# Multimodal Model (Image + Metadata)
# =============================================================================

def build_multimodal_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE, num_meta=17):
    """
    Fuses EfficientNet image features with patient metadata.
    Age + sex + localization adds +2-4% accuracy on HAM10000.
    """
    base = tf.keras.applications.EfficientNetV2M(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
        include_preprocessing=True,
    )
    # Image branch
    img_input = Input(shape=(img_size, img_size, 3), name="image")
    x = base(img_input, training=False)
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)
    img_feat = Concatenate()([gap, gmp])
    img_feat = Dense(256, activation="relu")(img_feat)
    img_feat = BatchNormalization()(img_feat)
    img_feat = Dropout(0.3)(img_feat)

    # Metadata branch
    meta_input = Input(shape=(num_meta,), name="metadata")
    m = Dense(64, activation="relu")(meta_input)
    m = BatchNormalization()(m)
    m = Dropout(0.2)(m)
    m = Dense(32, activation="relu")(m)

    # Late fusion
    fused = Concatenate()([img_feat, m])
    fused = Dense(128, activation="relu")(fused)
    fused = Dropout(0.3)(fused)
    output = Dense(num_classes, activation="softmax", dtype="float32")(fused)

    model = Model(inputs=[img_input, meta_input], outputs=output)
    return model, base


# =============================================================================
# Learning Rate Schedules
# =============================================================================

def cosine_schedule_with_warmup(warmup_epochs=3, total_epochs=15,
                                 eta_min=1e-7, eta_max=1e-3):
    """Cosine annealing with linear warmup."""
    import numpy as np

    def schedule(epoch, lr):
        if epoch < warmup_epochs:
            return eta_max * (epoch + 1) / warmup_epochs
        cycle_epoch = epoch - warmup_epochs
        T_max = total_epochs - warmup_epochs
        return eta_min + 0.5 * (eta_max - eta_min) * (
            1 + np.cos(np.pi * cycle_epoch / T_max)
        )
    return tf.keras.callbacks.LearningRateScheduler(schedule)


# =============================================================================
# Fine-tuning Utilities
# =============================================================================

def unfreeze_top_percent(base_model, percent=0.4):
    """Unfreeze the top `percent` of layers (excluding BatchNormalization)."""
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * (1 - percent))
    for i, layer in enumerate(base_model.layers):
        if i >= unfreeze_from and not isinstance(layer, BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False


def unfreeze_all(base_model):
    """Unfreeze all layers except BatchNormalization."""
    for layer in base_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

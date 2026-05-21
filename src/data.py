"""
Data pipeline for SkinScan AI.

Handles:
- Dataset download (HAM10000 via Kaggle)
- Leakage-free train/val/test split using lesion_id grouping
- Dermoscopy-specific preprocessing (hair removal, color constancy)
- High-performance tf.data pipeline with augmentation
- Patient metadata encoding
- ABCDE clinical feature extraction
"""

import os
import glob
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    DATA_DIR, HAM10000_DIR, HAM10000_IMAGES_DIR, HAM10000_CSV,
    IMG_SIZE, NUM_CLASSES, BATCH_SIZE, SEED, CLASS_NAMES,
)

AUTOTUNE = tf.data.AUTOTUNE


# =============================================================================
# Dataset Download
# =============================================================================

def download_ham10000():
    """Download HAM10000 dataset from Kaggle and organize it."""
    os.makedirs(HAM10000_DIR, exist_ok=True)
    os.system(
        f"kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 "
        f"-p {HAM10000_DIR} --unzip"
    )
    # Consolidate image folders into a single images/ directory
    os.makedirs(HAM10000_IMAGES_DIR, exist_ok=True)
    for part_dir in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        src_dir = os.path.join(HAM10000_DIR, part_dir)
        if os.path.isdir(src_dir):
            for img_file in os.listdir(src_dir):
                src = os.path.join(src_dir, img_file)
                dst = os.path.join(HAM10000_IMAGES_DIR, img_file)
                if not os.path.exists(dst):
                    os.rename(src, dst)
    print(f"HAM10000 images consolidated in {HAM10000_IMAGES_DIR}")
    n_images = len(glob.glob(os.path.join(HAM10000_IMAGES_DIR, "*.jpg")))
    print(f"Total images: {n_images}")


# =============================================================================
# Data Splitting (Leakage-Free)
# =============================================================================

def load_metadata():
    """Load HAM10000 metadata and add image_path column."""
    csv_path = HAM10000_CSV
    if not os.path.exists(csv_path):
        alt = os.path.join(HAM10000_DIR, "HAM10000_metadata")
        if os.path.exists(alt):
            csv_path = alt
        else:
            candidates = glob.glob(os.path.join(HAM10000_DIR, "*metadata*"))
            if candidates:
                csv_path = candidates[0]
    df = pd.read_csv(csv_path)
    df["image_path"] = df["image_id"].apply(
        lambda x: os.path.join(HAM10000_IMAGES_DIR, f"{x}.jpg")
    )
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    df["label"] = df["dx"].map(label_map)
    return df


def create_splits(df, val_size=0.15, test_size=0.15):
    """
    Create train/val/test splits grouped by lesion_id.
    Same lesion never appears in multiple splits — prevents data leakage.
    """
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(df, df["label"], groups=df["lesion_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    relative_test = test_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=SEED)
    val_idx, test_idx = next(gss2.split(temp_df, temp_df["label"], groups=temp_df["lesion_id"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    # Verify no leakage
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])
    assert len(train_lesions & val_lesions) == 0, "Train/Val leakage!"
    assert len(train_lesions & test_lesions) == 0, "Train/Test leakage!"
    assert len(val_lesions & test_lesions) == 0, "Val/Test leakage!"

    print(f"Split sizes — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def get_class_weights(train_df):
    """Compute balanced class weights for training."""
    weights = compute_class_weight(
        "balanced", classes=np.arange(NUM_CLASSES), y=train_df["label"].values
    )
    return dict(enumerate(weights))


# =============================================================================
# Dermoscopy-Specific Preprocessing
# =============================================================================

def remove_hair(image):
    """
    Dullrazor algorithm — removes dark hair artifacts from dermoscopy images.
    Proven to improve classification by ~1-2%.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    result = cv2.inpaint(image, hair_mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
    return result


def apply_shades_of_gray(image, power=6):
    """
    Shades of Gray color constancy correction.
    Removes device-specific color bias — critical for cross-device generalization.
    """
    image = image.astype(np.float32)
    norm = np.power(np.mean(np.power(image, power), axis=(0, 1)), 1.0 / power)
    scale = np.mean(norm) / (norm + 1e-7)
    corrected = np.clip(image * scale, 0, 255).astype(np.uint8)
    return corrected


def preprocess_dermoscopy(image_path, apply_hair_removal=True):
    """Complete dermoscopy preprocessing pipeline for a single image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = apply_shades_of_gray(img)
    if apply_hair_removal:
        img = remove_hair(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    return img


# =============================================================================
# ABCDE Clinical Feature Extraction
# =============================================================================

def extract_abcde_features(image):
    """
    Extract clinical ABCDE features (Asymmetry, Border, Color, Diameter)
    from a skin lesion image. Used as auxiliary input in multimodal model.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    asymmetry_score = 0.0
    border_score = 0.0
    compactness = 1.0
    asymmetry_v = 0.0
    asymmetry_h = 0.0

    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = mask.shape[1] // 2, mask.shape[0] // 2

        top = mask[:cy, :].sum()
        bottom = mask[cy:, :].sum()
        left = mask[:, :cx].sum()
        right = mask[:, cx:].sum()
        asymmetry_v = abs(top - bottom) / (top + bottom + 1e-6)
        asymmetry_h = abs(left - right) / (left + right + 1e-6)
        asymmetry_score = (asymmetry_v + asymmetry_h) / 2

        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)
        border_score = min(compactness / 10.0, 1.0)

    # Color variation
    lesion_pixels = image[mask > 0]
    color_score = 0.0
    if len(lesion_pixels) > 0:
        color_std = lesion_pixels.std(axis=0).mean()
        color_score = color_std / 128.0

    # Diameter (relative)
    diameter_score = 0.0
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        diameter_score = max(w, h) / max(image.shape[:2])

    features = np.array([
        asymmetry_score, border_score, color_score, diameter_score,
        asymmetry_v, asymmetry_h, compactness
    ], dtype=np.float32)
    return features


# =============================================================================
# Patient Metadata Encoding
# =============================================================================

def prepare_metadata_features(df):
    """Encode patient metadata (age, sex, localization) into feature vectors."""
    features = pd.DataFrame()
    features["age_norm"] = df["age"].fillna(df["age"].median()) / 100.0
    features["sex_male"] = (df["sex"] == "male").astype(float)
    features["sex_female"] = (df["sex"] == "female").astype(float)
    features["sex_unknown"] = df["sex"].isna().astype(float)
    loc_dummies = pd.get_dummies(df["localization"], prefix="loc")
    features = pd.concat([features, loc_dummies], axis=1)
    return features.values.astype(np.float32)


# =============================================================================
# tf.data Pipeline
# =============================================================================

def load_and_preprocess_image(image_path, label, augment=False):
    """Load and preprocess a single image inside tf.data pipeline."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    # EfficientNetV2 with include_preprocessing=True expects [0, 255]

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_hue(img, max_delta=0.05)
        img = tf.clip_by_value(img, 0, 255)

    label = tf.one_hot(label, depth=NUM_CLASSES)
    return img, label


def build_dataset(df, augment=False, shuffle=False, batch_size=BATCH_SIZE):
    """Build a performant tf.data.Dataset from a DataFrame."""
    paths = df["image_path"].values
    labels = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: load_and_preprocess_image(p, l, augment),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# =============================================================================
# Main entry point (for testing pipeline standalone)
# =============================================================================

if __name__ == "__main__":
    print("Loading metadata...")
    df = load_metadata()
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['dx'].value_counts()}")

    print("\nCreating splits...")
    train_df, val_df, test_df = create_splits(df)

    print("\nClass weights:")
    cw = get_class_weights(train_df)
    for cls_idx, weight in cw.items():
        print(f"  {CLASS_NAMES[cls_idx]}: {weight:.3f}")

    print("\nBuilding datasets...")
    train_ds = build_dataset(train_df, augment=True, shuffle=True)
    val_ds = build_dataset(val_df)

    for images, labels in train_ds.take(1):
        print(f"Batch image shape: {images.shape}")
        print(f"Batch label shape: {labels.shape}")
        print(f"Image value range: {images.numpy().min():.1f} – {images.numpy().max():.1f}")
    print("\nData pipeline OK.")

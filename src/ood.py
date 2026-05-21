"""
Out-of-Distribution (OOD) Detection for SkinScan AI.

Uses the trained EfficientNetV2-M as a feature extractor, then trains an
Isolation Forest on the embedding space. At inference, non-skin images
(dogs, cars, random photos) are flagged as OOD before reaching the classifier.

Usage:
  python src/ood.py            # Train OOD detector from saved model
  python src/ood.py --test     # Test with a sample image
"""

import os
import sys
import argparse

import numpy as np
import joblib
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BEST_MODEL_PATH, OOD_MODEL_PATH, OOD_CONTAMINATION,
    MODELS_DIR, IMG_SIZE, BATCH_SIZE,
)
from src.data import load_metadata, create_splits, build_dataset


def build_embedding_extractor(model_path=BEST_MODEL_PATH):
    """
    Load the trained classifier and create an embedding extractor
    that outputs the penultimate dense layer (before softmax).
    """
    model = tf.keras.models.load_model(model_path, compile=False)
    # The penultimate layer is the last Dense(256) before the softmax output
    # Find it by walking backwards from the output
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if isinstance(layer, tf.keras.layers.Dense) and layer.units != 7:
            embedding_layer = layer
            break

    extractor = tf.keras.Model(
        inputs=model.input,
        outputs=embedding_layer.output,
    )
    return extractor


def extract_embeddings(extractor, dataset, max_samples=5000):
    """Extract embeddings from a dataset using the feature extractor."""
    embeddings = []
    count = 0
    for images, _ in dataset:
        emb = extractor(images, training=False).numpy()
        embeddings.append(emb)
        count += len(emb)
        if count >= max_samples:
            break
    embeddings = np.concatenate(embeddings, axis=0)[:max_samples]
    print(f"Extracted {len(embeddings)} embeddings of shape {embeddings.shape[1:]}")
    return embeddings


def train_ood_detector(embeddings, contamination=OOD_CONTAMINATION):
    """
    Train an Isolation Forest on in-distribution embeddings.
    Isolation Forest isolates anomalies by random feature splitting —
    OOD samples require fewer splits to isolate.
    """
    from sklearn.ensemble import IsolationForest

    detector = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    detector.fit(embeddings)
    return detector


def save_ood_detector(detector, path=OOD_MODEL_PATH):
    """Save trained OOD detector to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(detector, path)
    print(f"OOD detector saved to: {path}")


def load_ood_detector(path=OOD_MODEL_PATH):
    """Load OOD detector from disk."""
    return joblib.load(path)


def is_ood(detector, extractor, image_array):
    """
    Check if a single image is Out-of-Distribution.

    Args:
        detector: Trained IsolationForest
        extractor: Keras embedding extractor model
        image_array: Preprocessed image array of shape (H, W, 3) or (1, H, W, 3)

    Returns:
        (is_anomaly: bool, anomaly_score: float)
        anomaly_score < 0 means OOD, more negative = more anomalous
    """
    if image_array.ndim == 3:
        image_array = image_array[np.newaxis, ...]
    embedding = extractor(image_array, training=False).numpy()
    score = detector.score_samples(embedding)[0]
    is_anomaly = detector.predict(embedding)[0] == -1
    return bool(is_anomaly), float(score)


def main():
    parser = argparse.ArgumentParser(description="Train or test OOD detector")
    parser.add_argument("--test", action="store_true", help="Test OOD with a sample")
    args = parser.parse_args()

    if args.test:
        print("Loading OOD detector and extractor...")
        detector = load_ood_detector()
        extractor = build_embedding_extractor()
        # Create a random noise image (should be OOD)
        fake_image = np.random.randint(0, 255, (1, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
        anomaly, score = is_ood(detector, extractor, fake_image)
        print(f"Random noise image — OOD: {anomaly}, Score: {score:.4f}")
        return

    # Train OOD detector
    print("=" * 60)
    print("TRAINING OOD DETECTOR")
    print("=" * 60)

    print("\nLoading embedding extractor from trained model...")
    extractor = build_embedding_extractor()

    print("\nLoading training data...")
    df = load_metadata()
    train_df, _, _ = create_splits(df)
    train_ds = build_dataset(train_df, augment=False, shuffle=True, batch_size=BATCH_SIZE)

    print("\nExtracting embeddings from training set...")
    embeddings = extract_embeddings(extractor, train_ds, max_samples=5000)

    print("\nTraining Isolation Forest...")
    detector = train_ood_detector(embeddings)

    # Quick validation: score training samples (should mostly be in-distribution)
    in_dist_scores = detector.score_samples(embeddings[:500])
    print(f"\nIn-distribution score stats:")
    print(f"  Mean: {in_dist_scores.mean():.4f}")
    print(f"  Std:  {in_dist_scores.std():.4f}")
    print(f"  Min:  {in_dist_scores.min():.4f}")

    # Score random noise (should be OOD)
    noise_embeddings = np.random.randn(100, embeddings.shape[1]).astype(np.float32)
    ood_scores = detector.score_samples(noise_embeddings)
    print(f"\nOOD (random noise) score stats:")
    print(f"  Mean: {ood_scores.mean():.4f}")
    print(f"  Std:  {ood_scores.std():.4f}")
    print(f"  Min:  {ood_scores.min():.4f}")

    save_ood_detector(detector)
    print("\nOOD detector training complete.")


if __name__ == "__main__":
    main()

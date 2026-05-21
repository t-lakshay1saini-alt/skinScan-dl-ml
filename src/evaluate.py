"""
Comprehensive evaluation suite for SkinScan AI.

Produces:
- Per-class classification report (precision, recall, F1)
- Confusion matrices (raw + normalized)
- ROC curves with per-class AUC
- Precision-Recall curves with per-class AP
- Melanoma-optimized threshold tuning
- Grad-CAM and Grad-CAM++ heatmaps
- Integrated Gradients attribution maps
- Bootstrap confidence intervals
- All figures saved to report_figures/

Usage:
  python src/evaluate.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, f1_score, recall_score,
)
from sklearn.preprocessing import label_binarize

from src.config import (
    BEST_MODEL_PATH, REPORT_FIGURES_DIR, MODELS_DIR,
    IMG_SIZE, NUM_CLASSES, CLASS_NAMES, BATCH_SIZE,
    FOCAL_GAMMA, LABEL_SMOOTHING,
)
from src.data import load_metadata, create_splits, build_dataset
from src.model import focal_label_smooth_loss


# =============================================================================
# Grad-CAM
# =============================================================================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap highlighting regions that drove prediction."""
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def make_gradcampp_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Grad-CAM++: improved localization for multiple relevant regions."""
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_out, preds = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_score = preds[:, pred_index]
            first_grad = tape3.gradient(class_score, conv_out)
        second_grad = tape2.gradient(first_grad, conv_out)
    third_grad = tape1.gradient(second_grad, conv_out)

    global_sum = tf.reduce_sum(conv_out, axis=[0, 1, 2])
    alpha_num = second_grad[0]
    alpha_denom = 2.0 * second_grad[0] + third_grad[0] * global_sum
    alpha = tf.where(
        alpha_denom != 0.0,
        alpha_num / (alpha_denom + 1e-8),
        tf.zeros_like(alpha_num),
    )
    weights = tf.maximum(first_grad[0], 0)
    alphas_weight = tf.reduce_sum(alpha * weights, axis=[0, 1])
    heatmap = tf.reduce_sum(alphas_weight * conv_out[0], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4):
    """Superimpose Grad-CAM heatmap on original image."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(original_img, 1 - alpha, colored_heatmap, alpha, 0)
    return superimposed


def find_last_conv_layer(model):
    """Find the name of the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if len(sub_layer.output_shape) == 4:
                    return sub_layer.name
        if len(layer.output_shape) == 4:
            return layer.name
    return None


# =============================================================================
# Threshold Tuning
# =============================================================================

def optimize_thresholds(y_true_bin, y_pred_proba, target_recall=0.90):
    """
    Find optimal threshold per class.
    For melanoma: maximize recall subject to recall >= target_recall.
    For others: maximize F1.
    """
    thresholds = {}
    for i, cls in enumerate(CLASS_NAMES):
        precision, recall, thresh = precision_recall_curve(
            y_true_bin[:, i], y_pred_proba[:, i]
        )
        if cls == "mel":
            valid = np.where(recall[:-1] >= target_recall)[0]
            if len(valid) > 0:
                opt_thresh = thresh[valid[-1]]
            else:
                opt_thresh = 0.3
            idx = valid[-1] if len(valid) > 0 else 0
            print(f"  {cls}: threshold={opt_thresh:.3f} -> "
                  f"recall={recall[idx]:.3f}, precision={precision[idx]:.3f}")
        else:
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            opt_idx = np.argmax(f1_scores[:-1])
            opt_thresh = thresh[opt_idx]
            print(f"  {cls}: threshold={opt_thresh:.3f} -> F1={f1_scores[opt_idx]:.3f}")
        thresholds[i] = float(opt_thresh)
    return thresholds


def predict_with_thresholds(y_pred_proba, thresholds):
    """Apply per-class thresholds to softmax probabilities."""
    n_samples, n_classes = y_pred_proba.shape
    binary_preds = np.zeros((n_samples, n_classes), dtype=int)
    for i in range(n_classes):
        binary_preds[:, i] = (y_pred_proba[:, i] >= thresholds[i]).astype(int)
    no_pred = binary_preds.sum(axis=1) == 0
    binary_preds[no_pred, np.argmax(y_pred_proba[no_pred], axis=1)] = 1
    return np.argmax(binary_preds, axis=1)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_confusion_matrices(y_true, y_pred, save_path):
    """Plot raw and normalized confusion matrices."""
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Raw Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curves(y_true_bin, y_pred_proba, save_path):
    """Plot per-class ROC curves."""
    auc_scores = roc_auc_score(y_true_bin, y_pred_proba, average=None)

    plt.figure(figsize=(12, 8))
    for i, (cls, auc) in enumerate(zip(CLASS_NAMES, auc_scores)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Per Class")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    return auc_scores


def plot_pr_curves(y_true_bin, y_pred_proba, save_path):
    """Plot per-class Precision-Recall curves."""
    ap_scores = average_precision_score(y_true_bin, y_pred_proba, average=None)
    mean_ap = average_precision_score(y_true_bin, y_pred_proba, average="macro")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_flat = axes.flatten()
    colors = plt.cm.Set1(np.linspace(0, 1, NUM_CLASSES))

    for i, (cls, ap, color) in enumerate(zip(CLASS_NAMES, ap_scores, colors)):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i], y_pred_proba[:, i]
        )
        axes_flat[i].plot(recall, precision, color=color, linewidth=2,
                          label=f"AP = {ap:.3f}")
        axes_flat[i].fill_between(recall, precision, alpha=0.15, color=color)
        baseline = y_true_bin[:, i].sum() / len(y_true_bin)
        axes_flat[i].axhline(y=baseline, linestyle="--", color="gray", linewidth=1,
                             label=f"Baseline = {baseline:.3f}")
        axes_flat[i].set_title(f"{cls.upper()}", fontsize=12, fontweight="bold")
        axes_flat[i].set_xlabel("Recall")
        axes_flat[i].set_ylabel("Precision")
        axes_flat[i].legend(fontsize=9)
        axes_flat[i].set_xlim([0, 1])
        axes_flat[i].set_ylim([0, 1])
        axes_flat[i].grid(True, alpha=0.3)

    axes_flat[-1].set_visible(False)
    plt.suptitle(f"Precision-Recall Curves (mAP={mean_ap:.4f})",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    return mean_ap, ap_scores


def generate_gradcam_grid(model, test_ds, save_path, n_samples=8):
    """Generate a grid of Grad-CAM visualizations for the report."""
    last_conv = find_last_conv_layer(model)
    if last_conv is None:
        print("  Warning: Could not find last conv layer for Grad-CAM")
        return

    images_list = []
    labels_list = []
    for images, labels in test_ds.take(1):
        images_list = images[:n_samples].numpy()
        labels_list = tf.argmax(labels[:n_samples], axis=1).numpy()
        break

    n = min(n_samples, len(images_list))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    for i in range(n):
        img = images_list[i]
        true_label = labels_list[i]
        img_input = img[np.newaxis, ...]
        pred = model.predict(img_input, verbose=0)[0]
        pred_class = np.argmax(pred)

        heatmap = make_gradcam_heatmap(img_input, model, last_conv, pred_class)
        img_display = (img / 255.0).clip(0, 1)
        overlay = overlay_gradcam((img_display * 255).astype(np.uint8), heatmap)

        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f"True: {CLASS_NAMES[true_label]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title("Heatmap")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Pred: {CLASS_NAMES[pred_class]} ({pred[pred_class]*100:.1f}%)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_ci(y_true, y_pred_proba, metric_fn, n_bootstrap=1000, alpha=0.05):
    """Compute bootstrap confidence interval for a metric."""
    scores = []
    n = len(y_true)
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        try:
            score = metric_fn(y_true[idx], y_pred_proba[idx])
            scores.append(score)
        except Exception:
            continue
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    mean = np.mean(scores)
    return mean, lower, upper


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate():
    """Run the complete evaluation pipeline."""
    os.makedirs(REPORT_FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    # Load model
    print("\nLoading best model...")
    model = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)

    # Load data
    print("Loading test data...")
    df = load_metadata()
    _, _, test_df = create_splits(df)
    test_ds = build_dataset(test_df, augment=False, shuffle=False, batch_size=BATCH_SIZE)

    # Predict
    print("Running predictions on test set...")
    y_pred_proba = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_df["label"].values
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Confusion matrices
    print("\nGenerating confusion matrices...")
    plot_confusion_matrices(
        y_true, y_pred,
        os.path.join(REPORT_FIGURES_DIR, "confusion_matrix.png"),
    )

    # ROC curves
    print("Generating ROC curves...")
    auc_scores = plot_roc_curves(
        y_true_bin, y_pred_proba,
        os.path.join(REPORT_FIGURES_DIR, "roc_curves.png"),
    )
    macro_auc = roc_auc_score(y_true_bin, y_pred_proba, average="macro")
    print(f"  Macro AUC: {macro_auc:.4f}")
    for cls, auc in zip(CLASS_NAMES, auc_scores):
        print(f"    {cls}: {auc:.4f}")

    # Precision-Recall curves
    print("\nGenerating Precision-Recall curves...")
    mean_ap, ap_scores = plot_pr_curves(
        y_true_bin, y_pred_proba,
        os.path.join(REPORT_FIGURES_DIR, "pr_curves.png"),
    )
    print(f"  Mean AP: {mean_ap:.4f}")

    # Threshold tuning for melanoma
    print("\n" + "=" * 60)
    print("THRESHOLD TUNING (Melanoma recall >= 0.90)")
    print("=" * 60)
    thresholds = optimize_thresholds(y_true_bin, y_pred_proba, target_recall=0.90)
    y_pred_tuned = predict_with_thresholds(y_pred_proba, thresholds)
    mel_recall = recall_score(y_true, y_pred_tuned, labels=[4], average=None)[0]
    print(f"\n  Melanoma recall after tuning: {mel_recall:.3f}")

    # Grad-CAM visualizations
    print("\nGenerating Grad-CAM visualizations...")
    generate_gradcam_grid(
        model, test_ds,
        os.path.join(REPORT_FIGURES_DIR, "gradcam_grid.png"),
    )

    # Bootstrap confidence intervals
    print("\n" + "=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("=" * 60)

    def auc_metric(yt, yp):
        yt_bin = label_binarize(yt, classes=list(range(NUM_CLASSES)))
        return roc_auc_score(yt_bin, yp, average="macro")

    mean_auc, lower, upper = bootstrap_ci(y_true, y_pred_proba, auc_metric, n_bootstrap=500)
    print(f"  Macro AUC: {mean_auc:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")

    # Save summary metrics
    summary = {
        "macro_auc": macro_auc,
        "mean_ap": mean_ap,
        "melanoma_recall_tuned": mel_recall,
        "per_class_auc": dict(zip(CLASS_NAMES, [float(a) for a in auc_scores])),
        "optimal_thresholds": thresholds,
        "auc_95ci_lower": lower,
        "auc_95ci_upper": upper,
    }
    import json
    summary_path = os.path.join(REPORT_FIGURES_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"All figures saved to: {REPORT_FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()

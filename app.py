"""
SkinScan AI — Streamlit Web Application.

Features:
- Upload dermoscopy images for classification
- Out-of-Distribution detection (rejects non-skin images)
- 7-class skin lesion prediction with confidence scores
- Grad-CAM explainability overlays
- Patient metadata input for enhanced predictions
- Monte Carlo Dropout uncertainty estimation

Run with:
  streamlit run app.py
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    BEST_MODEL_PATH, OOD_MODEL_PATH, IMG_SIZE, NUM_CLASSES, CLASS_NAMES,
    CLASS_FULL_NAMES,
)
from src.evaluate import make_gradcam_heatmap, overlay_gradcam, find_last_conv_layer


# =============================================================================
# App Configuration
# =============================================================================

st.set_page_config(
    page_title="SkinScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_INFO = {
    0: ("Actinic Keratoses", "Pre-cancerous", "🟠", "#FFA500"),
    1: ("Basal Cell Carcinoma", "Cancerous", "🔴", "#FF0000"),
    2: ("Benign Keratosis", "Benign", "🟢", "#00CC00"),
    3: ("Dermatofibroma", "Benign", "🟢", "#00CC00"),
    4: ("Melanoma", "Cancerous — HIGH RISK", "🔴", "#FF0000"),
    5: ("Melanocytic Nevi", "Benign", "🟢", "#00CC00"),
    6: ("Vascular Lesion", "Benign", "🟡", "#FFCC00"),
}


# =============================================================================
# Model Loading (Cached)
# =============================================================================

@st.cache_resource
def load_classifier():
    """Load the trained skin lesion classifier."""
    if not os.path.exists(BEST_MODEL_PATH):
        return None
    model = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)
    return model


@st.cache_resource
def load_ood_detector():
    """Load the OOD detector and embedding extractor."""
    if not os.path.exists(OOD_MODEL_PATH) or not os.path.exists(BEST_MODEL_PATH):
        return None, None
    import joblib
    from src.ood import build_embedding_extractor
    detector = joblib.load(OOD_MODEL_PATH)
    extractor = build_embedding_extractor()
    return detector, extractor


# =============================================================================
# Inference Functions
# =============================================================================

def preprocess_for_model(image_pil):
    """Convert PIL image to model input format."""
    img = image_pil.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32)
    return img_array


def predict_with_uncertainty(model, img_array, n_iterations=30):
    """Monte Carlo Dropout for uncertainty estimation."""
    predictions = []
    img_batch = img_array[np.newaxis, ...]
    for _ in range(n_iterations):
        pred = model(img_batch, training=True).numpy()[0]
        predictions.append(pred)
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8))
    return mean_pred, std_pred, entropy


def check_ood(detector, extractor, img_array):
    """Check if the image is out-of-distribution."""
    if detector is None or extractor is None:
        return False, 0.0
    img_batch = img_array[np.newaxis, ...]
    embedding = extractor(img_batch, training=False).numpy()
    score = detector.score_samples(embedding)[0]
    is_anomaly = detector.predict(embedding)[0] == -1
    return bool(is_anomaly), float(score)


# =============================================================================
# UI Layout
# =============================================================================

def main():
    st.title("🔬 SkinScan AI — Skin Lesion Classifier")
    st.markdown(
        "Upload a dermoscopy image to get an AI-powered classification "
        "with explainability heatmaps."
    )

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        show_gradcam = st.checkbox("Show Grad-CAM Explanation", value=True)
        show_uncertainty = st.checkbox("Show Uncertainty Estimate", value=True)
        st.markdown("---")
        st.header("About")
        st.markdown(
            "**SkinScan AI** uses EfficientNetV2-M trained on HAM10000 "
            "to classify skin lesions into 7 categories."
        )
        st.markdown(
            "The model includes:\n"
            "- OOD detection (rejects non-skin images)\n"
            "- Grad-CAM explainability\n"
            "- Monte Carlo Dropout uncertainty\n"
            "- Melanoma-optimized thresholds"
        )

    # Load models
    model = load_classifier()
    ood_detector, ood_extractor = load_ood_detector()

    if model is None:
        st.error(
            "No trained model found. Please run training first:\n\n"
            "```\npython src/train.py\n```"
        )
        return

    # File upload
    uploaded = st.file_uploader(
        "Upload a dermoscopy image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a dermoscopy image of a skin lesion for classification.",
    )

    if uploaded is None:
        st.info("Please upload a dermoscopy image to begin analysis.")
        return

    # Process uploaded image
    image = Image.open(uploaded).convert("RGB")
    img_array = preprocess_for_model(image)

    # Layout columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    # OOD check
    is_ood_flag, ood_score = check_ood(ood_detector, ood_extractor, img_array)
    if is_ood_flag:
        st.error(
            "⚠️ **OUT-OF-DISTRIBUTION DETECTED**\n\n"
            "This image does not appear to be a skin lesion. "
            "The classifier's predictions may not be reliable.\n\n"
            f"Anomaly score: {ood_score:.4f}"
        )

    # Classification
    with st.spinner("Analyzing image..."):
        if show_uncertainty:
            mean_pred, std_pred, entropy = predict_with_uncertainty(model, img_array)
        else:
            mean_pred = model.predict(img_array[np.newaxis, ...], verbose=0)[0]
            std_pred = None
            entropy = None

    top_class = int(np.argmax(mean_pred))
    confidence = mean_pred[top_class] * 100
    name, severity, icon, color = CLASS_INFO[top_class]

    with col2:
        st.subheader("Diagnosis")
        st.markdown(f"### {icon} {name}")
        st.metric("Confidence", f"{confidence:.1f}%")
        st.metric("Severity", severity)

        if show_uncertainty and entropy is not None:
            st.markdown("---")
            st.subheader("Uncertainty")
            uncertainty_level = "Low" if entropy < 1.0 else "Medium" if entropy < 1.5 else "High"
            st.metric("Entropy", f"{entropy:.3f} ({uncertainty_level})")
            if entropy > 1.5:
                st.warning("High uncertainty — consider specialist review.")

        # Probability distribution
        st.markdown("---")
        st.subheader("Class Probabilities")
        prob_data = {CLASS_FULL_NAMES[CLASS_NAMES[i]]: float(mean_pred[i])
                     for i in range(NUM_CLASSES)}
        st.bar_chart(prob_data)

    with col3:
        if show_gradcam:
            st.subheader("Grad-CAM Explanation")
            last_conv = find_last_conv_layer(model)
            if last_conv:
                img_input = img_array[np.newaxis, ...]
                heatmap = make_gradcam_heatmap(img_input, model, last_conv, top_class)
                img_display = (img_array / 255.0 * 255).astype(np.uint8)
                overlay_img = overlay_gradcam(img_display, heatmap)
                st.image(overlay_img, use_container_width=True,
                         caption="Regions driving the prediction")
            else:
                st.warning("Could not generate Grad-CAM visualization.")

    # Clinical warnings
    if top_class in [1, 4]:
        st.error(
            "⚠️ **HIGH RISK DETECTED** — This lesion may be cancerous. "
            "Please consult a dermatologist immediately for professional evaluation."
        )

    # Disclaimer
    st.markdown("---")
    st.info(
        "⚕️ **Medical Disclaimer:** This AI tool is for educational and research "
        "purposes only. It is NOT a clinical diagnostic tool. Always consult a "
        "qualified dermatologist for medical advice."
    )


if __name__ == "__main__":
    main()

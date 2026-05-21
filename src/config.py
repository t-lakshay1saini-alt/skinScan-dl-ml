"""Central configuration for SkinScan AI project."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
REPORT_FIGURES_DIR = os.path.join(BASE_DIR, "report_figures")

HAM10000_DIR = os.path.join(DATA_DIR, "ham10000")
HAM10000_IMAGES_DIR = os.path.join(HAM10000_DIR, "images")
HAM10000_CSV = os.path.join(HAM10000_DIR, "HAM10000_metadata.csv")

# Image settings
IMG_SIZE = 380
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Training settings
BATCH_SIZE = 32
NUM_CLASSES = 7
SEED = 42

# Class names (HAM10000 7-class)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesion",
}

# Phase 1 (warmup) settings
PHASE1_EPOCHS = 5
PHASE1_LR = 1e-3

# Phase 2 (fine-tune top 40%) settings
PHASE2_EPOCHS = 15
PHASE2_LR = 1e-4

# Phase 3 (full fine-tune) settings
PHASE3_EPOCHS = 10
PHASE3_LR = 1e-5

# Loss settings
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
LABEL_SMOOTHING = 0.1

# Dropout
DROPOUT_RATE = 0.4

# OOD detection
OOD_MODEL_PATH = os.path.join(MODELS_DIR, "ood_detector.pkl")
OOD_CONTAMINATION = 0.05

# Best model path
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")

"""
Training script for SkinScan AI.

Implements 3-phase fine-tuning:
  Phase 1: Frozen backbone warmup (5 epochs, LR=1e-3)
  Phase 2: Fine-tune top 40% of backbone (15 epochs, LR=1e-4)
  Phase 3: Full fine-tune with very small LR (10 epochs, LR=1e-5)

Usage:
  python src/train.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODELS_DIR, LOGS_DIR, BEST_MODEL_PATH, SEED, BATCH_SIZE,
    PHASE1_EPOCHS, PHASE1_LR, PHASE2_EPOCHS, PHASE2_LR,
    PHASE3_EPOCHS, PHASE3_LR, FOCAL_GAMMA, LABEL_SMOOTHING,
)
from src.data import load_metadata, create_splits, get_class_weights, build_dataset
from src.model import (
    build_efficientnetv2m, focal_label_smooth_loss,
    unfreeze_top_percent, unfreeze_all, cosine_schedule_with_warmup,
)


def setup_environment():
    """Configure GPU, mixed precision, and reproducibility."""
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Enable mixed precision for faster training on GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"Found {len(gpus)} GPU(s): {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        print(f"Mixed precision enabled: compute={policy.compute_dtype}, "
              f"variable={policy.variable_dtype}")
    else:
        print("No GPU found. Training on CPU (will be slow).")


def get_callbacks(run_id, phase_name):
    """Build callbacks for a training phase."""
    model_dir = os.path.join(MODELS_DIR, f"run_{run_id}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOGS_DIR, f"run_{run_id}", phase_name),
            histogram_freq=1,
            update_freq="epoch",
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(model_dir, "training_log.csv"),
            append=True,
        ),
    ]


def compile_model(model, learning_rate):
    """Compile model with focal loss and standard metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=focal_label_smooth_loss(gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTHING),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )


def train():
    """Run the full 3-phase training pipeline."""
    setup_environment()

    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_metadata()
    train_df, val_df, test_df = create_splits(df)
    class_weights = get_class_weights(train_df)

    print("\nClass weights:")
    from src.config import CLASS_NAMES
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {class_weights[idx]:.3f}")

    # Build datasets
    train_ds = build_dataset(train_df, augment=True, shuffle=True, batch_size=BATCH_SIZE)
    val_ds = build_dataset(val_df, augment=False, shuffle=False, batch_size=BATCH_SIZE)

    # Build model
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)
    model, base = build_efficientnetv2m()
    model.summary(print_fn=lambda x: print(x) if "Total" in x or "Trainable" in x else None)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # =========================================================================
    # PHASE 1: Frozen backbone warmup
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"PHASE 1: Frozen Backbone Warmup ({PHASE1_EPOCHS} epochs, LR={PHASE1_LR})")
    print("=" * 60)
    base.trainable = False
    compile_model(model, PHASE1_LR)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(run_id, "phase1"),
    )

    # =========================================================================
    # PHASE 2: Fine-tune top 40% of backbone
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"PHASE 2: Fine-tune Top 40% ({PHASE2_EPOCHS} epochs, LR={PHASE2_LR})")
    print("=" * 60)
    unfreeze_top_percent(base, percent=0.4)
    compile_model(model, PHASE2_LR)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(run_id, "phase2"),
    )

    # =========================================================================
    # PHASE 3: Full fine-tune
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"PHASE 3: Full Fine-tune ({PHASE3_EPOCHS} epochs, LR={PHASE3_LR})")
    print("=" * 60)
    unfreeze_all(base)
    compile_model(model, PHASE3_LR)

    history3 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE3_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(run_id, "phase3"),
    )

    # =========================================================================
    # Final evaluation on test set
    # =========================================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    test_ds = build_dataset(test_df, augment=False, shuffle=False, batch_size=BATCH_SIZE)

    # Load best checkpoint
    best_model = tf.keras.models.load_model(
        BEST_MODEL_PATH,
        custom_objects={"focal_loss": focal_label_smooth_loss(FOCAL_GAMMA, LABEL_SMOOTHING)},
        compile=False,
    )
    compile_model(best_model, PHASE3_LR)

    results = best_model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test AUC: {results[2]:.4f}")

    # Save test split for evaluation script
    test_df.to_csv(os.path.join(MODELS_DIR, f"run_{run_id}", "test_split.csv"), index=False)
    print(f"\nTraining complete. Best model saved to: {BEST_MODEL_PATH}")
    print(f"Run artifacts in: {MODELS_DIR}/run_{run_id}/")


if __name__ == "__main__":
    train()

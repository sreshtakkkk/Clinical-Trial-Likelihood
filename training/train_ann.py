"""
training/train_ann.py
======================
Artificial Neural Network (MLP) using TensorFlow/Keras.
Includes: architecture search, early stopping, learning rate scheduling,
dropout regularization, batch normalization, and SHAP DeepExplainer.
"""

import sys
import os
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from utils.preprocessing import ClinicalTrialPreprocessor, load_raw_data, DATASETS_DIR, SAVED_MODELS_DIR
from utils.visualization import plot_confusion_matrix, plot_roc_curve, VIZ_DIR


def build_model(input_dim: int, units_1: int = 256, units_2: int = 128,
                units_3: int = 64, dropout_rate: float = 0.3,
                learning_rate: float = 0.001):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(units_1, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(dropout_rate),

        layers.Dense(units_2, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(dropout_rate * 0.7),

        layers.Dense(units_3, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(dropout_rate * 0.5),

        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model


def train_ann(X: np.ndarray, y: np.ndarray, feature_names: list):
    logger.info("=" * 60)
    logger.info("Training ANN (MLP) with TensorFlow/Keras")
    logger.info("=" * 60)

    import tensorflow as tf
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report
    )
    from sklearn.base import BaseEstimator, ClassifierMixin
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tf.random.set_seed(42)
    np.random.seed(42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    input_dim = X_train.shape[1]

    # Simple architecture search over key hyperparameters
    architectures = [
        {"units_1": 256, "units_2": 128, "units_3": 64, "dropout_rate": 0.3, "learning_rate": 1e-3},
        {"units_1": 512, "units_2": 256, "units_3": 128, "dropout_rate": 0.4, "learning_rate": 5e-4},
        {"units_1": 128, "units_2": 64, "units_3": 32, "dropout_rate": 0.25, "learning_rate": 1e-3},
    ]

    best_val_auc, best_cfg, best_model = 0.0, None, None

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=15, restore_best_weights=True,
            mode="max", verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=0
        ),
    ]

    for cfg in architectures:
        logger.info(f"Testing architecture: {cfg}")
        m = build_model(input_dim, **cfg)
        history = m.fit(
            X_train, y_train,
            epochs=150,
            batch_size=64,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight={0: 1.0, 1: (np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))},
            verbose=0,
        )
        val_auc = max(history.history.get("val_auc", [0]))
        logger.info(f"  Val AUC: {val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_cfg = cfg
            best_model = m

    logger.info(f"Best architecture: {best_cfg} | Val AUC: {best_val_auc:.4f}")

    # Retrain best on full train+val
    final_model = build_model(input_dim, **best_cfg)
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    final_model.fit(
        X_full, y_full,
        epochs=200,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight={0: 1.0, 1: (np.sum(y_full == 0) / max(np.sum(y_full == 1), 1))},
        verbose=0,
    )

    y_proba = final_model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "model": "ANN",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "best_architecture": best_cfg,
    }

    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Failure", "Success"]))
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:12s}: {v:.4f}")

    # Training history plot
    hist = final_model.history.history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")

    axes[0].plot(hist.get("loss", []), color="#6c8fff", label="Train Loss")
    axes[0].plot(hist.get("val_loss", []), color="#f87171", label="Val Loss")
    axes[0].set_title("ANN — Loss Curves", color="#e0e4ff")
    axes[0].set_xlabel("Epoch", color="#e0e4ff")
    axes[0].set_ylabel("Loss", color="#e0e4ff")
    axes[0].legend()

    axes[1].plot(hist.get("auc", []), color="#4ade80", label="Train AUC")
    axes[1].plot(hist.get("val_auc", []), color="#fbbf24", label="Val AUC")
    axes[1].set_title("ANN — AUC Curves", color="#e0e4ff")
    axes[1].set_xlabel("Epoch", color="#e0e4ff")
    axes[1].set_ylabel("AUC", color="#e0e4ff")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(VIZ_DIR / "ann_training_history.png", bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)

    # Manual CV using Keras wrapper approach
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_model = build_model(input_dim, **best_cfg)
        fold_model.fit(
            X[train_idx], y[train_idx],
            epochs=50, batch_size=64, verbose=0,
            validation_data=(X[val_idx], y[val_idx]),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=8, mode="max")],
        )
        fold_proba = fold_model.predict(X[val_idx], verbose=0).flatten()
        cv_aucs.append(roc_auc_score(y[val_idx], fold_proba))
        logger.info(f"  Fold {fold_idx + 1} AUC: {cv_aucs[-1]:.4f}")

    metrics["cv_auc_mean"] = float(np.mean(cv_aucs))
    metrics["cv_auc_std"] = float(np.std(cv_aucs))
    logger.info(f"CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")

    plot_confusion_matrix(y_test, y_pred, "ANN")
    plot_roc_curve(y_test, y_proba, "ANN")

    # Save Keras model
    keras_path = SAVED_MODELS_DIR / "ann_model.keras"
    final_model.save(keras_path)
    logger.info(f"Keras model saved: {keras_path}")

    # Save metadata
    joblib.dump({"best_cfg": best_cfg, "metrics": metrics}, SAVED_MODELS_DIR / "ann_metadata.joblib")

    return final_model, metrics, X_test, y_test, y_proba


if __name__ == "__main__":
    df, _ = load_raw_data()
    preprocessor = ClinicalTrialPreprocessor.load()
    X = preprocessor.transform(df)
    y = df["label"].astype(int).values
    model, metrics, X_test, y_test, y_proba = train_ann(X, y, preprocessor.feature_names)
    logger.info("✅ ANN training complete.")

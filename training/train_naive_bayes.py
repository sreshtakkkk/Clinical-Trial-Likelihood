"""
training/train_naive_bayes.py
==============================
Gaussian Naive Bayes — Training Pipeline.
GNB works on continuous features after scaling.
Includes alpha (var_smoothing) tuning via GridSearchCV.
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from utils.preprocessing import ClinicalTrialPreprocessor, load_raw_data, DATASETS_DIR, SAVED_MODELS_DIR
from utils.visualization import plot_confusion_matrix, plot_roc_curve


def train_naive_bayes(X: np.ndarray, y: np.ndarray, feature_names: list):
    logger.info("=" * 60)
    logger.info("Training Gaussian Naive Bayes Classifier")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # GNB var_smoothing tuning
    param_grid = {
        "var_smoothing": np.logspace(-12, -1, 20)
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("Tuning var_smoothing via GridSearchCV...")
    grid_search = GridSearchCV(
        GaussianNB(), param_grid=param_grid,
        cv=skf, scoring="f1_weighted", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    logger.info(f"Best var_smoothing: {best_params['var_smoothing']:.2e}")

    model = GaussianNB(var_smoothing=best_params["var_smoothing"])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Naive Bayes",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "best_params": best_params,
    }

    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Failure", "Success"]))
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:12s}: {v:.4f}")

    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted", n_jobs=-1)
    logger.info(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    metrics["cv_f1_mean"] = cv_scores.mean()
    metrics["cv_f1_std"] = cv_scores.std()

    plot_confusion_matrix(y_test, y_pred, "Naive Bayes")
    plot_roc_curve(y_test, y_proba, "Naive Bayes")

    # Posterior probability distribution plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.visualization import VIZ_DIR

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    bins = np.linspace(0, 1, 50)
    ax.hist(y_proba[y_test == 0], bins=bins, alpha=0.7, color="#f87171", label="Failure", density=True)
    ax.hist(y_proba[y_test == 1], bins=bins, alpha=0.7, color="#4ade80", label="Success", density=True)
    ax.axvline(0.5, color="white", linestyle="--", lw=1.5, label="Decision Boundary")
    ax.set_xlabel("Posterior Probability P(Success)", color="#e0e4ff")
    ax.set_ylabel("Density", color="#e0e4ff")
    ax.set_title("Naive Bayes — Posterior Probability Distribution", color="#e0e4ff", fontsize=13)
    ax.legend()
    fig.savefig(VIZ_DIR / "nb_posterior_dist.png", bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)

    save_path = SAVED_MODELS_DIR / "naive_bayes_model.joblib"
    joblib.dump(model, save_path)
    logger.info(f"Model saved: {save_path}")

    return model, metrics, X_test, y_test, y_proba


if __name__ == "__main__":
    df, _ = load_raw_data()
    preprocessor = ClinicalTrialPreprocessor.load()
    X = preprocessor.transform(df)
    y = df["label"].astype(int).values
    model, metrics, X_test, y_test, y_proba = train_naive_bayes(X, y, preprocessor.feature_names)
    logger.info("✅ Naive Bayes training complete.")

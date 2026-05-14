"""
training/train_logistic_regression.py
=======================================
Logistic Regression — Training Pipeline.
Tests L1 (Lasso), L2 (Ridge), and ElasticNet regularization.
Includes coefficient-based interpretability plot.
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
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
from utils.visualization import plot_confusion_matrix, plot_roc_curve, VIZ_DIR


def train_logistic_regression(X: np.ndarray, y: np.ndarray, feature_names: list):
    logger.info("=" * 60)
    logger.info("Training Logistic Regression Classifier")
    logger.info("=" * 60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Test L1, L2, ElasticNet
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "class_weight": ["balanced", None],
        "max_iter": [1000],
    }

    logger.info("Running GridSearchCV (L1/L2 regularization)...")
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid=param_grid,
        cv=skf,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    logger.info(f"Best params: {best_params}")

    # Also test ElasticNet separately
    lr_en = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
        max_iter=2000, random_state=42, class_weight="balanced"
    )
    lr_en.fit(X_train, y_train)
    en_auc = roc_auc_score(y_test, lr_en.predict_proba(X_test)[:, 1])
    logger.info(f"ElasticNet ROC-AUC: {en_auc:.4f}")

    model = grid_search.best_estimator_
    best_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    if en_auc > best_auc:
        model = lr_en
        logger.info("Using ElasticNet model (better AUC)")
    else:
        logger.info("Using L1/L2 GridSearch model")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Logistic Regression",
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

    plot_confusion_matrix(y_test, y_pred, "Logistic Regression")
    plot_roc_curve(y_test, y_proba, "Logistic Regression")

    # Coefficient plot
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        coef_df = pd.DataFrame({
            "feature": feature_names[:len(coefs)],
            "coefficient": coefs
        }).sort_values("coefficient", key=abs, ascending=False).head(25)

        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0f1117")
        ax.set_facecolor("#1a1d27")
        colors = ["#4ade80" if c > 0 else "#f87171" for c in coef_df["coefficient"]]
        ax.barh(coef_df["feature"][::-1], coef_df["coefficient"][::-1],
                color=colors[::-1], edgecolor="#ffffff20")
        ax.axvline(0, color="white", lw=1, linestyle="--")
        ax.set_title("Logistic Regression — Top 25 Feature Coefficients",
                     color="#e0e4ff", fontsize=13, pad=12)
        ax.set_xlabel("Coefficient Value", color="#e0e4ff")
        ax.tick_params(colors="#9ba3c7", labelsize=8)
        plt.tight_layout()
        fig.savefig(VIZ_DIR / "lr_coefficients.png", bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig)

    save_path = SAVED_MODELS_DIR / "logistic_regression_model.joblib"
    joblib.dump(model, save_path)
    logger.info(f"Model saved: {save_path}")

    return model, metrics, X_test, y_test, y_proba


if __name__ == "__main__":
    df, _ = load_raw_data()
    preprocessor = ClinicalTrialPreprocessor.load()
    X = preprocessor.transform(df)
    y = df["label"].astype(int).values
    model, metrics, X_test, y_test, y_proba = train_logistic_regression(X, y, preprocessor.feature_names)
    logger.info("✅ Logistic Regression training complete.")

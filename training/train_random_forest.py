"""
training/train_random_forest.py
================================
Random Forest Classifier — Training Pipeline with RandomizedSearchCV,
cross-validation, OOB score, and feature importance.
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
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
from utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_feature_importance, plot_learning_curve


def train_random_forest(X: np.ndarray, y: np.ndarray, feature_names: list):
    logger.info("=" * 60)
    logger.info("Training Random Forest Classifier")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.3],
        "bootstrap": [True, False],
        "class_weight": ["balanced", None],
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("Running RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True),
        param_distributions=param_dist,
        n_iter=20,
        cv=skf,
        scoring="f1_weighted",
        random_state=42,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    logger.info(f"Best params: {best_params}")

    model = random_search.best_estimator_

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Random Forest",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "oob_score": model.oob_score_ if hasattr(model, "oob_score_") else None,
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

    plot_confusion_matrix(y_test, y_pred, "Random Forest")
    plot_roc_curve(y_test, y_proba, "Random Forest")

    feat_imp_df = pd.DataFrame({
        "feature": feature_names[:len(model.feature_importances_)],
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    plot_feature_importance(feat_imp_df, "Random Forest")
    plot_learning_curve(model, X, y, "Random Forest")

    save_path = SAVED_MODELS_DIR / "random_forest_model.joblib"
    joblib.dump(model, save_path)
    logger.info(f"Model saved: {save_path}")

    return model, metrics, X_test, y_test, y_proba


if __name__ == "__main__":
    df, _ = load_raw_data()
    preprocessor = ClinicalTrialPreprocessor.load()
    X = preprocessor.transform(df)
    y = df["label"].astype(int).values
    model, metrics, X_test, y_test, y_proba = train_random_forest(X, y, preprocessor.feature_names)
    logger.info("✅ Random Forest training complete.")

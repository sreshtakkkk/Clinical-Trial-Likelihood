"""
training/train_xgboost.py
==========================
XGBoost Classifier — Training Pipeline with GridSearchCV hyperparameter tuning,
cross-validation, SHAP explainability, and full evaluation metrics.
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from utils.preprocessing import ClinicalTrialPreprocessor, load_raw_data, DATASETS_DIR, SAVED_MODELS_DIR
from utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_feature_importance, plot_learning_curve


def train_xgboost(X: np.ndarray, y: np.ndarray, feature_names: list):
    logger.info("=" * 60)
    logger.info("Training XGBoost Classifier")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Hyperparameter tuning (GridSearchCV)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1.0, 2.0],
    }

    base_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("Running GridSearchCV (this may take a few minutes)...")
    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        ),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
        },
        cv=skf,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    logger.info(f"Best params: {best_params}")

    # ── Final model with best params + early stopping
    model = xgb.XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Predictions & Metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "XGBoost",
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

    # ── Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted", n_jobs=-1)
    logger.info(f"Stratified K-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    metrics["cv_f1_mean"] = cv_scores.mean()
    metrics["cv_f1_std"] = cv_scores.std()

    # ── Visualizations
    plot_confusion_matrix(y_test, y_pred, "XGBoost")
    plot_roc_curve(y_test, y_proba, "XGBoost")

    feat_imp_df = pd.DataFrame({
        "feature": feature_names[:len(model.feature_importances_)],
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    plot_feature_importance(feat_imp_df, "XGBoost")

    # ── SHAP
    try:
        from utils.feature_engineering import compute_shap_importance
        shap_df, _, _ = compute_shap_importance(model, X_test, feature_names[:X_test.shape[1]])
        logger.info(f"Top SHAP features:\n{shap_df.head(10).to_string(index=False)}")
    except Exception as e:
        logger.warning(f"SHAP skipped: {e}")

    # ── Save model
    save_path = SAVED_MODELS_DIR / "xgboost_model.joblib"
    joblib.dump(model, save_path)
    logger.info(f"Model saved: {save_path}")

    return model, metrics, X_test, y_test, y_proba


if __name__ == "__main__":
    df, _ = load_raw_data()
    preprocessor = ClinicalTrialPreprocessor()
    X, y, feature_names = preprocessor.fit_transform(df, DATASETS_DIR / "clintox.csv")
    preprocessor.save()

    model, metrics, X_test, y_test, y_proba = train_xgboost(X, y, feature_names)

    results_df = pd.DataFrame([{
        "Model": metrics["model"],
        "Accuracy": round(metrics["accuracy"], 4),
        "Precision": round(metrics["precision"], 4),
        "Recall": round(metrics["recall"], 4),
        "F1": round(metrics["f1"], 4),
        "ROC_AUC": round(metrics["roc_auc"], 4),
        "CV_F1_Mean": round(metrics["cv_f1_mean"], 4),
    }])
    results_df.to_csv(BASE_DIR / "evaluation" / "xgboost_results.csv", index=False)
    logger.info("✅ XGBoost training complete.")

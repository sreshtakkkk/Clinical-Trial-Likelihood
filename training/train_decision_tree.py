"""
training/train_decision_tree.py
================================
Decision Tree Classifier — Training Pipeline with GridSearchCV,
cost-complexity pruning, and tree visualization.
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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
from utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_feature_importance, VIZ_DIR


def train_decision_tree(X: np.ndarray, y: np.ndarray, feature_names: list):
    logger.info("=" * 60)
    logger.info("Training Decision Tree Classifier")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", None],
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("Running GridSearchCV for Decision Tree...")
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid={
            "criterion": ["gini", "entropy"],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "class_weight": ["balanced", None],
        },
        cv=skf,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    logger.info(f"Best params: {best_params}")

    # Cost-complexity pruning path
    model = grid_search.best_estimator_

    # Try post-pruning
    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[::max(1, len(path.ccp_alphas) // 10)]  # sample

    best_alpha, best_score = 0.0, 0.0
    for alpha in ccp_alphas[1:]:
        pruned = DecisionTreeClassifier(**best_params, ccp_alpha=alpha, random_state=42)
        scores = cross_val_score(pruned, X_train, y_train, cv=3, scoring="f1_weighted")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_alpha = alpha

    model = DecisionTreeClassifier(**best_params, ccp_alpha=best_alpha, random_state=42)
    model.fit(X_train, y_train)
    logger.info(f"Optimal CCP alpha: {best_alpha:.6f} | Tree depth: {model.get_depth()}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Decision Tree",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "tree_depth": model.get_depth(),
        "n_leaves": model.get_n_leaves(),
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

    plot_confusion_matrix(y_test, y_pred, "Decision Tree")
    plot_roc_curve(y_test, y_proba, "Decision Tree")

    feat_imp_df = pd.DataFrame({
        "feature": feature_names[:len(model.feature_importances_)],
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    plot_feature_importance(feat_imp_df, "Decision Tree")

    # Tree visualization (top 3 levels)
    fig, ax = plt.subplots(figsize=(20, 8), facecolor="#0f1117")
    plot_tree(
        model, max_depth=3,
        feature_names=feature_names[:X_train.shape[1]],
        class_names=["Failure", "Success"],
        filled=True, rounded=True, fontsize=8, ax=ax
    )
    ax.set_title("Decision Tree Structure (Max Depth 3)", fontsize=14, color="white")
    fig.savefig(VIZ_DIR / "decision_tree_structure.png", bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)

    save_path = SAVED_MODELS_DIR / "decision_tree_model.joblib"
    joblib.dump(model, save_path)
    logger.info(f"Model saved: {save_path}")

    return model, metrics, X_test, y_test, y_proba


if __name__ == "__main__":
    df, _ = load_raw_data()
    preprocessor = ClinicalTrialPreprocessor.load()
    X = preprocessor.transform(df)
    y = df["label"].astype(int).values
    model, metrics, X_test, y_test, y_proba = train_decision_tree(X, y, preprocessor.feature_names)
    logger.info("✅ Decision Tree training complete.")

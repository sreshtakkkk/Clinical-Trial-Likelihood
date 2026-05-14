"""
utils/feature_engineering.py
=============================
Feature Selection, RFE, and Advanced Feature Engineering.

Includes:
- Recursive Feature Elimination (RFE)
- Correlation-based filtering
- SHAP-based feature importance
- Variance threshold filtering
- SelectKBest with chi2 / mutual information
- Feature importance from tree models
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

from sklearn.feature_selection import (
    RFE, SelectKBest, mutual_info_classif, VarianceThreshold, SelectFromModel
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def remove_low_variance_features(X: np.ndarray, feature_names: list, threshold: float = 0.01):
    """Remove features with variance below threshold."""
    selector = VarianceThreshold(threshold=threshold)
    X_sel = selector.fit_transform(X)
    selected_mask = selector.get_support()
    selected_names = [n for n, m in zip(feature_names, selected_mask) if m]
    logger.info(f"Variance filter: {X.shape[1]} → {X_sel.shape[1]} features")
    return X_sel, selected_names, selector


def remove_correlated_features(X: np.ndarray, feature_names: list, threshold: float = 0.95):
    """Remove features with Pearson correlation above threshold."""
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    df_filtered = df.drop(columns=to_drop)
    logger.info(f"Correlation filter: dropped {len(to_drop)} features. Remaining: {df_filtered.shape[1]}")
    return df_filtered.values, df_filtered.columns.tolist(), to_drop


def select_k_best_features(X: np.ndarray, y: np.ndarray, feature_names: list, k: int = 100):
    """Select top-k features using mutual information."""
    # Shift X to non-negative for mutual_info_classif
    X_shifted = X - X.min(axis=0)
    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
    X_sel = selector.fit_transform(X_shifted, y)
    selected_mask = selector.get_support()
    selected_names = [n for n, m in zip(feature_names, selected_mask) if m]
    scores = selector.scores_
    score_df = pd.DataFrame({
        "feature": feature_names,
        "mutual_info_score": scores
    }).sort_values("mutual_info_score", ascending=False)
    logger.info(f"SelectKBest: {X.shape[1]} → {X_sel.shape[1]} features (MI scores)")
    return X_sel, selected_names, score_df, selector


def rfe_feature_selection(X: np.ndarray, y: np.ndarray, feature_names: list, n_features: int = 80):
    """Recursive Feature Elimination using Logistic Regression."""
    estimator = LogisticRegression(max_iter=500, solver="liblinear", C=1.0, random_state=42)
    rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]), step=10)
    X_sel = rfe.fit_transform(X, y)
    selected_names = [n for n, m in zip(feature_names, rfe.support_) if m]
    rankings = pd.DataFrame({
        "feature": feature_names,
        "rfe_rank": rfe.ranking_
    }).sort_values("rfe_rank")
    logger.info(f"RFE: {X.shape[1]} → {X_sel.shape[1]} features selected")
    return X_sel, selected_names, rankings, rfe


def tree_based_feature_importance(X: np.ndarray, y: np.ndarray, feature_names: list, top_n: int = 50):
    """Use Random Forest to get feature importances."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    top_features = importance_df.head(top_n)["feature"].tolist()
    top_indices = [feature_names.index(f) for f in top_features]
    X_top = X[:, top_indices]
    logger.info(f"Tree importance: top {top_n} features selected")
    return X_top, top_features, importance_df, rf


def compute_shap_importance(model, X: np.ndarray, feature_names: list, max_samples: int = 500):
    """
    Compute SHAP feature importance values.
    Returns a DataFrame sorted by mean |SHAP value|.
    """
    try:
        import shap
        X_sample = X[:min(max_samples, len(X))]
        model_type = type(model).__name__.lower()

        if "xgb" in model_type or "forest" in model_type or "tree" in model_type:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                shap.sample(X_sample, 100)
            )

        shap_values = explainer.shap_values(X_sample)

        # For multi-class, use class 1
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        mean_shap = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feature_names[:len(mean_shap)],
            "mean_abs_shap": mean_shap
        }).sort_values("mean_abs_shap", ascending=False)
        logger.info(f"SHAP importance computed for {len(shap_df)} features")
        return shap_df, shap_values, explainer

    except ImportError:
        logger.warning("SHAP not installed. Skipping SHAP importance.")
        return pd.DataFrame({"feature": feature_names, "mean_abs_shap": 0}), None, None
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return pd.DataFrame({"feature": feature_names, "mean_abs_shap": 0}), None, None


def full_feature_selection_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    strategy: str = "tree_importance",
    top_n: int = 80
):
    """
    Run full feature selection pipeline.
    strategy: 'variance', 'correlation', 'mutual_info', 'rfe', 'tree_importance'
    Returns (X_selected, selected_feature_names, report_df)
    """
    logger.info(f"Feature selection strategy: {strategy}")

    if strategy == "variance":
        X_s, names, _ = remove_low_variance_features(X, feature_names)
        report = pd.DataFrame({"feature": names, "selected": True})

    elif strategy == "correlation":
        X_v, names_v, _ = remove_low_variance_features(X, feature_names)
        X_s, names, _ = remove_correlated_features(X_v, names_v)
        report = pd.DataFrame({"feature": names, "selected": True})

    elif strategy == "mutual_info":
        X_s, names, report, _ = select_k_best_features(X, y, feature_names, k=top_n)

    elif strategy == "rfe":
        X_s, names, report, _ = rfe_feature_selection(X, y, feature_names, n_features=top_n)

    else:  # tree_importance (default)
        X_s, names, report, _ = tree_based_feature_importance(X, y, feature_names, top_n=top_n)

    return X_s, names, report

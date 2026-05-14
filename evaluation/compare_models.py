"""
evaluation/compare_models.py
=============================
Cross-validation comparison across all models with K-Fold
and Stratified K-Fold. Generates a final leaderboard.
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from utils.preprocessing import ClinicalTrialPreprocessor, load_raw_data, DATASETS_DIR, SAVED_MODELS_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.visualization import VIZ_DIR


SCORING = {
    "accuracy": "accuracy",
    "f1_weighted": "f1_weighted",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
    "roc_auc": "roc_auc",
}


def run_cross_validation_comparison():
    logger.info("=" * 60)
    logger.info("Cross-Validation Model Comparison")
    logger.info("=" * 60)

    df, _ = load_raw_data()
    try:
        preprocessor = ClinicalTrialPreprocessor.load()
        X = preprocessor.transform(df)
        y = df["label"].astype(int).values
    except Exception:
        preprocessor = ClinicalTrialPreprocessor()
        X, y, _ = preprocessor.fit_transform(df, DATASETS_DIR / "clintox.csv")

    models_to_compare = {}

    model_files = {
        "XGBoost": "xgboost_model.joblib",
        "Random Forest": "random_forest_model.joblib",
        "Decision Tree": "decision_tree_model.joblib",
        "Naive Bayes": "naive_bayes_model.joblib",
        "Logistic Regression": "logistic_regression_model.joblib",
    }

    for name, fname in model_files.items():
        p = SAVED_MODELS_DIR / fname
        if p.exists():
            try:
                models_to_compare[name] = joblib.load(p)
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")

    if not models_to_compare:
        logger.error("No trained models found. Run training scripts first.")
        return None

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = []

    for name, model in models_to_compare.items():
        logger.info(f"Running 10-Fold Stratified CV for {name}...")
        try:
            cv = cross_validate(
                model, X, y, cv=skf,
                scoring=SCORING, n_jobs=-1,
                return_train_score=True
            )
            row = {
                "Model": name,
                "CV_Accuracy_Mean": round(cv["test_accuracy"].mean(), 4),
                "CV_Accuracy_Std": round(cv["test_accuracy"].std(), 4),
                "CV_F1_Mean": round(cv["test_f1_weighted"].mean(), 4),
                "CV_F1_Std": round(cv["test_f1_weighted"].std(), 4),
                "CV_ROC_AUC_Mean": round(cv["test_roc_auc"].mean(), 4),
                "CV_ROC_AUC_Std": round(cv["test_roc_auc"].std(), 4),
                "Train_F1_Mean": round(cv["train_f1_weighted"].mean(), 4),
                "Overfit_Gap": round(
                    cv["train_f1_weighted"].mean() - cv["test_f1_weighted"].mean(), 4
                ),
                "Fit_Time_s": round(cv["fit_time"].mean(), 3),
            }
            cv_results.append(row)
            logger.info(
                f"  {name:22s} | AUC={row['CV_ROC_AUC_Mean']:.4f}±{row['CV_ROC_AUC_Std']:.4f} "
                f"| F1={row['CV_F1_Mean']:.4f} | FitTime={row['Fit_Time_s']}s"
            )
        except Exception as e:
            logger.error(f"CV failed for {name}: {e}")

    if not cv_results:
        return None

    cv_df = pd.DataFrame(cv_results).sort_values("CV_ROC_AUC_Mean", ascending=False).reset_index(drop=True)

    logger.info("\n" + "=" * 80)
    logger.info("📋 CROSS-VALIDATION LEADERBOARD (10-Fold Stratified)")
    logger.info("=" * 80)
    logger.info("\n" + cv_df.to_string(index=False))

    eval_dir = BASE_DIR / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    cv_df.to_csv(eval_dir / "cv_comparison.csv", index=False)

    # Box plot of CV F1 scores
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    model_names = cv_df["Model"].tolist()
    cv_means = cv_df["CV_F1_Mean"].tolist()
    cv_stds = cv_df["CV_F1_Std"].tolist()

    colors = ["#6c8fff", "#4ade80", "#fbbf24", "#f472b6", "#a78bfa"]
    bars = ax.bar(model_names, cv_means, yerr=cv_stds, color=colors[:len(model_names)],
                  edgecolor="#ffffff20", capsize=5, error_kw={"color": "white", "lw": 2})

    for bar, mean in zip(bars, cv_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=9, color="#e0e4ff")

    ax.set_ylabel("F1 Score (Weighted)", color="#e0e4ff", fontsize=12)
    ax.set_title("10-Fold Stratified CV — F1 Score Comparison (Mean ± Std)",
                 color="#e0e4ff", fontsize=13, pad=12)
    ax.set_ylim(0, 1.1)
    ax.tick_params(colors="#9ba3c7")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "cv_f1_comparison.png", bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"✅ CV comparison chart saved.")

    # Overfitting analysis chart
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    x = np.arange(len(cv_df))
    width = 0.35
    ax.bar(x - width / 2, cv_df["Train_F1_Mean"], width, label="Train F1", color="#6c8fff", alpha=0.8)
    ax.bar(x + width / 2, cv_df["CV_F1_Mean"], width, label="Val F1", color="#4ade80", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cv_df["Model"], rotation=15, ha="right", color="#9ba3c7")
    ax.set_ylabel("F1 Score", color="#e0e4ff")
    ax.set_title("Train vs Validation F1 — Overfitting Analysis", color="#e0e4ff", fontsize=13)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "overfitting_analysis.png", bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)

    return cv_df


if __name__ == "__main__":
    run_cross_validation_comparison()

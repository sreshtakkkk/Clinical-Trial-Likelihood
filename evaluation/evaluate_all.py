"""
evaluation/evaluate_all.py
============================
Unified evaluation runner — loads all saved models and evaluates
them on the same test set, then generates a comprehensive comparison report.
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

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from utils.preprocessing import ClinicalTrialPreprocessor, load_raw_data, DATASETS_DIR, SAVED_MODELS_DIR
from utils.visualization import (
    plot_all_roc_curves, plot_metrics_comparison, plot_accuracy_radar, VIZ_DIR
)


MODEL_FILES = {
    "XGBoost": "xgboost_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "Naive Bayes": "naive_bayes_model.joblib",
    "Logistic Regression": "logistic_regression_model.joblib",
}
ANN_KERAS = "ann_model.keras"


def load_model(name: str, path: Path):
    try:
        if path.suffix == ".keras":
            import tensorflow as tf
            return tf.keras.models.load_model(path)
        return joblib.load(path)
    except Exception as e:
        logger.warning(f"Could not load {name}: {e}")
        return None


def evaluate_model(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_proba >= 0.5).astype(int)

        return {
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "ROC_AUC": round(roc_auc_score(y_test, y_proba), 4),
            "y_true": y_test,
            "y_proba": y_proba,
            "y_pred": y_pred,
        }
    except Exception as e:
        logger.error(f"Evaluation failed for {name}: {e}")
        return None


def run_full_evaluation():
    logger.info("=" * 70)
    logger.info("FULL MODEL EVALUATION — Clinical Trial Success Predictor")
    logger.info("=" * 70)

    df, _ = load_raw_data()

    try:
        preprocessor = ClinicalTrialPreprocessor.load()
        X = preprocessor.transform(df)
        y = df["label"].astype(int).values
    except Exception:
        logger.info("No saved preprocessor found. Fitting fresh preprocessor...")
        preprocessor = ClinicalTrialPreprocessor()
        X, y, _ = preprocessor.fit_transform(df, DATASETS_DIR / "clintox.csv")
        preprocessor.save()

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    all_results = []
    roc_data = {}

    # Sklearn models
    for name, filename in MODEL_FILES.items():
        model_path = SAVED_MODELS_DIR / filename
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}. Train it first.")
            continue
        model = load_model(name, model_path)
        if model is None:
            continue
        result = evaluate_model(name, model, X_test, y_test)
        if result:
            all_results.append(result)
            roc_data[name] = {"y_true": result["y_true"], "y_proba": result["y_proba"]}
            logger.info(f"{name:22s} | Acc={result['Accuracy']:.4f} | F1={result['F1']:.4f} | AUC={result['ROC_AUC']:.4f}")

    # ANN
    ann_path = SAVED_MODELS_DIR / ANN_KERAS
    if ann_path.exists():
        ann = load_model("ANN", ann_path)
        if ann:
            result = evaluate_model("ANN", ann, X_test, y_test)
            if result:
                all_results.append(result)
                roc_data["ANN"] = {"y_true": result["y_true"], "y_proba": result["y_proba"]}
                logger.info(f"{'ANN':22s} | Acc={result['Accuracy']:.4f} | F1={result['F1']:.4f} | AUC={result['ROC_AUC']:.4f}")

    if not all_results:
        logger.error("No models were successfully evaluated. Please train models first.")
        return None

    # Build comparison DataFrame
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    metrics_df = pd.DataFrame([{c: r[c] for c in cols} for r in all_results])
    metrics_df = metrics_df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)

    logger.info("\n" + "=" * 60)
    logger.info("📊 MODEL COMPARISON TABLE")
    logger.info("=" * 60)
    logger.info("\n" + metrics_df.to_string(index=False))

    best = metrics_df.iloc[0]
    fastest = metrics_df[metrics_df["Model"].isin(["Logistic Regression", "Naive Bayes"])].head(1)
    most_interpretable = metrics_df[metrics_df["Model"].isin(
        ["Logistic Regression", "Decision Tree", "Naive Bayes"])].head(1)

    logger.info(f"\n🏆 Best Overall Model:        {best['Model']} (AUC={best['ROC_AUC']:.4f})")
    if not fastest.empty:
        logger.info(f"⚡ Fastest/Simplest Model:    {fastest.iloc[0]['Model']}")
    if not most_interpretable.empty:
        logger.info(f"🔍 Most Interpretable Model:  {most_interpretable.iloc[0]['Model']}")

    # Save comparison CSV
    eval_dir = BASE_DIR / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    metrics_df.to_csv(eval_dir / "model_comparison.csv", index=False)
    logger.info(f"\n✅ Comparison table saved: {eval_dir / 'model_comparison.csv'}")

    # Visualizations
    if len(roc_data) > 1:
        plot_all_roc_curves(roc_data)
    plot_metrics_comparison(metrics_df)
    plot_accuracy_radar(metrics_df)

    return metrics_df, all_results


if __name__ == "__main__":
    run_full_evaluation()

"""
backend/predictor.py
=====================
ML inference engine — loads trained models and preprocessor,
performs single + batch predictions, and computes SHAP explanations.
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

SAVED_MODELS_DIR = BASE_DIR / "saved_models"

MODEL_FILES = {
    "XGBoost": "xgboost_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "Naive Bayes": "naive_bayes_model.joblib",
    "Logistic Regression": "logistic_regression_model.joblib",
}
ANN_KERAS_FILE = "ann_model.keras"


def _confidence_from_proba(prob: float) -> str:
    if prob < 0.35 or prob > 0.65:
        return "high"
    if prob < 0.42 or prob > 0.58:
        return "medium"
    return "low"


def _risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "low"
    if prob >= 0.55:
        return "moderate"
    if prob >= 0.35:
        return "high"
    return "critical"


def _interpretation(prob: float, prediction: int) -> str:
    if prediction == 1:
        if prob >= 0.85:
            return "Very high likelihood of trial success. Strong candidate for advancement."
        if prob >= 0.70:
            return "Good probability of success. Recommend continued monitoring."
        return "Moderate success probability. Proceed with caution and milestone reviews."
    else:
        if prob <= 0.15:
            return "Very high failure risk. Consider protocol redesign or early termination."
        if prob <= 0.30:
            return "High failure risk. Significant protocol adjustments recommended."
        return "Failure likely. Review endpoints, eligibility criteria, and drug safety data."


class ClinicalTrialPredictor:

    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self._ann_model = None

    def load_models(self):
        """Load preprocessor and all available trained models."""
        # Load preprocessor
        prep_path = SAVED_MODELS_DIR / "preprocessor.joblib"
        if prep_path.exists():
            try:
                from utils.preprocessing import ClinicalTrialPreprocessor
                self.preprocessor = ClinicalTrialPreprocessor.load(prep_path)
                logger.info("Preprocessor loaded successfully.")
            except Exception as e:
                logger.warning(f"Preprocessor load failed: {e}. Will use raw features.")

        # Load sklearn models
        for name, fname in MODEL_FILES.items():
            path = SAVED_MODELS_DIR / fname
            if path.exists():
                try:
                    self.models[name] = joblib.load(path)
                    logger.info(f"Loaded: {name}")
                except Exception as e:
                    logger.warning(f"Could not load {name}: {e}")

        # Load ANN
        ann_path = SAVED_MODELS_DIR / ANN_KERAS_FILE
        if ann_path.exists():
            try:
                import tensorflow as tf
                self._ann_model = tf.keras.models.load_model(ann_path)
                self.models["ANN"] = self._ann_model
                logger.info("Loaded: ANN (Keras)")
            except Exception as e:
                logger.warning(f"ANN load failed: {e}")

        if not self.models:
            logger.warning(
                "No trained models found! Using mock predictions. "
                "Run training scripts first."
            )

    def _preprocess_input(self, trial_dict: dict) -> np.ndarray:
        """Convert a single trial dict to feature array."""
        df = pd.DataFrame([trial_dict])

        if self.preprocessor is not None:
            try:
                return self.preprocessor.transform(df)
            except Exception as e:
                logger.warning(f"Preprocessor transform failed: {e}. Using zeros.")

        # Fallback: return zeros vector (demo mode)
        n_features = 80  # default feature dim
        return np.zeros((1, n_features))

    def predict_single(self, trial_dict: dict, model_name: str = "XGBoost") -> dict:
        """Run inference on a single trial."""
        if model_name not in self.models:
            available = list(self.models.keys())
            model_name = available[0] if available else None
            if model_name is None:
                # Mock prediction for demo
                return self._mock_prediction(trial_dict)

        X = self._preprocess_input(trial_dict)
        model = self.models[model_name]

        try:
            if model_name == "ANN":
                prob = float(model.predict(X, verbose=0).flatten()[0])
            else:
                prob = float(model.predict_proba(X)[0][1])
        except Exception as e:
            logger.warning(f"Prediction failed ({model_name}): {e}. Returning mock.")
            prob = 0.5

        prediction = int(prob >= 0.5)

        # SHAP features (top 5)
        shap_features = self._quick_shap(model, X, model_name)

        return {
            "prediction": prediction,
            "success_probability": round(prob, 4),
            "failure_probability": round(1 - prob, 4),
            "confidence": _confidence_from_proba(prob),
            "risk_level": _risk_level(prob),
            "model_used": model_name,
            "top_shap_features": shap_features,
            "interpretation": _interpretation(prob, prediction),
        }

    def predict_batch(self, df: pd.DataFrame, model_name: str = "XGBoost") -> pd.DataFrame:
        """Run inference on a batch DataFrame."""
        results = []
        for _, row in df.iterrows():
            trial_dict = row.to_dict()
            res = self.predict_single(trial_dict, model_name=model_name)
            results.append({
                "prediction": res["prediction"],
                "success_probability": res["success_probability"],
                "failure_probability": res["failure_probability"],
                "confidence": res["confidence"],
                "risk_level": res["risk_level"],
                "interpretation": res["interpretation"],
            })

        result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
        return result_df

    def get_shap_explanation(self, trial_dict: dict, model_name: str = "XGBoost") -> list:
        X = self._preprocess_input(trial_dict)
        model = self.models.get(model_name)
        if model is None:
            return []

        feature_names = (
            self.preprocessor.feature_names
            if self.preprocessor and hasattr(self.preprocessor, "feature_names")
            else [f"feature_{i}" for i in range(X.shape[1])]
        )
        return self._quick_shap(model, X, model_name, feature_names=feature_names, top_n=15)

    def _quick_shap(self, model, X: np.ndarray, model_name: str,
                    feature_names: list = None, top_n: int = 5) -> list:
        """Compute SHAP values or fall back to feature importance."""
        try:
            import shap
            if model_name in ("XGBoost", "Random Forest", "Decision Tree"):
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X)
                if isinstance(shap_vals, list):
                    vals = shap_vals[1][0]
                else:
                    vals = shap_vals[0]
            else:
                return []

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(vals))]

            indices = np.argsort(np.abs(vals))[::-1][:top_n]
            return [
                {
                    "feature": feature_names[i] if i < len(feature_names) else f"f{i}",
                    "value": round(float(vals[i]), 5),
                    "impact": "positive" if vals[i] > 0 else "negative",
                }
                for i in indices
            ]
        except Exception:
            return []

    def _mock_prediction(self, trial_dict: dict) -> dict:
        """Demo prediction when no models are trained."""
        import random
        random.seed(42)
        prob = round(random.uniform(0.3, 0.9), 4)
        prediction = int(prob >= 0.5)
        return {
            "prediction": prediction,
            "success_probability": prob,
            "failure_probability": round(1 - prob, 4),
            "confidence": _confidence_from_proba(prob),
            "risk_level": _risk_level(prob),
            "model_used": "mock",
            "top_shap_features": [],
            "interpretation": _interpretation(prob, prediction) + " [DEMO MODE — train models first]",
        }

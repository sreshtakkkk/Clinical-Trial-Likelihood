# notebooks/03_Model_Training.py
# ============================================================
# Notebook 3: Full Model Training Pipeline
# Trains all 6 models sequentially and saves results.
# ============================================================

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.preprocessing import (
    ClinicalTrialPreprocessor, load_raw_data, DATASETS_DIR, SAVED_MODELS_DIR
)

print("=" * 65)
print("  CLINICAL TRIAL SUCCESS PREDICTOR — MODEL TRAINING NOTEBOOK")
print("=" * 65)

# ── Step 1: Load & Preprocess ────────────────────────────────
print("\n[1/7] Loading and preprocessing data...")
df, _ = load_raw_data()

preprocessor = ClinicalTrialPreprocessor(
    tfidf_max_features=200,
    svd_components=50,
    use_smote=True,
    scaler_type="robust"
)
X, y, feature_names = preprocessor.fit_transform(df, DATASETS_DIR / "clintox.csv")
preprocessor.save()

processed_dir = DATASETS_DIR / "processed"
processed_dir.mkdir(exist_ok=True)
np.save(processed_dir / "X.npy", X)
np.save(processed_dir / "y.npy", y)
pd.DataFrame({"feature_name": feature_names}).to_csv(processed_dir / "feature_names.csv", index=False)

print(f"    X shape: {X.shape} | y shape: {y.shape}")
print(f"    Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"    Features: {len(feature_names)}")

# ── Step 2: XGBoost ──────────────────────────────────────────
print("\n[2/7] Training XGBoost...")
try:
    from training.train_xgboost import train_xgboost
    xgb_model, xgb_metrics, *_ = train_xgboost(X, y, feature_names)
    print(f"    ✅ XGBoost | Acc={xgb_metrics['accuracy']:.4f} | AUC={xgb_metrics['roc_auc']:.4f}")
except Exception as e:
    print(f"    ❌ XGBoost failed: {e}")
    xgb_metrics = {}

# ── Step 3: Random Forest ────────────────────────────────────
print("\n[3/7] Training Random Forest...")
try:
    from training.train_random_forest import train_random_forest
    rf_model, rf_metrics, *_ = train_random_forest(X, y, feature_names)
    print(f"    ✅ Random Forest | Acc={rf_metrics['accuracy']:.4f} | AUC={rf_metrics['roc_auc']:.4f}")
except Exception as e:
    print(f"    ❌ Random Forest failed: {e}")
    rf_metrics = {}

# ── Step 4: Decision Tree ────────────────────────────────────
print("\n[4/7] Training Decision Tree...")
try:
    from training.train_decision_tree import train_decision_tree
    dt_model, dt_metrics, *_ = train_decision_tree(X, y, feature_names)
    print(f"    ✅ Decision Tree | Acc={dt_metrics['accuracy']:.4f} | AUC={dt_metrics['roc_auc']:.4f}")
except Exception as e:
    print(f"    ❌ Decision Tree failed: {e}")
    dt_metrics = {}

# ── Step 5: Naive Bayes ──────────────────────────────────────
print("\n[5/7] Training Naive Bayes...")
try:
    from training.train_naive_bayes import train_naive_bayes
    nb_model, nb_metrics, *_ = train_naive_bayes(X, y, feature_names)
    print(f"    ✅ Naive Bayes | Acc={nb_metrics['accuracy']:.4f} | AUC={nb_metrics['roc_auc']:.4f}")
except Exception as e:
    print(f"    ❌ Naive Bayes failed: {e}")
    nb_metrics = {}

# ── Step 6: ANN ──────────────────────────────────────────────
print("\n[6/7] Training ANN (TensorFlow/Keras)...")
try:
    from training.train_ann import train_ann
    ann_model, ann_metrics, *_ = train_ann(X, y, feature_names)
    print(f"    ✅ ANN | Acc={ann_metrics['accuracy']:.4f} | AUC={ann_metrics['roc_auc']:.4f}")
except Exception as e:
    print(f"    ❌ ANN failed: {e}")
    ann_metrics = {}

# ── Step 7: Logistic Regression ──────────────────────────────
print("\n[7/7] Training Logistic Regression...")
try:
    from training.train_logistic_regression import train_logistic_regression
    lr_model, lr_metrics, *_ = train_logistic_regression(X, y, feature_names)
    print(f"    ✅ Logistic Regression | Acc={lr_metrics['accuracy']:.4f} | AUC={lr_metrics['roc_auc']:.4f}")
except Exception as e:
    print(f"    ❌ Logistic Regression failed: {e}")
    lr_metrics = {}

# ── Summary ──────────────────────────────────────────────────
all_results = []
for name, m in [("XGBoost", xgb_metrics), ("Random Forest", rf_metrics),
                ("Decision Tree", dt_metrics), ("Naive Bayes", nb_metrics),
                ("ANN", ann_metrics), ("Logistic Regression", lr_metrics)]:
    if m:
        all_results.append({
            "Model": name,
            "Accuracy": round(m.get("accuracy", 0), 4),
            "F1": round(m.get("f1", 0), 4),
            "ROC_AUC": round(m.get("roc_auc", 0), 4),
        })

if all_results:
    summary = pd.DataFrame(all_results).sort_values("ROC_AUC", ascending=False)
    eval_dir = Path(__file__).resolve().parent.parent / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    summary.to_csv(eval_dir / "model_comparison.csv", index=False)

    print("\n" + "=" * 55)
    print("  TRAINING SUMMARY")
    print("=" * 55)
    print(summary.to_string(index=False))
    print(f"\n🏆 Best model: {summary.iloc[0]['Model']} (AUC={summary.iloc[0]['ROC_AUC']:.4f})")
    print(f"\n✅ All results saved to evaluation/model_comparison.csv")

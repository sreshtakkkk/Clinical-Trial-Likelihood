"""
app.py
=======
Unified CLI entrypoint for the Clinical Trial Success Predictor.

Usage:
  python app.py --mode preprocess
  python app.py --mode train
  python app.py --mode evaluate
  python app.py --mode api
  python app.py --mode all
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def run_preprocess():
    logger.info("🔧 Running data preprocessing pipeline...")
    from utils.preprocessing import (
        ClinicalTrialPreprocessor, load_raw_data, DATASETS_DIR
    )
    import numpy as np
    import pandas as pd

    df, _ = load_raw_data()
    preprocessor = ClinicalTrialPreprocessor(tfidf_max_features=200, svd_components=50)
    X, y, feature_names = preprocessor.fit_transform(df, DATASETS_DIR / "clintox.csv")
    preprocessor.save()

    processed_dir = DATASETS_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)
    np.save(processed_dir / "X.npy", X)
    np.save(processed_dir / "y.npy", y)
    pd.DataFrame({"feature_name": feature_names}).to_csv(
        processed_dir / "feature_names.csv", index=False
    )
    logger.info(f"✅ Preprocessing done: X={X.shape}, y={y.shape}")
    return X, y, feature_names


def run_eda():
    logger.info("📊 Running EDA visualization pipeline...")
    from utils.preprocessing import load_raw_data, DATASETS_DIR
    from utils.visualization import generate_full_eda_report
    import numpy as np

    df, _ = load_raw_data()
    X_path = DATASETS_DIR / "processed" / "X.npy"
    feat_path = DATASETS_DIR / "processed" / "feature_names.csv"

    X, feature_names = None, None
    if X_path.exists() and feat_path.exists():
        import pandas as pd
        X = np.load(X_path)
        feature_names = pd.read_csv(feat_path)["feature_name"].tolist()

    plots = generate_full_eda_report(df, X=X, feature_names=feature_names)
    logger.info(f"✅ EDA complete. Generated {sum(1 for v in plots.values() if v)} plots.")


def run_training():
    logger.info("🤖 Starting model training pipeline...")
    import numpy as np
    import pandas as pd
    from utils.preprocessing import ClinicalTrialPreprocessor, load_raw_data, DATASETS_DIR

    # Load or preprocess
    X_path = DATASETS_DIR / "processed" / "X.npy"
    y_path = DATASETS_DIR / "processed" / "y.npy"
    feat_path = DATASETS_DIR / "processed" / "feature_names.csv"

    if X_path.exists() and y_path.exists():
        X = np.load(X_path)
        y = np.load(y_path)
        feature_names = pd.read_csv(feat_path)["feature_name"].tolist()
        logger.info(f"Loaded preprocessed data: X={X.shape}, y={y.shape}")
    else:
        X, y, feature_names = run_preprocess()

    all_metrics = []

    # XGBoost
    try:
        from training.train_xgboost import train_xgboost
        _, metrics, _, _, _ = train_xgboost(X, y, feature_names)
        all_metrics.append(metrics)
        logger.info("✅ XGBoost trained")
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")

    # Random Forest
    try:
        from training.train_random_forest import train_random_forest
        _, metrics, _, _, _ = train_random_forest(X, y, feature_names)
        all_metrics.append(metrics)
        logger.info("✅ Random Forest trained")
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")

    # Decision Tree
    try:
        from training.train_decision_tree import train_decision_tree
        _, metrics, _, _, _ = train_decision_tree(X, y, feature_names)
        all_metrics.append(metrics)
        logger.info("✅ Decision Tree trained")
    except Exception as e:
        logger.error(f"Decision Tree training failed: {e}")

    # Naive Bayes
    try:
        from training.train_naive_bayes import train_naive_bayes
        _, metrics, _, _, _ = train_naive_bayes(X, y, feature_names)
        all_metrics.append(metrics)
        logger.info("✅ Naive Bayes trained")
    except Exception as e:
        logger.error(f"Naive Bayes training failed: {e}")

    # ANN
    try:
        from training.train_ann import train_ann
        _, metrics, _, _, _ = train_ann(X, y, feature_names)
        all_metrics.append(metrics)
        logger.info("✅ ANN trained")
    except Exception as e:
        logger.error(f"ANN training failed: {e}")

    # Logistic Regression
    try:
        from training.train_logistic_regression import train_logistic_regression
        _, metrics, _, _, _ = train_logistic_regression(X, y, feature_names)
        all_metrics.append(metrics)
        logger.info("✅ Logistic Regression trained")
    except Exception as e:
        logger.error(f"Logistic Regression training failed: {e}")

    logger.info(f"\n🎯 Training complete! {len(all_metrics)}/6 models trained successfully.")


def run_evaluation():
    logger.info("📈 Running evaluation and model comparison...")
    from evaluation.evaluate_all import run_full_evaluation
    from evaluation.compare_models import run_cross_validation_comparison

    metrics_df, _ = run_full_evaluation()
    cv_df = run_cross_validation_comparison()

    if metrics_df is not None:
        logger.info("\n🏆 Final Leaderboard:")
        logger.info(metrics_df[["Model", "Accuracy", "F1", "ROC_AUC"]].to_string(index=False))


def run_api():
    logger.info("🚀 Starting FastAPI server...")
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


def main():
    parser = argparse.ArgumentParser(
        description="Clinical Trial Success Predictor — Unified Entrypoint"
    )
    parser.add_argument(
        "--mode",
        choices=["preprocess", "eda", "train", "evaluate", "api", "all"],
        default="all",
        help="Pipeline mode to run"
    )
    args = parser.parse_args()

    if args.mode == "preprocess":
        run_preprocess()
    elif args.mode == "eda":
        run_eda()
    elif args.mode == "train":
        run_training()
    elif args.mode == "evaluate":
        run_evaluation()
    elif args.mode == "api":
        run_api()
    elif args.mode == "all":
        run_preprocess()
        run_eda()
        run_training()
        run_evaluation()
        logger.info("\n✅ Full pipeline complete!")
        logger.info("Run 'python app.py --mode api' to start the API server.")


if __name__ == "__main__":
    main()

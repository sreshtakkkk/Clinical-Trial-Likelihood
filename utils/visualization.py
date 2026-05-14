"""
utils/visualization.py
=======================
Comprehensive visualization utilities for EDA, model evaluation,
and result comparison for Clinical Trial Success Prediction.

Generates:
- Correlation heatmaps
- Class distribution plots
- Phase analysis charts
- ROC curves (all models)
- Confusion matrices
- Feature importance plots
- Learning curves
- Word clouds
- SHAP summary plots
- Accuracy comparison charts
"""

import os
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import learning_curve

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─── Style Setup ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3d4166",
    "axes.labelcolor": "#e0e4ff",
    "text.color": "#e0e4ff",
    "xtick.color": "#9ba3c7",
    "ytick.color": "#9ba3c7",
    "grid.color": "#2d3055",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
    "figure.dpi": 120,
})

PALETTE = {
    "primary": "#6c8fff",
    "success": "#4ade80",
    "danger": "#f87171",
    "warning": "#fbbf24",
    "info": "#38bdf8",
    "purple": "#a78bfa",
    "pink": "#f472b6",
    "cyan": "#22d3ee",
}

MODEL_COLORS = {
    "XGBoost": PALETTE["primary"],
    "Random Forest": PALETTE["success"],
    "Decision Tree": PALETTE["warning"],
    "Naive Bayes": PALETTE["pink"],
    "ANN": PALETTE["cyan"],
    "Logistic Regression": PALETTE["purple"],
}

VIZ_DIR = Path(__file__).resolve().parent.parent / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)


def _save(fig, filename: str):
    path = VIZ_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ─── EDA Plots ───────────────────────────────────────────────────────────────

def plot_class_distribution(y: np.ndarray, filename: str = "class_distribution.png"):
    labels = ["Failure (0)", "Success (1)"]
    counts = [np.sum(y == 0), np.sum(y == 1)]
    colors = [PALETTE["danger"], PALETTE["success"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Clinical Trial Label Distribution", fontsize=16, fontweight="bold", y=1.02)

    # Bar chart
    bars = axes[0].bar(labels, counts, color=colors, width=0.5, edgecolor="#ffffff20", linewidth=1)
    axes[0].set_title("Count per Class", pad=12)
    axes[0].set_ylabel("Count")
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{count:,}", ha="center", va="bottom", fontweight="bold")

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        counts, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "#ffffff30", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    axes[1].set_title("Class Balance", pad=12)

    plt.tight_layout()
    return _save(fig, filename)


def plot_phase_distribution(df: pd.DataFrame, filename: str = "phase_distribution.png"):
    if "phase" not in df.columns:
        return None
    phase_counts = df["phase"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(phase_counts.index, phase_counts.values,
                   color=list(PALETTE.values())[:len(phase_counts)],
                   edgecolor="#ffffff20", linewidth=1)
    ax.set_title("Clinical Trial Phase Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Trials")
    for bar, val in zip(bars, phase_counts.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    return _save(fig, filename)


def plot_correlation_heatmap(X: np.ndarray, feature_names: list,
                              top_n: int = 25, filename: str = "correlation_heatmap.png"):
    n = min(top_n, X.shape[1])
    X_sub = X[:, :n]
    names_sub = feature_names[:n]
    corr = np.corrcoef(X_sub.T)

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=0.5, ax=ax,
                xticklabels=names_sub, yticklabels=names_sub,
                annot=False, cbar_kws={"shrink": 0.8})
    ax.set_title(f"Feature Correlation Heatmap (top {n} features)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    return _save(fig, filename)


def plot_word_cloud(text_series: pd.Series, title: str = "Criteria Keywords",
                    filename: str = "wordcloud.png"):
    try:
        from wordcloud import WordCloud
        combined = " ".join(text_series.dropna().astype(str))
        if not combined.strip():
            return None
        wc = WordCloud(
            width=1200, height=600, background_color="#0f1117",
            colormap="cool", max_words=150,
            prefer_horizontal=0.8, min_font_size=10
        ).generate(combined)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()
        return _save(fig, filename)
    except ImportError:
        logger.warning("wordcloud not installed. Skipping word cloud.")
        return None


def plot_top_drugs(df: pd.DataFrame, top_n: int = 20, filename: str = "top_drugs.png"):
    if "drugs" not in df.columns:
        return None
    all_drugs = []
    for row in df["drugs"].dropna():
        all_drugs.extend([d.strip() for d in str(row).split("|") if d.strip()])
    drug_counts = pd.Series(all_drugs).value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(drug_counts.index[::-1], drug_counts.values[::-1],
            color=PALETTE["info"], edgecolor="#ffffff20")
    ax.set_title(f"Top {top_n} Most Common Drugs in Clinical Trials", fontsize=14, fontweight="bold")
    ax.set_xlabel("Frequency")
    plt.tight_layout()
    return _save(fig, filename)


def plot_top_diseases(df: pd.DataFrame, top_n: int = 20, filename: str = "top_diseases.png"):
    if "diseases" not in df.columns:
        return None
    all_diseases = []
    for row in df["diseases"].dropna():
        all_diseases.extend([d.strip() for d in str(row).split("|") if d.strip()])
    dis_counts = pd.Series(all_diseases).value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, top_n))
    ax.barh(dis_counts.index[::-1], dis_counts.values[::-1],
            color=colors[::-1], edgecolor="#ffffff20")
    ax.set_title(f"Top {top_n} Disease Categories", fontsize=14, fontweight="bold")
    ax.set_xlabel("Frequency")
    plt.tight_layout()
    return _save(fig, filename)


# ─── Model Evaluation Plots ──────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str,
                           filename: str = None):
    cm = confusion_matrix(y_true, y_pred)
    filename = filename or f"cm_{model_name.lower().replace(' ', '_')}.png"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Failure", "Success"],
        yticklabels=["Failure", "Success"],
        linewidths=1, linecolor="#ffffff20",
        cbar_kws={"shrink": 0.8}, ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, filename)


def plot_roc_curve(y_true, y_proba, model_name: str, filename: str = None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    filename = filename or f"roc_{model_name.lower().replace(' ', '_')}.png"
    color = MODEL_COLORS.get(model_name, PALETTE["primary"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#6b7280", lw=1.5, linestyle="--", label="Random Chance")
    ax.fill_between(fpr, tpr, alpha=0.15, color=color)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, filename)


def plot_all_roc_curves(results: dict, filename: str = "all_roc_curves.png"):
    """
    results: {model_name: {"y_true": ..., "y_proba": ...}}
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], color="#6b7280", lw=1.5, linestyle="--", label="Random Chance")

    for model_name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
        roc_auc = auc(fpr, tpr)
        color = MODEL_COLORS.get(model_name, PALETTE["primary"])
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{model_name} (AUC={roc_auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.05, color=color)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    return _save(fig, filename)


def plot_metrics_comparison(metrics_df: pd.DataFrame, filename: str = "metrics_comparison.png"):
    """metrics_df: columns = [Model, Accuracy, Precision, Recall, F1, ROC_AUC]"""
    metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"] if c in metrics_df.columns]
    x = np.arange(len(metrics_df))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 7))
    colors_list = list(PALETTE.values())

    for i, metric in enumerate(metric_cols):
        offset = (i - len(metric_cols) / 2) * width + width / 2
        bars = ax.bar(x + offset, metrics_df[metric], width,
                      label=metric, color=colors_list[i % len(colors_list)],
                      edgecolor="#ffffff20")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["Model"], rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Performance Comparison — All Metrics", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.3, loc="upper left")
    ax.grid(axis="y", alpha=0.4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    plt.tight_layout()
    return _save(fig, filename)


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str,
                             top_n: int = 20, filename: str = None):
    top = importance_df.head(top_n)
    filename = filename or f"feat_imp_{model_name.lower().replace(' ', '_')}.png"
    color = MODEL_COLORS.get(model_name, PALETTE["primary"])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"][::-1], top.iloc[:, 1].values[::-1],
            color=color, edgecolor="#ffffff20", alpha=0.85)
    ax.set_title(f"Feature Importance — {model_name} (Top {top_n})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return _save(fig, filename)


def plot_learning_curve(model, X: np.ndarray, y: np.ndarray, model_name: str,
                         filename: str = None):
    filename = filename or f"learning_curve_{model_name.lower().replace(' ', '_')}.png"
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="f1_weighted"
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, "o-", color=PALETTE["primary"],
            label="Training Score", lw=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color=PALETTE["primary"])
    ax.plot(train_sizes, val_mean, "s-", color=PALETTE["success"],
            label="Validation Score", lw=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color=PALETTE["success"])
    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("F1 Score (Weighted)", fontsize=12)
    ax.set_title(f"Learning Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.3)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    return _save(fig, filename)


def plot_accuracy_radar(metrics_df: pd.DataFrame, filename: str = "radar_chart.png"):
    """Spider/Radar chart for model comparison."""
    try:
        from matplotlib.patches import FancyArrowPatch
        import matplotlib.patheffects as pe

        categories = [c for c in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
                      if c in metrics_df.columns]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
        ax.set_facecolor("#1a1d27")

        for _, row in metrics_df.iterrows():
            values = row[categories].tolist() + row[categories[:1]].tolist()
            color = MODEL_COLORS.get(row["Model"], PALETTE["primary"])
            ax.plot(angles, values, "o-", linewidth=2.5, color=color, label=row["Model"])
            ax.fill(angles, values, alpha=0.08, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title("Model Comparison — Radar Chart", fontsize=14, fontweight="bold", pad=25)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9, framealpha=0.3)
        ax.grid(color="#3d4166", linewidth=0.8)
        plt.tight_layout()
        return _save(fig, filename)
    except Exception as e:
        logger.warning(f"Radar chart failed: {e}")
        return None


def generate_full_eda_report(df: pd.DataFrame, X: np.ndarray = None,
                               feature_names: list = None):
    """Run all EDA plots from the raw DataFrame."""
    logger.info("Generating EDA visualizations...")
    plots = {}

    if "label" in df.columns:
        plots["class_dist"] = plot_class_distribution(df["label"].values)

    plots["phase"] = plot_phase_distribution(df)

    if "criteria" in df.columns:
        from utils.nlp_utils import clean_text
        plots["wordcloud"] = plot_word_cloud(df["criteria"].apply(clean_text),
                                              title="Clinical Trial Criteria — Key Terms")

    plots["top_drugs"] = plot_top_drugs(df)
    plots["top_diseases"] = plot_top_diseases(df)

    if X is not None and feature_names is not None:
        plots["heatmap"] = plot_correlation_heatmap(X, feature_names)

    logger.info(f"EDA report complete. Generated {len([v for v in plots.values() if v])} plots.")
    return plots

# notebooks/01_EDA.py
# ============================================================
# Notebook 1: Exploratory Data Analysis
# Run as a Jupyter notebook or plain Python script.
# ============================================================

# ── Cell 1: Imports & Setup ──────────────────────────────────
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.preprocessing import load_raw_data, DATASETS_DIR
from utils.visualization import (
    plot_class_distribution, plot_phase_distribution,
    plot_top_drugs, plot_top_diseases, plot_word_cloud,
    VIZ_DIR
)

print("📦 Imports ready.")
print(f"Visualizations will be saved to: {VIZ_DIR}")

# ── Cell 2: Load Data ────────────────────────────────────────
df, df_clintox = load_raw_data()

print(f"\n📋 Main Dataset Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"\n🧪 ClinTox Shape: {df_clintox.shape}")

# ── Cell 3: Basic Statistics ─────────────────────────────────
print("\n═══ DATASET STATISTICS ═══")
print(df.describe(include="all").T[["count", "unique", "top", "freq"]].head(20))

print("\n═══ MISSING VALUES ═══")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"count": missing, "pct": missing_pct})
print(missing_df[missing_df["count"] > 0])

# ── Cell 4: Class Distribution ───────────────────────────────
if "label" in df.columns:
    print(f"\n═══ CLASS DISTRIBUTION ═══")
    vc = df["label"].value_counts()
    print(vc)
    print(f"Class imbalance ratio: {vc.max() / vc.min():.2f}:1")
    plot_class_distribution(df["label"].values)
    print("✅ Saved: class_distribution.png")

# ── Cell 5: Phase Analysis ───────────────────────────────────
if "phase" in df.columns:
    print(f"\n═══ TRIAL PHASE DISTRIBUTION ═══")
    print(df["phase"].value_counts())
    plot_phase_distribution(df)
    print("✅ Saved: phase_distribution.png")

# ── Cell 6: Success Rate by Phase ───────────────────────────
if "label" in df.columns and "phase" in df.columns:
    phase_success = df.groupby("phase")["label"].agg(["mean", "count"]).reset_index()
    phase_success.columns = ["Phase", "SuccessRate", "Count"]
    phase_success = phase_success.sort_values("SuccessRate", ascending=False)
    print("\n═══ SUCCESS RATE BY PHASE ═══")
    print(phase_success.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    bars = ax.bar(phase_success["Phase"], phase_success["SuccessRate"],
                  color="#6c8fff", edgecolor="#ffffff20")
    ax.set_xlabel("Trial Phase", color="#e0e4ff")
    ax.set_ylabel("Success Rate", color="#e0e4ff")
    ax.set_title("Success Rate by Clinical Trial Phase", color="#e0e4ff", fontsize=13)
    ax.tick_params(colors="#9ba3c7")
    plt.xticks(rotation=30, ha="right")
    for bar, val in zip(bars, phase_success["SuccessRate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", fontsize=8, color="#e0e4ff")
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "success_rate_by_phase.png", bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print("✅ Saved: success_rate_by_phase.png")

# ── Cell 7: Drug & Disease Frequency ─────────────────────────
plot_top_drugs(df, top_n=20)
print("✅ Saved: top_drugs.png")

plot_top_diseases(df, top_n=20)
print("✅ Saved: top_diseases.png")

# ── Cell 8: Criteria Text Analysis ──────────────────────────
if "criteria" in df.columns:
    print(f"\n═══ CRITERIA TEXT STATISTICS ═══")
    word_counts = df["criteria"].dropna().apply(lambda x: len(str(x).split()))
    print(f"  Mean words:   {word_counts.mean():.0f}")
    print(f"  Median words: {word_counts.median():.0f}")
    print(f"  Max words:    {word_counts.max():.0f}")

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.hist(word_counts, bins=50, color="#6c8fff", edgecolor="#ffffff10")
    ax.set_xlabel("Word Count in Criteria", color="#e0e4ff")
    ax.set_ylabel("Frequency", color="#e0e4ff")
    ax.set_title("Criteria Text Length Distribution", color="#e0e4ff")
    ax.tick_params(colors="#9ba3c7")
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "criteria_length_dist.png", bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print("✅ Saved: criteria_length_dist.png")

    from utils.nlp_utils import clean_text
    plot_word_cloud(df["criteria"].apply(clean_text))
    print("✅ Saved: wordcloud.png")

# ── Cell 9: ClinTox Analysis ─────────────────────────────────
if not df_clintox.empty:
    print(f"\n═══ CLINTOX DATASET ═══")
    df_clintox.columns = df_clintox.columns.str.lower()
    print(df_clintox.describe())
    print(f"\nFDA Approved: {df_clintox.get('fda_approved', pd.Series()).value_counts().to_dict()}")
    print(f"CT Toxic:     {df_clintox.get('ct_tox', pd.Series()).value_counts().to_dict()}")

print("\n\n✅ EDA Complete! All visualizations saved to:", VIZ_DIR)

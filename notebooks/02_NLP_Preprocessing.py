# notebooks/02_NLP_Preprocessing.py
# ============================================================
# Notebook 2: NLP Preprocessing Deep-Dive
# Covers: text cleaning, TF-IDF, SVD, keyword extraction,
#         bigram analysis, and criteria feature engineering.
# ============================================================

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from collections import Counter

from utils.preprocessing import load_raw_data, DATASETS_DIR
from utils.nlp_utils import clean_text, extract_clinical_keywords
from utils.visualization import VIZ_DIR

print("📦 NLP Preprocessing Notebook")

# ── Load ─────────────────────────────────────────────────────
df, _ = load_raw_data()

if "criteria" not in df.columns:
    print("⚠️  No 'criteria' column found. Skipping NLP notebook.")
    exit()

# ── Text Cleaning ─────────────────────────────────────────────
print("\n═══ TEXT CLEANING ═══")
sample_raw = str(df["criteria"].dropna().iloc[0])[:300]
print(f"Raw sample (first 300 chars):\n{sample_raw}\n")

df["criteria_clean"] = df["criteria"].apply(clean_text)
sample_clean = df["criteria_clean"].iloc[0][:300]
print(f"Cleaned:\n{sample_clean}")

# Token statistics
all_tokens = " ".join(df["criteria_clean"].dropna()).split()
print(f"\nTotal tokens:    {len(all_tokens):,}")
print(f"Unique tokens:   {len(set(all_tokens)):,}")
print(f"Avg tokens/doc:  {np.mean([len(t.split()) for t in df['criteria_clean'].dropna()]):.1f}")

# Most common terms
top_terms = Counter(all_tokens).most_common(20)
print(f"\nTop 20 clinical terms:")
for term, count in top_terms:
    print(f"  {term:20s} {count:,}")

# ── Bigram Analysis ──────────────────────────────────────────
print("\n═══ BIGRAM ANALYSIS ═══")
bigram_vec = CountVectorizer(ngram_range=(2, 2), max_features=20)
bigram_mat = bigram_vec.fit_transform(df["criteria_clean"].fillna(""))
bigram_counts = bigram_mat.sum(axis=0).A1
bigram_df = pd.DataFrame({
    "bigram": bigram_vec.get_feature_names_out(),
    "count": bigram_counts
}).sort_values("count", ascending=False)
print(bigram_df.head(15).to_string(index=False))

# Plot top bigrams
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f1117")
ax.set_facecolor("#1a1d27")
top_bg = bigram_df.head(15)
ax.barh(top_bg["bigram"][::-1], top_bg["count"][::-1], color="#a78bfa", edgecolor="#ffffff20")
ax.set_title("Top 15 Bigrams in Criteria Text", color="#e0e4ff", fontsize=13)
ax.set_xlabel("Count", color="#e0e4ff")
ax.tick_params(colors="#9ba3c7", labelsize=9)
plt.tight_layout()
fig.savefig(VIZ_DIR / "top_bigrams.png", bbox_inches="tight", facecolor="#0f1117")
plt.close(fig)
print("✅ Saved: top_bigrams.png")

# ── TF-IDF ───────────────────────────────────────────────────
print("\n═══ TF-IDF VECTORIZATION ═══")
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
tfidf_matrix = tfidf.fit_transform(df["criteria_clean"].fillna(""))
print(f"TF-IDF matrix: {tfidf_matrix.shape}")

feature_names = tfidf.get_feature_names_out()
mean_tfidf = tfidf_matrix.mean(axis=0).A1
top_tfidf_df = pd.DataFrame({"feature": feature_names, "mean_tfidf": mean_tfidf})
top_tfidf_df = top_tfidf_df.sort_values("mean_tfidf", ascending=False).head(20)
print("\nTop 20 TF-IDF features:")
print(top_tfidf_df.to_string(index=False))

# Plot top TF-IDF
fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f1117")
ax.set_facecolor("#1a1d27")
ax.barh(top_tfidf_df["feature"][::-1], top_tfidf_df["mean_tfidf"][::-1],
        color="#38bdf8", edgecolor="#ffffff20")
ax.set_title("Top 20 TF-IDF Features (Mean Score)", color="#e0e4ff", fontsize=13)
ax.set_xlabel("Mean TF-IDF", color="#e0e4ff")
ax.tick_params(colors="#9ba3c7", labelsize=9)
plt.tight_layout()
fig.savefig(VIZ_DIR / "top_tfidf_features.png", bbox_inches="tight", facecolor="#0f1117")
plt.close(fig)
print("✅ Saved: top_tfidf_features.png")

# ── SVD / LSA ────────────────────────────────────────────────
print("\n═══ TRUNCATED SVD (LSA) ═══")
svd = TruncatedSVD(n_components=50, random_state=42)
X_lsa = svd.fit_transform(tfidf_matrix)
explained = svd.explained_variance_ratio_.cumsum()
print(f"SVD output shape: {X_lsa.shape}")
print(f"Explained variance @ 10 components: {explained[9]:.3f}")
print(f"Explained variance @ 50 components: {explained[49]:.3f}")

fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0f1117")
ax.set_facecolor("#1a1d27")
ax.plot(range(1, 51), explained * 100, color="#6c8fff", marker="o", markersize=3)
ax.fill_between(range(1, 51), explained * 100, alpha=0.15, color="#6c8fff")
ax.set_xlabel("Number of SVD Components", color="#e0e4ff")
ax.set_ylabel("Cumulative Explained Variance (%)", color="#e0e4ff")
ax.set_title("TF-IDF + SVD: Explained Variance", color="#e0e4ff", fontsize=13)
ax.axvline(50, color="#f87171", linestyle="--", lw=1, label="Selected (50)")
ax.tick_params(colors="#9ba3c7")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(VIZ_DIR / "svd_explained_variance.png", bbox_inches="tight", facecolor="#0f1117")
plt.close(fig)
print("✅ Saved: svd_explained_variance.png")

# ── Clinical Keyword Features ─────────────────────────────────
print("\n═══ CLINICAL KEYWORD EXTRACTION ═══")
kw_features = df["criteria"].apply(extract_clinical_keywords)
kw_df = pd.DataFrame(kw_features.tolist())
print(f"Keyword features extracted: {kw_df.shape[1]} features")
print(kw_df.describe().T[["mean", "std", "max"]].round(3))

if "label" in df.columns:
    df_joined = pd.concat([kw_df, df["label"]], axis=1)
    print("\nMean keyword features by class:")
    print(df_joined.groupby("label").mean().T.round(3))

print("\n✅ NLP Preprocessing notebook complete!")

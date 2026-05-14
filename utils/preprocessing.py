"""
utils/preprocessing.py
======================
Core Data Preprocessing Pipeline for Clinical Trial Success Prediction.

Handles:
- Missing value imputation (smart strategy per column type)
- Duplicate removal
- Categorical encoding (Label + One-Hot)
- NLP text cleaning & TF-IDF vectorization for 'criteria'
- SMILES string feature extraction (basic descriptors)
- Class imbalance handling with SMOTE
- Feature scaling (StandardScaler / RobustScaler)
- ClinTox dataset merging for toxicity enrichment
- Outlier detection & capping (IQR method)
- Derived feature generation (risk scores, trial complexity)
"""

import os
import re
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
SAVED_MODELS_DIR.mkdir(exist_ok=True)

# ─── NLP Utilities ───────────────────────────────────────────────────────────
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# Healthcare-domain stop words to KEEP (informative)
_CLINICAL_KEEP = {
    "not", "no", "without", "exclude", "include", "severe",
    "moderate", "mild", "chronic", "acute", "prior", "current"
}
_stop_words -= _CLINICAL_KEEP


def clean_text(text: str) -> str:
    """
    Clean and normalize clinical trial criteria text.
    Steps: lowercase → remove special chars → tokenize → remove stops → lemmatize
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)           # keep only alpha + space
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _stop_words and len(t) > 2
    ]
    return " ".join(tokens)


# ─── SMILES Feature Extraction ────────────────────────────────────────────────

def extract_smiles_features(smiles_series: pd.Series) -> pd.DataFrame:
    """
    Extract basic molecular descriptors from SMILES strings.
    Falls back to length/character-count features if RDKit is unavailable.
    Returns a DataFrame of numeric molecular features.
    """
    features = []

    for smi in smiles_series:
        feat = {}
        if not isinstance(smi, str) or not smi.strip():
            features.append(feat)
            continue

        # Basic string-level descriptors (always available)
        feat["smi_length"] = len(smi)
        feat["smi_ring_count"] = smi.count("1") + smi.count("2") + smi.count("3")
        feat["smi_branch_count"] = smi.count("(")
        feat["smi_double_bonds"] = smi.count("=")
        feat["smi_triple_bonds"] = smi.count("#")
        feat["smi_aromatic_atoms"] = sum(1 for c in smi if c.islower())
        feat["smi_nitrogen"] = smi.upper().count("N")
        feat["smi_oxygen"] = smi.upper().count("O")
        feat["smi_fluorine"] = smi.upper().count("F")
        feat["smi_chlorine"] = smi.upper().count("CL") // 2 if "CL" in smi.upper() else 0
        feat["smi_heteroatom_count"] = feat["smi_nitrogen"] + feat["smi_oxygen"] + feat["smi_fluorine"]

        # Try RDKit for richer features
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors

            mol = Chem.MolFromSmiles(smi)
            if mol:
                feat["mol_weight"] = Descriptors.MolWt(mol)
                feat["mol_logp"] = Descriptors.MolLogP(mol)
                feat["mol_hbd"] = rdMolDescriptors.CalcNumHBD(mol)
                feat["mol_hba"] = rdMolDescriptors.CalcNumHBA(mol)
                feat["mol_tpsa"] = Descriptors.TPSA(mol)
                feat["mol_rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
                feat["mol_rings"] = rdMolDescriptors.CalcNumRings(mol)
                feat["mol_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
        except ImportError:
            pass
        except Exception:
            pass

        features.append(feat)

    df = pd.DataFrame(features).fillna(0)
    return df


# ─── ClinTox Enrichment ───────────────────────────────────────────────────────

def load_and_merge_clintox(main_df: pd.DataFrame, clintox_path: Path) -> pd.DataFrame:
    """
    Load ClinTox dataset and merge toxicity features onto main DataFrame
    by matching SMILES strings. Adds FDA_APPROVED and CT_TOX columns,
    plus aggregate toxicity scores per trial.
    """
    logger.info("Loading ClinTox dataset...")
    try:
        ctox = pd.read_csv(clintox_path)
        ctox.columns = ctox.columns.str.strip().str.lower()
        ctox = ctox.rename(columns={
            "smiles": "smiles_key",
            "fda_approved": "fda_approved",
            "ct_tox": "ct_tox"
        })
        ctox = ctox.dropna(subset=["smiles_key"]).drop_duplicates(subset=["smiles_key"])

        # Build a lookup dict
        fda_map = dict(zip(ctox["smiles_key"], ctox["fda_approved"]))
        tox_map = dict(zip(ctox["smiles_key"], ctox["ct_tox"]))

        # Main df may have multiple SMILES per trial (pipe-separated)
        def aggregate_tox(smi_str):
            if not isinstance(smi_str, str):
                return 0.0, 0.0
            smis = [s.strip() for s in smi_str.split("|")]
            fda_vals = [fda_map.get(s, np.nan) for s in smis]
            tox_vals = [tox_map.get(s, np.nan) for s in smis]
            fda_mean = np.nanmean(fda_vals) if any(~np.isnan(v) for v in fda_vals if not np.isnan(v)) else 0.0
            tox_mean = np.nanmean(tox_vals) if any(~np.isnan(v) for v in tox_vals if not np.isnan(v)) else 0.0
            return float(fda_mean) if not np.isnan(fda_mean) else 0.0, \
                   float(tox_mean) if not np.isnan(tox_mean) else 0.0

        tox_results = main_df["smiless"].apply(aggregate_tox)
        main_df["fda_approved_score"] = tox_results.apply(lambda x: x[0])
        main_df["ct_tox_score"] = tox_results.apply(lambda x: x[1])
        logger.info(f"ClinTox enrichment complete. FDA/Tox scores added.")
    except FileNotFoundError:
        logger.warning(f"ClinTox file not found at {clintox_path}. Skipping toxicity enrichment.")
        main_df["fda_approved_score"] = 0.0
        main_df["ct_tox_score"] = 0.0
    except Exception as e:
        logger.warning(f"ClinTox merge failed: {e}. Using defaults.")
        main_df["fda_approved_score"] = 0.0
        main_df["ct_tox_score"] = 0.0
    return main_df


# ─── Feature Engineering ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate derived risk-scoring and complexity features."""

    # Phase risk (higher phase = more advanced = lower risk numerically)
    phase_risk_map = {
        "Phase 1": 4, "Phase 2": 3, "Phase 3": 2, "Phase 4": 1,
        "Phase 1/Phase 2": 3.5, "Phase 2/Phase 3": 2.5,
        "Early Phase 1": 5, "N/A": 3, "Not Applicable": 3
    }
    if "phase" in df.columns:
        df["phase_risk_score"] = df["phase"].map(phase_risk_map).fillna(3)

    # Disease complexity (number of distinct diseases per trial)
    if "diseases" in df.columns:
        df["disease_count"] = df["diseases"].apply(
            lambda x: len(str(x).split("|")) if isinstance(x, str) else 1
        )

    # Drug count (number of drugs in the trial)
    if "drugs" in df.columns:
        df["drug_count"] = df["drugs"].apply(
            lambda x: len(str(x).split("|")) if isinstance(x, str) else 1
        )

    # ICD code diversity
    if "icdcodes" in df.columns:
        df["icd_code_count"] = df["icdcodes"].apply(
            lambda x: len(str(x).split("|")) if isinstance(x, str) else 0
        )

    # Criteria text complexity
    if "criteria" in df.columns:
        df["criteria_word_count"] = df["criteria"].apply(
            lambda x: len(str(x).split()) if isinstance(x, str) else 0
        )
        df["criteria_has_exclusion"] = df["criteria"].apply(
            lambda x: int("exclusion" in str(x).lower())
        )
        df["criteria_has_age_limit"] = df["criteria"].apply(
            lambda x: int(bool(re.search(r"\b(age|years old|year-old)\b", str(x).lower())))
        )

    # Why-stop risk encoding
    if "why_stop" in df.columns:
        def encode_why_stop(val):
            if not isinstance(val, str) or val.strip() == "":
                return 0
            v = val.lower()
            if any(w in v for w in ["safety", "adverse", "toxicity", "harm"]):
                return 3  # High risk
            if any(w in v for w in ["efficacy", "futility", "lack"]):
                return 2  # Medium risk
            if any(w in v for w in ["enrollment", "funding", "withdrawn"]):
                return 1  # Low risk
            return 1
        df["why_stop_risk"] = df["why_stop"].apply(encode_why_stop)

    # Combined risk score
    risk_cols = [c for c in ["phase_risk_score", "ct_tox_score", "why_stop_risk"] if c in df.columns]
    if risk_cols:
        df["composite_risk_score"] = df[risk_cols].mean(axis=1)

    logger.info(f"Feature engineering complete. New features: {[c for c in df.columns if c not in ['nctid','status','label']][-10:]}")
    return df


# ─── Outlier Detection & Capping ──────────────────────────────────────────────

def cap_outliers_iqr(df: pd.DataFrame, cols: list, factor: float = 1.5) -> pd.DataFrame:
    """Cap numeric outliers using IQR method."""
    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# ─── Main Preprocessing Pipeline ─────────────────────────────────────────────

class ClinicalTrialPreprocessor:
    """
    End-to-end preprocessing pipeline for clinical trial data.
    Fits on training data and transforms any split.
    """

    def __init__(
        self,
        tfidf_max_features: int = 200,
        tfidf_ngram_range: tuple = (1, 2),
        svd_components: int = 50,
        use_smote: bool = True,
        scaler_type: str = "robust",
    ):
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.svd_components = svd_components
        self.use_smote = use_smote
        self.scaler_type = scaler_type

        self.tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=svd_components, random_state=42)
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.label_encoders = {}
        self.ohe = None
        self.ohe_cols = []
        self.feature_names = []
        self.smote = SMOTE(random_state=42, k_neighbors=5)
        self._fitted = False

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode high-cardinality cats; OHE low-cardinality ones."""
        label_enc_cols = []
        ohe_candidate_cols = ["phase"]

        self.ohe_cols = [c for c in ohe_candidate_cols if c in df.columns]

        if fit and self.ohe_cols:
            self.ohe = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore", drop="first"
            )
            ohe_arr = self.ohe.fit_transform(df[self.ohe_cols].astype(str))
        elif not fit and self.ohe_cols and self.ohe:
            ohe_arr = self.ohe.transform(df[self.ohe_cols].astype(str))
        else:
            ohe_arr = np.empty((len(df), 0))

        ohe_feature_names = []
        if self.ohe and self.ohe_cols:
            ohe_feature_names = [
                f"ohe_{col}_{val}"
                for col, cats in zip(self.ohe_cols, self.ohe.categories_)
                for val in cats[1:]
            ]
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_feature_names, index=df.index)

        return pd.concat([df.drop(columns=self.ohe_cols, errors="ignore"), ohe_df], axis=1)

    def fit_transform(self, raw_df: pd.DataFrame, clintox_path: Path = None):
        """
        Full fit+transform on training data.
        Returns (X_array, y_array, feature_names)
        """
        logger.info("=" * 60)
        logger.info("Starting Preprocessing Pipeline (fit_transform)...")
        df = raw_df.copy()

        # ── 1. Drop duplicates
        before = len(df)
        df = df.drop_duplicates(subset=["nctid"] if "nctid" in df.columns else None)
        logger.info(f"Removed {before - len(df)} duplicates. Rows: {len(df)}")

        # ── 2. Target extraction
        if "label" not in df.columns:
            raise ValueError("Dataset must contain a 'label' column.")
        y = df["label"].astype(int).values
        df = df.drop(columns=["label", "nctid", "status"], errors="ignore")

        # ── 3. ClinTox enrichment
        if clintox_path:
            df = load_and_merge_clintox(df, clintox_path)

        # ── 4. SMILES feature extraction
        if "smiless" in df.columns:
            logger.info("Extracting SMILES molecular features...")
            smiles_feats = extract_smiles_features(df["smiless"])
            smiles_feats.index = df.index
            df = pd.concat([df, smiles_feats], axis=1)
        df = df.drop(columns=["smiless", "drugs", "icdcodes", "diseases"], errors="ignore")

        # ── 5. Feature engineering
        df = engineer_features(df)

        # ── 6. NLP: TF-IDF on 'criteria'
        if "criteria" in df.columns:
            logger.info("Cleaning and vectorizing 'criteria' text...")
            criteria_clean = df["criteria"].apply(clean_text)
            tfidf_matrix = self.tfidf.fit_transform(criteria_clean)
            tfidf_svd = self.svd.fit_transform(tfidf_matrix)
            tfidf_cols = [f"tfidf_svd_{i}" for i in range(tfidf_svd.shape[1])]
            tfidf_df = pd.DataFrame(tfidf_svd, columns=tfidf_cols, index=df.index)
            df = df.drop(columns=["criteria"], errors="ignore")
            df = pd.concat([df, tfidf_df], axis=1)
            logger.info(f"TF-IDF+SVD: {tfidf_svd.shape[1]} components generated.")

        df = df.drop(columns=["why_stop"], errors="ignore")

        # ── 7. Categorical encoding
        df = self._encode_categoricals(df, fit=True)

        # ── 8. Keep only numeric columns
        df = df.select_dtypes(include=[np.number])

        # ── 9. Missing value imputation
        numeric_cols = df.columns.tolist()
        imputer = KNNImputer(n_neighbors=5)
        X = imputer.fit_transform(df)
        self._imputer = imputer
        self.feature_names = numeric_cols

        # ── 10. Outlier capping
        X_df = pd.DataFrame(X, columns=numeric_cols)
        outlier_cols = [c for c in numeric_cols if "score" in c or "count" in c]
        X_df = cap_outliers_iqr(X_df, outlier_cols)
        X = X_df.values

        # ── 11. Feature scaling
        X = self.scaler.fit_transform(X)

        # ── 12. SMOTE for class imbalance
        if self.use_smote:
            logger.info(f"Class distribution before SMOTE: {np.bincount(y)}")
            try:
                X, y = self.smote.fit_resample(X, y)
                logger.info(f"Class distribution after SMOTE:  {np.bincount(y)}")
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}. Proceeding without oversampling.")

        self._fitted = True
        logger.info(f"Preprocessing complete. Final shape: X={X.shape}, y={y.shape}")
        return X, y, self.feature_names

    def transform(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted pipeline (no SMOTE)."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        df = raw_df.copy()
        df = df.drop(columns=["label", "nctid", "status"], errors="ignore")

        df = load_and_merge_clintox(df, DATASETS_DIR / "clintox.csv")

        if "smiless" in df.columns:
            smiles_feats = extract_smiles_features(df["smiless"])
            smiles_feats.index = df.index
            df = pd.concat([df, smiles_feats], axis=1)
        df = df.drop(columns=["smiless", "drugs", "icdcodes", "diseases"], errors="ignore")

        df = engineer_features(df)

        if "criteria" in df.columns:
            criteria_clean = df["criteria"].apply(clean_text)
            tfidf_matrix = self.tfidf.transform(criteria_clean)
            tfidf_svd = self.svd.transform(tfidf_matrix)
            tfidf_cols = [f"tfidf_svd_{i}" for i in range(tfidf_svd.shape[1])]
            tfidf_df = pd.DataFrame(tfidf_svd, columns=tfidf_cols, index=df.index)
            df = df.drop(columns=["criteria"], errors="ignore")
            df = pd.concat([df, tfidf_df], axis=1)

        df = df.drop(columns=["why_stop"], errors="ignore")
        df = self._encode_categoricals(df, fit=False)
        df = df.select_dtypes(include=[np.number])

        # Align columns
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_names]

        X = self._imputer.transform(df)
        X = self.scaler.transform(X)
        return X

    def save(self, path: Path = None):
        path = path or (SAVED_MODELS_DIR / "preprocessor.joblib")
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: Path = None) -> "ClinicalTrialPreprocessor":
        path = path or (SAVED_MODELS_DIR / "preprocessor.joblib")
        return joblib.load(path)


# ─── Dataset Loader ───────────────────────────────────────────────────────────

def load_raw_data(raw_path: Path = None, clintox_path: Path = None) -> tuple:
    """
    Load raw datasets and return (df_main, df_clintox).
    """
    raw_path = raw_path or (DATASETS_DIR / "raw_data.csv")
    clintox_path = clintox_path or (DATASETS_DIR / "clintox.csv")

    logger.info(f"Loading main dataset from: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)

    logger.info(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    logger.info(f"Class distribution:\n{df['label'].value_counts() if 'label' in df.columns else 'No label column'}")

    ctox = pd.DataFrame()
    if clintox_path.exists():
        ctox = pd.read_csv(clintox_path)
        logger.info(f"ClinTox dataset: {ctox.shape[0]} rows × {ctox.shape[1]} columns")

    return df, ctox


if __name__ == "__main__":
    df, ctox = load_raw_data()
    preprocessor = ClinicalTrialPreprocessor(tfidf_max_features=200, svd_components=50)
    X, y, feature_names = preprocessor.fit_transform(df, clintox_path=DATASETS_DIR / "clintox.csv")
    preprocessor.save()
    print(f"\n✅ Preprocessing done: X={X.shape}, y={y.shape}")
    print(f"Features ({len(feature_names)}): {feature_names[:10]} ...")

    # Save processed arrays
    processed_dir = DATASETS_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)
    np.save(processed_dir / "X.npy", X)
    np.save(processed_dir / "y.npy", y)
    pd.DataFrame({"feature_name": feature_names}).to_csv(processed_dir / "feature_names.csv", index=False)
    print(f"✅ Saved processed data to {processed_dir}")

"""
utils/nlp_utils.py
==================
NLP utilities for clinical trial text processing.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))
_CLINICAL_KEEP = {"not", "no", "without", "exclude", "include", "severe", "moderate", "mild"}
_stop_words -= _CLINICAL_KEEP


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [_lemmatizer.lemmatize(t) for t in tokens
              if t not in _stop_words and len(t) > 2]
    return " ".join(tokens)


def extract_clinical_keywords(text: str) -> dict:
    text_lower = str(text).lower()
    return {
        "has_age_criteria": int(bool(re.search(r"\b(age|years old)\b", text_lower))),
        "has_gender_criteria": int(bool(re.search(r"\b(male|female|gender)\b", text_lower))),
        "has_prior_treatment": int(bool(re.search(r"\b(prior|previous|naive)\b", text_lower))),
        "has_comorbidity": int(bool(re.search(r"\b(diabetes|hypertension|cancer)\b", text_lower))),
        "has_pregnancy": int(bool(re.search(r"\b(pregnan|lactating|breastfeed)\b", text_lower))),
        "has_lab_values": int(bool(re.search(r"\b(hemoglobin|creatinine|platelet|wbc)\b", text_lower))),
        "exclusion_criteria_count": len(re.findall(r"exclusion", text_lower)),
        "inclusion_criteria_count": len(re.findall(r"inclusion", text_lower)),
        "criteria_length": len(text.split()),
    }

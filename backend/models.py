"""
backend/models.py
==================
Pydantic request/response schemas for the Clinical Trial Predictor API.
"""

from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # Core trial fields
    phase: Optional[str] = Field(default="Phase 2", description="Trial phase (e.g. 'Phase 3')")
    diseases: Optional[str] = Field(default="", description="Pipe-separated disease names")
    drugs: Optional[str] = Field(default="", description="Pipe-separated drug names")
    icdcodes: Optional[str] = Field(default="", description="Pipe-separated ICD codes")
    smiless: Optional[str] = Field(default="", description="Pipe-separated SMILES strings")
    criteria: Optional[str] = Field(default="", description="Inclusion/exclusion criteria text")
    why_stop: Optional[str] = Field(default="", description="Why the trial stopped (if applicable)")

    # Model selection
    model: Optional[str] = Field(default="XGBoost", description="ML model to use")

    class Config:
        json_schema_extra = {
            "example": {
                "phase": "Phase 3",
                "diseases": "Type 2 Diabetes|Hypertension",
                "drugs": "Metformin|Linagliptin",
                "icdcodes": "E11|I10",
                "smiless": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "criteria": "Inclusion: Adults 18-75. Exclusion: Severe renal impairment.",
                "why_stop": "",
                "model": "XGBoost",
            }
        }


class ShapFeature(BaseModel):
    feature: str
    value: float
    impact: str  # 'positive' | 'negative'


class PredictResponse(BaseModel):
    trial_id: Optional[str] = None
    prediction: int = Field(description="0=Failure, 1=Success")
    success_probability: float = Field(description="Probability of trial success (0-1)")
    failure_probability: float
    confidence: str = Field(description="low | medium | high")
    risk_level: str = Field(description="low | moderate | high | critical")
    model_used: str
    top_shap_features: Optional[List[ShapFeature]] = []
    interpretation: str = Field(description="Human-readable interpretation")


class BatchPredictionRow(BaseModel):
    row_index: int
    prediction: int
    success_probability: float
    confidence: str
    risk_level: str


class BatchPredictResponse(BaseModel):
    total_trials: int
    successful_trials: int
    failed_trials: int
    avg_success_probability: float
    model_used: str
    predictions: List[Dict[str, Any]]


class ModelInfo(BaseModel):
    name: str
    loaded: bool
    description: str
    supports_proba: bool

"""
backend/database.py
====================
PostgreSQL connection and ORM models via SQLAlchemy.
"""

import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://clinicaluser:clinicalpw@localhost:5432/clinicaltrials"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    trial_id = Column(String(50), unique=True, index=True)
    model_used = Column(String(50))
    prediction = Column(Integer)
    success_probability = Column(Float)
    risk_level = Column(String(20))
    confidence = Column(String(20))
    input_data = Column(JSON)
    shap_features = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), unique=True)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    cv_f1_mean = Column(Float)
    cv_f1_std = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)

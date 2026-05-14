# 🧬 Predictive Modeling for Clinical-Trial Success Likelihood Using Machine Learning

A production-grade, full-stack ML system for predicting the success likelihood of clinical trials using multi-modal data including NLP text analysis, molecular SMILES features, and structured clinical metadata.

---

## 📁 Project Structure

```
project/
├── backend/                    # FastAPI REST API
│   ├── main.py                 # App entrypoint & routes
│   ├── models.py               # Pydantic request/response models
│   ├── database.py             # PostgreSQL connection
│   └── predictor.py            # ML inference engine
│
├── frontend/                   # React.js dashboard
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   ├── pages/              # Dashboard, Predict, Compare
│   │   ├── hooks/              # Custom React hooks
│   │   └── utils/              # API helpers
│   ├── package.json
│   └── tailwind.config.js
│
├── datasets/                   # Raw & processed datasets
│   ├── raw_data.csv
│   ├── clintox.csv
│   └── processed/
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_NLP_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Comparison.ipynb
│
├── training/                   # Model training scripts
│   ├── train_xgboost.py
│   ├── train_decision_tree.py
│   ├── train_random_forest.py
│   ├── train_naive_bayes.py
│   ├── train_ann.py
│   └── train_logistic_regression.py
│
├── evaluation/                 # Evaluation & comparison
│   ├── evaluate_all.py
│   └── compare_models.py
│
├── utils/                      # Shared utilities
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── nlp_utils.py
│   ├── smiles_utils.py
│   └── visualization.py
│
├── saved_models/               # Serialized model artifacts
├── visualizations/             # Generated plots & charts
├── api/                        # API route modules
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── app.py                      # Unified entrypoint
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repo>
cd project
pip install -r requirements.txt
```

### 2. Prepare Datasets

Place your datasets in `datasets/`:
- `datasets/raw_data.csv`
- `datasets/clintox.csv`

### 3. Run Full Training Pipeline

```bash
python app.py --mode train
```

### 4. Launch API Server

```bash
uvicorn backend.main:app --reload --port 8000
```

### 5. Launch Frontend

```bash
cd frontend
npm install
npm run dev
```

### 6. Docker (All-in-One)

```bash
docker-compose up --build
```

---

## 🧠 ML Models Implemented

| Model | Type | Tuning | Explainability |
|-------|------|--------|----------------|
| XGBoost | Gradient Boosting | GridSearchCV | SHAP |
| Random Forest | Ensemble | RandomizedSearchCV | Feature Importance |
| Decision Tree | Tree-based | GridSearchCV | Tree Visualization |
| Naive Bayes | Probabilistic | Alpha tuning | Posterior probs |
| ANN (MLP) | Neural Network | Keras Tuner | SHAP DeepExplainer |
| Logistic Regression | Linear | L1/L2/ElasticNet | Coefficients |

---

## 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (macro + per-class)
- Confusion Matrix
- Classification Report
- K-Fold & Stratified K-Fold Cross-Validation
- SHAP feature explanations

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Single trial prediction |
| POST | `/predict/batch` | CSV batch prediction |
| GET | `/models` | List available models |
| GET | `/metrics` | All model metrics |
| GET | `/shap/{trial_id}` | SHAP explanation |

---

## 🏥 Healthcare Impact

This system enables:
- **Phase prediction**: Identify trials likely to fail early
- **Resource optimization**: Prioritize high-probability trials
- **Risk scoring**: Quantify trial risk from molecular + clinical data
- **Regulatory insights**: Toxicity flags from ClinTox integration

---

## 📦 Tech Stack

- **ML**: Python, Scikit-learn, XGBoost, TensorFlow/Keras, SHAP
- **NLP**: NLTK, spaCy, TF-IDF, WordCloud
- **Chemistry**: RDKit, ClinTox molecular features
- **Backend**: FastAPI, PostgreSQL, SQLAlchemy
- **Frontend**: React.js, Tailwind CSS, Recharts
- **Deployment**: Docker, Docker Compose

---

## 📄 License

MIT License — for research and educational purposes.

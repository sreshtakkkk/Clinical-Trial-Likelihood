# ============================================================
# Dockerfile — Clinical Trial Success Predictor
# Multi-stage: Python ML backend + Node.js frontend build
# ============================================================

FROM python:3.11-slim AS backend-base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# NLTK downloads
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt','stopwords','wordnet','omw-1.4']]"

# Copy source
COPY . .

# Create required directories
RUN mkdir -p datasets/processed saved_models visualizations evaluation

# Expose API port
EXPOSE 8000

# Default: start API
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

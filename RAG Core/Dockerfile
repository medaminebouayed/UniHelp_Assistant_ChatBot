# ── RAG Core Dockerfile ──
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models (cache in image layer)
# These are cached in /root/.cache/huggingface and mounted as a volume
COPY download_models.py ./
RUN python download_models.py

# Copy application code
COPY rag/ ./rag/
COPY rag_api/ ./rag_api/
COPY utils/ ./utils/
COPY upload_doc.py ./
COPY data/ ./data/

# Create data directories if missing
RUN mkdir -p /app/data /app/data/augmented /app/data/synthetic

ENV PYTHONPATH=/app

EXPOSE 8002

CMD ["uvicorn", "rag_api.main:app", "--host", "0.0.0.0", "--port", "8002"]

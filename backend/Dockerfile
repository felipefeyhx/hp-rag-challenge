FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (cached layer)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code
COPY app ./app
COPY scripts ./scripts

# Data folders (documents mounted read-only, chroma + db as volumes)
RUN mkdir -p /app/data/chroma /app/data/db /app/data/documents

EXPOSE 8000

# Ingest documents on start (idempotent), then boot the API.
CMD ["sh", "-c", "python -m scripts.ingest && uvicorn app.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000} --workers 4"]

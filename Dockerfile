FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# System deps (optional, keep slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY src/requirements.txt /app/src/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/src/requirements.txt

# Copy source
COPY src /app/src
COPY pyproject.toml /app/pyproject.toml

EXPOSE 5050

CMD ["python", "-m", "src.server.app"]



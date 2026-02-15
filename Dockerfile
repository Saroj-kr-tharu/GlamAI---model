# Stage 1: Builder – install Python deps
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed to compile native wheels (opencv, mediapipe, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Production – lean runtime image
FROM python:3.11-slim AS runtime

LABEL maintainer="face-analysis-api" \
    org.opencontainers.image.title="Face Analysis API" \
    org.opencontainers.image.description="Flask API – face landmark extraction, classification & makeup recommendations"

# Runtime-only system libs (no compilers)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    tini && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user BEFORE referencing it
RUN groupadd --gid 1000 appuser && \
    useradd  --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# ── Application code ──
COPY app.py main.py generation.py retrieve.py \
    layer1_extraction.py layer2_metrics.py layer3_classify.py \
    ./

# MediaPipe model bundle
COPY face_landmarker.task ./

# Knowledge base (JSON files for RAG)
COPY knowledge/ ./knowledge/

# Copy requirements.txt for reference
COPY requirements.txt ./



# ── Environment ──
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    OLLAMA_HOST=http://ollama:11434

EXPOSE 5000


HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# tini ensures proper PID-1 signal handling
ENTRYPOINT ["tini", "--"]

# Launch with gunicorn 
CMD ["gunicorn", \
    "--bind",            "0.0.0.0:5000", \
    "--workers",         "2", \
    "--threads",         "2", \
    "--timeout",         "300", \
    "--graceful-timeout","30", \
    "--access-logfile",  "-", \
    "--error-logfile",   "-", \
    "app:app"]

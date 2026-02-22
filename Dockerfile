# Stage 1: Builder – install Python deps
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Download MediaPipe face landmarker model (~29MB) at build time
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/* && \
    wget -q "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
    -O /build/face_landmarker.task && \
    test -s /build/face_landmarker.task || (echo "ERROR: face_landmarker.task is empty or failed to download" && exit 1)


# Stage 2: Production – lean runtime image
FROM python:3.11-slim AS runtime

LABEL maintainer="glamai" \
    org.opencontainers.image.title="GlamAI API" \
    org.opencontainers.image.description="Flask API – face landmark extraction, classification & personalized makeup recommendations"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    tini && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local



WORKDIR /app

# Application code (main.py excluded – CLI-only, not needed at runtime)
COPY app.py generation.py retrieve.py \
    layer1_extraction.py layer2_metrics.py layer3_classify.py \
    ./

# Copy the verified model from builder (not from host)
COPY --from=builder /build/face_landmarker.task ./

COPY knowledge/ ./knowledge/
COPY requirements.txt ./

# Create uploads directory and set ownership
RUN mkdir -p /app/uploads 



ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    OLLAMA_HOST=http://139.59.85.203.nip.io/ollama

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://139.59.85.203.nip.io/model || exit 1

ENTRYPOINT ["tini", "--"]

CMD ["gunicorn", \
    "--bind",            "0.0.0.0:5000", \
    "--workers",         "2", \
    "--threads",         "2", \
    "--timeout",         "300", \
    "--graceful-timeout","30", \
    "--access-logfile",  "-", \
    "--error-logfile",   "-", \
    "app:app"]

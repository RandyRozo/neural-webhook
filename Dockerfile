# ===================================
# Stage 1: Dependencias y Build
# ===================================
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===================================
# Stage 2: Imagen de Produccion
# ===================================
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN groupadd --system appuser && \
    useradd --system --gid appuser appuser

RUN mkdir -p /app/logs /app/evidencias_neural && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser app/ /app/
COPY --chown=appuser:appuser requirements.txt /app/

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    PRODUCTION=true

LABEL org.opencontainers.image.title="Neural Webhook Service" \
    org.opencontainers.image.description="Microservicio webhook para camaras Neural con ANPR y Oracle Cloud Object Storage" \
    org.opencontainers.image.version="1.0.0" \
    org.opencontainers.image.vendor="Neural Team"

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "/app/main.py"]

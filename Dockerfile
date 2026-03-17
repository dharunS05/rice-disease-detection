# ═══════════════════════════════════════════════════════════════════
# Rice Disease Detection — Dockerfile
# Anna University Final Year Project 2024–25
#
# Targets: Hugging Face Spaces (Docker SDK) + local Docker
# Port: 7860 (required by Hugging Face Spaces)
#
# Build:
#   docker build -t rice-disease .
#
# Run locally:
#   docker run -p 7860:7860 rice-disease
#
# Hugging Face Spaces:
#   Set Space SDK to "Docker" in the Space settings.
#   The Space will auto-build and serve on port 7860.
#   Upload trained_models/ files via the Space file manager or Git LFS.
# ═══════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────
# libgomp1    : required by LightGBM / some sklearn builds
# libglib2.0  : required by OpenCV (if used later)
# curl        : useful for health checks in CI
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────
# Copy requirements first so Docker layer-caches the install step.
# Changing only source code won't re-trigger pip install.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project source ────────────────────────────────────────────
# .dockerignore should exclude: __pycache__, *.pyc, .git, notebooks/
COPY . .

# ── Hugging Face Spaces: non-root user requirement ─────────────────
# HF Spaces runs containers as UID 1000. Pre-create the user so file
# permissions are correct when Space mounts model files via Git LFS.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ── Runtime environment ────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Suppress TF info/warning logs in production
    TF_CPP_MIN_LOG_LEVEL=2 \
    # Gradio telemetry off
    GRADIO_ANALYTICS_ENABLED=false \
    # HF Spaces requires the app to bind 0.0.0.0:7860
    PORT=7860

EXPOSE 7860

# ── Health check ───────────────────────────────────────────────────
# Checks the FastAPI /health endpoint every 30 seconds.
# Space will show "Unhealthy" if models fail to load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start the application ──────────────────────────────────────────
# uvicorn serves both:
#   - Gradio UI at "/"  (rendered by HF Spaces)
#   - FastAPI REST at "/api/v1/..."  + "/docs"
#
# --workers 1 : TF and PyTorch models are not fork-safe; use 1 worker.
# --timeout-keep-alive 30 : HF Spaces load balancer keep-alive.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]
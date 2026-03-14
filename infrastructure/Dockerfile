# Stage 1: Build & Dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Final Runtime
FROM python:3.10-slim AS runner

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code
COPY model_server/ ./model_server/

# Standard non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch the model server
CMD ["uvicorn", "model_server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

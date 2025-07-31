# Neural Operator Foundation Lab - Production Container
FROM python:3.9-slim as base

# Security: Create non-root user
RUN groupadd -r neural && useradd -r -g neural neural

# System dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements*.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY --chown=neural:neural . .

# Install package in development mode
RUN pip install --no-cache-dir -e .

# Security: Switch to non-root user
USER neural

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import neural_operator_lab; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "neural_operator_lab"]

# Multi-stage build for smaller production image
FROM base as production

# Remove development dependencies in production
RUN pip uninstall -y pytest pytest-cov black isort flake8 mypy pre-commit

# Final production image
FROM python:3.9-slim as final

# Security updates
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r neural && useradd -r -g neural neural

# Copy only necessary files from production stage
COPY --from=production --chown=neural:neural /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=production --chown=neural:neural /usr/local/bin /usr/local/bin
COPY --from=production --chown=neural:neural /app /app

WORKDIR /app
USER neural

# Labels for better maintainability
LABEL maintainer="daniel@terragon.ai"
LABEL description="Neural Operator Foundation Lab - Training & Benchmarking Suite"
LABEL version="0.1.0"

CMD ["python", "-m", "neural_operator_lab"]
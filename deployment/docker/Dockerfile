# Production Dockerfile for Neural Operator Foundation Lab
FROM nvidia/cuda:12.1-devel-ubuntu22.04

LABEL maintainer="Terragon Labs <info@terragonlabs.com>"
LABEL version="1.0.0"
LABEL description="Neural Operator Foundation Lab - Production Ready Container"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Create non-root user
RUN groupadd --gid 1000 neural_op && \
    useradd --uid 1000 --gid neural_op --shell /bin/bash --create-home neural_op

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libhdf5-dev \
    libopenmpi-dev \
    openmpi-bin \
    openssh-client \
    rsync \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create application directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip3 install -r requirements.txt && \
    pip3 install -r requirements-prod.txt && \
    pip3 cache purge

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/outputs /app/checkpoints /app/configs && \
    chown -R neural_op:neural_op /app

# Switch to non-root user
USER neural_op

# Set up configuration
COPY deployment/configs/production.yaml /app/configs/production.yaml

# Expose ports for distributed training and monitoring
EXPOSE 8000 8080 29500

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import neural_operator_lab; print('OK')" || exit 1

# Default command
CMD ["python3", "-m", "neural_operator_lab.cli.main", "--config", "/app/configs/production.yaml"]
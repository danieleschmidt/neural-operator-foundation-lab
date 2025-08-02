#!/bin/bash
# Development container setup script

set -e

echo "🚀 Setting up Neural Operator Foundation Lab development environment..."

# Install the package in development mode
if [ -f "/workspace/pyproject.toml" ]; then
    echo "📦 Installing package in development mode..."
    pip install -e "/workspace[dev,test,docs]"
fi

# Install pre-commit hooks
if [ -f "/workspace/.pre-commit-config.yaml" ]; then
    echo "🔧 Installing pre-commit hooks..."
    cd /workspace
    pre-commit install
    pre-commit install --hook-type commit-msg
fi

# Create necessary directories
echo "📁 Creating workspace directories..."
mkdir -p /workspace/{data,models,checkpoints,logs,experiments,outputs,cache}

# Set up Jupyter Lab extensions
echo "🔬 Setting up Jupyter Lab..."
pip install --quiet \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server[all] \
    jupyterlab_code_formatter

# Set up environment variables
echo "🔧 Setting up environment..."
if [ ! -f "/workspace/.env" ] && [ -f "/workspace/.env.example" ]; then
    echo "📋 Creating .env from .env.example..."
    cp /workspace/.env.example /workspace/.env
fi

# Set up Git safe directory
echo "🔒 Setting up Git safe directory..."
git config --global --add safe.directory /workspace

# Download sample data (if needed)
if [ ! -d "/workspace/data/samples" ]; then
    echo "📊 Setting up sample data..."
    mkdir -p /workspace/data/samples
    # Add sample data download here if needed
fi

# Verify installation
echo "✅ Verifying installation..."
cd /workspace
python -c "import neural_operator_lab; print(f'Neural Operator Lab v{neural_operator_lab.__version__} loaded successfully!')" || true

# Display helpful information
echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📚 Useful commands:"
echo "  make test          - Run the test suite"
echo "  make lint          - Run code quality checks"
echo "  make docs          - Build documentation"
echo "  make install-dev   - Reinstall in development mode"
echo "  jupyter lab        - Start Jupyter Lab"
echo ""
echo "🌐 Available services:"
echo "  Jupyter Lab:  http://localhost:8888 (token: development)"
echo "  TensorBoard:  http://localhost:6006"
echo "  MLflow UI:    http://localhost:5000"
echo ""
echo "Happy coding! 🚀"
#!/bin/bash
# Development environment setup script for Neural Operator Foundation Lab

set -e

echo "🚀 Setting up Neural Operator Foundation Lab development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "📥 Installing development dependencies..."
pip install -r requirements-dev.txt

# Install package in development mode
echo "🔨 Installing package in development mode..."
pip install -e .

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Run initial checks
echo "🧪 Running initial checks..."
echo "  - Code formatting..."
black --check src/ tests/ || (echo "❌ Code formatting issues found. Run 'make format' to fix." && exit 1)

echo "  - Import sorting..."
isort --check-only src/ tests/ || (echo "❌ Import sorting issues found. Run 'make format' to fix." && exit 1)

echo "  - Linting..."
flake8 src/ tests/ || (echo "❌ Linting issues found. Fix them before proceeding." && exit 1)

echo "  - Type checking..."
mypy src/neural_operator_lab || (echo "❌ Type checking issues found. Fix them before proceeding." && exit 1)

echo "  - Running tests..."
pytest tests/ -v || (echo "❌ Some tests failed. Fix them before proceeding." && exit 1)

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Start developing: make help"
echo "  3. Run tests: make test"
echo "  4. Format code: make format"
echo ""
echo "Happy coding! 🚀"
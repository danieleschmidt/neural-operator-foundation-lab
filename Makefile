.PHONY: help install install-dev test lint format type-check clean docs build upload

help:
	@echo "Neural Operator Foundation Lab - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install     Install package for production"
	@echo "  install-dev Install package for development"
	@echo ""
	@echo "Development:"
	@echo "  test        Run all tests"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code with black and isort"
	@echo "  type-check  Run type checking with mypy"
	@echo "  pre-commit  Run pre-commit hooks"
	@echo ""
	@echo "Documentation:"
	@echo "  docs        Build documentation"
	@echo "  docs-serve  Serve documentation locally"
	@echo ""
	@echo "Build & Release:"
	@echo "  build       Build package"
	@echo "  clean       Clean build artifacts"
	@echo "  upload      Upload to PyPI (maintainers only)"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ --cov=neural_operator_lab --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -x -v

lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/neural_operator_lab

pre-commit:
	pre-commit run --all-files

docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

build:
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

upload:
	python -m twine upload dist/*

# Development shortcuts
dev: install-dev
check: lint type-check test
all: clean format check docs build
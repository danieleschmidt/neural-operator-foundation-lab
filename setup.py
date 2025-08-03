"""Setup script for Neural Operator Foundation Lab."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.2.0",
]

setup(
    name="neural-operator-foundation-lab",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@terragon.ai",
    description="A comprehensive framework for training and benchmarking neural operators on high-dimensional PDEs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural-operator-foundation-lab",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
        "jax": [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neural-operator-lab=neural_operator_lab.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "neural_operator_lab": [
            "data/examples/*.h5",
            "configs/*.yaml",
        ],
    },
    zip_safe=False,
    keywords=[
        "neural operators",
        "pde",
        "machine learning",
        "physics",
        "deep learning",
        "scientific computing",
        "fourier neural operator",
        "transformer",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/neural-operator-foundation-lab/issues",
        "Source": "https://github.com/yourusername/neural-operator-foundation-lab",
        "Documentation": "https://neural-operator-lab.readthedocs.io",
    },
)
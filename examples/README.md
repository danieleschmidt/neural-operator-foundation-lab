# Examples

This directory contains example scripts and Jupyter notebooks demonstrating how to use Neural Operator Foundation Lab.

## Getting Started

1. **Install the package**:
   ```bash
   pip install -e .
   ```

2. **Install example dependencies**:
   ```bash
   pip install jupyter matplotlib plotly
   ```

3. **Run examples**:
   ```bash
   # Python scripts
   python examples/basic_training.py
   
   # Jupyter notebooks
   jupyter notebook examples/
   ```

## Example Categories

### Basic Usage
- `basic_training.py` - Simple neural operator training
- `data_loading.py` - Loading and preprocessing PDE data
- `model_evaluation.py` - Evaluating trained models

### Advanced Features
- `physics_informed_training.ipynb` - Physics-informed neural operators
- `uncertainty_quantification.ipynb` - Probabilistic neural operators
- `multi_gpu_training.py` - Distributed training setup

### Applications
- `turbulence_modeling.ipynb` - 3D Navier-Stokes equations
- `electromagnetic_fields.ipynb` - Maxwell's equations
- `structural_mechanics.ipynb` - Elasticity problems

## Running Examples

Each example includes:
- Clear documentation and comments
- Expected runtime and resource requirements
- Sample outputs and visualizations
- Links to relevant tutorials

For questions about examples, please see our [documentation](../docs/) or open an [issue](https://github.com/terragon-labs/neural-operator-foundation-lab/issues).
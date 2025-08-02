# Neural Operator Foundation Lab - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Model Selection](#model-selection)
4. [Training Your First Model](#training-your-first-model)
5. [Data Preparation](#data-preparation)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

---

## Getting Started

### Prerequisites
- Python 3.9 or higher
- CUDA 11.0+ (for GPU acceleration)
- 16GB+ RAM (32GB+ recommended for large models)
- Linux or macOS (Windows support via WSL2)

### Installation

#### Quick Install (Recommended)
```bash
pip install neural-operator-lab
```

#### Development Install
```bash
git clone https://github.com/yourusername/neural-operator-foundation-lab.git
cd neural-operator-foundation-lab
pip install -e ".[dev]"
```

#### With Optional Dependencies
```bash
# JAX backend support
pip install -e ".[jax]"

# Visualization extras
pip install -e ".[viz]"

# Complete installation
pip install -e ".[all]"
```

### Verify Installation
```python
import neural_operator_lab as nol
print(f"Neural Operator Lab v{nol.__version__}")

# Check GPU availability
print(f"CUDA available: {nol.cuda.is_available()}")
print(f"GPU count: {nol.cuda.device_count()}")
```

---

## Basic Usage

### Hello World Example
```python
from neural_operator_lab import FourierNeuralOperator, PDEDataset
from neural_operator_lab.training import Trainer

# Load a simple 2D Burgers' equation dataset
dataset = PDEDataset.load_builtin("burgers_2d")

# Create FNO model
model = FourierNeuralOperator(
    input_dim=2,      # 2D spatial + initial condition
    output_dim=1,     # Solution field
    modes=[16, 16],   # Fourier modes
    width=64          # Hidden dimension
)

# Train the model
trainer = Trainer(model=model, dataset=dataset)
trainer.train(epochs=50)

# Make predictions
prediction = model(dataset.test_data[0])
```

### Quick Benchmark
```python
from neural_operator_lab.benchmarks import quick_benchmark

# Run a quick performance test
results = quick_benchmark(
    model_type="fno",
    pde_type="navier_stokes_2d",
    resolution=64
)

print(f"Training time: {results.train_time:.2f}s")
print(f"Test error: {results.test_error:.2e}")
print(f"Inference speed: {results.inference_fps:.1f} FPS")
```

---

## Model Selection

### When to Use Each Architecture

| Model Type | Best For | Pros | Cons |
|------------|----------|------|------|
| **FNO** | Regular grids, periodic BCs | Fast, memory efficient | Limited to Fourier basis |
| **PNO** | Uncertainty quantification | Bayesian inference | Slower training |
| **TNO** | Complex, multi-scale problems | SOTA accuracy | High memory usage |
| **Graph NO** | Irregular meshes | Flexible geometry | Complex preprocessing |

### Performance Comparison
```python
from neural_operator_lab.benchmarks import ModelComparison

comparison = ModelComparison(
    models=["fno", "pno", "tno"],
    dataset="navier_stokes_3d",
    resolution=128
)

results = comparison.run()
results.plot_accuracy_vs_speed()
results.save_report("model_comparison.pdf")
```

### Model Recommendations

#### For Beginners
```python
# Start with FNO - fastest and most stable
model = FourierNeuralOperator(
    input_dim=3, output_dim=1,
    modes=[16, 16, 16], width=32
)
```

#### For Research
```python
# TNO for cutting-edge performance
model = TransformerNeuralOperator(
    input_dim=3, output_dim=1,
    hidden_dim=512, num_layers=8,
    num_heads=8
)
```

#### For Production
```python
# PNO for uncertainty quantification
model = ProbabilisticNeuralOperator(
    input_dim=3, output_dim=1,
    trunk_layers=[256, 256, 256],
    uncertainty_type="epistemic"
)
```

---

## Training Your First Model

### Step 1: Prepare Your Data
```python
from neural_operator_lab.data import PDEDataset

# Option A: Use built-in datasets
dataset = PDEDataset.load_builtin("darcy_flow_2d")

# Option B: Load from simulation files
dataset = PDEDataset.from_simulation_files(
    data_path="path/to/simulations/*.h5",
    input_fields=["initial_condition", "boundary"],
    output_fields=["solution"],
    normalize=True
)

# Option C: Create from numpy arrays
dataset = PDEDataset.from_arrays(
    inputs=input_arrays,
    outputs=output_arrays,
    grid=coordinate_grid
)
```

### Step 2: Configure Training
```python
from neural_operator_lab.training import TrainingConfig

config = TrainingConfig(
    batch_size=32,
    learning_rate=1e-3,
    epochs=100,
    optimizer="adam",
    scheduler="cosine",
    mixed_precision=True,  # Enable for speed
    gradient_clipping=1.0
)
```

### Step 3: Train with Monitoring
```python
from neural_operator_lab.training import Trainer

trainer = Trainer(
    model=model,
    dataset=dataset,
    config=config,
    log_dir="logs/experiment_1"
)

# Train with callbacks
trainer.add_callback("early_stopping", patience=10)
trainer.add_callback("checkpoint", save_best=True)
trainer.add_callback("tensorboard", log_dir="logs")

history = trainer.train()

# Plot training curves
trainer.plot_history()
```

### Step 4: Evaluate Results
```python
# Test set evaluation
test_metrics = trainer.evaluate()
print(f"Test L2 error: {test_metrics['l2_error']:.2e}")
print(f"Test relative error: {test_metrics['rel_error']:.2%}")

# Rollout evaluation (for time-series PDEs)
rollout_metrics = trainer.evaluate_rollout(
    num_steps=100,
    initial_conditions=dataset.test_data[:10]
)
print(f"Rollout stability: {rollout_metrics['stability']:.3f}")
```

---

## Data Preparation

### Data Formats

#### HDF5 Format (Recommended)
```python
# Expected structure:
# simulation.h5
# ├── inputs/
# │   ├── initial_conditions [N, H, W, C]
# │   └── parameters [N, P]
# ├── outputs/
# │   └── solutions [N, T, H, W, C]
# └── metadata/
#     ├── grid [H, W, 2]
#     └── timesteps [T]

dataset = PDEDataset.from_hdf5("simulation.h5")
```

#### NetCDF Format
```python
# For climate/atmospheric data
dataset = PDEDataset.from_netcdf(
    "climate_data.nc",
    input_vars=["temperature", "pressure"],
    output_vars=["precipitation"],
    time_dim="time",
    spatial_dims=["lat", "lon"]
)
```

#### NumPy Arrays
```python
# For custom data
dataset = PDEDataset.from_arrays(
    inputs=np.array([...]),      # [N, spatial..., features]
    outputs=np.array([...]),     # [N, spatial..., features]
    grid=np.array([...])         # [spatial..., dimensions]
)
```

### Data Preprocessing

#### Normalization
```python
from neural_operator_lab.data.preprocessing import normalize_data

# Automatic normalization
dataset = normalize_data(dataset, method="gaussian")

# Custom normalization
dataset = normalize_data(
    dataset,
    method="custom",
    input_stats={"mean": 0.5, "std": 0.2},
    output_stats={"mean": 0.0, "std": 1.0}
)
```

#### Augmentation
```python
from neural_operator_lab.data.augmentation import PDEAugmentation

augmenter = PDEAugmentation(
    rotation=True,              # Random rotations
    flip=True,                  # Random flips
    noise_level=0.01,          # Gaussian noise
    elastic_deformation=0.1     # Elastic deformation
)

augmented_dataset = augmenter(dataset)
```

#### Subsampling
```python
# Reduce resolution for faster training
dataset = dataset.subsample_space(factor=2)  # 512x512 -> 256x256
dataset = dataset.subsample_time(factor=2)   # Every other timestep
```

---

## Advanced Features

### Mixed Precision Training
```python
trainer = Trainer(
    model=model,
    dataset=dataset,
    mixed_precision=True,      # Enable automatic mixed precision
    gradient_scaling=True      # Handle gradient underflow
)
```

### Distributed Training
```python
from neural_operator_lab.distributed import DistributedTrainer

# Multi-GPU training
trainer = DistributedTrainer(
    model=model,
    dataset=dataset,
    world_size=4,              # Number of GPUs
    backend="nccl"             # Communication backend
)

trainer.train()
```

### Physics-Informed Training
```python
from neural_operator_lab.losses import PhysicsInformedLoss

# Define PDE residual
def navier_stokes_residual(u, p, nu=0.01):
    # Compute PDE residual
    return residual

# Create physics-informed loss
loss_fn = PhysicsInformedLoss(
    data_weight=1.0,
    pde_weight=0.1,
    boundary_weight=0.05
)
loss_fn.set_pde_residual(navier_stokes_residual)

trainer = Trainer(model=model, dataset=dataset, loss_fn=loss_fn)
```

### Model Compression
```python
from neural_operator_lab.compression import compress_model

# Quantization
compressed_model = compress_model(
    model, method="quantization", bits=8
)

# Pruning
compressed_model = compress_model(
    model, method="pruning", sparsity=0.5
)
```

### Transfer Learning
```python
# Load pre-trained model
pretrained = TransformerNeuralOperator.load_pretrained(
    "neurips2025/universal-physics-transformer"
)

# Fine-tune on specific PDE
pretrained.freeze_backbone()
pretrained.add_task_head(output_dim=new_output_dim)

trainer = Trainer(
    model=pretrained,
    dataset=new_dataset,
    learning_rate=1e-5  # Lower LR for fine-tuning
)
```

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Solutions:
# 1. Reduce batch size
config.batch_size = 16  # Instead of 32

# 2. Enable gradient checkpointing
model.enable_gradient_checkpointing()

# 3. Use mixed precision
config.mixed_precision = True

# 4. Reduce model size
model = FourierNeuralOperator(width=32)  # Instead of 64
```

#### Training Instability
```python
# Solutions:
# 1. Reduce learning rate
config.learning_rate = 1e-4  # Instead of 1e-3

# 2. Enable gradient clipping
config.gradient_clipping = 1.0

# 3. Use learning rate scheduling
config.scheduler = "cosine"
config.warmup_epochs = 10

# 4. Add regularization
config.weight_decay = 1e-4
config.dropout = 0.1
```

#### Poor Convergence
```python
# Check data quality
dataset.plot_samples()
dataset.check_statistics()

# Verify data normalization
print(f"Input mean: {dataset.input_stats['mean']}")
print(f"Input std: {dataset.input_stats['std']}")

# Add physics constraints
loss_fn = PhysicsInformedLoss(pde_weight=0.1)
```

#### Slow Training
```python
# Enable optimizations
config.mixed_precision = True
config.compile_model = True        # PyTorch 2.0 compilation
config.dataloader_workers = 4     # Parallel data loading

# Use efficient data format
dataset = dataset.convert_to_memory_map()  # Memory-mapped files
```

### Performance Optimization

#### Memory Usage
```python
# Monitor memory usage
from neural_operator_lab.utils import MemoryProfiler

profiler = MemoryProfiler()
with profiler:
    trainer.train()
    
profiler.plot_memory_usage()
```

#### Speed Profiling
```python
# Profile training speed
from neural_operator_lab.utils import SpeedProfiler

profiler = SpeedProfiler()
profiler.profile_training(trainer, num_steps=100)
profiler.print_bottlenecks()
```

---

## FAQ

### General Questions

**Q: Which model should I start with?**
A: Start with FourierNeuralOperator (FNO) for regular grid problems. It's fast, stable, and well-documented.

**Q: How much data do I need?**
A: Typically 1000-10000 samples for simple PDEs, 10000+ for complex 3D problems. Use data augmentation to increase effective dataset size.

**Q: Can I use my own PDE data?**
A: Yes! Convert to HDF5 format or use `PDEDataset.from_arrays()`. See the data preparation section for details.

### Technical Questions

**Q: How do I handle irregular geometries?**
A: Use Graph Neural Operators or convert to regular grids with appropriate masking.

**Q: Can I train on multiple GPUs?**
A: Yes, use `DistributedTrainer` for multi-GPU training. Supports both data and model parallelism.

**Q: How do I add custom loss functions?**
A: Inherit from `LossFunction` base class and implement the `forward` method. See examples in `neural_operator_lab.losses`.

**Q: Can I deploy models in production?**
A: Yes, models can be exported to ONNX or TorchScript for deployment. Use model compression for efficiency.

### Troubleshooting Questions

**Q: Training is very slow, what can I do?**
A: Enable mixed precision, use multiple GPUs, reduce batch size if memory-bound, or use model compilation.

**Q: My model isn't converging, help!**
A: Check data normalization, reduce learning rate, add gradient clipping, or try a different architecture.

**Q: I get CUDA out of memory errors**
A: Reduce batch size, enable gradient checkpointing, use mixed precision, or train on CPU for debugging.

---

## Getting Help

### Community Resources
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and community help
- **Discord**: Real-time chat and collaboration
- **Documentation**: Comprehensive guides and API reference

### Professional Support
- **Consulting**: Available for complex projects
- **Training**: Workshops and tutorials
- **Custom Development**: Specialized features and optimizations

### Contributing
We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Requesting features
- Contributing code
- Writing documentation

---

*Last updated: 2025-08-01*  
*For the latest version of this guide, visit: [https://neural-operator-lab.readthedocs.io](https://neural-operator-lab.readthedocs.io)*
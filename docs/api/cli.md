# CLI Reference

The Neural Operator Foundation Lab provides a comprehensive command-line interface for training, evaluation, and benchmarking neural operators.

## Main Command

```bash
python -m neural_operator_lab.cli.main [COMMAND] [OPTIONS]
```

## Global Options

- `--config PATH`: Configuration file path (YAML or JSON)
- `--verbose, -v`: Increase verbosity (-v, -vv, -vvv)
- `--log-file PATH`: Log file path
- `--version`: Show version information

## Commands

### train

Train a neural operator model.

```bash
python -m neural_operator_lab.cli.main train [OPTIONS]
```

#### Options

- `--model-type TEXT`: Type of model to train [default: transformer_neural_operator]
- `--data-path PATH`: Path to training data
- `--epochs INTEGER`: Number of training epochs [default: 100]
- `--batch-size INTEGER`: Batch size [default: 32]
- `--learning-rate FLOAT`: Learning rate [default: 0.0001]
- `--optimizer TEXT`: Optimizer type [default: adamw]
- `--scheduler TEXT`: Learning rate scheduler [default: cosine]
- `--output-dir PATH`: Output directory for results [default: ./outputs]
- `--checkpoint-dir PATH`: Directory for model checkpoints [default: ./checkpoints]
- `--resume-from PATH`: Resume training from checkpoint
- `--distributed`: Enable distributed training
- `--mixed-precision`: Enable mixed precision training
- `--physics-loss`: Enable physics-informed loss
- `--early-stopping`: Enable early stopping
- `--patience INTEGER`: Early stopping patience [default: 20]

#### Examples

```bash
# Basic training
python -m neural_operator_lab.cli.main train \
    --model-type fourier_neural_operator \
    --data-path /data/navier_stokes \
    --epochs 200 \
    --batch-size 16

# Training with physics-informed loss
python -m neural_operator_lab.cli.main train \
    --config configs/navier_stokes.yaml \
    --physics-loss \
    --mixed-precision \
    --distributed

# Resume training from checkpoint
python -m neural_operator_lab.cli.main train \
    --resume-from checkpoints/model_epoch_50.pt \
    --epochs 100
```

### evaluate

Evaluate a trained model.

```bash
python -m neural_operator_lab.cli.main evaluate [OPTIONS]
```

#### Options

- `--model PATH`: Path to trained model [required]
- `--data-path PATH`: Path to evaluation data [required]
- `--output-file PATH`: Output file for results [default: evaluation_results.json]
- `--batch-size INTEGER`: Batch size for evaluation [default: 32]
- `--metrics TEXT`: Metrics to compute (comma-separated) [default: mse,relative_l2]
- `--rollout-steps INTEGER`: Number of steps for rollout evaluation
- `--compute-uncertainty`: Compute uncertainty estimates (for probabilistic models)
- `--save-predictions`: Save model predictions
- `--visualization`: Generate visualization plots

#### Examples

```bash
# Basic evaluation
python -m neural_operator_lab.cli.main evaluate \
    --model checkpoints/best_model.pt \
    --data-path /data/test

# Rollout evaluation with uncertainty
python -m neural_operator_lab.cli.main evaluate \
    --model checkpoints/probabilistic_model.pt \
    --data-path /data/test \
    --rollout-steps 100 \
    --compute-uncertainty \
    --save-predictions

# Comprehensive evaluation with visualization
python -m neural_operator_lab.cli.main evaluate \
    --model checkpoints/tno_model.pt \
    --data-path /data/test \
    --metrics mse,relative_l2,spectral_error,conservation \
    --visualization
```

### benchmark

Run benchmarking suite.

```bash
python -m neural_operator_lab.cli.main benchmark [OPTIONS]
```

#### Options

- `--model PATH`: Path to model to benchmark [required]
- `--suite TEXT`: Benchmark suite to run [default: standard]
- `--datasets TEXT`: Specific datasets to benchmark (comma-separated)
- `--output-dir PATH`: Output directory for benchmark results [default: ./benchmark_results]
- `--compare-with PATH`: Compare with baseline model
- `--generate-report`: Generate PDF benchmark report
- `--upload-results`: Upload results to benchmark database

#### Available Suites

- `standard`: Standard PDE benchmarks (Navier-Stokes, Darcy, Burgers)
- `comprehensive`: Extended benchmark suite with multiple PDEs
- `performance`: Performance and scalability benchmarks
- `uncertainty`: Uncertainty quantification benchmarks

#### Examples

```bash
# Standard benchmark suite
python -m neural_operator_lab.cli.main benchmark \
    --model checkpoints/fno_model.pt \
    --suite standard \
    --generate-report

# Compare two models
python -m neural_operator_lab.cli.main benchmark \
    --model checkpoints/new_model.pt \
    --compare-with checkpoints/baseline_model.pt \
    --suite comprehensive

# Specific dataset benchmarks
python -m neural_operator_lab.cli.main benchmark \
    --model checkpoints/model.pt \
    --datasets navier_stokes_2d,heat_equation \
    --upload-results
```

### config

Configuration management utilities.

```bash
python -m neural_operator_lab.cli.main config [OPTIONS]
```

#### Options

- `--create`: Create new configuration file
- `--template TEXT`: Configuration template type [default: default]
- `--output PATH`: Output configuration file path
- `--validate PATH`: Validate configuration file
- `--merge PATH PATH`: Merge two configuration files
- `--show-schema`: Show configuration schema

#### Available Templates

- `default`: Default configuration
- `small`: Small model configuration
- `large`: Large model configuration
- `distributed`: Distributed training configuration
- `physics_informed`: Physics-informed training configuration

#### Examples

```bash
# Create default configuration
python -m neural_operator_lab.cli.main config \
    --create \
    --template default \
    --output config.yaml

# Validate configuration
python -m neural_operator_lab.cli.main config \
    --validate config.yaml

# Merge configurations
python -m neural_operator_lab.cli.main config \
    --merge base_config.yaml override_config.yaml \
    --output merged_config.yaml

# Show configuration schema
python -m neural_operator_lab.cli.main config --show-schema
```

### data

Data management and preprocessing utilities.

```bash
python -m neural_operator_lab.cli.main data [OPTIONS]
```

#### Options

- `--preprocess PATH`: Preprocess raw simulation data
- `--format TEXT`: Data format [default: h5]
- `--resolution INTEGER`: Target resolution for preprocessing
- `--normalize`: Normalize data
- `--augment`: Apply data augmentation
- `--split FLOAT FLOAT`: Train/validation split ratios
- `--output-dir PATH`: Output directory for processed data

#### Examples

```bash
# Preprocess simulation data
python -m neural_operator_lab.cli.main data \
    --preprocess /data/raw_simulations/ \
    --format h5 \
    --resolution 256 \
    --normalize \
    --split 0.8 0.2

# Generate augmented dataset
python -m neural_operator_lab.cli.main data \
    --preprocess /data/turbulence/ \
    --augment \
    --output-dir /data/turbulence_augmented/
```

### serve

Serve trained model as REST API.

```bash
python -m neural_operator_lab.cli.main serve [OPTIONS]
```

#### Options

- `--model PATH`: Path to trained model [required]
- `--host TEXT`: Host address [default: 0.0.0.0]
- `--port INTEGER`: Port number [default: 8000]
- `--workers INTEGER`: Number of worker processes [default: 1]
- `--gpu`: Use GPU for inference
- `--batch-size INTEGER`: Maximum batch size for inference [default: 1]

#### Example

```bash
# Serve model API
python -m neural_operator_lab.cli.main serve \
    --model checkpoints/production_model.pt \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu \
    --workers 4
```

### export

Export trained models to different formats.

```bash
python -m neural_operator_lab.cli.main export [OPTIONS]
```

#### Options

- `--model PATH`: Path to trained model [required]
- `--format TEXT`: Export format [required]
- `--output PATH`: Output file path [required]
- `--optimize`: Optimize exported model
- `--quantize`: Apply quantization

#### Available Formats

- `onnx`: ONNX format
- `torchscript`: TorchScript format
- `tensorrt`: TensorRT format (requires TensorRT)
- `coreml`: Core ML format (macOS only)

#### Examples

```bash
# Export to ONNX
python -m neural_operator_lab.cli.main export \
    --model checkpoints/model.pt \
    --format onnx \
    --output model.onnx \
    --optimize

# Export to TorchScript
python -m neural_operator_lab.cli.main export \
    --model checkpoints/model.pt \
    --format torchscript \
    --output model.pt
```

## Configuration Files

The CLI supports hierarchical configuration files in YAML or JSON format.

### Example Configuration

```yaml
# config.yaml
experiment:
  name: "turbulence_modeling"
  description: "High-resolution turbulence simulation"
  tags: ["navier_stokes", "3d", "turbulence"]

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.0005
  optimizer: "adamw"
  scheduler: "cosine"
  mixed_precision: true
  early_stopping_patience: 20

model:
  model_type: "transformer_neural_operator"
  input_dim: 4
  output_dim: 4
  hidden_dim: 256
  num_layers: 8
  num_heads: 8

data:
  train_data_path: "/data/turbulence/train"
  val_data_path: "/data/turbulence/val"
  resolution: [128, 128, 128]
  variables: ["u", "v", "w", "p"]
  normalize: true
  augmentation: true

distributed:
  enabled: true
  backend: "nccl"
  nodes: 2
  gpus_per_node: 8
```

## Environment Variables

The CLI supports configuration through environment variables:

- `NEURAL_OP_CONFIG_PATH`: Default configuration file path
- `NEURAL_OP_DATA_PATH`: Default data directory
- `NEURAL_OP_OUTPUT_PATH`: Default output directory
- `NEURAL_OP_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `NEURAL_OP_CACHE_DIR`: Cache directory for models and data
- `NEURAL_OP_DISTRIBUTED`: Enable distributed training by default
- `NEURAL_OP_GPU_ID`: Default GPU ID to use

## Logging

The CLI provides comprehensive logging with configurable levels:

```bash
# Minimal output
python -m neural_operator_lab.cli.main train

# Verbose output
python -m neural_operator_lab.cli.main train -v

# Debug output with log file
python -m neural_operator_lab.cli.main train -vv --log-file training.log

# Maximum verbosity
python -m neural_operator_lab.cli.main train -vvv
```

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Configuration error
- `3`: Data error
- `4`: Model error
- `130`: Interrupted by user (Ctrl+C)

## Shell Completion

Generate shell completion scripts:

```bash
# Bash completion
python -m neural_operator_lab.cli.main --install-completion bash

# Zsh completion
python -m neural_operator_lab.cli.main --install-completion zsh

# Fish completion
python -m neural_operator_lab.cli.main --install-completion fish
```

## Examples

### Complete Training Pipeline

```bash
#!/bin/bash

# Set environment
export NEURAL_OP_LOG_LEVEL=INFO
export NEURAL_OP_DATA_PATH=/data/pde_datasets

# Create configuration
python -m neural_operator_lab.cli.main config \
    --create --template physics_informed \
    --output physics_config.yaml

# Preprocess data
python -m neural_operator_lab.cli.main data \
    --preprocess $NEURAL_OP_DATA_PATH/raw \
    --resolution 256 \
    --normalize --augment

# Train model
python -m neural_operator_lab.cli.main train \
    --config physics_config.yaml \
    --distributed \
    --mixed-precision \
    --early-stopping

# Evaluate trained model
python -m neural_operator_lab.cli.main evaluate \
    --model checkpoints/best_model.pt \
    --data-path $NEURAL_OP_DATA_PATH/test \
    --rollout-steps 100 \
    --visualization

# Run benchmarks
python -m neural_operator_lab.cli.main benchmark \
    --model checkpoints/best_model.pt \
    --suite comprehensive \
    --generate-report

# Export for production
python -m neural_operator_lab.cli.main export \
    --model checkpoints/best_model.pt \
    --format onnx \
    --output production_model.onnx \
    --optimize
```

### Distributed Training on Multiple Nodes

```bash
# Node 0 (master)
python -m neural_operator_lab.cli.main train \
    --config distributed_config.yaml \
    --distributed \
    --rank 0 \
    --world-size 16 \
    --master-addr node0 \
    --master-port 29500

# Node 1
python -m neural_operator_lab.cli.main train \
    --config distributed_config.yaml \
    --distributed \
    --rank 8 \
    --world-size 16 \
    --master-addr node0 \
    --master-port 29500
```
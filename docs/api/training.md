# Training API Reference

This document provides comprehensive API documentation for the training infrastructure in the Neural Operator Foundation Lab.

## Core Training Classes

### Trainer

The main training class that orchestrates the training process.

```python
class Trainer
```

#### Constructor Parameters

- `model` (NeuralOperatorBase): The neural operator model to train
- `train_loader` (DataLoader): Training data loader
- `val_loader` (DataLoader): Validation data loader
- `optimizer` (torch.optim.Optimizer): Optimizer instance
- `scheduler` (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler
- `loss_fn` (Callable): Loss function
- `device` (torch.device): Training device
- `config` (TrainingConfig): Training configuration

#### Methods

##### `train(self, epochs: int, callbacks: Optional[List[TrainingCallback]] = None) -> Dict[str, List[float]]`

Execute the complete training loop.

**Parameters:**
- `epochs` (int): Number of training epochs
- `callbacks` (Optional[List[TrainingCallback]]): Training callbacks

**Returns:**
- Dict[str, List[float]]: Training history with losses and metrics

**Example:**

```python
from neural_operator_lab.training import Trainer
from neural_operator_lab.training.callbacks import EarlyStopping, ModelCheckpoint

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device
)

callbacks = [
    EarlyStopping(patience=10, monitor='val_loss'),
    ModelCheckpoint(filepath='best_model.pt', save_best_only=True)
]

history = trainer.train(epochs=100, callbacks=callbacks)
```

##### `evaluate(self, data_loader: DataLoader, return_predictions: bool = False) -> Dict[str, float]`

Evaluate model on given dataset.

**Parameters:**
- `data_loader` (DataLoader): Data loader for evaluation
- `return_predictions` (bool): Whether to return predictions

**Returns:**
- Dict[str, float]: Evaluation metrics

##### `save_checkpoint(self, filepath: str, include_optimizer: bool = True)`

Save training checkpoint.

**Parameters:**
- `filepath` (str): Path to save checkpoint
- `include_optimizer` (bool): Whether to include optimizer state

##### `load_checkpoint(self, filepath: str, load_optimizer: bool = True)`

Load training checkpoint.

**Parameters:**
- `filepath` (str): Path to checkpoint file
- `load_optimizer` (bool): Whether to load optimizer state

### DistributedTrainer

Distributed training implementation supporting multiple GPUs and nodes.

```python
class DistributedTrainer(Trainer)
```

#### Constructor Parameters

All parameters from `Trainer` plus:

- `world_size` (int): Total number of processes
- `rank` (int): Rank of current process
- `local_rank` (int): Local rank within node
- `backend` (str): Distributed backend ('nccl', 'gloo')
- `init_method` (str): Initialization method
- `strategy` (ParallelismStrategy): Parallelism strategy

#### Example Usage

```python
from neural_operator_lab.training import DistributedTrainer
from neural_operator_lab.distributed.strategies import create_parallelism_strategy

# Create parallelism strategy
strategy = create_parallelism_strategy(
    config=parallelism_config,
    world_size=world_size
)

# Initialize distributed trainer
trainer = DistributedTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    world_size=world_size,
    rank=rank,
    strategy=strategy
)

trainer.train(epochs=100)
```

## Training Callbacks

### TrainingCallback

Abstract base class for training callbacks.

```python
class TrainingCallback(ABC)
```

#### Abstract Methods

- `on_train_begin(trainer)`: Called at the start of training
- `on_train_end(trainer)`: Called at the end of training
- `on_epoch_begin(trainer, epoch)`: Called at the start of each epoch
- `on_epoch_end(trainer, epoch, train_losses, val_losses)`: Called at the end of each epoch
- `on_batch_begin(trainer, batch_idx, batch)`: Called at the start of each batch
- `on_batch_end(trainer, batch_idx, batch_metrics)`: Called at the end of each batch

### EarlyStopping

Stop training when monitored metric stops improving.

```python
class EarlyStopping(TrainingCallback)
```

#### Constructor Parameters

- `patience` (int): Number of epochs to wait for improvement
- `min_delta` (float): Minimum change to qualify as improvement
- `monitor` (str): Metric to monitor ('val_loss', 'val_accuracy', etc.)
- `mode` (str): 'min' for minimization, 'max' for maximization
- `restore_best_weights` (bool): Whether to restore best weights

#### Example

```python
early_stopping = EarlyStopping(
    patience=10,
    min_delta=1e-4,
    monitor='val_loss',
    mode='min',
    restore_best_weights=True
)
```

### ModelCheckpoint

Save model checkpoints during training.

```python
class ModelCheckpoint(TrainingCallback)
```

#### Constructor Parameters

- `filepath` (str): Checkpoint save path (supports formatting)
- `monitor` (str): Metric to monitor for best model
- `mode` (str): 'min' or 'max'
- `save_best_only` (bool): Only save if model improves
- `save_weights_only` (bool): Save only model weights
- `period` (int): Frequency of checkpoint saving

#### Example

```python
checkpoint = ModelCheckpoint(
    filepath='checkpoints/model_epoch_{epoch:02d}_loss_{val_loss:.4f}.pt',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False,
    period=1
)
```

### MetricsTracker

Track training metrics and generate summaries.

```python
class MetricsTracker(TrainingCallback)
```

#### Methods

##### `get_summary(self) -> Dict[str, Dict[str, float]]`

Get statistical summary of tracked metrics.

##### `reset(self)`

Reset metric history.

##### `get_metric_history(self, metric_name: str) -> List[float]`

Get history for specific metric.

#### Example

```python
metrics_tracker = MetricsTracker()

# After training
summary = metrics_tracker.get_summary()
print(f"Best validation loss: {summary['val_loss']['best']}")
print(f"Average training time: {summary['epoch_time']['mean']:.2f}s")
```

### LearningRateScheduler

Wrapper for PyTorch learning rate schedulers.

```python
class LearningRateScheduler(TrainingCallback)
```

#### Constructor Parameters

- `scheduler` (torch.optim.lr_scheduler._LRScheduler): PyTorch scheduler
- `step_on` (str): When to step ('epoch', 'batch')

#### Example

```python
import torch.optim.lr_scheduler as lr_scheduler

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
lr_callback = LearningRateScheduler(scheduler, step_on='epoch')
```

### MemoryProfiler

Monitor GPU memory usage during training.

```python
class MemoryProfiler(TrainingCallback)
```

#### Methods

##### `get_memory_summary(self) -> Dict[str, float]`

Get memory usage statistics.

#### Example

```python
memory_profiler = MemoryProfiler()

# After training
memory_stats = memory_profiler.get_memory_summary()
print(f"Peak memory usage: {memory_stats['peak_memory_gb']:.2f} GB")
```

### TensorBoardLogger

Log metrics to TensorBoard.

```python
class TensorBoardLogger(TrainingCallback)
```

#### Constructor Parameters

- `log_dir` (str): TensorBoard log directory
- `log_frequency` (int): Logging frequency in batches

#### Example

```python
tb_logger = TensorBoardLogger(
    log_dir='runs/experiment_001',
    log_frequency=100
)
```

## Optimizers and Schedulers

### create_optimizer

Factory function for creating optimizers.

```python
def create_optimizer(
    parameters,
    optimizer_name: str = 'adamw',
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    **kwargs
) -> torch.optim.Optimizer
```

**Parameters:**
- `parameters`: Model parameters
- `optimizer_name` (str): Optimizer type ('adam', 'adamw', 'sgd', 'rmsprop')
- `learning_rate` (float): Learning rate
- `weight_decay` (float): Weight decay
- `**kwargs`: Additional optimizer parameters

**Example:**

```python
from neural_operator_lab.training.optimizers import create_optimizer

optimizer = create_optimizer(
    model.parameters(),
    optimizer_name='adamw',
    learning_rate=1e-4,
    weight_decay=1e-5,
    betas=(0.9, 0.999)
)
```

### create_scheduler

Factory function for creating learning rate schedulers.

```python
def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'cosine',
    warmup_steps: int = 0,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler
```

**Parameters:**
- `optimizer` (torch.optim.Optimizer): Optimizer instance
- `scheduler_name` (str): Scheduler type
- `warmup_steps` (int): Number of warmup steps
- `**kwargs`: Additional scheduler parameters

**Example:**

```python
from neural_operator_lab.training.optimizers import create_scheduler

scheduler = create_scheduler(
    optimizer,
    scheduler_name='cosine',
    warmup_steps=1000,
    T_max=100000
)
```

### Available Optimizers

#### AdamW with Weight Decay

```python
optimizer = create_optimizer(
    model.parameters(),
    optimizer_name='adamw',
    learning_rate=5e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

#### SGD with Momentum

```python
optimizer = create_optimizer(
    model.parameters(),
    optimizer_name='sgd',
    learning_rate=1e-2,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)
```

#### AdaBound

Combines benefits of Adam and SGD.

```python
optimizer = create_optimizer(
    model.parameters(),
    optimizer_name='adabound',
    learning_rate=1e-3,
    final_lr=0.1,
    gamma=1e-3
)
```

### Available Schedulers

#### Cosine Annealing with Warm Restarts

```python
scheduler = create_scheduler(
    optimizer,
    scheduler_name='cosine_restarts',
    T_0=10,
    T_mult=2,
    eta_min=1e-6
)
```

#### Linear Warmup + Cosine Decay

```python
scheduler = create_scheduler(
    optimizer,
    scheduler_name='cosine',
    warmup_steps=2000,
    T_max=100000,
    eta_min=1e-6
)
```

#### Reduce on Plateau

```python
scheduler = create_scheduler(
    optimizer,
    scheduler_name='plateau',
    mode='min',
    factor=0.5,
    patience=10,
    threshold=1e-4
)
```

## Loss Functions

### PhysicsInformedLoss

Combined data and physics loss for physics-informed training.

```python
class PhysicsInformedLoss(nn.Module)
```

#### Constructor Parameters

- `data_weight` (float): Weight for data loss term
- `pde_weight` (float): Weight for PDE residual loss
- `boundary_weight` (float): Weight for boundary condition loss
- `conservation_weight` (float): Weight for conservation law loss

#### Methods

##### `set_pde_residual(self, residual_fn: Callable)`

Set PDE residual function.

##### `set_boundary_conditions(self, boundary_fn: Callable)`

Set boundary condition function.

#### Example

```python
from neural_operator_lab.training.losses import PhysicsInformedLoss

def navier_stokes_residual(u, coords):
    # Compute Navier-Stokes residual
    return residual

loss_fn = PhysicsInformedLoss(
    data_weight=1.0,
    pde_weight=0.1,
    boundary_weight=0.05
)
loss_fn.set_pde_residual(navier_stokes_residual)
```

### ConservationLoss

Enforce conservation laws during training.

```python
class ConservationLoss(nn.Module)
```

#### Constructor Parameters

- `conservation_types` (List[str]): Types of conservation to enforce
- `weights` (Dict[str, float]): Weights for each conservation type

#### Example

```python
conservation_loss = ConservationLoss(
    conservation_types=['mass', 'momentum', 'energy'],
    weights={'mass': 1.0, 'momentum': 0.5, 'energy': 0.1}
)
```

## Training Configuration

### TrainingConfig

Configuration dataclass for training parameters.

```python
@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 20
    save_frequency: int = 10
    val_frequency: int = 1
```

### Example Training Script

```python
#!/usr/bin/env python3

import torch
from neural_operator_lab.models import create_model
from neural_operator_lab.training import Trainer
from neural_operator_lab.training.callbacks import *
from neural_operator_lab.training.optimizers import create_optimizer, create_scheduler
from neural_operator_lab.config import Config

def main():
    # Load configuration
    config = Config.load('config.yaml')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(config.model.model_type, config.model.__dict__)
    model = model.to(device)
    
    # Create data loaders
    train_loader = create_data_loader(config.data, split='train')
    val_loader = create_data_loader(config.data, split='val')
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model.parameters(),
        config.training.optimizer,
        config.training.learning_rate,
        config.training.weight_decay
    )
    
    scheduler = create_scheduler(
        optimizer,
        config.training.scheduler,
        config.training.warmup_steps
    )
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            patience=config.training.early_stopping_patience,
            monitor='val_loss'
        ),
        ModelCheckpoint(
            filepath='best_model.pt',
            monitor='val_loss',
            save_best_only=True
        ),
        MetricsTracker(),
        TensorBoardLogger(log_dir='runs/experiment'),
        MemoryProfiler()
    ]
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # Train model
    history = trainer.train(
        epochs=config.training.epochs,
        callbacks=callbacks
    )
    
    print("Training completed!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")

if __name__ == "__main__":
    main()
```
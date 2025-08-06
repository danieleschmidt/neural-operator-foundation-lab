"""Advanced distributed training strategies for neural operators."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass

from ..base import NeuralOperatorBase


@dataclass
class ParallelismConfig:
    """Configuration for different parallelism strategies."""
    
    # Data parallelism
    data_parallel: bool = True
    gradient_accumulation_steps: int = 1
    
    # Model parallelism
    model_parallel: bool = False
    model_parallel_size: int = 1
    
    # Pipeline parallelism
    pipeline_parallel: bool = False
    pipeline_parallel_size: int = 1
    num_microbatches: int = 4
    
    # Tensor parallelism
    tensor_parallel: bool = False
    tensor_parallel_size: int = 1
    
    # Optimization
    activation_checkpointing: bool = True
    cpu_offloading: bool = False


class ParallelismStrategy(ABC):
    """Abstract base class for parallelism strategies."""
    
    def __init__(self, config: ParallelismConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def setup_model(self, model: NeuralOperatorBase) -> NeuralOperatorBase:
        """Setup model for this parallelism strategy."""
        pass
    
    @abstractmethod
    def setup_optimizer(self, optimizer) -> Any:
        """Setup optimizer for this parallelism strategy."""
        pass
    
    @abstractmethod
    def forward_backward(self, model, batch, loss_fn) -> Dict[str, torch.Tensor]:
        """Execute forward and backward pass."""
        pass


class DataParallelStrategy(ParallelismStrategy):
    """Data parallelism strategy with gradient synchronization."""
    
    def __init__(self, config: ParallelismConfig, world_size: int = 1):
        super().__init__(config)
        self.world_size = world_size
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
    
    def setup_model(self, model: NeuralOperatorBase) -> NeuralOperatorBase:
        """Setup model for data parallelism."""
        if self.world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            
            model = DDP(
                model,
                find_unused_parameters=True,
                gradient_as_bucket_view=True
            )
            
            self.logger.info(f"Setup DistributedDataParallel with {self.world_size} processes")
        
        return model
    
    def setup_optimizer(self, optimizer) -> Any:
        """Setup optimizer for data parallelism."""
        # Scale learning rate by world size for large batch training
        if self.world_size > 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.world_size
            
            self.logger.info(f"Scaled learning rate by world size: {self.world_size}")
        
        return optimizer
    
    def forward_backward(self, model, batch, loss_fn) -> Dict[str, torch.Tensor]:
        """Execute data parallel forward and backward pass."""
        inputs, targets = batch[:2]
        grid = batch[2] if len(batch) > 2 else None
        
        # Forward pass
        predictions = model(inputs, grid)
        
        # Compute loss
        if hasattr(loss_fn, '__call__'):
            if len(batch) > 2:  # Has grid
                loss_dict = loss_fn(predictions, targets, grid, inputs)
            else:
                loss_dict = loss_fn(predictions, targets)
        else:
            loss_dict = {'total_loss': loss_fn(predictions, targets)}
        
        # Scale loss for gradient accumulation
        if self.gradient_accumulation_steps > 1:
            for key in loss_dict:
                loss_dict[key] = loss_dict[key] / self.gradient_accumulation_steps
        
        return loss_dict


class ModelParallelStrategy(ParallelismStrategy):
    """Model parallelism strategy for large models."""
    
    def __init__(self, config: ParallelismConfig, model_parallel_size: int):
        super().__init__(config)
        self.model_parallel_size = model_parallel_size
        self.device_map = self._create_device_map()
    
    def _create_device_map(self) -> Dict[str, int]:
        """Create device mapping for model parallel layers."""
        if not torch.cuda.is_available():
            return {}
        
        num_devices = min(torch.cuda.device_count(), self.model_parallel_size)
        device_map = {}
        
        # Simple round-robin device assignment
        for i in range(self.model_parallel_size):
            device_map[f'layer_{i}'] = i % num_devices
        
        return device_map
    
    def setup_model(self, model: NeuralOperatorBase) -> NeuralOperatorBase:
        """Setup model for model parallelism."""
        
        # Split model layers across devices
        if hasattr(model, 'blocks') and self.device_map:
            layers_per_device = len(model.blocks) // len(self.device_map)
            
            for i, block in enumerate(model.blocks):
                device_id = i // layers_per_device
                device_id = min(device_id, len(self.device_map) - 1)
                
                target_device = list(self.device_map.values())[device_id]
                block.to(f'cuda:{target_device}')
                
                self.logger.info(f"Moved block {i} to device {target_device}")
        
        return model
    
    def setup_optimizer(self, optimizer) -> Any:
        """Setup optimizer for model parallelism."""
        # No special optimizer setup needed for model parallelism
        return optimizer
    
    def forward_backward(self, model, batch, loss_fn) -> Dict[str, torch.Tensor]:
        """Execute model parallel forward and backward pass."""
        inputs, targets = batch[:2]
        grid = batch[2] if len(batch) > 2 else None
        
        # Move inputs through devices during forward pass
        current_device = 0
        x = inputs.to(f'cuda:{current_device}')
        
        if hasattr(model, 'blocks'):
            # Process through model parallel layers
            layers_per_device = len(model.blocks) // len(self.device_map)
            
            for i, block in enumerate(model.blocks):
                target_device = i // layers_per_device
                target_device = min(target_device, len(self.device_map) - 1)
                
                if target_device != current_device:
                    x = x.to(f'cuda:{target_device}')
                    current_device = target_device
                
                x = block(x)
        
        # Final output processing
        if hasattr(model, 'output_projection'):
            x = model.output_projection(x)
        
        predictions = x
        
        # Move targets to same device as predictions
        targets = targets.to(predictions.device)
        
        # Compute loss
        if hasattr(loss_fn, '__call__'):
            loss_dict = loss_fn(predictions, targets)
        else:
            loss_dict = {'total_loss': loss_fn(predictions, targets)}
        
        return loss_dict


class PipelineParallelStrategy(ParallelismStrategy):
    """Pipeline parallelism strategy for memory efficiency."""
    
    def __init__(self, config: ParallelismConfig, pipeline_size: int, num_microbatches: int = 4):
        super().__init__(config)
        self.pipeline_size = pipeline_size
        self.num_microbatches = num_microbatches
        self.pipeline_stages = []
    
    def setup_model(self, model: NeuralOperatorBase) -> NeuralOperatorBase:
        """Setup model for pipeline parallelism."""
        
        # Split model into pipeline stages
        if hasattr(model, 'blocks'):
            blocks_per_stage = len(model.blocks) // self.pipeline_size
            
            for stage in range(self.pipeline_size):
                start_idx = stage * blocks_per_stage
                end_idx = start_idx + blocks_per_stage if stage < self.pipeline_size - 1 else len(model.blocks)
                
                stage_blocks = model.blocks[start_idx:end_idx]
                stage_model = PipelineStage(stage_blocks, stage)
                
                # Move to appropriate device
                device_id = stage % torch.cuda.device_count() if torch.cuda.is_available() else 0
                stage_model.to(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
                
                self.pipeline_stages.append(stage_model)
                
                self.logger.info(f"Created pipeline stage {stage} with {len(stage_blocks)} blocks on device {device_id}")
        
        return model
    
    def setup_optimizer(self, optimizer) -> Any:
        """Setup optimizer for pipeline parallelism."""
        # Create separate optimizers for each pipeline stage
        stage_optimizers = []
        
        for stage in self.pipeline_stages:
            stage_optimizer = type(optimizer)(
                stage.parameters(),
                **optimizer.defaults
            )
            stage_optimizers.append(stage_optimizer)
        
        return stage_optimizers
    
    def forward_backward(self, model, batch, loss_fn) -> Dict[str, torch.Tensor]:
        """Execute pipeline parallel forward and backward pass."""
        inputs, targets = batch[:2]
        
        # Split batch into microbatches
        microbatch_size = inputs.size(0) // self.num_microbatches
        microbatches = []
        
        for i in range(self.num_microbatches):
            start_idx = i * microbatch_size
            end_idx = start_idx + microbatch_size
            
            microbatch_input = inputs[start_idx:end_idx]
            microbatch_target = targets[start_idx:end_idx]
            
            microbatches.append((microbatch_input, microbatch_target))
        
        # Pipeline execution
        total_loss = 0.0
        
        for microbatch_input, microbatch_target in microbatches:
            # Forward pass through pipeline stages
            x = microbatch_input
            
            for stage in self.pipeline_stages:
                x = x.to(stage.device)
                x = stage(x)
            
            predictions = x
            microbatch_target = microbatch_target.to(predictions.device)
            
            # Compute loss for microbatch
            if hasattr(loss_fn, '__call__'):
                microbatch_loss_dict = loss_fn(predictions, microbatch_target)
                microbatch_loss = microbatch_loss_dict.get('total_loss', microbatch_loss_dict)
            else:
                microbatch_loss = loss_fn(predictions, microbatch_target)
            
            # Backward pass through pipeline stages
            microbatch_loss.backward()
            total_loss += microbatch_loss.item()
        
        # Average loss across microbatches
        avg_loss = total_loss / self.num_microbatches
        
        return {'total_loss': torch.tensor(avg_loss)}


class PipelineStage(nn.Module):
    """A single stage in pipeline parallelism."""
    
    def __init__(self, blocks: nn.ModuleList, stage_id: int):
        super().__init__()
        self.blocks = blocks
        self.stage_id = stage_id
        self.device = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through pipeline stage."""
        for block in self.blocks:
            x = block(x)
        return x
    
    def to(self, device):
        """Move stage to device."""
        super().to(device)
        self.device = device
        return self


class HybridParallelStrategy(ParallelismStrategy):
    """Hybrid parallelism combining multiple strategies."""
    
    def __init__(self, config: ParallelismConfig, 
                 data_parallel_size: int = 1,
                 model_parallel_size: int = 1,
                 pipeline_parallel_size: int = 1):
        super().__init__(config)
        
        self.data_parallel_size = data_parallel_size
        self.model_parallel_size = model_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        
        # Initialize sub-strategies
        self.strategies = []
        
        if config.data_parallel and data_parallel_size > 1:
            self.strategies.append(
                DataParallelStrategy(config, data_parallel_size)
            )
        
        if config.model_parallel and model_parallel_size > 1:
            self.strategies.append(
                ModelParallelStrategy(config, model_parallel_size)
            )
        
        if config.pipeline_parallel and pipeline_parallel_size > 1:
            self.strategies.append(
                PipelineParallelStrategy(config, pipeline_parallel_size)
            )
    
    def setup_model(self, model: NeuralOperatorBase) -> NeuralOperatorBase:
        """Setup model for hybrid parallelism."""
        
        # Apply strategies in order: pipeline -> model -> data
        for strategy in reversed(self.strategies):
            model = strategy.setup_model(model)
        
        self.logger.info(
            f"Setup hybrid parallelism: "
            f"data={self.data_parallel_size}, "
            f"model={self.model_parallel_size}, "
            f"pipeline={self.pipeline_parallel_size}"
        )
        
        return model
    
    def setup_optimizer(self, optimizer) -> Any:
        """Setup optimizer for hybrid parallelism."""
        
        # Apply optimizer setup from each strategy
        for strategy in self.strategies:
            optimizer = strategy.setup_optimizer(optimizer)
        
        return optimizer
    
    def forward_backward(self, model, batch, loss_fn) -> Dict[str, torch.Tensor]:
        """Execute hybrid parallel forward and backward pass."""
        
        # Use the most complex strategy available
        if self.strategies:
            return self.strategies[-1].forward_backward(model, batch, loss_fn)
        else:
            # Fallback to simple forward-backward
            inputs, targets = batch[:2]
            grid = batch[2] if len(batch) > 2 else None
            
            predictions = model(inputs, grid)
            
            if hasattr(loss_fn, '__call__'):
                loss_dict = loss_fn(predictions, targets)
            else:
                loss_dict = {'total_loss': loss_fn(predictions, targets)}
            
            return loss_dict


def create_parallelism_strategy(config: ParallelismConfig,
                               world_size: int = 1,
                               **kwargs) -> ParallelismStrategy:
    """Factory function to create appropriate parallelism strategy."""
    
    # Determine optimal parallelism configuration
    total_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    data_parallel_size = kwargs.get('data_parallel_size', world_size)
    model_parallel_size = kwargs.get('model_parallel_size', 1)
    pipeline_parallel_size = kwargs.get('pipeline_parallel_size', 1)
    
    # Validate configuration
    total_parallelism = data_parallel_size * model_parallel_size * pipeline_parallel_size
    if total_parallelism > total_devices:
        logging.warning(
            f"Total parallelism {total_parallelism} exceeds available devices {total_devices}. "
            f"Adjusting configuration."
        )
        
        # Auto-adjust to fit available devices
        data_parallel_size = min(data_parallel_size, total_devices)
        model_parallel_size = 1
        pipeline_parallel_size = 1
    
    # Create strategy based on configuration
    if (data_parallel_size > 1 and 
        (model_parallel_size > 1 or pipeline_parallel_size > 1)):
        # Hybrid parallelism
        return HybridParallelStrategy(
            config, data_parallel_size, model_parallel_size, pipeline_parallel_size
        )
    elif model_parallel_size > 1:
        return ModelParallelStrategy(config, model_parallel_size)
    elif pipeline_parallel_size > 1:
        return PipelineParallelStrategy(config, pipeline_parallel_size)
    else:
        return DataParallelStrategy(config, data_parallel_size)
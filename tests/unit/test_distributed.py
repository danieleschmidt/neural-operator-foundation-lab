"""Tests for distributed training functionality."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from neural_operator_lab.distributed.strategies import (
    ParallelismConfig, DataParallelStrategy, ModelParallelStrategy,
    PipelineParallelStrategy, HybridParallelStrategy, 
    create_parallelism_strategy, PipelineStage
)


class MockNeuralOperator(nn.Module):
    """Mock neural operator for distributed testing."""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_blocks=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Create mock blocks for pipeline parallelism
        self.blocks = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_blocks)
        ])
        
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, grid=None):
        """Forward pass through mock operator."""
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        for block in self.blocks:
            x_flat = torch.relu(block(x_flat))
        
        output = self.output_projection(x_flat)
        return output.view_as(x)


class TestParallelismConfig:
    """Test parallelism configuration."""
    
    def test_default_config(self):
        """Test default parallelism configuration."""
        config = ParallelismConfig()
        
        assert config.data_parallel is True
        assert config.model_parallel is False
        assert config.pipeline_parallel is False
        assert config.gradient_accumulation_steps == 1
        assert config.activation_checkpointing is True
        assert config.cpu_offloading is False
    
    def test_custom_config(self):
        """Test custom parallelism configuration."""
        config = ParallelismConfig(
            model_parallel=True,
            model_parallel_size=2,
            pipeline_parallel=True,
            pipeline_parallel_size=4,
            cpu_offloading=True
        )
        
        assert config.model_parallel is True
        assert config.model_parallel_size == 2
        assert config.pipeline_parallel is True
        assert config.pipeline_parallel_size == 4
        assert config.cpu_offloading is True


class TestDataParallelStrategy:
    """Test data parallel strategy."""
    
    def test_initialization(self):
        """Test data parallel strategy initialization."""
        config = ParallelismConfig()
        strategy = DataParallelStrategy(config, world_size=4)
        
        assert strategy.world_size == 4
        assert strategy.gradient_accumulation_steps == 1
    
    def test_setup_model_single_gpu(self):
        """Test model setup for single GPU."""
        config = ParallelismConfig()
        strategy = DataParallelStrategy(config, world_size=1)
        model = MockNeuralOperator()
        
        # Should return original model for single GPU
        setup_model = strategy.setup_model(model)
        assert setup_model is model
    
    @patch('torch.nn.parallel.DistributedDataParallel')
    def test_setup_model_multi_gpu(self, mock_ddp):
        """Test model setup for multi-GPU."""
        config = ParallelismConfig()
        strategy = DataParallelStrategy(config, world_size=4)
        model = MockNeuralOperator()
        
        mock_ddp.return_value = model  # Mock DDP wrapper
        
        setup_model = strategy.setup_model(model)
        
        # Should wrap model with DDP
        mock_ddp.assert_called_once_with(
            model,
            find_unused_parameters=True,
            gradient_as_bucket_view=True
        )
    
    def test_setup_optimizer(self):
        """Test optimizer setup with learning rate scaling."""
        config = ParallelismConfig()
        strategy = DataParallelStrategy(config, world_size=4)
        
        model = MockNeuralOperator()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        original_lr = optimizer.param_groups[0]['lr']
        
        setup_optimizer = strategy.setup_optimizer(optimizer)
        
        # Learning rate should be scaled by world size
        assert setup_optimizer.param_groups[0]['lr'] == original_lr * 4
    
    def test_forward_backward(self):
        """Test forward-backward pass."""
        config = ParallelismConfig()
        strategy = DataParallelStrategy(config, world_size=1)
        
        model = MockNeuralOperator()
        batch = (torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8))
        
        def mock_loss_fn(pred, target):
            return {'total_loss': nn.MSELoss()(pred, target)}
        
        loss_dict = strategy.forward_backward(model, batch, mock_loss_fn)
        
        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss']).all()
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation scaling."""
        config = ParallelismConfig(gradient_accumulation_steps=4)
        strategy = DataParallelStrategy(config, world_size=1)
        
        model = MockNeuralOperator()
        batch = (torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8))
        
        def mock_loss_fn(pred, target):
            return {'total_loss': nn.MSELoss()(pred, target)}
        
        loss_dict = strategy.forward_backward(model, batch, mock_loss_fn)
        
        # Loss should be scaled by accumulation steps
        assert torch.isfinite(loss_dict['total_loss']).all()


class TestModelParallelStrategy:
    """Test model parallel strategy."""
    
    def test_initialization(self):
        """Test model parallel strategy initialization."""
        config = ParallelismConfig()
        strategy = ModelParallelStrategy(config, model_parallel_size=2)
        
        assert strategy.model_parallel_size == 2
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_device_map_creation(self, mock_device_count, mock_cuda_available):
        """Test device mapping creation."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 4
        
        config = ParallelismConfig()
        strategy = ModelParallelStrategy(config, model_parallel_size=2)
        
        assert len(strategy.device_map) == 2
        assert all(isinstance(v, int) for v in strategy.device_map.values())
    
    @patch('torch.cuda.is_available')
    def test_setup_model_no_cuda(self, mock_cuda_available):
        """Test model setup without CUDA."""
        mock_cuda_available.return_value = False
        
        config = ParallelismConfig()
        strategy = ModelParallelStrategy(config, model_parallel_size=2)
        model = MockNeuralOperator()
        
        # Should return original model if no CUDA
        setup_model = strategy.setup_model(model)
        assert setup_model is model
    
    def test_forward_backward(self):
        """Test model parallel forward-backward pass."""
        config = ParallelismConfig()
        strategy = ModelParallelStrategy(config, model_parallel_size=2)
        
        model = MockNeuralOperator()
        batch = (torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8))
        
        def mock_loss_fn(pred, target):
            return {'total_loss': nn.MSELoss()(pred, target)}
        
        loss_dict = strategy.forward_backward(model, batch, mock_loss_fn)
        
        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss']).all()


class TestPipelineParallelStrategy:
    """Test pipeline parallel strategy."""
    
    def test_initialization(self):
        """Test pipeline parallel strategy initialization."""
        config = ParallelismConfig()
        strategy = PipelineParallelStrategy(config, pipeline_size=4, num_microbatches=8)
        
        assert strategy.pipeline_size == 4
        assert strategy.num_microbatches == 8
        assert len(strategy.pipeline_stages) == 0
    
    def test_setup_model(self):
        """Test pipeline model setup."""
        config = ParallelismConfig()
        strategy = PipelineParallelStrategy(config, pipeline_size=2)
        model = MockNeuralOperator(num_blocks=4)
        
        setup_model = strategy.setup_model(model)
        
        # Should create pipeline stages
        assert len(strategy.pipeline_stages) == 2
        assert all(isinstance(stage, PipelineStage) for stage in strategy.pipeline_stages)
    
    def test_setup_optimizer(self):
        """Test pipeline optimizer setup."""
        config = ParallelismConfig()
        strategy = PipelineParallelStrategy(config, pipeline_size=2)
        
        # Create mock stages
        stage1 = Mock()
        stage1.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        stage2 = Mock()
        stage2.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        strategy.pipeline_stages = [stage1, stage2]
        
        optimizer = torch.optim.Adam([torch.randn(5, 5, requires_grad=True)], lr=1e-4)
        
        stage_optimizers = strategy.setup_optimizer(optimizer)
        
        # Should return list of optimizers for each stage
        assert isinstance(stage_optimizers, list)
        assert len(stage_optimizers) == 2
    
    def test_forward_backward_microbatching(self):
        """Test pipeline forward-backward with microbatching."""
        config = ParallelismConfig()
        strategy = PipelineParallelStrategy(config, pipeline_size=2, num_microbatches=4)
        
        # Mock pipeline stages
        stage1 = Mock()
        stage2 = Mock()
        stage1.device = torch.device('cpu')
        stage2.device = torch.device('cpu')
        
        # Mock forward passes
        def stage1_forward(x):
            return x * 2
        
        def stage2_forward(x):
            return x + 1
        
        stage1.side_effect = stage1_forward
        stage2.side_effect = stage2_forward
        strategy.pipeline_stages = [stage1, stage2]
        
        model = MockNeuralOperator()
        batch = (torch.randn(8, 3, 8, 8), torch.randn(8, 3, 8, 8))  # Divisible by 4 microbatches
        
        def mock_loss_fn(pred, target):
            return torch.tensor(1.0)
        
        loss_dict = strategy.forward_backward(model, batch, mock_loss_fn)
        
        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss']).all()


class TestPipelineStage:
    """Test pipeline stage functionality."""
    
    def test_initialization(self):
        """Test pipeline stage initialization."""
        blocks = nn.ModuleList([nn.Linear(10, 10), nn.Linear(10, 10)])
        stage = PipelineStage(blocks, stage_id=0)
        
        assert stage.stage_id == 0
        assert len(stage.blocks) == 2
        assert stage.device is None
    
    def test_forward_pass(self):
        """Test pipeline stage forward pass."""
        blocks = nn.ModuleList([nn.Linear(10, 10), nn.ReLU()])
        stage = PipelineStage(blocks, stage_id=0)
        
        input_tensor = torch.randn(2, 10)
        output = stage(input_tensor)
        
        assert output.shape == input_tensor.shape
        assert torch.isfinite(output).all()
    
    def test_device_placement(self):
        """Test pipeline stage device placement."""
        blocks = nn.ModuleList([nn.Linear(10, 10)])
        stage = PipelineStage(blocks, stage_id=0)
        
        device = torch.device('cpu')
        stage_on_device = stage.to(device)
        
        assert stage_on_device.device == device
        assert stage_on_device is stage  # Should return self


class TestHybridParallelStrategy:
    """Test hybrid parallel strategy."""
    
    def test_initialization(self):
        """Test hybrid parallel strategy initialization."""
        config = ParallelismConfig(
            data_parallel=True,
            model_parallel=True,
            pipeline_parallel=True
        )
        strategy = HybridParallelStrategy(
            config,
            data_parallel_size=2,
            model_parallel_size=2,
            pipeline_parallel_size=2
        )
        
        assert strategy.data_parallel_size == 2
        assert strategy.model_parallel_size == 2
        assert strategy.pipeline_parallel_size == 2
        assert len(strategy.strategies) == 3  # All strategies enabled
    
    def test_single_strategy(self):
        """Test hybrid strategy with single strategy."""
        config = ParallelismConfig(data_parallel=True)
        strategy = HybridParallelStrategy(config, data_parallel_size=4)
        
        assert len(strategy.strategies) == 1
        assert isinstance(strategy.strategies[0], DataParallelStrategy)
    
    def test_setup_model(self):
        """Test hybrid model setup."""
        config = ParallelismConfig(data_parallel=True)
        strategy = HybridParallelStrategy(config, data_parallel_size=2)
        
        model = MockNeuralOperator()
        setup_model = strategy.setup_model(model)
        
        # Should apply strategies in reverse order
        assert setup_model is not None
    
    def test_fallback_forward_backward(self):
        """Test fallback forward-backward when no strategies."""
        config = ParallelismConfig(
            data_parallel=False,
            model_parallel=False,
            pipeline_parallel=False
        )
        strategy = HybridParallelStrategy(config)
        
        model = MockNeuralOperator()
        batch = (torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8))
        
        def mock_loss_fn(pred, target):
            return {'total_loss': nn.MSELoss()(pred, target)}
        
        loss_dict = strategy.forward_backward(model, batch, mock_loss_fn)
        
        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss']).all()


class TestStrategyFactory:
    """Test parallelism strategy factory."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_create_data_parallel_strategy(self, mock_device_count, mock_cuda_available):
        """Test creating data parallel strategy."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 4
        
        config = ParallelismConfig()
        strategy = create_parallelism_strategy(config, world_size=4)
        
        assert isinstance(strategy, DataParallelStrategy)
        assert strategy.world_size == 4
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_create_hybrid_strategy(self, mock_device_count, mock_cuda_available):
        """Test creating hybrid strategy."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 8
        
        config = ParallelismConfig()
        strategy = create_parallelism_strategy(
            config,
            world_size=4,
            data_parallel_size=2,
            model_parallel_size=2
        )
        
        assert isinstance(strategy, HybridParallelStrategy)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_auto_adjust_configuration(self, mock_device_count, mock_cuda_available):
        """Test automatic configuration adjustment."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2  # Limited devices
        
        config = ParallelismConfig()
        
        with patch('logging.warning') as mock_warning:
            strategy = create_parallelism_strategy(
                config,
                world_size=1,
                data_parallel_size=4,  # Exceeds available devices
                model_parallel_size=4
            )
            
            # Should warn about adjusting configuration
            mock_warning.assert_called()
        
        # Should fall back to data parallel only
        assert isinstance(strategy, DataParallelStrategy)
    
    def test_create_model_parallel_strategy(self):
        """Test creating model parallel strategy."""
        config = ParallelismConfig()
        strategy = create_parallelism_strategy(
            config,
            world_size=1,
            model_parallel_size=2
        )
        
        assert isinstance(strategy, ModelParallelStrategy)
    
    def test_create_pipeline_parallel_strategy(self):
        """Test creating pipeline parallel strategy."""
        config = ParallelismConfig()
        strategy = create_parallelism_strategy(
            config,
            world_size=1,
            pipeline_parallel_size=4
        )
        
        assert isinstance(strategy, PipelineParallelStrategy)
"""Tests for training callbacks."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from neural_operator_lab.training.callbacks import (
    TrainingCallback, EarlyStopping, ModelCheckpoint, 
    MetricsTracker, LearningRateScheduler, GradientClipping,
    MemoryProfiler, ConsoleLogger, TensorBoardLogger,
    CallbackManager
)


class MockTrainer:
    """Mock trainer for testing callbacks."""
    
    def __init__(self):
        self.model = Mock()
        self.optimizer = Mock()
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.metrics = {}
        self.config = Mock()
        self.logger = Mock()


class TestTrainingCallback:
    """Test base training callback functionality."""
    
    def test_abstract_callback(self):
        """Test that TrainingCallback is abstract."""
        with pytest.raises(TypeError):
            TrainingCallback()
    
    def test_callback_interface(self):
        """Test callback interface methods."""
        
        class DummyCallback(TrainingCallback):
            def on_train_begin(self, trainer):
                pass
        
        callback = DummyCallback()
        trainer = MockTrainer()
        
        # All methods should be callable without errors
        callback.on_train_begin(trainer)
        callback.on_train_end(trainer)
        callback.on_epoch_begin(trainer, 0)
        callback.on_epoch_end(trainer, 0, {'loss': 1.0}, {'loss': 1.5})
        callback.on_batch_begin(trainer, 0, {})
        callback.on_batch_end(trainer, 0, {})


class TestEarlyStopping:
    """Test early stopping callback."""
    
    def test_initialization(self):
        """Test early stopping initialization."""
        callback = EarlyStopping(patience=5, min_delta=0.01, monitor='val_loss')
        
        assert callback.patience == 5
        assert callback.min_delta == 0.01
        assert callback.monitor == 'val_loss'
        assert callback.mode == 'min'
        assert callback.best_score is None
        assert callback.patience_counter == 0
        assert callback.early_stop is False
    
    def test_mode_validation(self):
        """Test mode validation."""
        # Valid modes
        EarlyStopping(mode='min')
        EarlyStopping(mode='max')
        
        # Invalid mode
        with pytest.raises(ValueError):
            EarlyStopping(mode='invalid')
    
    def test_improvement_detection_min_mode(self):
        """Test improvement detection in min mode."""
        callback = EarlyStopping(patience=3, monitor='val_loss', mode='min')
        trainer = MockTrainer()
        
        # First epoch - should improve
        callback.on_epoch_end(trainer, 0, {}, {'val_loss': 1.0})
        assert callback.best_score == 1.0
        assert callback.patience_counter == 0
        assert not callback.early_stop
        
        # Second epoch - improvement
        callback.on_epoch_end(trainer, 1, {}, {'val_loss': 0.8})
        assert callback.best_score == 0.8
        assert callback.patience_counter == 0
        assert not callback.early_stop
        
        # Third epoch - no improvement
        callback.on_epoch_end(trainer, 2, {}, {'val_loss': 0.9})
        assert callback.best_score == 0.8
        assert callback.patience_counter == 1
        assert not callback.early_stop
    
    def test_early_stopping_trigger(self):
        """Test early stopping trigger."""
        callback = EarlyStopping(patience=2, monitor='val_loss', mode='min')
        trainer = MockTrainer()
        
        # Initial improvement
        callback.on_epoch_end(trainer, 0, {}, {'val_loss': 1.0})
        
        # No improvement for patience epochs
        callback.on_epoch_end(trainer, 1, {}, {'val_loss': 1.1})
        callback.on_epoch_end(trainer, 2, {}, {'val_loss': 1.2})
        
        # Should trigger early stopping
        assert callback.early_stop
        assert callback.patience_counter == 2
    
    def test_max_mode(self):
        """Test early stopping in max mode."""
        callback = EarlyStopping(patience=2, monitor='val_acc', mode='max')
        trainer = MockTrainer()
        
        # First epoch
        callback.on_epoch_end(trainer, 0, {}, {'val_acc': 0.8})
        assert callback.best_score == 0.8
        
        # Improvement
        callback.on_epoch_end(trainer, 1, {}, {'val_acc': 0.9})
        assert callback.best_score == 0.9
        assert callback.patience_counter == 0
        
        # No improvement
        callback.on_epoch_end(trainer, 2, {}, {'val_acc': 0.85})
        assert callback.patience_counter == 1
    
    def test_missing_metric(self):
        """Test handling of missing metric."""
        callback = EarlyStopping(patience=2, monitor='missing_metric')
        trainer = MockTrainer()
        
        # Should not crash when metric is missing
        callback.on_epoch_end(trainer, 0, {}, {'val_loss': 1.0})
        assert callback.patience_counter == 1  # Should increment when metric missing
    
    def test_min_delta_threshold(self):
        """Test minimum delta threshold."""
        callback = EarlyStopping(patience=2, min_delta=0.1, monitor='val_loss', mode='min')
        trainer = MockTrainer()
        
        # Initial score
        callback.on_epoch_end(trainer, 0, {}, {'val_loss': 1.0})
        
        # Small improvement (below min_delta)
        callback.on_epoch_end(trainer, 1, {}, {'val_loss': 0.95})
        assert callback.patience_counter == 1  # Should not count as improvement
        
        # Large improvement (above min_delta)
        callback.on_epoch_end(trainer, 2, {}, {'val_loss': 0.8})
        assert callback.patience_counter == 0  # Should reset counter


class TestModelCheckpoint:
    """Test model checkpoint callback."""
    
    def test_initialization(self, tmp_path):
        """Test model checkpoint initialization."""
        callback = ModelCheckpoint(
            filepath=tmp_path / "checkpoint.pt",
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        assert callback.filepath == tmp_path / "checkpoint.pt"
        assert callback.monitor == 'val_loss'
        assert callback.save_best_only is True
        assert callback.mode == 'min'
        assert callback.best_score is None
    
    def test_save_best_only(self, tmp_path):
        """Test saving only best models."""
        filepath = tmp_path / "best_model.pt"
        callback = ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        trainer = MockTrainer()
        trainer.model.state_dict.return_value = {'weight': torch.randn(5, 5)}
        
        # First epoch - should save
        callback.on_epoch_end(trainer, 0, {}, {'val_loss': 1.0})
        assert filepath.exists()
        assert callback.best_score == 1.0
        
        # Worse performance - should not save
        filepath.unlink()  # Remove file
        callback.on_epoch_end(trainer, 1, {}, {'val_loss': 1.5})
        assert not filepath.exists()
        
        # Better performance - should save
        callback.on_epoch_end(trainer, 2, {}, {'val_loss': 0.8})
        assert filepath.exists()
        assert callback.best_score == 0.8
    
    def test_save_all(self, tmp_path):
        """Test saving all checkpoints."""
        callback = ModelCheckpoint(
            filepath=tmp_path / "model_{epoch}.pt",
            save_best_only=False
        )
        
        trainer = MockTrainer()
        trainer.model.state_dict.return_value = {'weight': torch.randn(5, 5)}
        
        # Should save every epoch
        callback.on_epoch_end(trainer, 0, {}, {'val_loss': 1.0})
        callback.on_epoch_end(trainer, 1, {}, {'val_loss': 1.5})
        callback.on_epoch_end(trainer, 2, {}, {'val_loss': 0.8})
        
        assert (tmp_path / "model_0.pt").exists()
        assert (tmp_path / "model_1.pt").exists()
        assert (tmp_path / "model_2.pt").exists()
    
    def test_save_weights_only(self, tmp_path):
        """Test saving weights only."""
        filepath = tmp_path / "weights.pt"
        callback = ModelCheckpoint(
            filepath=filepath,
            save_weights_only=True,
            save_best_only=False
        )
        
        trainer = MockTrainer()
        trainer.model.state_dict.return_value = {'weight': torch.randn(5, 5)}
        
        callback.on_epoch_end(trainer, 0, {}, {})
        
        # Should save state dict only
        trainer.model.state_dict.assert_called_once()
    
    def test_save_full_checkpoint(self, tmp_path):
        """Test saving full checkpoint."""
        filepath = tmp_path / "full_checkpoint.pt"
        callback = ModelCheckpoint(
            filepath=filepath,
            save_weights_only=False,
            save_best_only=False
        )
        
        trainer = MockTrainer()
        trainer.model.state_dict.return_value = {'weight': torch.randn(5, 5)}
        trainer.optimizer.state_dict.return_value = {'param_groups': []}
        trainer.current_epoch = 5
        
        callback.on_epoch_end(trainer, 5, {}, {'val_loss': 1.0})
        
        # Should save model, optimizer, and epoch
        assert filepath.exists()


class TestMetricsTracker:
    """Test metrics tracker callback."""
    
    def test_initialization(self):
        """Test metrics tracker initialization."""
        callback = MetricsTracker()
        
        assert callback.history == {}
        assert callback.current_metrics == {}
    
    def test_metric_tracking(self):
        """Test metric tracking functionality."""
        callback = MetricsTracker()
        trainer = MockTrainer()
        
        # Track metrics over epochs
        callback.on_epoch_end(trainer, 0, {'train_loss': 1.0}, {'val_loss': 1.2})
        callback.on_epoch_end(trainer, 1, {'train_loss': 0.8}, {'val_loss': 1.0})
        callback.on_epoch_end(trainer, 2, {'train_loss': 0.6}, {'val_loss': 0.9})
        
        # Check history
        assert len(callback.history['train_loss']) == 3
        assert len(callback.history['val_loss']) == 3
        assert callback.history['train_loss'] == [1.0, 0.8, 0.6]
        assert callback.history['val_loss'] == [1.2, 1.0, 0.9]
    
    def test_get_metric_summary(self):
        """Test metric summary generation."""
        callback = MetricsTracker()
        trainer = MockTrainer()
        
        # Add some metrics
        callback.on_epoch_end(trainer, 0, {'train_loss': 1.0}, {'val_loss': 1.2})
        callback.on_epoch_end(trainer, 1, {'train_loss': 0.8}, {'val_loss': 1.0})
        callback.on_epoch_end(trainer, 2, {'train_loss': 0.6}, {'val_loss': 0.9})
        
        summary = callback.get_summary()
        
        assert 'train_loss' in summary
        assert 'val_loss' in summary
        assert summary['train_loss']['best'] == 0.6
        assert summary['val_loss']['best'] == 0.9
        assert abs(summary['train_loss']['mean'] - 0.8) < 1e-6
    
    def test_reset_history(self):
        """Test history reset functionality."""
        callback = MetricsTracker()
        trainer = MockTrainer()
        
        callback.on_epoch_end(trainer, 0, {'train_loss': 1.0}, {})
        assert len(callback.history) > 0
        
        callback.reset()
        assert callback.history == {}
        assert callback.current_metrics == {}


class TestLearningRateScheduler:
    """Test learning rate scheduler callback."""
    
    def test_step_scheduler(self):
        """Test step-based scheduler."""
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        
        callback = LearningRateScheduler(scheduler, step_on='epoch')
        trainer = MockTrainer()
        trainer.optimizer = optimizer
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Should not change LR on first epoch
        callback.on_epoch_end(trainer, 0, {}, {})
        assert optimizer.param_groups[0]['lr'] == initial_lr
        
        # Should not change LR on second epoch (step_size=2)
        callback.on_epoch_end(trainer, 1, {}, {})
        assert optimizer.param_groups[0]['lr'] == initial_lr
        
        # Should change LR on third epoch
        callback.on_epoch_end(trainer, 2, {}, {})
        assert optimizer.param_groups[0]['lr'] == initial_lr * 0.5
    
    def test_batch_scheduler(self):
        """Test batch-based scheduler."""
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        
        callback = LearningRateScheduler(scheduler, step_on='batch')
        trainer = MockTrainer()
        trainer.optimizer = optimizer
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Should change LR after each batch
        callback.on_batch_end(trainer, 0, {})
        assert optimizer.param_groups[0]['lr'] != initial_lr


class TestCallbackManager:
    """Test callback manager functionality."""
    
    def test_initialization(self):
        """Test callback manager initialization."""
        callback1 = Mock(spec=TrainingCallback)
        callback2 = Mock(spec=TrainingCallback)
        
        manager = CallbackManager([callback1, callback2])
        
        assert len(manager.callbacks) == 2
        assert callback1 in manager.callbacks
        assert callback2 in manager.callbacks
    
    def test_add_remove_callbacks(self):
        """Test adding and removing callbacks."""
        manager = CallbackManager()
        callback = Mock(spec=TrainingCallback)
        
        # Add callback
        manager.add_callback(callback)
        assert callback in manager.callbacks
        
        # Remove callback
        manager.remove_callback(callback)
        assert callback not in manager.callbacks
    
    def test_callback_execution(self):
        """Test callback execution."""
        callback1 = Mock(spec=TrainingCallback)
        callback2 = Mock(spec=TrainingCallback)
        
        manager = CallbackManager([callback1, callback2])
        trainer = MockTrainer()
        
        # Test train begin
        manager.on_train_begin(trainer)
        callback1.on_train_begin.assert_called_once_with(trainer)
        callback2.on_train_begin.assert_called_once_with(trainer)
        
        # Test epoch end
        train_losses = {'loss': 1.0}
        val_losses = {'loss': 1.2}
        manager.on_epoch_end(trainer, 5, train_losses, val_losses)
        callback1.on_epoch_end.assert_called_once_with(trainer, 5, train_losses, val_losses)
        callback2.on_epoch_end.assert_called_once_with(trainer, 5, train_losses, val_losses)
    
    def test_early_stopping_check(self):
        """Test early stopping check."""
        early_stop_callback = Mock(spec=EarlyStopping)
        early_stop_callback.early_stop = True
        
        normal_callback = Mock(spec=TrainingCallback)
        
        manager = CallbackManager([early_stop_callback, normal_callback])
        
        # Should detect early stopping
        should_stop = manager.should_stop_training()
        assert should_stop is True
    
    def test_no_early_stopping(self):
        """Test no early stopping."""
        callback1 = Mock(spec=TrainingCallback)
        callback2 = Mock(spec=TrainingCallback)
        
        manager = CallbackManager([callback1, callback2])
        
        # Should not stop training
        should_stop = manager.should_stop_training()
        assert should_stop is False
    
    def test_exception_handling(self):
        """Test exception handling in callbacks."""
        failing_callback = Mock(spec=TrainingCallback)
        failing_callback.on_train_begin.side_effect = Exception("Test error")
        
        working_callback = Mock(spec=TrainingCallback)
        
        manager = CallbackManager([failing_callback, working_callback])
        trainer = MockTrainer()
        
        # Should not crash and should continue with other callbacks
        with patch('logging.getLogger') as mock_logger:
            manager.on_train_begin(trainer)
            
            # Working callback should still be called
            working_callback.on_train_begin.assert_called_once_with(trainer)
            
            # Should log the error
            mock_logger.return_value.error.assert_called()


class TestMemoryProfiler:
    """Test memory profiler callback."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        callback = MemoryProfiler()
        trainer = MockTrainer()
        
        callback.on_train_begin(trainer)
        callback.on_epoch_begin(trainer, 0)
        callback.on_epoch_end(trainer, 0, {}, {})
        
        # Should have recorded memory usage
        assert len(callback.memory_history) > 0
    
    def test_cpu_fallback(self):
        """Test CPU fallback when CUDA unavailable."""
        callback = MemoryProfiler()
        trainer = MockTrainer()
        
        # Should not crash on CPU
        callback.on_train_begin(trainer)
        callback.on_epoch_begin(trainer, 0)
        callback.on_epoch_end(trainer, 0, {}, {})


class TestLoggers:
    """Test logging callbacks."""
    
    def test_console_logger(self):
        """Test console logger callback."""
        callback = ConsoleLogger(log_frequency=1)
        trainer = MockTrainer()
        
        # Should not crash
        callback.on_train_begin(trainer)
        callback.on_epoch_end(trainer, 0, {'loss': 1.0}, {'loss': 1.2})
        callback.on_train_end(trainer)
    
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_tensorboard_logger(self, mock_writer):
        """Test TensorBoard logger callback."""
        callback = TensorBoardLogger(log_dir="./logs")
        trainer = MockTrainer()
        
        # Mock writer instance
        mock_writer_instance = Mock()
        mock_writer.return_value = mock_writer_instance
        
        callback.on_train_begin(trainer)
        callback.on_epoch_end(trainer, 0, {'loss': 1.0}, {'val_loss': 1.2})
        
        # Should have created writer and logged scalars
        mock_writer.assert_called_once_with("./logs")
        assert mock_writer_instance.add_scalar.called
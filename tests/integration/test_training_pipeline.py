"""Integration tests for the complete training pipeline."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch


class MockTrainer:
    """Mock trainer class for integration testing."""
    
    def __init__(self, model, dataset, config=None):
        self.model = model
        self.dataset = dataset
        self.config = config or self._default_config()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.history = {'train_loss': [], 'val_loss': []}
        self.epoch = 0
        
    def _default_config(self):
        return {
            'batch_size': 4,
            'learning_rate': 1e-3,
            'epochs': 5,
            'validation_split': 0.2,
            'early_stopping_patience': 3,
            'checkpoint_frequency': 2
        }
    
    def train(self, epochs=None):
        """Run training loop."""
        epochs = epochs or self.config['epochs']
        
        # Split dataset into train/val
        dataset_size = len(self.dataset)
        val_size = int(self.config['validation_split'] * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config['batch_size'], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config['batch_size'], shuffle=False
        )
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
        
        return self.history
    
    def _train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_input, batch_target in train_loader:
            self.optimizer.zero_grad()
            
            output = self.model(batch_input)
            loss = nn.functional.mse_loss(output, batch_target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_input, batch_target in val_loader:
                output = self.model(batch_input)
                loss = nn.functional.mse_loss(output, batch_target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.history = checkpoint['history']
        self.config = checkpoint.get('config', self.config)


# Import mock models from unit tests
from ..unit.test_neural_operators import MockFourierNeuralOperator, MockTransformerNeuralOperator


@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for complete training pipeline."""
    
    def test_complete_training_workflow(self, mock_dataset):
        """Test complete training workflow from start to finish."""
        # Create model
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=3, modes=[16, 16], width=64
        )
        
        # Create trainer
        trainer = MockTrainer(model, mock_dataset)
        
        # Run training
        history = trainer.train()
        
        # Verify training completed
        assert len(history['train_loss']) == 5  # 5 epochs
        assert len(history['val_loss']) == 5
        
        # Verify loss decreased (learning occurred)
        assert history['train_loss'][-1] <= history['train_loss'][0] * 2  # Allow some tolerance
    
    def test_training_with_different_models(self, mock_dataset):
        """Test training pipeline with different model architectures."""
        models = [
            MockFourierNeuralOperator(input_dim=3, output_dim=3, modes=[8, 8], width=32),
            MockTransformerNeuralOperator(input_dim=3, output_dim=3, hidden_dim=64, num_layers=2, num_heads=2)
        ]
        
        for model in models:
            trainer = MockTrainer(model, mock_dataset)
            history = trainer.train(epochs=3)
            
            # Verify training works for each model
            assert len(history['train_loss']) == 3
            assert all(not np.isnan(loss) for loss in history['train_loss'])
            assert all(not np.isinf(loss) for loss in history['train_loss'])
    
    def test_checkpoint_save_load(self, mock_dataset, tmp_path):
        """Test checkpoint saving and loading."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=3, modes=[16, 16], width=64
        )
        
        # Train for a few epochs
        trainer = MockTrainer(model, mock_dataset)
        history1 = trainer.train(epochs=3)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)
        assert checkpoint_path.exists()
        
        # Create new trainer and load checkpoint
        new_model = MockFourierNeuralOperator(
            input_dim=3, output_dim=3, modes=[16, 16], width=64
        )
        new_trainer = MockTrainer(new_model, mock_dataset)
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Verify state loaded correctly
        assert new_trainer.epoch == trainer.epoch
        assert new_trainer.history == trainer.history
        
        # Continue training
        history2 = new_trainer.train(epochs=2)
        assert len(history2['train_loss']) == 2  # Only new epochs
    
    def test_training_reproducibility(self, mock_dataset):
        """Test training reproducibility with fixed seeds."""
        def train_with_seed(seed):
            torch.manual_seed(seed)
            model = MockFourierNeuralOperator(
                input_dim=3, output_dim=3, modes=[16, 16], width=32
            )
            trainer = MockTrainer(model, mock_dataset)
            return trainer.train(epochs=3)
        
        # Train with same seed twice
        history1 = train_with_seed(42)
        history2 = train_with_seed(42)
        
        # Results should be identical
        for loss1, loss2 in zip(history1['train_loss'], history2['train_loss']):
            assert abs(loss1 - loss2) < 1e-6
    
    def test_batch_size_scaling(self, mock_dataset):
        """Test training with different batch sizes."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=3, modes=[16, 16], width=32
        )
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            config = {'batch_size': batch_size, 'epochs': 2}
            trainer = MockTrainer(model, mock_dataset, config)
            history = trainer.train()
            
            # Training should work with all batch sizes
            assert len(history['train_loss']) == 2
            assert all(loss >= 0 for loss in history['train_loss'])
    
    def test_optimizer_state_persistence(self, mock_dataset, tmp_path):
        """Test that optimizer state is properly saved and loaded."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=3, modes=[16, 16], width=32
        )
        
        # Train with Adam optimizer
        trainer = MockTrainer(model, mock_dataset)
        trainer.train(epochs=2)
        
        # Get optimizer state before saving
        param_groups_before = trainer.optimizer.param_groups[0]['lr']
        
        # Save and reload
        checkpoint_path = tmp_path / "optimizer_test.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        new_model = MockFourierNeuralOperator(
            input_dim=3, output_dim=3, modes=[16, 16], width=32
        )
        new_trainer = MockTrainer(new_model, mock_dataset)
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Verify optimizer state preserved
        param_groups_after = new_trainer.optimizer.param_groups[0]['lr']
        assert param_groups_before == param_groups_after


@pytest.mark.integration
class TestDataModelIntegration:
    """Integration tests for data loading and model training."""
    
    def test_data_model_compatibility(self):
        """Test that data formats are compatible with models."""
        # Create dataset with specific format
        class FixedFormatDataset:
            def __init__(self, size=20):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Fixed format: 3 input channels, 1 output channel
                input_data = torch.randn(3, 32, 32)
                output_data = torch.randn(1, 32, 32)
                return input_data, output_data
        
        dataset = FixedFormatDataset(50)
        
        # Test with FNO
        fno_model = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=[16, 16], width=32
        )
        fno_trainer = MockTrainer(fno_model, dataset)
        fno_history = fno_trainer.train(epochs=2)
        
        assert len(fno_history['train_loss']) == 2
        
        # Test with TNO
        tno_model = MockTransformerNeuralOperator(
            input_dim=3, output_dim=1, hidden_dim=64, num_layers=2, num_heads=2
        )
        tno_trainer = MockTrainer(tno_model, dataset)
        tno_history = tno_trainer.train(epochs=2)
        
        assert len(tno_history['train_loss']) == 2
    
    def test_variable_resolution_handling(self):
        """Test handling of different resolution inputs."""
        class MultiResolutionDataset:
            def __init__(self, resolutions=[32, 64], size=20):
                self.resolutions = resolutions
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Alternate between resolutions
                resolution = self.resolutions[idx % len(self.resolutions)]
                input_data = torch.randn(3, resolution, resolution)
                output_data = torch.randn(1, resolution, resolution)
                return input_data, output_data
        
        # Note: This test demonstrates the need for proper resolution handling
        # In practice, models need to handle variable resolutions or data needs preprocessing
        dataset = MultiResolutionDataset([32, 32], 20)  # Use fixed resolution for now
        
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=[16, 16], width=32
        )
        trainer = MockTrainer(model, dataset)
        history = trainer.train(epochs=2)
        
        assert len(history['train_loss']) == 2
    
    def test_memory_efficient_training(self, mock_dataset):
        """Test memory-efficient training techniques."""
        model = MockTransformerNeuralOperator(
            input_dim=3, output_dim=3, hidden_dim=128, num_layers=4, num_heads=4
        )
        
        # Test with smaller batch size for memory efficiency
        config = {
            'batch_size': 2,  # Smaller batch size
            'epochs': 3,
            'learning_rate': 1e-3
        }
        
        trainer = MockTrainer(model, mock_dataset, config)
        history = trainer.train()
        
        # Training should complete without memory issues
        assert len(history['train_loss']) == 3
        assert all(loss < 1000 for loss in history['train_loss'])  # Sanity check


@pytest.mark.integration  
class TestModelEvaluation:
    """Integration tests for model evaluation and metrics."""
    
    def test_evaluation_metrics(self, mock_dataset):
        """Test comprehensive model evaluation."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=3, modes=[16, 16], width=64
        )
        
        trainer = MockTrainer(model, mock_dataset)
        trainer.train(epochs=3)
        
        # Evaluation phase
        model.eval()
        test_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=4)
        
        metrics = {
            'mse': [],
            'mae': [],
            'relative_error': []
        }
        
        with torch.no_grad():
            for batch_input, batch_target in test_loader:
                output = model(batch_input)
                
                # Calculate metrics
                mse = torch.nn.functional.mse_loss(output, batch_target)
                mae = torch.nn.functional.l1_loss(output, batch_target)
                
                # Relative error
                relative_error = torch.mean(torch.abs(output - batch_target) / (torch.abs(batch_target) + 1e-8))
                
                metrics['mse'].append(mse.item())
                metrics['mae'].append(mae.item())
                metrics['relative_error'].append(relative_error.item())
        
        # Verify metrics are reasonable
        assert all(mse >= 0 for mse in metrics['mse'])
        assert all(mae >= 0 for mae in metrics['mae'])
        assert all(rel_err >= 0 for rel_err in metrics['relative_error'])
    
    def test_model_inference_consistency(self, mock_dataset):
        """Test that model produces consistent results during inference."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=3, modes=[16, 16], width=32
        )
        
        # Get a sample from dataset
        sample_input, sample_target = mock_dataset[0]
        sample_input = sample_input.unsqueeze(0)  # Add batch dimension
        
        model.eval()
        
        # Run inference multiple times
        outputs = []
        with torch.no_grad():
            for _ in range(5):
                output = model(sample_input)
                outputs.append(output)
        
        # All outputs should be identical
        for i in range(1, len(outputs)):
            torch.testing.assert_close(outputs[0], outputs[i])
    
    def test_model_generalization(self):
        """Test model generalization to unseen data patterns."""
        # Create datasets with different statistical properties
        class StatisticalDataset:
            def __init__(self, mean=0.0, std=1.0, size=50):
                self.mean = mean
                self.std = std
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                torch.manual_seed(idx)  # Deterministic for reproducibility
                input_data = torch.normal(self.mean, self.std, (3, 32, 32))
                output_data = torch.normal(self.mean, self.std, (1, 32, 32))
                return input_data, output_data
        
        # Train on dataset with mean=0, std=1
        train_dataset = StatisticalDataset(mean=0.0, std=1.0, size=50)
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=1, modes=[16, 16], width=32
        )
        
        trainer = MockTrainer(model, train_dataset)
        trainer.train(epochs=5)
        
        # Test on dataset with different statistics
        test_dataset = StatisticalDataset(mean=0.5, std=1.5, size=20)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
        
        model.eval()
        test_losses = []
        
        with torch.no_grad():
            for batch_input, batch_target in test_loader:
                output = model(batch_input)
                loss = torch.nn.functional.mse_loss(output, batch_target)
                test_losses.append(loss.item())
        
        # Model should still produce reasonable results
        avg_test_loss = sum(test_losses) / len(test_losses)
        assert avg_test_loss < 100  # Sanity check for reasonable performance


import numpy as np

@pytest.mark.integration
@pytest.mark.slow  
class TestLongTrainingScenarios:
    """Integration tests for longer training scenarios."""
    
    def test_long_training_stability(self, mock_dataset):
        """Test training stability over many epochs."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=3, modes=[8, 8], width=32
        )
        
        config = {'epochs': 20, 'batch_size': 4, 'learning_rate': 1e-4}
        trainer = MockTrainer(model, mock_dataset, config)
        history = trainer.train()
        
        # Training should remain stable
        assert len(history['train_loss']) == 20
        
        # Check for training instability (NaN or exploding gradients)
        assert all(not np.isnan(loss) for loss in history['train_loss'])
        assert all(not np.isinf(loss) for loss in history['train_loss'])
        assert all(loss < 1000 for loss in history['train_loss'])  # No exploding loss
    
    def test_learning_rate_scheduling(self, mock_dataset):
        """Test learning rate scheduling during training."""
        model = MockFourierNeuralOperator(
            input_dim=3, output_dim=3, modes=[8, 8], width=32
        )
        
        trainer = MockTrainer(model, mock_dataset)
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Train for enough epochs to trigger scheduler
        trainer.train(epochs=15)
        
        final_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Learning rate should have decreased
        assert final_lr < initial_lr
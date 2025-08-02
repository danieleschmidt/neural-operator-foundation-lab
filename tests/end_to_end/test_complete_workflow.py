"""End-to-end tests for complete Neural Operator workflows."""

import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch


class MockCompleteWorkflow:
    """Mock implementation of complete neural operator workflow."""
    
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.model = None
        self.dataset = None
        self.trainer = None
        self.results = {}
    
    def _load_config(self, config_path):
        """Load configuration from file or use defaults."""
        default_config = {
            "model": {
                "type": "fno",
                "input_dim": 3,
                "output_dim": 1,
                "modes": [16, 16],
                "width": 64
            },
            "data": {
                "dataset_type": "synthetic",
                "resolution": 64,
                "num_samples": 1000,
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 1e-3,
                "epochs": 10,
                "optimizer": "adam",
                "scheduler": "step",
                "early_stopping": True,
                "patience": 5
            },
            "evaluation": {
                "metrics": ["mse", "mae", "relative_error"],
                "save_predictions": True,
                "visualize_results": False
            },
            "output": {
                "save_model": True,
                "save_history": True,
                "output_dir": "./experiments/test_run"
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge configurations (simplified)
            for key, value in user_config.items():
                if key in default_config:
                    default_config[key].update(value)
        
        return default_config
    
    def create_synthetic_dataset(self):
        """Create synthetic PDE dataset."""
        class SyntheticPDEDataset:
            def __init__(self, config):
                self.config = config
                self.resolution = config["data"]["resolution"]
                self.num_samples = config["data"]["num_samples"]
                self.input_dim = config["model"]["input_dim"]
                self.output_dim = config["model"]["output_dim"]
                
                # Generate deterministic synthetic data
                torch.manual_seed(42)
                self.data = self._generate_pde_data()
            
            def _generate_pde_data(self):
                """Generate synthetic PDE data that resembles real physics."""
                data = []
                for i in range(self.num_samples):
                    # Create spatial grid
                    x = torch.linspace(0, 1, self.resolution)
                    y = torch.linspace(0, 1, self.resolution)
                    X, Y = torch.meshgrid(x, y, indexing='ij')
                    
                    # Generate initial conditions (Gaussian blobs, sine waves, etc.)
                    if i % 3 == 0:
                        # Gaussian blob
                        center_x, center_y = torch.rand(2) * 0.6 + 0.2
                        width = torch.rand(1) * 0.1 + 0.05
                        initial = torch.exp(-((X - center_x)**2 + (Y - center_y)**2) / width**2)
                    elif i % 3 == 1:
                        # Sine wave pattern
                        freq_x, freq_y = torch.randint(1, 5, (2,))
                        initial = torch.sin(freq_x * np.pi * X) * torch.cos(freq_y * np.pi * Y)
                    else:
                        # Random smooth field
                        initial = torch.randn(self.resolution, self.resolution)
                        initial = torch.nn.functional.conv2d(
                            initial.unsqueeze(0).unsqueeze(0),
                            torch.ones(1, 1, 5, 5) / 25,
                            padding=2
                        ).squeeze()
                    
                    # Create input (include derivatives, boundary conditions, etc.)
                    input_channels = []
                    input_channels.append(initial)
                    
                    # Add gradient information
                    if self.input_dim > 1:
                        grad_x = torch.gradient(initial, dim=0)[0]
                        input_channels.append(grad_x)
                    
                    if self.input_dim > 2:
                        grad_y = torch.gradient(initial, dim=1)[0]
                        input_channels.append(grad_y)
                    
                    # Stack input channels
                    input_tensor = torch.stack(input_channels[:self.input_dim])
                    
                    # Generate target (evolved solution - simplified physics)
                    # For demonstration, apply diffusion-like evolution
                    target = initial.clone()
                    for _ in range(3):  # Simple diffusion steps
                        target = torch.nn.functional.conv2d(
                            target.unsqueeze(0).unsqueeze(0),
                            torch.tensor([[[[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]]]]),
                            padding=1
                        ).squeeze()
                    
                    if self.output_dim == 1:
                        target = target.unsqueeze(0)
                    else:
                        # Multi-channel output
                        target_channels = [target]
                        for _ in range(self.output_dim - 1):
                            # Add related fields
                            related_field = target + torch.randn_like(target) * 0.1
                            target_channels.append(related_field)
                        target = torch.stack(target_channels)
                    
                    data.append((input_tensor, target))
                
                return data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        self.dataset = SyntheticPDEDataset(self.config)
        return self.dataset
    
    def create_model(self):
        """Create neural operator model based on configuration."""
        # Import mock models
        from ..unit.test_neural_operators import (
            MockFourierNeuralOperator, 
            MockTransformerNeuralOperator,
            MockProbabilisticNeuralOperator
        )
        
        model_config = self.config["model"]
        model_type = model_config["type"]
        
        if model_type == "fno":
            self.model = MockFourierNeuralOperator(
                input_dim=model_config["input_dim"],
                output_dim=model_config["output_dim"],
                modes=model_config["modes"],
                width=model_config["width"]
            )
        elif model_type == "tno":
            self.model = MockTransformerNeuralOperator(
                input_dim=model_config["input_dim"],
                output_dim=model_config["output_dim"],
                hidden_dim=model_config.get("hidden_dim", 256),
                num_layers=model_config.get("num_layers", 4),
                num_heads=model_config.get("num_heads", 8)
            )
        elif model_type == "pno":
            self.model = MockProbabilisticNeuralOperator(
                input_dim=model_config["input_dim"],
                output_dim=model_config["output_dim"],
                trunk_layers=model_config.get("trunk_layers", [64, 64, 64]),
                branch_layers=model_config.get("branch_layers", [64, 64])
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self.model
    
    def split_dataset(self):
        """Split dataset into train/val/test sets."""
        if not self.dataset:
            raise ValueError("Dataset not created. Call create_synthetic_dataset() first.")
        
        data_config = self.config["data"]
        total_size = len(self.dataset)
        
        train_size = int(data_config["train_split"] * total_size)
        val_size = int(data_config["val_split"] * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_trainer(self, train_dataset, val_dataset):
        """Create trainer for the model."""
        from ..integration.test_training_pipeline import MockTrainer
        
        # Combine train and val datasets for the mock trainer
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        
        training_config = {
            'batch_size': self.config["training"]["batch_size"],
            'learning_rate': self.config["training"]["learning_rate"],
            'epochs': self.config["training"]["epochs"],
            'validation_split': len(val_dataset) / len(combined_dataset)
        }
        
        self.trainer = MockTrainer(self.model, combined_dataset, training_config)
        return self.trainer
    
    def train_model(self):
        """Train the model."""
        if not self.trainer:
            raise ValueError("Trainer not created. Call create_trainer() first.")
        
        history = self.trainer.train()
        self.results["training_history"] = history
        return history
    
    def evaluate_model(self, test_dataset):
        """Evaluate the model on test set."""
        if not self.model:
            raise ValueError("Model not created.")
        
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=self.config["training"]["batch_size"],
            shuffle=False
        )
        
        metrics = {
            'mse': [],
            'mae': [],
            'relative_error': []
        }
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_input, batch_target in test_loader:
                output = self.model(batch_input)
                
                # Calculate metrics
                mse = torch.nn.functional.mse_loss(output, batch_target)
                mae = torch.nn.functional.l1_loss(output, batch_target)
                relative_error = torch.mean(
                    torch.abs(output - batch_target) / (torch.abs(batch_target) + 1e-8)
                )
                
                metrics['mse'].append(mse.item())
                metrics['mae'].append(mae.item())
                metrics['relative_error'].append(relative_error.item())
                
                if self.config["evaluation"]["save_predictions"]:
                    predictions.append(output.cpu())
                    targets.append(batch_target.cpu())
        
        # Aggregate metrics
        final_metrics = {
            'mse': np.mean(metrics['mse']),
            'mae': np.mean(metrics['mae']),
            'relative_error': np.mean(metrics['relative_error'])
        }
        
        self.results["test_metrics"] = final_metrics
        
        if self.config["evaluation"]["save_predictions"]:
            self.results["predictions"] = torch.cat(predictions)
            self.results["targets"] = torch.cat(targets)
        
        return final_metrics
    
    def save_results(self, output_dir):
        """Save all results to specified directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.config["output"]["save_model"] and self.model:
            torch.save(self.model.state_dict(), output_path / "model.pt")
        
        # Save training history
        if self.config["output"]["save_history"] and "training_history" in self.results:
            with open(output_path / "training_history.json", 'w') as f:
                json.dump(self.results["training_history"], f, indent=2)
        
        # Save test metrics
        if "test_metrics" in self.results:
            with open(output_path / "test_metrics.json", 'w') as f:
                json.dump(self.results["test_metrics"], f, indent=2)
        
        # Save predictions (if enabled)
        if "predictions" in self.results:
            torch.save({
                "predictions": self.results["predictions"],
                "targets": self.results["targets"]
            }, output_path / "predictions.pt")
        
        # Save configuration
        with open(output_path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def run_complete_workflow(self, output_dir=None):
        """Run the complete workflow from start to finish."""
        # Step 1: Create dataset
        self.create_synthetic_dataset()
        
        # Step 2: Create model
        self.create_model()
        
        # Step 3: Split dataset
        train_dataset, val_dataset, test_dataset = self.split_dataset()
        
        # Step 4: Create trainer
        self.create_trainer(train_dataset, val_dataset)
        
        # Step 5: Train model
        self.train_model()
        
        # Step 6: Evaluate model
        self.evaluate_model(test_dataset)
        
        # Step 7: Save results
        if output_dir:
            self.save_results(output_dir)
        
        return self.results


@pytest.mark.end_to_end
class TestCompleteWorkflow:
    """End-to-end tests for complete neural operator workflows."""
    
    def test_default_workflow(self, tmp_path):
        """Test complete workflow with default configuration."""
        workflow = MockCompleteWorkflow()
        results = workflow.run_complete_workflow(str(tmp_path))
        
        # Verify all components completed successfully
        assert "training_history" in results
        assert "test_metrics" in results
        
        # Verify training occurred
        training_history = results["training_history"]
        assert len(training_history["train_loss"]) == 10  # Default epochs
        assert len(training_history["val_loss"]) == 10
        
        # Verify evaluation metrics are reasonable
        test_metrics = results["test_metrics"]
        assert "mse" in test_metrics
        assert "mae" in test_metrics
        assert "relative_error" in test_metrics
        assert all(metric >= 0 for metric in test_metrics.values())
        
        # Verify files were saved
        assert (tmp_path / "model.pt").exists()
        assert (tmp_path / "training_history.json").exists()
        assert (tmp_path / "test_metrics.json").exists()
        assert (tmp_path / "config.json").exists()
    
    def test_custom_config_workflow(self, tmp_path):
        """Test workflow with custom configuration."""
        # Create custom config
        custom_config = {
            "model": {
                "type": "tno",
                "input_dim": 2,
                "output_dim": 2,
                "hidden_dim": 128,
                "num_layers": 3,
                "num_heads": 4
            },
            "data": {
                "resolution": 32,
                "num_samples": 100
            },
            "training": {
                "batch_size": 4,
                "epochs": 5,
                "learning_rate": 1e-4
            }
        }
        
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(custom_config, f)
        
        workflow = MockCompleteWorkflow(str(config_path))
        results = workflow.run_complete_workflow(str(tmp_path / "output"))
        
        # Verify custom configuration was used
        assert len(results["training_history"]["train_loss"]) == 5  # Custom epochs
        
        # Verify model type was respected
        assert workflow.model.__class__.__name__ == "MockTransformerNeuralOperator"
    
    def test_different_model_types(self, tmp_path):
        """Test workflow with different neural operator models."""
        model_types = ["fno", "tno", "pno"]
        
        for model_type in model_types:
            model_config = {
                "model": {
                    "type": model_type,
                    "input_dim": 3,
                    "output_dim": 1
                },
                "training": {
                    "epochs": 3  # Quick test
                },
                "data": {
                    "num_samples": 50  # Small dataset
                }
            }
            
            config_path = tmp_path / f"config_{model_type}.json"
            with open(config_path, 'w') as f:
                json.dump(model_config, f)
            
            workflow = MockCompleteWorkflow(str(config_path))
            results = workflow.run_complete_workflow(
                str(tmp_path / f"output_{model_type}")
            )
            
            # Each model type should complete successfully
            assert "training_history" in results
            assert "test_metrics" in results
            assert len(results["training_history"]["train_loss"]) == 3
    
    def test_workflow_reproducibility(self, tmp_path):
        """Test that workflow results are reproducible."""
        config = {
            "training": {"epochs": 5},
            "data": {"num_samples": 100}
        }
        
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Run workflow twice
        workflow1 = MockCompleteWorkflow(str(config_path))
        results1 = workflow1.run_complete_workflow(str(tmp_path / "run1"))
        
        workflow2 = MockCompleteWorkflow(str(config_path))
        results2 = workflow2.run_complete_workflow(str(tmp_path / "run2"))
        
        # Results should be identical (due to fixed seeds in synthetic data)
        history1 = results1["training_history"]["train_loss"]
        history2 = results2["training_history"]["train_loss"]
        
        for loss1, loss2 in zip(history1, history2):
            assert abs(loss1 - loss2) < 1e-6
    
    def test_workflow_error_handling(self, tmp_path):
        """Test workflow error handling for invalid configurations."""
        # Invalid model type
        invalid_config = {
            "model": {"type": "invalid_model"},
            "training": {"epochs": 1}
        }
        
        config_path = tmp_path / "invalid_config.json"
        with open(config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        workflow = MockCompleteWorkflow(str(config_path))
        
        with pytest.raises(ValueError, match="Unknown model type"):
            workflow.run_complete_workflow()
    
    def test_partial_workflow_execution(self, tmp_path):
        """Test executing individual steps of the workflow."""
        workflow = MockCompleteWorkflow()
        
        # Test individual steps
        dataset = workflow.create_synthetic_dataset()
        assert len(dataset) == 1000  # Default num_samples
        
        model = workflow.create_model()
        assert model is not None
        
        train_dataset, val_dataset, test_dataset = workflow.split_dataset()
        assert len(train_dataset) == 800  # 80% of 1000
        assert len(val_dataset) == 100   # 10% of 1000
        assert len(test_dataset) == 100  # 10% of 1000
        
        trainer = workflow.create_trainer(train_dataset, val_dataset)
        assert trainer is not None
    
    @pytest.mark.slow
    def test_long_training_workflow(self, tmp_path):
        """Test workflow with longer training duration."""
        config = {
            "training": {
                "epochs": 25,
                "batch_size": 8,
                "learning_rate": 1e-4
            },
            "data": {
                "num_samples": 500,
                "resolution": 32
            },
            "model": {
                "type": "fno",
                "width": 32  # Smaller model for speed
            }
        }
        
        config_path = tmp_path / "long_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        workflow = MockCompleteWorkflow(str(config_path))
        results = workflow.run_complete_workflow(str(tmp_path / "long_output"))
        
        # Verify long training completed
        assert len(results["training_history"]["train_loss"]) == 25
        
        # Training should show improvement over time
        train_losses = results["training_history"]["train_loss"]
        early_loss = np.mean(train_losses[:5])
        late_loss = np.mean(train_losses[-5:])
        
        # Allow for some fluctuation but expect general improvement
        assert late_loss <= early_loss * 1.5


@pytest.mark.end_to_end
class TestRealWorldScenarios:
    """End-to-end tests for real-world usage scenarios."""
    
    def test_scientific_research_workflow(self, tmp_path):
        """Test workflow mimicking scientific research use case."""
        # Research scenario: comparing different models on same problem
        research_config = {
            "data": {
                "num_samples": 200,
                "resolution": 64
            },
            "training": {
                "epochs": 10,
                "batch_size": 4
            },
            "evaluation": {
                "save_predictions": True
            }
        }
        
        model_types = ["fno", "tno"]
        results_comparison = {}
        
        for model_type in model_types:
            config = research_config.copy()
            config["model"] = {"type": model_type}
            
            config_path = tmp_path / f"research_{model_type}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            workflow = MockCompleteWorkflow(str(config_path))
            results = workflow.run_complete_workflow(
                str(tmp_path / f"research_{model_type}")
            )
            
            results_comparison[model_type] = results["test_metrics"]
        
        # Both models should complete successfully
        assert "fno" in results_comparison
        assert "tno" in results_comparison
        
        # Results should be comparable (within reasonable range)
        fno_mse = results_comparison["fno"]["mse"]
        tno_mse = results_comparison["tno"]["mse"]
        
        assert fno_mse > 0
        assert tno_mse > 0
        # Both should be reasonable (not orders of magnitude different)
        assert min(fno_mse, tno_mse) / max(fno_mse, tno_mse) > 0.01
    
    def test_production_deployment_workflow(self, tmp_path):
        """Test workflow for production deployment scenario."""
        # Production scenario: optimized for inference speed
        production_config = {
            "model": {
                "type": "fno",
                "width": 32,  # Smaller for speed
                "modes": [8, 8]  # Fewer modes for speed
            },
            "training": {
                "epochs": 15,
                "batch_size": 16,
                "learning_rate": 1e-3
            },
            "data": {
                "num_samples": 1000,
                "resolution": 32  # Lower resolution for speed
            },
            "output": {
                "save_model": True,
                "save_history": False  # Don't need history in production
            }
        }
        
        config_path = tmp_path / "production.json"
        with open(config_path, 'w') as f:
            json.dump(production_config, f)
        
        workflow = MockCompleteWorkflow(str(config_path))
        results = workflow.run_complete_workflow(str(tmp_path / "production"))
        
        # Verify production requirements
        assert "test_metrics" in results
        assert (tmp_path / "production" / "model.pt").exists()
        
        # Model should be reasonably accurate
        assert results["test_metrics"]["relative_error"] < 1.0  # Less than 100% error
        
        # Test inference speed (simplified)
        model = workflow.model
        model.eval()
        
        test_input = torch.randn(1, 3, 32, 32)  # Single sample
        
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(test_input)
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / 100
        assert avg_inference_time < 0.1  # Should be fast for production
    
    def test_educational_workflow(self, tmp_path):
        """Test workflow for educational/tutorial use case."""
        # Educational scenario: clear, simple configuration
        educational_config = {
            "model": {
                "type": "fno",
                "input_dim": 1,
                "output_dim": 1,
                "modes": [8, 8],
                "width": 32
            },
            "data": {
                "num_samples": 100,  # Small for quick demonstration
                "resolution": 32
            },
            "training": {
                "epochs": 5,  # Few epochs for quick results
                "batch_size": 8
            },
            "evaluation": {
                "save_predictions": True,
                "visualize_results": False  # Would be True in real scenario
            }
        }
        
        config_path = tmp_path / "educational.json"
        with open(config_path, 'w') as f:
            json.dump(educational_config, f)
        
        workflow = MockCompleteWorkflow(str(config_path))
        results = workflow.run_complete_workflow(str(tmp_path / "educational"))
        
        # Educational requirements: should complete quickly and successfully
        assert "training_history" in results
        assert "test_metrics" in results
        assert len(results["training_history"]["train_loss"]) == 5
        
        # Should produce reasonable learning curve for educational purposes
        train_losses = results["training_history"]["train_loss"]
        assert all(loss >= 0 for loss in train_losses)
        assert not any(np.isnan(loss) for loss in train_losses)
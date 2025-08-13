#!/usr/bin/env python3
"""
Advanced Physics Simulation Example

Demonstrates comprehensive usage of the Neural Operator Foundation Lab
for solving complex physics problems with cross-domain learning.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import neural operator lab components
try:
    from neural_operator_lab.models.foundation import (
        UniversalPhysicsTransformer, 
        UniversalPhysicsConfig
    )
    from neural_operator_lab.data.datasets import PhysicsDataset
    from neural_operator_lab.training.trainer import AdaptiveTrainer
    from neural_operator_lab.performance_optimizer import profile, cached
    from neural_operator_lab.research_framework import (
        StatisticalAnalyzer, 
        BenchmarkSuite
    )
    from neural_operator_lab.robust_framework import robust_execution
    from neural_operator_lab.intelligent_cache import AdaptiveCache
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Please install neural_operator_lab package")
    raise


class MultiPhysicsSimulator:
    """
    Advanced multi-physics simulator demonstrating the capabilities
    of the Neural Operator Foundation Lab.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the multi-physics simulator."""
        
        # Default configuration
        self.config = {
            'model': {
                'input_dim': 64,
                'output_dim': 64,
                'hidden_dim': 512,
                'num_layers': 8,
                'num_heads': 16,
                'physics_types': [
                    'fluid_dynamics',
                    'heat_transfer', 
                    'electromagnetics',
                    'structural_mechanics'
                ],
                'cross_physics_attention': True,
                'few_shot_learning': True
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 1e-4,
                'num_epochs': 100,
                'early_stopping_patience': 10
            },
            'caching': {
                'enabled': True,
                'max_size_mb': 1000,
                'compression': True
            }
        }
        
        # Update with provided config
        if config:
            self._update_config(self.config, config)
        
        # Initialize components
        self.model = None
        self.trainer = None
        self.cache = AdaptiveCache(**self.config['caching'])
        self.analyzer = StatisticalAnalyzer()
        
        logger.info("Multi-physics simulator initialized")
    
    def _update_config(self, base: Dict, update: Dict):
        """Recursively update configuration."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                self._update_config(base[key], value)
            else:
                base[key] = value
    
    @robust_execution(max_retries=3)
    def initialize_model(self) -> bool:
        """Initialize the universal physics transformer model."""
        try:
            # Create model configuration
            model_config = UniversalPhysicsConfig(**self.config['model'])
            
            # Initialize universal physics transformer
            self.model = UniversalPhysicsTransformer(model_config)
            
            # Initialize trainer
            self.trainer = AdaptiveTrainer(
                model=self.model,
                **self.config['training']
            )
            
            logger.info(f"Model initialized with {self.model.count_parameters()} parameters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return False
    
    @cached(maxsize=100, ttl=3600)
    @profile('data_generation')
    def generate_physics_data(
        self, 
        physics_type: str, 
        resolution: int = 64,
        num_samples: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic physics data for training and testing.
        
        Args:
            physics_type: Type of physics ('fluid', 'heat', 'em', 'structural')
            resolution: Spatial resolution
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (input_data, target_data)
        """
        logger.info(f"Generating {num_samples} samples of {physics_type} data")
        
        if physics_type == 'fluid':
            return self._generate_fluid_dynamics_data(resolution, num_samples)
        elif physics_type == 'heat':
            return self._generate_heat_transfer_data(resolution, num_samples)
        elif physics_type == 'em':
            return self._generate_electromagnetic_data(resolution, num_samples)
        elif physics_type == 'structural':
            return self._generate_structural_mechanics_data(resolution, num_samples)
        else:
            raise ValueError(f"Unknown physics type: {physics_type}")
    
    def _generate_fluid_dynamics_data(
        self, 
        resolution: int, 
        num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 2D fluid dynamics data (simplified Navier-Stokes)."""
        
        # Create coordinate grids
        x = torch.linspace(0, 2*np.pi, resolution)
        y = torch.linspace(0, 2*np.pi, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        input_data = []
        target_data = []
        
        for i in range(num_samples):
            # Random initial conditions
            k1, k2 = np.random.randint(1, 4, 2)
            A, B = np.random.uniform(0.5, 2.0, 2)
            
            # Velocity field (u, v)
            u = A * torch.sin(k1 * X) * torch.cos(k2 * Y)
            v = -B * torch.cos(k1 * X) * torch.sin(k2 * Y)
            
            # Pressure field
            p = torch.sin(X) * torch.cos(Y)
            
            # Input: [u, v, p]
            input_field = torch.stack([u, v, p], dim=0)
            
            # Target: evolved state after time step (simplified)
            dt = 0.01
            nu = 0.01  # viscosity
            
            # Simplified evolution (this would be more complex in reality)
            u_new = u - dt * (u * torch.gradient(u, dim=0)[0] + v * torch.gradient(u, dim=1)[0]) + nu * (torch.gradient(torch.gradient(u, dim=0)[0], dim=0)[0] + torch.gradient(torch.gradient(u, dim=1)[0], dim=1)[0])
            v_new = v - dt * (u * torch.gradient(v, dim=0)[0] + v * torch.gradient(v, dim=1)[0]) + nu * (torch.gradient(torch.gradient(v, dim=0)[0], dim=0)[0] + torch.gradient(torch.gradient(v, dim=1)[0], dim=1)[0])
            p_new = p  # Simplified pressure evolution
            
            target_field = torch.stack([u_new, v_new, p_new], dim=0)
            
            input_data.append(input_field)
            target_data.append(target_field)
        
        return torch.stack(input_data), torch.stack(target_data)
    
    def _generate_heat_transfer_data(
        self, 
        resolution: int, 
        num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 2D heat transfer data."""
        
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        input_data = []
        target_data = []
        
        for i in range(num_samples):
            # Random initial temperature distribution
            n_sources = np.random.randint(1, 4)
            T = torch.zeros_like(X)
            
            for _ in range(n_sources):
                cx, cy = np.random.uniform(0.2, 0.8, 2)
                sigma = np.random.uniform(0.05, 0.15)
                amplitude = np.random.uniform(50, 100)
                
                T += amplitude * torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
            
            # Add background temperature
            T += np.random.uniform(20, 30)
            
            input_field = T.unsqueeze(0)  # Single channel
            
            # Heat equation evolution
            dt = 0.001
            alpha = 1.0  # thermal diffusivity
            
            # Simple finite difference approximation
            T_new = T + alpha * dt * (
                torch.gradient(torch.gradient(T, dim=0)[0], dim=0)[0] + 
                torch.gradient(torch.gradient(T, dim=1)[0], dim=1)[0]
            )
            
            target_field = T_new.unsqueeze(0)
            
            input_data.append(input_field)
            target_data.append(target_field)
        
        return torch.stack(input_data), torch.stack(target_data)
    
    def _generate_electromagnetic_data(
        self, 
        resolution: int, 
        num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate electromagnetic field data."""
        
        x = torch.linspace(0, 2*np.pi, resolution)
        y = torch.linspace(0, 2*np.pi, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        input_data = []
        target_data = []
        
        for i in range(num_samples):
            # Random electromagnetic field
            k = np.random.uniform(1, 3)
            omega = np.random.uniform(1, 2)
            
            # Electric field components
            Ex = torch.sin(k * X) * torch.cos(omega * Y)
            Ey = torch.cos(k * X) * torch.sin(omega * Y)
            Ez = torch.zeros_like(X)
            
            # Magnetic field components (simplified)
            Bx = torch.cos(k * X) * torch.sin(omega * Y)
            By = -torch.sin(k * X) * torch.cos(omega * Y)  
            Bz = torch.zeros_like(X)
            
            input_field = torch.stack([Ex, Ey, Ez, Bx, By, Bz], dim=0)
            
            # Maxwell equation evolution (simplified)
            dt = 0.01
            c = 1.0  # speed of light
            
            # Simplified evolution
            Ex_new = Ex + dt * c * torch.gradient(Bz, dim=1)[0]
            Ey_new = Ey - dt * c * torch.gradient(Bz, dim=0)[0]
            Ez_new = Ez
            
            Bx_new = Bx - dt * torch.gradient(Ez, dim=1)[0] / c
            By_new = By + dt * torch.gradient(Ez, dim=0)[0] / c
            Bz_new = Bz
            
            target_field = torch.stack([Ex_new, Ey_new, Ez_new, Bx_new, By_new, Bz_new], dim=0)
            
            input_data.append(input_field)
            target_data.append(target_field)
        
        return torch.stack(input_data), torch.stack(target_data)
    
    def _generate_structural_mechanics_data(
        self, 
        resolution: int, 
        num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate structural mechanics data."""
        
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        input_data = []
        target_data = []
        
        for i in range(num_samples):
            # Random stress field
            sigma_x = torch.sin(2 * np.pi * X) * torch.cos(2 * np.pi * Y)
            sigma_y = torch.cos(2 * np.pi * X) * torch.sin(2 * np.pi * Y)
            tau_xy = 0.5 * torch.sin(2 * np.pi * X) * torch.sin(2 * np.pi * Y)
            
            input_field = torch.stack([sigma_x, sigma_y, tau_xy], dim=0)
            
            # Simplified deformation calculation
            E = 200e9  # Young's modulus (Pa)
            nu = 0.3   # Poisson's ratio
            
            epsilon_x = (sigma_x - nu * sigma_y) / E
            epsilon_y = (sigma_y - nu * sigma_x) / E
            gamma_xy = tau_xy / (E / (2 * (1 + nu)))
            
            target_field = torch.stack([epsilon_x, epsilon_y, gamma_xy], dim=0)
            
            input_data.append(input_field)
            target_data.append(target_field)
        
        return torch.stack(input_data), torch.stack(target_data)
    
    @profile('training')
    def train_multi_physics_model(
        self, 
        physics_types: List[str],
        resolution: int = 64,
        samples_per_type: int = 1000
    ) -> Dict[str, float]:
        """
        Train the model on multiple physics domains.
        
        Args:
            physics_types: List of physics types to train on
            resolution: Spatial resolution
            samples_per_type: Number of samples per physics type
            
        Returns:
            Training metrics
        """
        if not self.model or not self.trainer:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        logger.info(f"Training on physics types: {physics_types}")
        
        # Generate training data for each physics type
        all_inputs = []
        all_targets = []
        all_physics_labels = []
        
        for i, physics_type in enumerate(physics_types):
            logger.info(f"Generating data for {physics_type}")
            
            inputs, targets = self.generate_physics_data(
                physics_type, resolution, samples_per_type
            )
            
            all_inputs.append(inputs)
            all_targets.append(targets)
            
            # Physics type labels for cross-physics attention
            physics_labels = torch.full((samples_per_type,), i, dtype=torch.long)
            all_physics_labels.append(physics_labels)
        
        # Combine all data
        combined_inputs = torch.cat(all_inputs, dim=0)
        combined_targets = torch.cat(all_targets, dim=0)
        combined_labels = torch.cat(all_physics_labels, dim=0)
        
        logger.info(f"Combined dataset shape: {combined_inputs.shape}")
        
        # Create dataset
        dataset = PhysicsDataset(
            inputs=combined_inputs,
            targets=combined_targets,
            physics_labels=combined_labels
        )
        
        # Train the model
        metrics = self.trainer.train(dataset)
        
        logger.info(f"Training completed. Final metrics: {metrics}")
        return metrics
    
    @profile('evaluation')
    def evaluate_cross_physics_transfer(
        self,
        source_physics: str,
        target_physics: str,
        num_shots: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate few-shot learning capabilities across physics domains.
        
        Args:
            source_physics: Source physics domain (already trained)
            target_physics: Target physics domain (few-shot)
            num_shots: Number of examples for few-shot learning
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating transfer from {source_physics} to {target_physics}")
        
        # Generate few-shot examples for target domain
        few_shot_inputs, few_shot_targets = self.generate_physics_data(
            target_physics, num_samples=num_shots
        )
        
        # Generate test data for target domain
        test_inputs, test_targets = self.generate_physics_data(
            target_physics, num_samples=100
        )
        
        # Few-shot adaptation
        self.model.adapt_few_shot(few_shot_inputs, few_shot_targets)
        
        # Evaluate on test data
        with torch.no_grad():
            predictions = self.model(test_inputs)
            
            # Compute metrics
            mse = torch.mean((predictions - test_targets) ** 2).item()
            mae = torch.mean(torch.abs(predictions - test_targets)).item()
            
            # Physics-specific metrics
            relative_error = torch.mean(
                torch.abs(predictions - test_targets) / (torch.abs(test_targets) + 1e-8)
            ).item()
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'relative_error': relative_error,
            'num_shots': num_shots
        }
        
        logger.info(f"Transfer evaluation metrics: {metrics}")
        return metrics
    
    def run_comprehensive_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Run comprehensive benchmarking across all physics domains."""
        
        benchmark_suite = BenchmarkSuite()
        
        physics_types = ['fluid', 'heat', 'em', 'structural']
        results = {}
        
        for physics_type in physics_types:
            logger.info(f"Benchmarking {physics_type}")
            
            def benchmark_func():
                inputs, targets = self.generate_physics_data(physics_type, num_samples=100)
                
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                with torch.no_grad():
                    predictions = self.model(inputs)
                end_time.record()
                
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
                
                mse = torch.mean((predictions - targets) ** 2).item()
                
                return {
                    'inference_time': inference_time,
                    'mse': mse,
                    'throughput': len(inputs) / inference_time
                }
            
            benchmark_suite.register_benchmark(f'{physics_type}_benchmark', benchmark_func)
            results[physics_type] = benchmark_suite.run_benchmark(f'{physics_type}_benchmark')
        
        return results
    
    def analyze_model_performance(self) -> Dict[str, any]:
        """Comprehensive performance analysis of the trained model."""
        
        analysis_results = {}
        
        # Model statistics
        analysis_results['model_stats'] = self.model.get_model_summary()
        
        # Memory usage
        if torch.cuda.is_available():
            analysis_results['gpu_memory'] = {
                'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'cached': torch.cuda.memory_reserved() / 1024**2,      # MB
            }
        
        # Cache performance
        if self.cache:
            analysis_results['cache_stats'] = self.cache.get_statistics()
        
        # Performance profiling
        from neural_operator_lab.performance_optimizer import get_performance_stats
        analysis_results['performance_stats'] = get_performance_stats()
        
        return analysis_results
    
    def visualize_results(
        self, 
        physics_type: str, 
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """Visualize model predictions for a specific physics type."""
        
        # Generate test data
        inputs, targets = self.generate_physics_data(physics_type, num_samples=10)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(inputs)
        
        # Select sample to visualize
        input_sample = inputs[sample_idx]
        target_sample = targets[sample_idx]
        pred_sample = predictions[sample_idx]
        
        # Create visualization
        num_channels = input_sample.shape[0]
        fig, axes = plt.subplots(3, num_channels, figsize=(4*num_channels, 12))
        
        if num_channels == 1:
            axes = axes.reshape(3, 1)
        
        for i in range(num_channels):
            # Input
            im1 = axes[0, i].imshow(input_sample[i].cpu().numpy(), cmap='viridis')
            axes[0, i].set_title(f'Input Channel {i}')
            plt.colorbar(im1, ax=axes[0, i])
            
            # Target
            im2 = axes[1, i].imshow(target_sample[i].cpu().numpy(), cmap='viridis')
            axes[1, i].set_title(f'Target Channel {i}')
            plt.colorbar(im2, ax=axes[1, i])
            
            # Prediction
            im3 = axes[2, i].imshow(pred_sample[i].cpu().numpy(), cmap='viridis')
            axes[2, i].set_title(f'Prediction Channel {i}')
            plt.colorbar(im3, ax=axes[2, i])
        
        plt.suptitle(f'{physics_type.title()} Physics Simulation Results')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()


def main():
    """Main example execution."""
    
    logger.info("🚀 Starting Advanced Physics Simulation Example")
    
    # Initialize simulator
    simulator = MultiPhysicsSimulator()
    
    # Initialize model
    success = simulator.initialize_model()
    if not success:
        logger.error("Failed to initialize model")
        return
    
    # Train on multiple physics domains
    physics_types = ['fluid', 'heat', 'em']
    training_metrics = simulator.train_multi_physics_model(
        physics_types=physics_types,
        resolution=32,  # Smaller for faster demo
        samples_per_type=100
    )
    
    logger.info(f"Training metrics: {training_metrics}")
    
    # Evaluate cross-physics transfer learning
    transfer_metrics = simulator.evaluate_cross_physics_transfer(
        source_physics='fluid',
        target_physics='structural',
        num_shots=5
    )
    
    logger.info(f"Transfer learning metrics: {transfer_metrics}")
    
    # Run comprehensive benchmarks
    benchmark_results = simulator.run_comprehensive_benchmark()
    logger.info(f"Benchmark results: {benchmark_results}")
    
    # Analyze performance
    performance_analysis = simulator.analyze_model_performance()
    logger.info(f"Performance analysis: {performance_analysis}")
    
    # Visualize results for fluid dynamics
    simulator.visualize_results('fluid', sample_idx=0)
    
    logger.info("✅ Advanced Physics Simulation Example completed successfully!")


if __name__ == "__main__":
    main()
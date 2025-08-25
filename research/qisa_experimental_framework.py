"""QISA Experimental Framework

Comprehensive experimental framework for testing the Quantum-Inspired Spectral Attention
Neural Operator (QISA) against baseline methods with rigorous statistical analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Import our novel QISA model
import sys
sys.path.append('/root/repo/src')
from neural_operator_lab.models.quantum_spectral_neural_operator import (
    QuantumInspiredSpectralAttentionNeuralOperator,
    QISAConfig,
    create_qisa_model
)
from neural_operator_lab.research.experiment_framework import (
    HypothesisTester, ExperimentConfig, Hypothesis, ExperimentManager
)
from neural_operator_lab.research.statistical_analysis import StatisticalAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class QISAExperimentConfig:
    """Configuration for QISA experiments."""
    # Dataset parameters
    dataset_name: str = "synthetic_navier_stokes_3d"
    resolution: int = 64
    sequence_length: int = 100
    num_samples: int = 1000
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 50
    weight_decay: float = 1e-6
    
    # Model parameters
    hidden_dim: int = 256
    num_layers: int = 6
    quantum_depth: int = 4
    spectral_modes: int = 32
    
    # Experimental parameters
    num_runs: int = 5
    random_seeds: List[int] = None
    enable_mixed_precision: bool = True
    
    # Comparison baselines
    compare_against: List[str] = None
    
    def __post_init__(self):
        if self.random_seeds is None:
            np.random.seed(42)
            self.random_seeds = np.random.randint(0, 10000, self.num_runs).tolist()
        
        if self.compare_against is None:
            self.compare_against = [
                "FourierNeuralOperator",
                "TransformerNeuralOperator", 
                "ProbabilisticNeuralOperator"
            ]


class SyntheticPDEDataGenerator:
    """Generate synthetic PDE data for controlled experiments."""
    
    def __init__(self, config: QISAExperimentConfig):
        self.config = config
        self.resolution = config.resolution
        self.sequence_length = config.sequence_length
        
    def generate_navier_stokes_3d(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 3D Navier-Stokes data with known spectral properties."""
        # Create spatial grid
        x = torch.linspace(0, 2*np.pi, self.resolution)
        y = torch.linspace(0, 2*np.pi, self.resolution) 
        z = torch.linspace(0, 2*np.pi, self.resolution)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        # Time steps
        dt = 0.01
        t_steps = torch.arange(self.sequence_length) * dt
        
        data_samples = []
        targets = []
        
        for sample in range(num_samples):
            # Random initial conditions with controlled spectral content
            k_max = self.resolution // 4  # Maximum wave number
            
            # Velocity field components (divergence-free initialization)
            u_init = self._generate_spectral_field(X, Y, Z, k_max, sample)
            v_init = self._generate_spectral_field(X, Y, Z, k_max, sample + 1000) 
            w_init = self._generate_spectral_field(X, Y, Z, k_max, sample + 2000)
            
            # Ensure divergence-free condition: ∇·u = 0
            u_init, v_init, w_init = self._enforce_divergence_free(u_init, v_init, w_init)
            
            # Pressure field (derived from velocity)
            p_init = self._compute_pressure_field(u_init, v_init, w_init)
            
            # Create sequence by evolving with simplified dynamics
            sequence_data = []
            current_u, current_v, current_w, current_p = u_init, v_init, w_init, p_init
            
            for t_step in range(self.sequence_length):
                # Stack velocity and pressure fields
                state = torch.stack([current_u, current_v, current_w, current_p], dim=-1)
                sequence_data.append(state.flatten(start_dim=0, end_dim=2))  # [res^3, 4]
                
                # Simple evolution (Euler step)
                if t_step < self.sequence_length - 1:
                    current_u, current_v, current_w, current_p = self._evolve_navier_stokes(
                        current_u, current_v, current_w, current_p, dt
                    )
            
            # Convert to tensor [seq_len, res^3, 4]
            sequence_tensor = torch.stack(sequence_data)
            
            # Input: all timesteps except last, Target: all timesteps except first
            data_samples.append(sequence_tensor[:-1])  # [seq_len-1, res^3, 4]
            targets.append(sequence_tensor[1:])         # [seq_len-1, res^3, 4]
        
        return torch.stack(data_samples), torch.stack(targets)
    
    def _generate_spectral_field(self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, 
                                k_max: int, seed: int) -> torch.Tensor:
        """Generate field with controlled spectral content."""
        torch.manual_seed(seed)
        field = torch.zeros_like(X)
        
        # Add spectral components with decreasing amplitude
        for kx in range(1, k_max + 1):
            for ky in range(1, k_max + 1):
                for kz in range(1, k_max + 1):
                    # Amplitude decreases with wave number (realistic energy spectrum)
                    amplitude = 1.0 / (kx**2 + ky**2 + kz**2)**(5/6)  # Kolmogorov spectrum
                    phase = torch.rand(1).item() * 2 * np.pi
                    
                    field += amplitude * torch.sin(kx * X + ky * Y + kz * Z + phase)
        
        return field
    
    def _enforce_divergence_free(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enforce divergence-free condition using spectral projection."""
        # Simple projection: modify w to satisfy ∇·u = 0
        # du/dx + dv/dy + dw/dz = 0 => dw/dz = -(du/dx + dv/dy)
        
        # Compute derivatives using central differences
        dx = 2 * np.pi / self.resolution
        
        du_dx = torch.gradient(u, spacing=dx, dim=0)[0]
        dv_dy = torch.gradient(v, spacing=dx, dim=1)[0]
        
        # Integrate to get w that satisfies divergence-free condition
        # This is a simplification; in practice, would use spectral methods
        w_corrected = w - torch.cumsum(du_dx + dv_dy, dim=2) * dx
        
        return u, v, w_corrected
    
    def _compute_pressure_field(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Compute pressure field from velocity using Poisson equation."""
        # Simplified pressure field computation
        # In practice, would solve Poisson equation: ∇²p = -∇·(u·∇u)
        
        # For synthetic data, create pressure with similar spectral properties
        pressure = 0.1 * (u**2 + v**2 + w**2)  # Dynamic pressure approximation
        pressure = pressure - torch.mean(pressure)  # Zero mean pressure
        
        return pressure
    
    def _evolve_navier_stokes(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, 
                             p: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simple Navier-Stokes evolution for synthetic data."""
        # Viscosity
        nu = 0.01
        
        # Simplified evolution (Euler step)
        # du/dt = -u·∇u - ∇p + ν∇²u
        
        # Convection term (simplified)
        conv_u = u * torch.gradient(u, dim=0)[0] + v * torch.gradient(u, dim=1)[0] + w * torch.gradient(u, dim=2)[0]
        conv_v = u * torch.gradient(v, dim=0)[0] + v * torch.gradient(v, dim=1)[0] + w * torch.gradient(v, dim=2)[0]
        conv_w = u * torch.gradient(w, dim=0)[0] + v * torch.gradient(w, dim=1)[0] + w * torch.gradient(w, dim=2)[0]
        
        # Pressure gradient (simplified)
        dp_dx = torch.gradient(p, dim=0)[0]
        dp_dy = torch.gradient(p, dim=1)[0] 
        dp_dz = torch.gradient(p, dim=2)[0]
        
        # Viscous term (simplified Laplacian)
        visc_u = nu * (torch.gradient(torch.gradient(u, dim=0)[0], dim=0)[0] + 
                      torch.gradient(torch.gradient(u, dim=1)[0], dim=1)[0] +
                      torch.gradient(torch.gradient(u, dim=2)[0], dim=2)[0])
        visc_v = nu * (torch.gradient(torch.gradient(v, dim=0)[0], dim=0)[0] + 
                      torch.gradient(torch.gradient(v, dim=1)[0], dim=1)[0] +
                      torch.gradient(torch.gradient(v, dim=2)[0], dim=2)[0])
        visc_w = nu * (torch.gradient(torch.gradient(w, dim=0)[0], dim=0)[0] + 
                      torch.gradient(torch.gradient(w, dim=1)[0], dim=1)[0] +
                      torch.gradient(torch.gradient(w, dim=2)[0], dim=2)[0])
        
        # Update velocities
        u_new = u + dt * (-conv_u - dp_dx + visc_u)
        v_new = v + dt * (-conv_v - dp_dy + visc_v)
        w_new = w + dt * (-conv_w - dp_dz + visc_w)
        
        # Update pressure (simplified)
        p_new = p + 0.1 * dt * (u_new**2 + v_new**2 + w_new**2 - u**2 - v**2 - w**2)
        
        return u_new, v_new, w_new, p_new
    
    def generate_burgers_1d(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 1D Burgers equation data."""
        # Simplified 1D Burgers equation data generation
        x = torch.linspace(0, 2*np.pi, self.resolution)
        dt = 0.01
        nu = 0.01  # viscosity
        
        data_samples = []
        targets = []
        
        for sample in range(num_samples):
            # Random initial condition
            torch.manual_seed(sample)
            u_init = torch.sin(x) + 0.1 * torch.randn_like(x)
            
            sequence_data = []
            current_u = u_init
            
            for t_step in range(self.sequence_length):
                sequence_data.append(current_u.unsqueeze(-1))  # [res, 1]
                
                if t_step < self.sequence_length - 1:
                    # Burgers equation: du/dt + u*du/dx = nu*d²u/dx²
                    du_dx = torch.gradient(current_u)[0]
                    d2u_dx2 = torch.gradient(du_dx)[0]
                    
                    current_u = current_u + dt * (-current_u * du_dx + nu * d2u_dx2)
            
            sequence_tensor = torch.stack(sequence_data)  # [seq_len, res, 1]
            data_samples.append(sequence_tensor[:-1])
            targets.append(sequence_tensor[1:])
        
        return torch.stack(data_samples), torch.stack(targets)


class QISAExperimentRunner:
    """Run comprehensive QISA experiments with statistical validation."""
    
    def __init__(self, config: QISAExperimentConfig, output_dir: str = "qisa_experiments"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.data_generator = SyntheticPDEDataGenerator(config)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "experiment.log"),
                logging.StreamHandler()
            ]
        )
        
        self.results = []
        
    def create_baseline_models(self) -> Dict[str, nn.Module]:
        """Create baseline models for comparison."""
        models = {}
        
        # Import baseline models
        from neural_operator_lab.models.fourier import FourierNeuralOperator
        from neural_operator_lab.models.transformer import TransformerNeuralOperator
        from neural_operator_lab.models.probabilistic import ProbabilisticNeuralOperator
        
        input_dim = 4 if "navier_stokes" in self.config.dataset_name else 1
        output_dim = input_dim
        
        if "FourierNeuralOperator" in self.config.compare_against:
            models["FNO"] = FourierNeuralOperator(
                input_dim=input_dim,
                output_dim=output_dim,
                modes=32,
                hidden_channels=self.config.hidden_dim
            )
        
        if "TransformerNeuralOperator" in self.config.compare_against:
            models["TNO"] = TransformerNeuralOperator(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                num_heads=8
            )
        
        if "ProbabilisticNeuralOperator" in self.config.compare_against:
            models["PNO"] = ProbabilisticNeuralOperator(
                input_dim=input_dim,
                output_dim=output_dim,
                trunk_layers=[self.config.hidden_dim] * self.config.num_layers,
                branch_layers=[self.config.hidden_dim // 2] * 2
            )
        
        return models
    
    def create_qisa_model(self) -> QuantumInspiredSpectralAttentionNeuralOperator:
        """Create QISA model with experiment configuration."""
        input_dim = 4 if "navier_stokes" in self.config.dataset_name else 1
        output_dim = input_dim
        
        qisa_config = QISAConfig(
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            quantum_depth=self.config.quantum_depth,
            spectral_modes=self.config.spectral_modes,
            enable_mixed_precision=self.config.enable_mixed_precision
        )
        
        return QuantumInspiredSpectralAttentionNeuralOperator(
            input_dim=input_dim,
            output_dim=output_dim,
            config=qisa_config
        )
    
    def generate_experiment_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate experimental data."""
        logger.info(f"Generating {self.config.num_samples} samples of {self.config.dataset_name}")
        
        if "navier_stokes" in self.config.dataset_name:
            return self.data_generator.generate_navier_stokes_3d(self.config.num_samples)
        elif "burgers" in self.config.dataset_name:
            return self.data_generator.generate_burgers_1d(self.config.num_samples)
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}")
    
    def train_model(self, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor], 
                   val_data: Tuple[torch.Tensor, torch.Tensor], seed: int) -> Dict[str, Any]:
        """Train a single model with given data."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        train_x, train_y = train_data
        val_x, val_y = val_data
        
        train_x, train_y = train_x.to(device), train_y.to(device)
        val_x, val_y = val_x.to(device), val_y.to(device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )
        
        # Training metrics
        train_losses = []
        val_losses = []
        spectral_losses = []
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            model.train()
            epoch_train_loss = 0.0
            epoch_spectral_loss = 0.0
            
            # Training loop
            for batch_idx in range(0, train_x.shape[0], self.config.batch_size):
                batch_x = train_x[batch_idx:batch_idx + self.config.batch_size]
                batch_y = train_y[batch_idx:batch_idx + self.config.batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(model, QuantumInspiredSpectralAttentionNeuralOperator):
                    # QISA model returns spectral loss
                    pred, spectral_loss = model(batch_x, return_spectral_loss=True)
                    data_loss = F.mse_loss(pred, batch_y)
                    total_loss = data_loss + 0.1 * spectral_loss  # Weighted spectral loss
                    epoch_spectral_loss += spectral_loss.item()
                else:
                    # Baseline models
                    pred = model(batch_x)
                    total_loss = F.mse_loss(pred, batch_y)
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_train_loss += total_loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x)
                if isinstance(val_pred, tuple):
                    val_pred = val_pred[0]  # Extract prediction from tuple
                val_loss = F.mse_loss(val_pred, val_y)
            
            scheduler.step()
            
            avg_train_loss = epoch_train_loss / (train_x.shape[0] // self.config.batch_size + 1)
            avg_spectral_loss = epoch_spectral_loss / (train_x.shape[0] // self.config.batch_size + 1)
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss.item())
            spectral_losses.append(avg_spectral_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}")
        
        training_time = time.time() - start_time
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'spectral_losses': spectral_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'training_time': training_time,
            'seed': seed
        }
    
    def evaluate_model(self, model: nn.Module, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        device = next(model.parameters()).device
        test_x, test_y = test_data
        test_x, test_y = test_x.to(device), test_y.to(device)
        
        model.eval()
        metrics = {}
        
        with torch.no_grad():
            # Inference time
            start_time = time.time()
            pred = model(test_x)
            if isinstance(pred, tuple):
                pred = pred[0]
            inference_time = time.time() - start_time
            
            # Basic metrics
            mse_loss = F.mse_loss(pred, test_y)
            relative_l2 = torch.norm(pred - test_y) / torch.norm(test_y)
            max_error = torch.max(torch.abs(pred - test_y))
            
            metrics.update({
                'mse_loss': mse_loss.item(),
                'relative_l2_error': relative_l2.item(),
                'max_error': max_error.item(),
                'inference_time': inference_time,
                'inference_time_per_sample': inference_time / test_x.shape[0]
            })
            
            # Spectral metrics
            pred_np = pred.cpu().numpy()
            target_np = test_y.cpu().numpy()
            
            # Power spectral density comparison
            pred_fft = np.fft.rfft(pred_np, axis=1)
            target_fft = np.fft.rfft(target_np, axis=1)
            
            pred_psd = np.mean(np.abs(pred_fft) ** 2, axis=(0, 2))
            target_psd = np.mean(np.abs(target_fft) ** 2, axis=(0, 2))
            
            spectral_error = np.mean((pred_psd - target_psd) ** 2)
            
            metrics['spectral_error'] = spectral_error
            
            # Conservation properties (for physical PDEs)
            if pred.shape[-1] >= 3:  # Velocity field
                # Energy conservation
                pred_energy = torch.mean(torch.sum(pred[..., :3] ** 2, dim=-1))
                target_energy = torch.mean(torch.sum(test_y[..., :3] ** 2, dim=-1))
                energy_error = torch.abs(pred_energy - target_energy) / target_energy
                metrics['energy_conservation_error'] = energy_error.item()
                
                # Momentum conservation (should be close to zero for periodic domains)
                pred_momentum = torch.mean(torch.sum(pred[..., :3], dim=-1))
                target_momentum = torch.mean(torch.sum(test_y[..., :3], dim=-1))
                momentum_error = torch.abs(pred_momentum - target_momentum)
                metrics['momentum_conservation_error'] = momentum_error.item()
        
        # Model complexity metrics
        metrics['model_parameters'] = sum(p.numel() for p in model.parameters())
        metrics['model_size_mb'] = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        return metrics
    
    def run_comparative_experiment(self) -> Dict[str, Any]:
        """Run comprehensive comparative experiment."""
        logger.info("Starting QISA comparative experiment")
        
        # Generate data
        data_x, data_y = self.generate_experiment_data()
        
        # Split data
        train_size = int(0.7 * self.config.num_samples)
        val_size = int(0.15 * self.config.num_samples)
        
        train_x, train_y = data_x[:train_size], data_y[:train_size]
        val_x, val_y = data_x[train_size:train_size + val_size], data_y[train_size:train_size + val_size]
        test_x, test_y = data_x[train_size + val_size:], data_y[train_size + val_size:]
        
        logger.info(f"Data split - Train: {train_x.shape[0]}, Val: {val_x.shape[0]}, Test: {test_x.shape[0]}")
        
        # Create models
        baseline_models = self.create_baseline_models()
        qisa_model = self.create_qisa_model()
        
        all_models = {"QISA": qisa_model}
        all_models.update(baseline_models)
        
        logger.info(f"Models to compare: {list(all_models.keys())}")
        
        # Run experiments for each model
        experiment_results = {}
        
        for model_name, model in all_models.items():
            logger.info(f"\nTraining {model_name}...")
            model_results = []
            
            for run_idx, seed in enumerate(self.config.random_seeds):
                logger.info(f"Run {run_idx + 1}/{self.config.num_runs} (seed={seed})")
                
                # Create fresh model instance for each run
                if model_name == "QISA":
                    fresh_model = self.create_qisa_model()
                else:
                    # Create fresh baseline model (simplified)
                    fresh_model = type(model)(**self._get_model_config(model))
                
                # Train model
                train_result = self.train_model(
                    fresh_model,
                    (train_x, train_y),
                    (val_x, val_y),
                    seed
                )
                
                # Evaluate model
                eval_metrics = self.evaluate_model(fresh_model, (test_x, test_y))
                
                # Combine results
                run_result = {
                    'run_id': run_idx,
                    'seed': seed,
                    'model_name': model_name,
                    **train_result,
                    **eval_metrics
                }
                
                model_results.append(run_result)
                
                # Get quantum state info for QISA
                if model_name == "QISA":
                    quantum_info = fresh_model.get_quantum_state_info()
                    run_result['quantum_info'] = quantum_info
            
            experiment_results[model_name] = model_results
        
        # Statistical analysis
        statistical_results = self._perform_statistical_analysis(experiment_results)
        
        # Save results
        self._save_experiment_results(experiment_results, statistical_results)
        
        # Generate visualizations
        self._generate_visualizations(experiment_results, statistical_results)
        
        return {
            'experiment_results': experiment_results,
            'statistical_results': statistical_results,
            'config': asdict(self.config)
        }
    
    def _get_model_config(self, model: nn.Module) -> Dict[str, Any]:
        """Extract configuration for recreating baseline models."""
        # Simplified - in practice would need more sophisticated config extraction
        config = {}
        
        if hasattr(model, 'input_dim'):
            config['input_dim'] = model.input_dim
        if hasattr(model, 'output_dim'):
            config['output_dim'] = model.output_dim
        if hasattr(model, 'hidden_dim'):
            config['hidden_dim'] = model.hidden_dim
        elif hasattr(model, 'hidden_channels'):
            config['hidden_channels'] = model.hidden_channels
        
        return config
    
    def _perform_statistical_analysis(self, experiment_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Perform rigorous statistical analysis of results."""
        statistical_results = {}
        
        # Extract metrics for comparison
        metrics_to_compare = [
            'final_val_loss', 'mse_loss', 'relative_l2_error', 
            'spectral_error', 'inference_time_per_sample',
            'energy_conservation_error', 'momentum_conservation_error'
        ]
        
        model_names = list(experiment_results.keys())
        
        # Pairwise comparisons
        for metric in metrics_to_compare:
            statistical_results[metric] = {}
            
            # Extract values for each model
            model_values = {}
            for model_name in model_names:
                values = []
                for result in experiment_results[model_name]:
                    if metric in result and result[metric] is not None:
                        values.append(result[metric])
                model_values[model_name] = values
            
            # Compare QISA against each baseline
            if 'QISA' in model_values:
                qisa_values = model_values['QISA']
                
                for baseline_name in model_names:
                    if baseline_name != 'QISA' and baseline_name in model_values:
                        baseline_values = model_values[baseline_name]
                        
                        if len(qisa_values) > 0 and len(baseline_values) > 0:
                            comparison = self.statistical_analyzer.compare_methods(
                                qisa_values, baseline_values, 
                                'QISA', baseline_name, metric
                            )
                            
                            statistical_results[metric][f'QISA_vs_{baseline_name}'] = {
                                'qisa_mean': np.mean(qisa_values),
                                'qisa_std': np.std(qisa_values),
                                'baseline_mean': np.mean(baseline_values),
                                'baseline_std': np.std(baseline_values),
                                'p_value': comparison.test_results['comparison'].p_value,
                                'is_significant': comparison.test_results['comparison'].is_significant,
                                'effect_size': comparison.test_results.get('effect_size', {}).effect_size if hasattr(comparison.test_results.get('effect_size', {}), 'effect_size') else None,
                                'improvement_percentage': ((np.mean(baseline_values) - np.mean(qisa_values)) / np.mean(baseline_values) * 100) if 'error' in metric or 'loss' in metric else ((np.mean(qisa_values) - np.mean(baseline_values)) / np.mean(baseline_values) * 100)
                            }
        
        return statistical_results
    
    def _save_experiment_results(self, experiment_results: Dict, statistical_results: Dict):
        """Save comprehensive experiment results."""
        # Save raw results
        with open(self.output_dir / "experiment_results.json", 'w') as f:
            json.dump(experiment_results, f, indent=2, default=str)
        
        # Save statistical analysis
        with open(self.output_dir / "statistical_analysis.json", 'w') as f:
            json.dump(statistical_results, f, indent=2, default=str)
        
        # Save configuration
        with open(self.output_dir / "experiment_config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report(experiment_results, statistical_results)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_summary_report(self, experiment_results: Dict, statistical_results: Dict):
        """Generate comprehensive summary report."""
        report_lines = [
            "# QISA Experimental Results Summary",
            "",
            f"**Dataset:** {self.config.dataset_name}",
            f"**Resolution:** {self.config.resolution}",
            f"**Number of runs:** {self.config.num_runs}",
            f"**Training epochs:** {self.config.num_epochs}",
            "",
            "## Model Performance Summary",
            ""
        ]
        
        # Performance summary table
        for model_name, results in experiment_results.items():
            if results:
                metrics_summary = {}
                for key in ['final_val_loss', 'mse_loss', 'relative_l2_error', 'spectral_error']:
                    values = [r[key] for r in results if key in r and r[key] is not None]
                    if values:
                        metrics_summary[key] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
                
                report_lines.extend([
                    f"### {model_name}",
                    f"- Parameters: {results[0].get('model_parameters', 'N/A'):,}",
                    f"- Model size: {results[0].get('model_size_mb', 'N/A'):.2f} MB",
                    f"- Training time: {np.mean([r['training_time'] for r in results]):.1f}s ± {np.std([r['training_time'] for r in results]):.1f}s",
                    ""
                ])
                
                for metric, stats in metrics_summary.items():
                    report_lines.append(f"- {metric}: {stats['mean']:.6f} ± {stats['std']:.6f}")
                
                report_lines.append("")
        
        # Statistical significance results
        report_lines.extend([
            "## Statistical Analysis",
            "",
            "### QISA vs Baseline Comparisons",
            ""
        ])
        
        for metric, comparisons in statistical_results.items():
            if comparisons:
                report_lines.append(f"#### {metric}")
                for comparison_name, stats in comparisons.items():
                    significance = "✓" if stats.get('is_significant', False) else "✗"
                    improvement = stats.get('improvement_percentage', 0)
                    report_lines.extend([
                        f"- {comparison_name}:",
                        f"  - Significant: {significance} (p={stats.get('p_value', 'N/A'):.4f})",
                        f"  - Improvement: {improvement:.1f}%",
                        f"  - Effect size: {stats.get('effect_size', 'N/A'):.3f}" if stats.get('effect_size') is not None else "",
                        ""
                    ])
        
        # Quantum-specific results for QISA
        if 'QISA' in experiment_results:
            qisa_results = experiment_results['QISA']
            if qisa_results and 'quantum_info' in qisa_results[0]:
                quantum_info = qisa_results[0]['quantum_info']
                report_lines.extend([
                    "## Quantum Analysis (QISA)",
                    "",
                    f"- Total quantum parameters: {quantum_info.get('total_quantum_params', 'N/A'):,}",
                    f"- Average density matrix trace: {np.mean(quantum_info.get('density_matrix_traces', [])):.4f}",
                    f"- Quantum gate angle range: [{np.min(quantum_info.get('quantum_gate_angles', [])):.3f}, {np.max(quantum_info.get('quantum_gate_angles', [])):.3f}]",
                    ""
                ])
        
        # Save report
        with open(self.output_dir / "experiment_summary.md", 'w') as f:
            f.write('\n'.join(report_lines))
    
    def _generate_visualizations(self, experiment_results: Dict, statistical_results: Dict):
        """Generate comprehensive visualizations."""
        plt.style.use('seaborn-v0_8')
        
        # Performance comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('QISA vs Baseline Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            ('final_val_loss', 'Final Validation Loss'),
            ('mse_loss', 'MSE Loss'),
            ('relative_l2_error', 'Relative L2 Error'),
            ('spectral_error', 'Spectral Error'),
            ('inference_time_per_sample', 'Inference Time (s/sample)'),
            ('model_parameters', 'Model Parameters')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            # Collect data for boxplot
            plot_data = []
            plot_labels = []
            
            for model_name, results in experiment_results.items():
                values = [r[metric] for r in results if metric in r and r[metric] is not None]
                if values:
                    plot_data.append(values)
                    plot_labels.append(model_name)
            
            if plot_data:
                # Create boxplot
                bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
                
                # Color QISA differently
                colors = ['lightcoral' if 'QISA' in label else 'lightblue' for label in plot_labels]
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_title(title, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Log scale for parameters
                if metric == 'model_parameters':
                    ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Training curves comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training loss curves
        for model_name, results in experiment_results.items():
            if results and 'train_losses' in results[0]:
                # Average training curves across runs
                all_curves = [r['train_losses'] for r in results]
                min_length = min(len(curve) for curve in all_curves)
                
                # Truncate all curves to same length
                truncated_curves = [curve[:min_length] for curve in all_curves]
                mean_curve = np.mean(truncated_curves, axis=0)
                std_curve = np.std(truncated_curves, axis=0)
                
                epochs = range(len(mean_curve))
                axes[0].plot(epochs, mean_curve, label=model_name, linewidth=2)
                axes[0].fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training Loss')
        axes[0].set_title('Training Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Validation loss curves
        for model_name, results in experiment_results.items():
            if results and 'val_losses' in results[0]:
                all_curves = [r['val_losses'] for r in results]
                min_length = min(len(curve) for curve in all_curves)
                truncated_curves = [curve[:min_length] for curve in all_curves]
                mean_curve = np.mean(truncated_curves, axis=0)
                std_curve = np.std(truncated_curves, axis=0)
                
                epochs = range(len(mean_curve))
                axes[1].plot(epochs, mean_curve, label=model_name, linewidth=2)
                axes[1].fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Loss')
        axes[1].set_title('Validation Loss Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistical significance heatmap
        if statistical_results:
            self._plot_significance_heatmap(statistical_results)
        
        logger.info("Visualizations saved")
    
    def _plot_significance_heatmap(self, statistical_results: Dict):
        """Plot statistical significance heatmap."""
        metrics = list(statistical_results.keys())
        comparisons = set()
        
        # Collect all comparison names
        for metric_results in statistical_results.values():
            comparisons.update(metric_results.keys())
        
        comparisons = sorted(list(comparisons))
        
        # Create significance matrix
        significance_matrix = np.zeros((len(metrics), len(comparisons)))
        improvement_matrix = np.zeros((len(metrics), len(comparisons)))
        
        for i, metric in enumerate(metrics):
            for j, comparison in enumerate(comparisons):
                if comparison in statistical_results[metric]:
                    stats = statistical_results[metric][comparison]
                    significance_matrix[i, j] = 1 if stats.get('is_significant', False) else 0
                    improvement_matrix[i, j] = stats.get('improvement_percentage', 0)
        
        # Plot heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Significance heatmap
        im1 = axes[0].imshow(significance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0].set_title('Statistical Significance (p < 0.05)', fontweight='bold')
        axes[0].set_xticks(range(len(comparisons)))
        axes[0].set_xticklabels(comparisons, rotation=45, ha='right')
        axes[0].set_yticks(range(len(metrics)))
        axes[0].set_yticklabels(metrics)
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(comparisons)):
                text = '✓' if significance_matrix[i, j] == 1 else '✗'
                axes[0].text(j, i, text, ha='center', va='center', fontweight='bold')
        
        # Improvement heatmap
        im2 = axes[1].imshow(improvement_matrix, cmap='RdBu_r', aspect='auto', center=0)
        axes[1].set_title('Improvement Percentage (%)', fontweight='bold')
        axes[1].set_xticks(range(len(comparisons)))
        axes[1].set_xticklabels(comparisons, rotation=45, ha='right')
        axes[1].set_yticks(range(len(metrics)))
        axes[1].set_yticklabels(metrics)
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(comparisons)):
                text = f'{improvement_matrix[i, j]:.1f}%'
                color = 'white' if abs(improvement_matrix[i, j]) > 20 else 'black'
                axes[1].text(j, i, text, ha='center', va='center', color=color, fontweight='bold')
        
        # Add colorbars
        plt.colorbar(im1, ax=axes[0], label='Significant (1) / Not Significant (0)')
        plt.colorbar(im2, ax=axes[1], label='Improvement (%)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "statistical_significance.png", dpi=300, bbox_inches='tight')
        plt.close()


def run_qisa_research_campaign():
    """Run comprehensive QISA research campaign."""
    
    # Configuration for experiments
    configs = [
        # Different datasets
        QISAExperimentConfig(
            dataset_name="synthetic_navier_stokes_3d",
            resolution=64,
            num_samples=500,
            num_epochs=30,
            num_runs=3
        ),
        QISAExperimentConfig(
            dataset_name="synthetic_burgers_1d", 
            resolution=128,
            num_samples=500,
            num_epochs=30,
            num_runs=3
        ),
        # Different model sizes
        QISAExperimentConfig(
            dataset_name="synthetic_navier_stokes_3d",
            resolution=64,
            num_samples=500,
            hidden_dim=128,
            num_layers=4,
            num_epochs=30,
            num_runs=3
        ),
        QISAExperimentConfig(
            dataset_name="synthetic_navier_stokes_3d",
            resolution=64,
            num_samples=500,
            hidden_dim=512,
            num_layers=8,
            num_epochs=30,
            num_runs=3
        )
    ]
    
    all_results = []
    
    for i, config in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING EXPERIMENT {i+1}/{len(configs)}")
        logger.info(f"{'='*60}")
        
        experiment_runner = QISAExperimentRunner(
            config, 
            output_dir=f"qisa_experiments/experiment_{i+1}"
        )
        
        results = experiment_runner.run_comparative_experiment()
        all_results.append(results)
    
    # Generate campaign summary
    campaign_summary = {
        'total_experiments': len(configs),
        'experiment_results': all_results,
        'summary': {
            'datasets_tested': list(set(config.dataset_name for config in configs)),
            'model_sizes_tested': list(set((config.hidden_dim, config.num_layers) for config in configs)),
            'total_model_runs': sum(config.num_runs * (1 + len(config.compare_against)) for config in configs)
        }
    }
    
    # Save campaign summary
    with open("qisa_experiments/campaign_summary.json", 'w') as f:
        json.dump(campaign_summary, f, indent=2, default=str)
    
    logger.info("\n🎉 QISA RESEARCH CAMPAIGN COMPLETE!")
    logger.info(f"Total experiments: {campaign_summary['total_experiments']}")
    logger.info(f"Results saved to: qisa_experiments/")
    
    return campaign_summary


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run single experiment
    config = QISAExperimentConfig(
        dataset_name="synthetic_navier_stokes_3d",
        resolution=32,  # Smaller for testing
        num_samples=100,  # Smaller for testing
        num_epochs=10,   # Shorter for testing
        num_runs=2       # Fewer runs for testing
    )
    
    runner = QISAExperimentRunner(config, "qisa_test_experiment")
    results = runner.run_comparative_experiment()
    
    print("\n📊 EXPERIMENT COMPLETED!")
    print(f"Results saved to: qisa_test_experiment/")
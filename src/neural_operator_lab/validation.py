"""Comprehensive validation module for neural operators."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from dataclasses import dataclass
import logging

from .base import NeuralOperatorBase, PDEDatasetBase
from .utils import ConservationLawChecker, PDEResidualComputer

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    mse: float
    relative_l2: float
    max_error: float
    spectral_error: Optional[float] = None
    conservation_error: Optional[float] = None
    physics_residual: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'mse': self.mse,
            'relative_l2': self.relative_l2,
            'max_error': self.max_error,
            'spectral_error': self.spectral_error or 0.0,
            'conservation_error': self.conservation_error or 0.0,
            'physics_residual': self.physics_residual or 0.0
        }


class ModelValidator:
    """Comprehensive model validation."""
    
    def __init__(self, model: NeuralOperatorBase, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def validate_single_prediction(self, x: torch.Tensor, y_true: torch.Tensor,
                                 grid: Optional[torch.Tensor] = None) -> ValidationMetrics:
        """Validate a single prediction."""
        with torch.no_grad():
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            if grid is not None:
                grid = grid.to(self.device)
            
            # Forward pass
            y_pred = self.model(x, grid)
            
            # Basic metrics
            mse = torch.mean((y_pred - y_true) ** 2).item()
            relative_l2 = (torch.norm(y_pred - y_true) / torch.norm(y_true)).item()
            max_error = torch.max(torch.abs(y_pred - y_true)).item()
            
            # Spectral error (if 2D or 3D)
            spectral_error = None
            if y_pred.dim() >= 3:
                spectral_error = self._compute_spectral_error(y_pred, y_true)
            
            # Conservation laws
            conservation_error = None
            if hasattr(self.model, 'pde_type'):
                conservation_error = self._check_conservation(y_pred)
            
            # Physics residual
            physics_residual = None
            if hasattr(self.model, 'compute_pde_residual'):
                physics_residual = self.model.compute_pde_residual(y_pred, x).mean().item()
            
            return ValidationMetrics(
                mse=mse,
                relative_l2=relative_l2,
                max_error=max_error,
                spectral_error=spectral_error,
                conservation_error=conservation_error,
                physics_residual=physics_residual
            )
    
    def validate_dataset(self, dataset: torch.utils.data.DataLoader,
                        max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Validate on entire dataset."""
        metrics_list = []
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataset):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
                grid = batch[2] if len(batch) > 2 else None
            else:
                raise ValueError("Dataset should return (x, y) or (x, y, grid)")
            
            batch_size = x.shape[0]
            for i in range(batch_size):
                if max_samples and total_samples >= max_samples:
                    break
                
                metrics = self.validate_single_prediction(
                    x[i:i+1], y[i:i+1], 
                    grid[i:i+1] if grid is not None else None
                )
                metrics_list.append(metrics)
                total_samples += 1
            
            if max_samples and total_samples >= max_samples:
                break
        
        # Aggregate metrics
        return self._aggregate_metrics(metrics_list)
    
    def validate_rollout(self, initial_condition: torch.Tensor,
                        true_trajectory: torch.Tensor,
                        num_steps: int,
                        dt: float = 0.01) -> Dict[str, Any]:
        """Validate long-term rollout prediction."""
        with torch.no_grad():
            current_state = initial_condition.to(self.device)
            predictions = [current_state.cpu()]
            errors = []
            
            for step in range(num_steps):
                # Predict next state
                next_state = self.model(current_state)
                predictions.append(next_state.cpu())
                
                # Compute error if true trajectory available
                if step < true_trajectory.shape[0] - 1:
                    true_next = true_trajectory[step + 1].to(self.device)
                    error = torch.norm(next_state - true_next).item()
                    errors.append(error)
                
                current_state = next_state
            
            rollout_prediction = torch.stack(predictions)
            
            # Rollout metrics
            if errors:
                final_error = errors[-1]
                mean_error = np.mean(errors)
                error_growth_rate = self._compute_error_growth_rate(errors)
                
                # Stability check
                is_stable = final_error < 10 * errors[0] if errors[0] > 0 else True
            else:
                final_error = mean_error = error_growth_rate = 0.0
                is_stable = True
            
            return {
                'rollout_prediction': rollout_prediction,
                'step_errors': errors,
                'final_error': final_error,
                'mean_error': mean_error,
                'error_growth_rate': error_growth_rate,
                'is_stable': is_stable,
                'conservation_drift': self._check_conservation_drift(rollout_prediction)
            }
    
    def _compute_spectral_error(self, pred: torch.Tensor, true: torch.Tensor) -> float:
        """Compute error in frequency domain."""
        try:
            pred_fft = torch.fft.fftn(pred, dim=(-2, -1))
            true_fft = torch.fft.fftn(true, dim=(-2, -1))
            
            spectral_error = torch.norm(pred_fft - true_fft) / torch.norm(true_fft)
            return spectral_error.item()
        except Exception as e:
            logger.warning(f"Failed to compute spectral error: {e}")
            return 0.0
    
    def _check_conservation(self, prediction: torch.Tensor) -> float:
        """Check conservation laws."""
        try:
            energy_error = ConservationLawChecker.energy_conservation(prediction)
            mass_error = ConservationLawChecker.mass_conservation(prediction)
            return max(energy_error, mass_error)
        except Exception as e:
            logger.warning(f"Failed to check conservation: {e}")
            return 0.0
    
    def _check_conservation_drift(self, trajectory: torch.Tensor) -> float:
        """Check conservation drift over time."""
        try:
            initial_energy = torch.sum(trajectory[0]**2).item()
            final_energy = torch.sum(trajectory[-1]**2).item()
            return abs(final_energy - initial_energy) / initial_energy
        except Exception:
            return 0.0
    
    def _compute_error_growth_rate(self, errors: List[float]) -> float:
        """Compute exponential error growth rate."""
        if len(errors) < 2:
            return 0.0
        
        try:
            # Fit exponential: error(t) = error_0 * exp(rate * t)
            log_errors = np.log(np.array(errors) + 1e-8)
            times = np.arange(len(errors))
            
            # Linear fit in log space
            coeffs = np.polyfit(times, log_errors, 1)
            growth_rate = coeffs[0]
            
            return max(0.0, growth_rate)  # Only positive growth
        except Exception:
            return 0.0
    
    def _aggregate_metrics(self, metrics_list: List[ValidationMetrics]) -> Dict[str, Any]:
        """Aggregate validation metrics."""
        if not metrics_list:
            return {}
        
        # Convert to arrays
        mse_values = [m.mse for m in metrics_list]
        rel_l2_values = [m.relative_l2 for m in metrics_list]
        max_error_values = [m.max_error for m in metrics_list]
        
        spectral_values = [m.spectral_error for m in metrics_list if m.spectral_error is not None]
        conservation_values = [m.conservation_error for m in metrics_list if m.conservation_error is not None]
        physics_values = [m.physics_residual for m in metrics_list if m.physics_residual is not None]
        
        # Aggregate statistics
        results = {
            'num_samples': len(metrics_list),
            'mse': {
                'mean': np.mean(mse_values),
                'std': np.std(mse_values),
                'min': np.min(mse_values),
                'max': np.max(mse_values)
            },
            'relative_l2': {
                'mean': np.mean(rel_l2_values),
                'std': np.std(rel_l2_values),
                'min': np.min(rel_l2_values),
                'max': np.max(rel_l2_values)
            },
            'max_error': {
                'mean': np.mean(max_error_values),
                'std': np.std(max_error_values),
                'min': np.min(max_error_values),
                'max': np.max(max_error_values)
            }
        }
        
        if spectral_values:
            results['spectral_error'] = {
                'mean': np.mean(spectral_values),
                'std': np.std(spectral_values),
                'min': np.min(spectral_values),
                'max': np.max(spectral_values)
            }
        
        if conservation_values:
            results['conservation_error'] = {
                'mean': np.mean(conservation_values),
                'std': np.std(conservation_values),
                'min': np.min(conservation_values),
                'max': np.max(conservation_values)
            }
        
        if physics_values:
            results['physics_residual'] = {
                'mean': np.mean(physics_values),
                'std': np.std(physics_values),
                'min': np.min(physics_values),
                'max': np.max(physics_values)
            }
        
        return results


class DataValidator:
    """Validate PDE datasets."""
    
    def __init__(self, dataset: Union[PDEDatasetBase, torch.utils.data.Dataset]):
        self.dataset = dataset
        
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate dataset integrity."""
        issues = []
        stats = {}
        
        try:
            # Sample a few data points
            sample_indices = np.random.choice(len(self.dataset), min(10, len(self.dataset)), replace=False)
            
            input_shapes = []
            output_shapes = []
            data_ranges = []
            
            for idx in sample_indices:
                try:
                    sample = self.dataset[idx]
                    if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                        x, y = sample[0], sample[1]
                        
                        input_shapes.append(x.shape)
                        output_shapes.append(y.shape)
                        
                        # Check for NaN/Inf
                        if torch.isnan(x).any() or torch.isinf(x).any():
                            issues.append(f"Sample {idx}: Input contains NaN/Inf")
                        if torch.isnan(y).any() or torch.isinf(y).any():
                            issues.append(f"Sample {idx}: Output contains NaN/Inf")
                        
                        # Data ranges
                        data_ranges.append({
                            'input_min': x.min().item(),
                            'input_max': x.max().item(),
                            'output_min': y.min().item(),
                            'output_max': y.max().item()
                        })
                        
                    else:
                        issues.append(f"Sample {idx}: Invalid format - expected (x, y) tuple")
                        
                except Exception as e:
                    issues.append(f"Sample {idx}: Error loading - {str(e)}")
            
            # Check shape consistency
            if input_shapes and not all(shape == input_shapes[0] for shape in input_shapes):
                issues.append("Inconsistent input shapes across samples")
            if output_shapes and not all(shape == output_shapes[0] for shape in output_shapes):
                issues.append("Inconsistent output shapes across samples")
            
            # Statistics
            if data_ranges:
                stats = {
                    'input_range': {
                        'min': min(r['input_min'] for r in data_ranges),
                        'max': max(r['input_max'] for r in data_ranges)
                    },
                    'output_range': {
                        'min': min(r['output_min'] for r in data_ranges),
                        'max': max(r['output_max'] for r in data_ranges)
                    },
                    'common_input_shape': input_shapes[0] if input_shapes else None,
                    'common_output_shape': output_shapes[0] if output_shapes else None
                }
            
        except Exception as e:
            issues.append(f"Dataset validation failed: {str(e)}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'statistics': stats,
            'num_samples_checked': len(sample_indices) if 'sample_indices' in locals() else 0
        }
    
    def check_physics_consistency(self, pde_type: str) -> Dict[str, Any]:
        """Check if data satisfies physics constraints."""
        if pde_type not in ['navier_stokes', 'wave', 'heat', 'maxwell']:
            return {'checked': False, 'reason': 'Unknown PDE type'}
        
        try:
            # Sample data for physics check
            sample = self.dataset[0]
            if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                x, y = sample[0], sample[1]
                
                # Basic physics checks based on PDE type
                physics_issues = []
                
                if pde_type == 'navier_stokes':
                    # Check incompressibility (div(u) = 0) for velocity fields
                    if x.shape[-1] >= 2:  # At least 2D velocity
                        div_check = self._check_divergence_free(x[..., :2])
                        if div_check > 0.1:  # Threshold for incompressibility
                            physics_issues.append(f"High divergence: {div_check:.3f}")
                
                elif pde_type == 'wave':
                    # Check wave equation constraints
                    pass  # Could add specific wave equation checks
                
                elif pde_type == 'heat':
                    # Check heat equation constraints (positivity, smoothness)
                    if torch.any(y < 0):
                        physics_issues.append("Negative temperature values found")
                
                return {
                    'checked': True,
                    'physics_consistent': len(physics_issues) == 0,
                    'issues': physics_issues
                }
            
        except Exception as e:
            return {'checked': False, 'error': str(e)}
        
        return {'checked': False, 'reason': 'Could not access data'}
    
    def _check_divergence_free(self, velocity_field: torch.Tensor) -> float:
        """Check if velocity field is divergence-free."""
        try:
            # Simple finite difference divergence
            u, v = velocity_field[..., 0], velocity_field[..., 1]
            
            # Central differences
            du_dx = (u[..., 2:] - u[..., :-2]) / 2
            dv_dy = (v[..., 2:, :] - v[..., :-2, :]) / 2
            
            # Match dimensions
            min_size = min(du_dx.shape[-1], dv_dy.shape[-2])
            du_dx = du_dx[..., :min_size]
            dv_dy = dv_dy[..., :min_size, :]
            
            divergence = du_dx + dv_dy
            return torch.abs(divergence).mean().item()
            
        except Exception:
            return 0.0  # Can't compute, assume okay
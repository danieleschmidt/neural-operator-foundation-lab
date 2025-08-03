"""Probabilistic Neural Operator (PNO) implementation with uncertainty quantification."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import numpy as np

from ..base import NeuralOperatorBase
from .fourier import FNOBlock


class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty."""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight mean and log variance
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 2)
        
        # Bias mean and log variance
        self.bias_mean = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.zeros(out_features) - 2)
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass with optional stochastic sampling."""
        if sample and self.training:
            # Sample weights from posterior
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mean + weight_std * torch.randn_like(self.weight_mean)
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mean + bias_std * torch.randn_like(self.bias_mean)
        else:
            # Use mean values
            weight = self.weight_mean
            bias = self.bias_mean
        
        return nn.functional.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence from prior."""
        # KL divergence for weights
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mean**2 + weight_var) / (self.prior_std**2) - 
            self.weight_logvar + 
            np.log(self.prior_std**2) - 1
        )
        
        # KL divergence for bias
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mean**2 + bias_var) / (self.prior_std**2) - 
            self.bias_logvar + 
            np.log(self.prior_std**2) - 1
        )
        
        return weight_kl + bias_kl


class UncertaintyHead(nn.Module):
    """Uncertainty quantification head for PNO."""
    
    def __init__(self, input_dim: int, output_dim: int, uncertainty_type: str = 'aleatoric'):
        super().__init__()
        self.uncertainty_type = uncertainty_type
        
        if uncertainty_type == 'aleatoric':
            # Predict both mean and variance
            self.mean_head = nn.Linear(input_dim, output_dim)
            self.logvar_head = nn.Linear(input_dim, output_dim)
        elif uncertainty_type == 'epistemic':
            # Use Bayesian layers
            self.mean_head = BayesianLinear(input_dim, output_dim)
        else:
            raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")
    
    def forward(self, x: torch.Tensor, num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and uncertainty."""
        if self.uncertainty_type == 'aleatoric':
            mean = self.mean_head(x)
            logvar = self.logvar_head(x)
            std = torch.exp(0.5 * logvar)
            return mean, std
        
        elif self.uncertainty_type == 'epistemic':
            if num_samples == 1:
                mean = self.mean_head(x, sample=True)
                return mean, torch.zeros_like(mean)
            else:
                # Monte Carlo sampling
                samples = []
                for _ in range(num_samples):
                    sample = self.mean_head(x, sample=True)
                    samples.append(sample)
                
                samples = torch.stack(samples, dim=0)
                mean = torch.mean(samples, dim=0)
                std = torch.std(samples, dim=0)
                return mean, std


class ProbabilisticNeuralOperator(NeuralOperatorBase):
    """Probabilistic Neural Operator with uncertainty quantification."""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 trunk_layers: list = [512, 512, 512],
                 branch_layers: list = [256, 256],
                 num_basis: int = 128,
                 uncertainty_type: str = 'epistemic',
                 dropout_rate: float = 0.1,
                 modes: list = [16, 16],
                 **kwargs):
        """Initialize PNO.
        
        Args:
            input_dim: Input function dimension
            output_dim: Output function dimension
            trunk_layers: Hidden dimensions for trunk network
            branch_layers: Hidden dimensions for branch network
            num_basis: Number of basis functions
            uncertainty_type: 'epistemic' or 'aleatoric'
            dropout_rate: Dropout probability for uncertainty
            modes: Fourier modes for spectral layers
        """
        super().__init__(input_dim, output_dim, **kwargs)
        
        self.num_basis = num_basis
        self.uncertainty_type = uncertainty_type
        self.dropout_rate = dropout_rate
        
        # Trunk network (processes coordinates)
        trunk_dims = [len(modes)] + trunk_layers + [num_basis]
        self.trunk_layers = nn.ModuleList()
        
        for i in range(len(trunk_dims) - 1):
            if uncertainty_type == 'epistemic':
                layer = BayesianLinear(trunk_dims[i], trunk_dims[i+1])
            else:
                layer = nn.Linear(trunk_dims[i], trunk_dims[i+1])
            self.trunk_layers.append(layer)
        
        # Branch network (processes input functions)
        branch_dims = [input_dim] + branch_layers + [num_basis]
        self.branch_layers = nn.ModuleList()
        
        for i in range(len(branch_dims) - 1):
            if uncertainty_type == 'epistemic':
                layer = BayesianLinear(branch_dims[i], branch_dims[i+1])
            else:
                layer = nn.Linear(branch_dims[i], branch_dims[i+1])
            self.branch_layers.append(layer)
        
        # Spectral processing layers
        self.spectral_layers = nn.ModuleList([
            FNOBlock(modes, trunk_layers[0], activation='gelu')
            for _ in range(2)
        ])
        
        # Uncertainty head
        self.uncertainty_head = UncertaintyHead(num_basis, output_dim, uncertainty_type)
        
        # Dropout for additional uncertainty
        self.dropout = nn.Dropout(dropout_rate)
        
        # Activation
        self.activation = nn.GELU()
    
    def forward_features(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract features using trunk-branch decomposition."""
        batch_size = x.size(0)
        
        if grid is None:
            # Create default coordinate grid
            spatial_dims = x.shape[1:-1]  # Exclude batch and channel
            coords = torch.meshgrid([torch.linspace(0, 1, s, device=x.device) for s in spatial_dims], indexing='ij')
            grid = torch.stack(coords, dim=-1)
            grid = grid.unsqueeze(0).expand(batch_size, *[-1]*grid.dim())
        
        # Branch network: process input functions
        branch_out = x
        for i, layer in enumerate(self.branch_layers):
            if isinstance(layer, BayesianLinear):
                branch_out = layer(branch_out, sample=self.training)
            else:
                branch_out = layer(branch_out)
            
            if i < len(self.branch_layers) - 1:
                branch_out = self.activation(branch_out)
                branch_out = self.dropout(branch_out)
        
        # Trunk network: process coordinates
        trunk_out = grid
        for i, layer in enumerate(self.trunk_layers):
            if isinstance(layer, BayesianLinear):
                trunk_out = layer(trunk_out, sample=self.training)
            else:
                trunk_out = layer(trunk_out)
            
            if i < len(self.trunk_layers) - 1:
                trunk_out = self.activation(trunk_out)
                trunk_out = self.dropout(trunk_out)
        
        # Apply spectral processing
        combined = branch_out * trunk_out  # Element-wise product
        
        for spectral_layer in self.spectral_layers:
            combined = spectral_layer(combined)
        
        return combined
    
    def forward_operator(self, features: torch.Tensor) -> torch.Tensor:
        """Apply PNO transformation with uncertainty."""
        # Get mean prediction
        mean, _ = self.uncertainty_head(features)
        return mean
    
    def forward_with_uncertainty(self, x: torch.Tensor, 
                               grid: Optional[torch.Tensor] = None,
                               num_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both prediction and uncertainty."""
        features = self.forward_features(x, grid)
        
        if self.uncertainty_type == 'epistemic' and num_samples > 1:
            # Monte Carlo sampling for epistemic uncertainty
            predictions = []
            for _ in range(num_samples):
                pred, _ = self.uncertainty_head(features, num_samples=1)
                predictions.append(pred)
            
            predictions = torch.stack(predictions, dim=0)
            mean = torch.mean(predictions, dim=0)
            uncertainty = torch.std(predictions, dim=0)
            
        else:
            # Single forward pass
            mean, uncertainty = self.uncertainty_head(features, num_samples=num_samples)
        
        return mean, uncertainty
    
    def compute_uncertainty_loss(self, pred: torch.Tensor, target: torch.Tensor,
                               uncertainty: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty-aware loss."""
        if self.uncertainty_type == 'aleatoric':
            # Negative log-likelihood for aleatoric uncertainty
            mse = (pred - target)**2
            nll = 0.5 * (mse / (uncertainty**2 + 1e-8) + torch.log(uncertainty**2 + 1e-8))
            return torch.mean(nll)
        
        elif self.uncertainty_type == 'epistemic':
            # Standard MSE + KL divergence
            mse_loss = nn.functional.mse_loss(pred, target)
            
            # Compute KL divergence from Bayesian layers
            kl_loss = 0
            for layer in self.trunk_layers + self.branch_layers:
                if isinstance(layer, BayesianLinear):
                    kl_loss += layer.kl_divergence()
            
            # Scale KL by number of data points (approximate)
            kl_weight = 1.0 / pred.numel()
            
            return mse_loss + kl_weight * kl_loss
        
        else:
            return nn.functional.mse_loss(pred, target)
    
    def get_loss(self, pred: torch.Tensor, target: torch.Tensor,
                 inputs: Optional[torch.Tensor] = None,
                 uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute total loss including uncertainty."""
        if uncertainty is not None:
            return self.compute_uncertainty_loss(pred, target, uncertainty)
        else:
            return super().get_loss(pred, target, inputs)
    
    def calibrate_uncertainty(self, val_loader, num_samples: int = 20) -> Dict[str, float]:
        """Calibrate uncertainty estimates on validation data."""
        self.eval()
        
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                if isinstance(batch, (list, tuple)) and len(batch) > 2:
                    grid = batch[2]
                else:
                    grid = None
                
                pred, uncertainty = self.forward_with_uncertainty(x, grid, num_samples)
                
                all_predictions.append(pred)
                all_targets.append(y)
                all_uncertainties.append(uncertainty)
        
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        uncertainties = torch.cat(all_uncertainties, dim=0)
        
        # Compute calibration metrics
        errors = torch.abs(predictions - targets)
        
        # Expected Calibration Error (ECE)
        num_bins = 10
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        
        ece = 0
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this uncertainty bin
            in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = (errors[in_bin] < uncertainties[in_bin]).float().mean()
                bin_confidence = uncertainties[in_bin].mean()
                ece += torch.abs(bin_accuracy - bin_confidence) * (in_bin.sum().float() / uncertainties.numel())
        
        # Reliability metrics
        correlation = torch.corrcoef(torch.stack([errors.flatten(), uncertainties.flatten()]))[0, 1]
        
        return {
            'expected_calibration_error': ece.item(),
            'uncertainty_correlation': correlation.item(),
            'mean_uncertainty': uncertainties.mean().item(),
            'mean_error': errors.mean().item()
        }


def create_pno_2d(input_dim: int = 3, output_dim: int = 1,
                  uncertainty_type: str = 'epistemic', **kwargs) -> ProbabilisticNeuralOperator:
    """Create 2D PNO for problems with uncertainty quantification."""
    return ProbabilisticNeuralOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        modes=[16, 16],
        uncertainty_type=uncertainty_type,
        **kwargs
    )


def create_pno_3d(input_dim: int = 4, output_dim: int = 4,
                  uncertainty_type: str = 'epistemic', **kwargs) -> ProbabilisticNeuralOperator:
    """Create 3D PNO for volumetric problems with uncertainty."""
    return ProbabilisticNeuralOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        modes=[8, 8, 8],
        uncertainty_type=uncertainty_type,
        **kwargs
    )
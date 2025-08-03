"""Physics-informed loss functions for neural operator training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Callable, Tuple, Any
import numpy as np

from ..utils import PDEResidualComputer, ConservationLawChecker


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss combining data and physics constraints."""
    
    def __init__(self,
                 data_weight: float = 1.0,
                 pde_weight: float = 0.1,
                 boundary_weight: float = 0.05,
                 conservation_weight: float = 0.01,
                 pde_residual_fn: Optional[Callable] = None):
        """Initialize physics-informed loss.
        
        Args:
            data_weight: Weight for data fitting loss
            pde_weight: Weight for PDE residual loss
            boundary_weight: Weight for boundary condition loss
            conservation_weight: Weight for conservation law loss
            pde_residual_fn: Custom PDE residual function
        """
        super().__init__()
        
        self.data_weight = data_weight
        self.pde_weight = pde_weight
        self.boundary_weight = boundary_weight
        self.conservation_weight = conservation_weight
        self.pde_residual_fn = pde_residual_fn
        
        # Initialize PDE residual computer
        self.residual_computer = PDEResidualComputer()
        self.conservation_checker = ConservationLawChecker()
    
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                coords: Optional[torch.Tensor] = None,
                inputs: Optional[torch.Tensor] = None,
                boundary_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute physics-informed loss.
        
        Args:
            pred: Model predictions
            target: Ground truth targets
            coords: Coordinate grid
            inputs: Input data for physics computations
            boundary_mask: Mask for boundary points
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Data fitting loss
        data_loss = F.mse_loss(pred, target)
        losses['data_loss'] = data_loss
        
        # PDE residual loss
        if self.pde_weight > 0 and coords is not None:
            pde_loss = self._compute_pde_loss(pred, coords, inputs)
            losses['pde_loss'] = pde_loss
        else:
            losses['pde_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Boundary condition loss
        if self.boundary_weight > 0 and boundary_mask is not None:
            boundary_loss = self._compute_boundary_loss(pred, target, boundary_mask)
            losses['boundary_loss'] = boundary_loss
        else:
            losses['boundary_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Conservation law loss
        if self.conservation_weight > 0:
            conservation_loss = self._compute_conservation_loss(pred)
            losses['conservation_loss'] = conservation_loss
        else:
            losses['conservation_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Total loss
        total_loss = (self.data_weight * losses['data_loss'] +
                     self.pde_weight * losses['pde_loss'] +
                     self.boundary_weight * losses['boundary_loss'] +
                     self.conservation_weight * losses['conservation_loss'])
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_pde_loss(self,
                         pred: torch.Tensor,
                         coords: torch.Tensor,
                         inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute PDE residual loss."""
        if self.pde_residual_fn is not None:
            return self.pde_residual_fn(pred, coords, inputs)
        
        # Default PDE losses for common equations
        if pred.size(-1) == 1:  # Scalar field (e.g., heat, wave)
            return self._heat_equation_residual(pred, coords)
        elif pred.size(-1) >= 2:  # Vector field (e.g., Navier-Stokes)
            return self._navier_stokes_residual(pred, coords)
        
        return torch.tensor(0.0, device=pred.device)
    
    def _heat_equation_residual(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Heat equation residual: ∂u/∂t - α∇²u = 0."""
        # Simplified implementation for demonstration
        if u.dim() < 4:  # Need time dimension
            return torch.tensor(0.0, device=u.device)
        
        # Compute Laplacian (simplified finite difference)
        laplacian = self._compute_laplacian_fd(u, coords)
        
        # Time derivative (simplified)
        if u.size(1) > 1:  # Time dimension
            du_dt = torch.diff(u, dim=1)
            
            # Match dimensions
            min_size = min(du_dt.size(-2), laplacian.size(-2))
            du_dt = du_dt[..., :min_size, :min_size, :]
            laplacian = laplacian[..., :min_size, :min_size, :]
            
            # Heat equation residual
            alpha = 0.1  # Thermal diffusivity
            residual = du_dt - alpha * laplacian
            
            return torch.mean(residual**2)
        
        return torch.tensor(0.0, device=u.device)
    
    def _navier_stokes_residual(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Navier-Stokes residual (simplified)."""
        if u.size(-1) < 2:
            return torch.tensor(0.0, device=u.device)
        
        # Extract velocity components
        velocity = u[..., :2] if u.size(-1) >= 2 else u
        
        # Compute divergence (incompressibility)
        div_u = self._compute_divergence_fd(velocity, coords)
        
        # Incompressibility constraint
        incompressibility_loss = torch.mean(div_u**2)
        
        return incompressibility_loss
    
    def _compute_laplacian_fd(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using finite differences."""
        if u.dim() < 3:
            return torch.zeros_like(u)
        
        # 2D Laplacian
        laplacian = torch.zeros_like(u)
        
        # Second derivatives
        if u.size(-3) > 2:  # x-direction
            d2u_dx2 = u[..., 2:, :, :] - 2*u[..., 1:-1, :, :] + u[..., :-2, :, :]
            laplacian[..., 1:-1, :, :] += d2u_dx2
        
        if u.size(-2) > 2:  # y-direction
            d2u_dy2 = u[..., :, 2:, :] - 2*u[..., :, 1:-1, :] + u[..., :, :-2, :]
            laplacian[..., :, 1:-1, :] += d2u_dy2
        
        return laplacian
    
    def _compute_divergence_fd(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute divergence using finite differences."""
        if u.size(-1) < 2 or u.dim() < 3:
            return torch.zeros(u.shape[:-1], device=u.device)
        
        # Extract velocity components
        u_x = u[..., 0]
        u_y = u[..., 1]
        
        # Compute partial derivatives
        du_dx = torch.zeros_like(u_x)
        du_dy = torch.zeros_like(u_y)
        
        if u_x.size(-2) > 1:
            du_dx[..., 1:, :] = torch.diff(u_x, dim=-2)
        
        if u_y.size(-1) > 1:
            du_dy[..., :, 1:] = torch.diff(u_y, dim=-1)
        
        # Divergence
        divergence = du_dx + du_dy
        
        return divergence
    
    def _compute_boundary_loss(self,
                              pred: torch.Tensor,
                              target: torch.Tensor,
                              boundary_mask: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss."""
        # Apply boundary mask
        pred_boundary = pred * boundary_mask
        target_boundary = target * boundary_mask
        
        # MSE on boundary points
        boundary_loss = F.mse_loss(pred_boundary, target_boundary)
        
        return boundary_loss
    
    def _compute_conservation_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute conservation law violations."""
        conservation_loss = torch.tensor(0.0, device=pred.device)
        
        # Energy conservation (for vector fields)
        if pred.size(-1) >= 2:
            energy = torch.sum(pred**2, dim=-1)
            
            # Penalize large energy variations
            if energy.dim() > 2:  # Has time dimension
                energy_diff = torch.diff(energy, dim=1)
                conservation_loss += torch.mean(energy_diff**2)
        
        # Mass conservation (total mass should be conserved)
        if pred.dim() >= 3:
            total_mass = torch.sum(pred, dim=(-3, -2))
            if total_mass.dim() > 1:  # Time series
                mass_diff = torch.diff(total_mass, dim=1)
                conservation_loss += torch.mean(mass_diff**2)
        
        return conservation_loss
    
    def set_pde_residual(self, residual_fn: Callable):
        """Set custom PDE residual function."""
        self.pde_residual_fn = residual_fn
    
    def update_weights(self, **kwargs):
        """Update loss component weights during training."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class SpectralLoss(nn.Module):
    """Spectral domain loss for neural operators."""
    
    def __init__(self, 
                 spectral_weight: float = 1.0,
                 high_freq_weight: float = 2.0):
        """Initialize spectral loss.
        
        Args:
            spectral_weight: Weight for spectral domain loss
            high_freq_weight: Additional weight for high frequency components
        """
        super().__init__()
        self.spectral_weight = spectral_weight
        self.high_freq_weight = high_freq_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute spectral loss."""
        # FFT of predictions and targets
        pred_fft = torch.fft.fftn(pred, dim=(-3, -2)) if pred.dim() >= 3 else torch.fft.fft(pred, dim=-1)
        target_fft = torch.fft.fftn(target, dim=(-3, -2)) if target.dim() >= 3 else torch.fft.fft(target, dim=-1)
        
        # Spectral magnitude
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Basic spectral loss
        spectral_loss = F.mse_loss(pred_mag, target_mag)
        
        # High frequency emphasis
        if pred.dim() >= 3:
            # Create high frequency mask
            h, w = pred_fft.shape[-2:]
            center_h, center_w = h // 2, w // 2
            
            # Distance from center in frequency domain
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
            
            # High frequency mask (outer regions)
            high_freq_mask = (dist > min(h, w) * 0.3).float().to(pred.device)
            
            # Weighted loss
            weighted_pred = pred_mag * (1 + self.high_freq_weight * high_freq_mask)
            weighted_target = target_mag * (1 + self.high_freq_weight * high_freq_mask)
            
            high_freq_loss = F.mse_loss(weighted_pred, weighted_target)
        else:
            high_freq_loss = spectral_loss
        
        # Phase loss
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        phase_loss = F.mse_loss(pred_phase, target_phase)
        
        return {
            'spectral_loss': spectral_loss,
            'high_freq_loss': high_freq_loss,
            'phase_loss': phase_loss,
            'total_spectral_loss': spectral_loss + 0.1 * phase_loss + 0.5 * high_freq_loss
        }


class ConservationLoss(nn.Module):
    """Loss for enforcing conservation laws."""
    
    def __init__(self,
                 energy_weight: float = 1.0,
                 mass_weight: float = 1.0,
                 momentum_weight: float = 1.0):
        """Initialize conservation loss.
        
        Args:
            energy_weight: Weight for energy conservation
            mass_weight: Weight for mass conservation
            momentum_weight: Weight for momentum conservation
        """
        super().__init__()
        self.energy_weight = energy_weight
        self.mass_weight = mass_weight
        self.momentum_weight = momentum_weight
        
        self.conservation_checker = ConservationLawChecker()
    
    def forward(self, pred: torch.Tensor, 
                initial_state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute conservation loss."""
        losses = {}
        
        # Energy conservation
        if self.energy_weight > 0:
            energy_violation = self.conservation_checker.energy_conservation(pred)
            losses['energy_loss'] = torch.tensor(energy_violation, device=pred.device)
        
        # Mass conservation
        if self.mass_weight > 0:
            mass_violation = self.conservation_checker.mass_conservation(pred)
            losses['mass_loss'] = torch.tensor(mass_violation, device=pred.device)
        
        # Momentum conservation (for vector fields)
        if self.momentum_weight > 0 and pred.size(-1) >= 2:
            momentum_loss = self._compute_momentum_conservation(pred)
            losses['momentum_loss'] = momentum_loss
        else:
            losses['momentum_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Total conservation loss
        total_loss = (self.energy_weight * losses['energy_loss'] +
                     self.mass_weight * losses['mass_loss'] +
                     self.momentum_weight * losses['momentum_loss'])
        
        losses['total_conservation_loss'] = total_loss
        
        return losses
    
    def _compute_momentum_conservation(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute momentum conservation violation."""
        if pred.dim() < 4 or pred.size(1) < 2:  # Need time dimension
            return torch.tensor(0.0, device=pred.device)
        
        # Total momentum over spatial dimensions
        momentum = torch.sum(pred, dim=(-3, -2))  # Sum over spatial dims
        
        # Momentum should be conserved over time
        if momentum.size(1) > 1:
            momentum_diff = torch.diff(momentum, dim=1)
            return torch.mean(momentum_diff**2)
        
        return torch.tensor(0.0, device=pred.device)


class AdaptiveLoss(nn.Module):
    """Adaptive loss that adjusts weights during training."""
    
    def __init__(self, 
                 base_loss: nn.Module,
                 adaptation_rate: float = 0.01,
                 target_ratios: Optional[Dict[str, float]] = None):
        """Initialize adaptive loss.
        
        Args:
            base_loss: Base loss function
            adaptation_rate: Rate of weight adaptation
            target_ratios: Target ratios for different loss components
        """
        super().__init__()
        self.base_loss = base_loss
        self.adaptation_rate = adaptation_rate
        self.target_ratios = target_ratios or {}
        
        # Track loss history for adaptation
        self.loss_history = {}
        self.step_count = 0
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute adaptive loss."""
        losses = self.base_loss(*args, **kwargs)
        
        # Update loss history
        for key, value in losses.items():
            if key not in self.loss_history:
                self.loss_history[key] = []
            self.loss_history[key].append(value.item())
        
        self.step_count += 1
        
        # Adapt weights periodically
        if self.step_count % 100 == 0:
            self._adapt_weights()
        
        return losses
    
    def _adapt_weights(self):
        """Adapt loss weights based on history."""
        if len(self.loss_history) < 2:
            return
        
        # Compute recent averages
        recent_losses = {}
        for key, history in self.loss_history.items():
            if len(history) >= 10:
                recent_losses[key] = np.mean(history[-10:])
        
        # Adapt weights based on relative magnitudes
        if 'data_loss' in recent_losses and 'pde_loss' in recent_losses:
            ratio = recent_losses['data_loss'] / (recent_losses['pde_loss'] + 1e-8)
            
            # Increase PDE weight if data loss dominates
            if ratio > 10:
                if hasattr(self.base_loss, 'pde_weight'):
                    self.base_loss.pde_weight *= (1 + self.adaptation_rate)
            # Decrease PDE weight if PDE loss dominates
            elif ratio < 0.1:
                if hasattr(self.base_loss, 'pde_weight'):
                    self.base_loss.pde_weight *= (1 - self.adaptation_rate)


def create_physics_loss(pde_type: str, **kwargs) -> PhysicsInformedLoss:
    """Create physics-informed loss for specific PDE types.
    
    Args:
        pde_type: Type of PDE ('navier_stokes', 'heat', 'wave', 'burgers')
        **kwargs: Additional arguments for loss configuration
        
    Returns:
        Configured physics-informed loss
    """
    
    if pde_type == 'navier_stokes':
        return PhysicsInformedLoss(
            data_weight=1.0,
            pde_weight=0.1,
            conservation_weight=0.05,
            pde_residual_fn=navier_stokes_residual,
            **kwargs
        )
    
    elif pde_type == 'heat':
        return PhysicsInformedLoss(
            data_weight=1.0,
            pde_weight=0.2,
            boundary_weight=0.1,
            pde_residual_fn=heat_equation_residual,
            **kwargs
        )
    
    elif pde_type == 'wave':
        return PhysicsInformedLoss(
            data_weight=1.0,
            pde_weight=0.15,
            boundary_weight=0.05,
            pde_residual_fn=wave_equation_residual,
            **kwargs
        )
    
    elif pde_type == 'burgers':
        return PhysicsInformedLoss(
            data_weight=1.0,
            pde_weight=0.1,
            pde_residual_fn=burgers_equation_residual,
            **kwargs
        )
    
    else:
        return PhysicsInformedLoss(**kwargs)


# Specific PDE residual functions

def navier_stokes_residual(u: torch.Tensor, coords: torch.Tensor, 
                          inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Incompressible Navier-Stokes residual."""
    if u.size(-1) < 2:
        return torch.tensor(0.0, device=u.device)
    
    # Velocity components
    velocity = u[..., :2]
    
    # Incompressibility: ∇·u = 0
    div_u = torch.zeros(velocity.shape[:-1], device=velocity.device)
    
    if velocity.size(-3) > 1:
        div_u += torch.diff(velocity[..., 0], dim=-2)
    if velocity.size(-2) > 1:
        div_u += torch.diff(velocity[..., 1], dim=-1)
    
    return torch.mean(div_u**2)


def heat_equation_residual(u: torch.Tensor, coords: torch.Tensor,
                          inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Heat equation residual: ∂u/∂t - α∇²u = 0."""
    # Simplified implementation
    if u.dim() < 3:
        return torch.tensor(0.0, device=u.device)
    
    # Simple smoothness penalty as proxy for Laplacian
    smoothness = 0
    for dim in range(-2, 0):  # Last two spatial dimensions
        if u.size(dim) > 2:
            second_diff = u.diff(dim=dim).diff(dim=dim)
            smoothness += torch.mean(second_diff**2)
    
    return smoothness


def wave_equation_residual(u: torch.Tensor, coords: torch.Tensor,
                          inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Wave equation residual: ∂²u/∂t² - c²∇²u = 0."""
    # Simplified implementation
    return heat_equation_residual(u, coords, inputs)  # Similar structure


def burgers_equation_residual(u: torch.Tensor, coords: torch.Tensor,
                             inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Burgers equation residual: ∂u/∂t + u∂u/∂x - ν∇²u = 0."""
    if u.dim() < 2:
        return torch.tensor(0.0, device=u.device)
    
    # Nonlinearity penalty (simplified)
    if u.size(-1) > 1:
        u_grad = torch.diff(u, dim=-1)
        nonlinear_term = u[..., :-1] * u_grad
        return torch.mean(nonlinear_term**2)
    
    return torch.tensor(0.0, device=u.device)
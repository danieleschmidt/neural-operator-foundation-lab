"""Advanced input validation and security for neural operators.

This module provides comprehensive input validation, sanitization, and security
measures specifically designed for neural operator architectures.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
import warnings
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    sanitized_input: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class SecurityValidator(ABC):
    """Abstract base class for security validators."""
    
    @abstractmethod
    def validate(self, x: torch.Tensor) -> ValidationResult:
        """Validate input tensor for security threats."""
        pass


class AdversarialDetector(SecurityValidator):
    """Detect potential adversarial inputs based on statistical properties."""
    
    def __init__(
        self,
        max_value_threshold: float = 1e6,
        min_value_threshold: float = -1e6,
        std_multiplier_threshold: float = 10.0,
        gradient_norm_threshold: float = 1e3
    ):
        self.max_value_threshold = max_value_threshold
        self.min_value_threshold = min_value_threshold
        self.std_multiplier_threshold = std_multiplier_threshold
        self.gradient_norm_threshold = gradient_norm_threshold
        
    def validate(self, x: torch.Tensor) -> ValidationResult:
        """Detect potential adversarial patterns."""
        warnings_list = []
        
        # Check for extreme values
        max_val = torch.max(x).item()
        min_val = torch.min(x).item()
        
        if max_val > self.max_value_threshold or min_val < self.min_value_threshold:
            return ValidationResult(
                is_valid=False,
                error_message=f"Extreme values detected: min={min_val:.2e}, max={max_val:.2e}"
            )
        
        # Check for statistical anomalies
        mean_val = torch.mean(x).item()
        std_val = torch.std(x).item()
        
        # Detect unusual standard deviation patterns
        if std_val > self.std_multiplier_threshold * abs(mean_val):
            warnings_list.append(f"High variance detected: std/mean ratio = {std_val/abs(mean_val):.2f}")
        
        # Check gradient magnitude if requires_grad
        if x.requires_grad and x.grad is not None:
            grad_norm = torch.norm(x.grad).item()
            if grad_norm > self.gradient_norm_threshold:
                warnings_list.append(f"Large gradient norm detected: {grad_norm:.2e}")
        
        # Check for frequency domain anomalies (for spatial data)
        if len(x.shape) >= 3:  # Spatial data
            try:
                x_fft = torch.fft.fft2(x[0, :, :] if len(x.shape) == 4 else x[0])
                fft_magnitude = torch.abs(x_fft)
                
                # Check for concentrated energy in high frequencies (potential adversarial noise)
                high_freq_energy = torch.sum(fft_magnitude[-fft_magnitude.shape[0]//4:, -fft_magnitude.shape[1]//4:])
                total_energy = torch.sum(fft_magnitude)
                
                if high_freq_energy / total_energy > 0.3:
                    warnings_list.append("High frequency energy concentration detected")
                    
            except Exception:
                pass  # Skip FFT analysis if not applicable
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings_list,
            metadata={
                'max_value': max_val,
                'min_value': min_val,
                'mean': mean_val,
                'std': std_val
            }
        )


class NumericalStabilityValidator(SecurityValidator):
    """Validate numerical stability of inputs."""
    
    def __init__(
        self,
        nan_tolerance: bool = False,
        inf_tolerance: bool = False,
        condition_number_threshold: float = 1e12
    ):
        self.nan_tolerance = nan_tolerance
        self.inf_tolerance = inf_tolerance
        self.condition_number_threshold = condition_number_threshold
        
    def validate(self, x: torch.Tensor) -> ValidationResult:
        """Check for numerical stability issues."""
        # Check for NaN values
        if torch.isnan(x).any():
            if not self.nan_tolerance:
                return ValidationResult(
                    is_valid=False,
                    error_message="NaN values detected in input"
                )
        
        # Check for infinite values
        if torch.isinf(x).any():
            if not self.inf_tolerance:
                return ValidationResult(
                    is_valid=False,
                    error_message="Infinite values detected in input"
                )
        
        warnings_list = []
        
        # Check condition number for matrix-like inputs
        if len(x.shape) >= 2:
            try:
                # Flatten to 2D for condition number calculation
                x_2d = x.reshape(x.shape[0], -1)
                if x_2d.shape[1] > 1:
                    # Compute condition number
                    U, S, V = torch.svd(x_2d)
                    condition_number = (S.max() / S.min()).item()
                    
                    if condition_number > self.condition_number_threshold:
                        warnings_list.append(f"High condition number: {condition_number:.2e}")
            except Exception:
                pass  # Skip if SVD fails
        
        # Check dynamic range
        if x.numel() > 0:
            min_abs = torch.min(torch.abs(x[x != 0])) if torch.any(x != 0) else torch.tensor(0.0)
            max_abs = torch.max(torch.abs(x))
            
            if min_abs > 0:
                dynamic_range = (max_abs / min_abs).item()
                if dynamic_range > 1e10:
                    warnings_list.append(f"Large dynamic range: {dynamic_range:.2e}")
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings_list
        )


class GeometricValidator(SecurityValidator):
    """Validate geometric and spatial properties of inputs."""
    
    def __init__(
        self,
        expected_spatial_dims: Optional[List[int]] = None,
        min_resolution: int = 4,
        max_resolution: int = 2048,
        check_spatial_continuity: bool = True
    ):
        self.expected_spatial_dims = expected_spatial_dims
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.check_spatial_continuity = check_spatial_continuity
        
    def validate(self, x: torch.Tensor) -> ValidationResult:
        """Validate geometric properties of spatial inputs."""
        warnings_list = []
        
        # Check spatial dimensions
        if len(x.shape) >= 3:  # Spatial data
            spatial_shape = x.shape[1:-1] if len(x.shape) == 4 else x.shape[1:]
            
            # Check resolution bounds
            for dim_size in spatial_shape:
                if dim_size < self.min_resolution:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Spatial resolution too low: {dim_size} < {self.min_resolution}"
                    )
                if dim_size > self.max_resolution:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Spatial resolution too high: {dim_size} > {self.max_resolution}"
                    )
            
            # Check expected dimensions
            if self.expected_spatial_dims is not None:
                if len(spatial_shape) != len(self.expected_spatial_dims):
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Unexpected spatial dimensions: {len(spatial_shape)} != {len(self.expected_spatial_dims)}"
                    )
            
            # Check spatial continuity
            if self.check_spatial_continuity and len(spatial_shape) == 2:
                try:
                    # Compute spatial gradients
                    if len(x.shape) == 4:  # (B, H, W, C)
                        sample = x[0, :, :, 0]
                    else:  # (B, H, W) or similar
                        sample = x[0]
                    
                    grad_x = torch.diff(sample, dim=0)
                    grad_y = torch.diff(sample, dim=1)
                    
                    # Check for extreme gradients (potential discontinuities)
                    max_grad_x = torch.max(torch.abs(grad_x)).item()
                    max_grad_y = torch.max(torch.abs(grad_y)).item()
                    
                    mean_val = torch.mean(torch.abs(sample)).item()
                    
                    if max_grad_x > 10 * mean_val or max_grad_y > 10 * mean_val:
                        warnings_list.append("Large spatial gradients detected - potential discontinuities")
                        
                except Exception:
                    pass  # Skip continuity check if not applicable
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings_list,
            metadata={'spatial_shape': spatial_shape if len(x.shape) >= 3 else None}
        )


class QuantumInputValidator(SecurityValidator):
    """Specialized validator for quantum-inspired neural operator inputs."""
    
    def __init__(
        self,
        check_quantum_constraints: bool = True,
        max_entanglement_strength: float = 1.0,
        phase_continuity_threshold: float = 2*np.pi
    ):
        self.check_quantum_constraints = check_quantum_constraints
        self.max_entanglement_strength = max_entanglement_strength
        self.phase_continuity_threshold = phase_continuity_threshold
        
    def validate(self, x: torch.Tensor) -> ValidationResult:
        """Validate quantum-specific properties of inputs."""
        warnings_list = []
        
        if self.check_quantum_constraints:
            # Check for complex-valued inputs (quantum amplitudes)
            if torch.is_complex(x):
                # Validate quantum normalization constraints
                magnitude_squared = torch.abs(x) ** 2
                
                # For quantum states, sum of probability amplitudes should be reasonable
                if len(x.shape) >= 2:
                    prob_sums = torch.sum(magnitude_squared, dim=-1)
                    
                    # Check if probabilities are approximately normalized
                    if torch.any(prob_sums > 2.0):
                        warnings_list.append("High probability amplitude sums detected")
                    
                    if torch.any(prob_sums < 0.1):
                        warnings_list.append("Very low probability amplitude sums detected")
                
                # Check phase continuity
                phases = torch.angle(x)
                if torch.numel(phases) > 1:
                    phase_diffs = torch.diff(phases.flatten())
                    max_phase_jump = torch.max(torch.abs(phase_diffs)).item()
                    
                    if max_phase_jump > self.phase_continuity_threshold:
                        warnings_list.append(f"Large phase discontinuity: {max_phase_jump:.2f} rad")
            
            # Check for quantum entanglement-like correlations
            if len(x.shape) >= 3:
                try:
                    # Simple correlation analysis between spatial regions
                    if len(x.shape) == 4:  # (B, H, W, C)
                        h, w = x.shape[1], x.shape[2]
                        region1 = x[:, :h//2, :w//2, :].flatten()
                        region2 = x[:, h//2:, w//2:, :].flatten()
                        
                        if len(region1) > 1 and len(region2) > 1:
                            correlation = torch.corrcoef(torch.stack([region1, region2]))[0, 1]
                            
                            if torch.abs(correlation) > self.max_entanglement_strength:
                                warnings_list.append(f"High spatial correlation: {correlation:.3f}")
                                
                except Exception:
                    pass  # Skip correlation analysis if not applicable
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings_list
        )


class ComprehensiveInputValidator:
    """Comprehensive input validation combining multiple validators."""
    
    def __init__(
        self,
        enable_adversarial_detection: bool = True,
        enable_numerical_stability: bool = True,
        enable_geometric_validation: bool = True,
        enable_quantum_validation: bool = True,
        custom_validators: Optional[List[SecurityValidator]] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.validators = []
        
        if enable_adversarial_detection:
            self.validators.append(AdversarialDetector())
            
        if enable_numerical_stability:
            self.validators.append(NumericalStabilityValidator())
            
        if enable_geometric_validation:
            self.validators.append(GeometricValidator())
            
        if enable_quantum_validation:
            self.validators.append(QuantumInputValidator())
        
        if custom_validators:
            self.validators.extend(custom_validators)
        
        self.validation_count = 0
        self.failed_validations = 0
        
    def validate_comprehensive(
        self, 
        x: torch.Tensor,
        strict_mode: bool = False,
        return_sanitized: bool = False
    ) -> ValidationResult:
        """Run comprehensive validation using all enabled validators.
        
        Args:
            x: Input tensor to validate
            strict_mode: If True, fail on any validator failure
            return_sanitized: If True, attempt to sanitize invalid inputs
            
        Returns:
            Comprehensive validation result
        """
        self.validation_count += 1
        
        all_warnings = []
        all_metadata = {}
        sanitized_input = None
        
        for validator in self.validators:
            try:
                result = validator.validate(x)
                
                # Collect warnings and metadata
                all_warnings.extend(result.warnings)
                all_metadata.update(result.metadata)
                
                # Handle validation failures
                if not result.is_valid:
                    self.failed_validations += 1
                    
                    if strict_mode:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"{validator.__class__.__name__}: {result.error_message}",
                            warnings=all_warnings,
                            metadata=all_metadata
                        )
                    else:
                        all_warnings.append(f"Validation failed: {result.error_message}")
                        
                        # Attempt sanitization if requested
                        if return_sanitized and sanitized_input is None:
                            sanitized_input = self._sanitize_input(x, result.error_message)
                
            except Exception as e:
                error_msg = f"Validator {validator.__class__.__name__} failed: {str(e)}"
                self.logger.error(error_msg)
                all_warnings.append(error_msg)
        
        # Log warnings if any
        if all_warnings:
            self.logger.warning(f"Input validation warnings: {all_warnings}")
        
        return ValidationResult(
            is_valid=True,
            warnings=all_warnings,
            sanitized_input=sanitized_input,
            metadata={
                **all_metadata,
                'validation_count': self.validation_count,
                'failed_validations': self.failed_validations,
                'validator_count': len(self.validators)
            }
        )
    
    def _sanitize_input(self, x: torch.Tensor, error_message: str) -> torch.Tensor:
        """Attempt to sanitize invalid input."""
        sanitized = x.clone()
        
        # Handle common issues
        if "NaN" in error_message:
            sanitized = torch.where(torch.isnan(sanitized), torch.zeros_like(sanitized), sanitized)
        
        if "Infinite" in error_message:
            sanitized = torch.clamp(sanitized, min=-1e6, max=1e6)
        
        if "Extreme values" in error_message:
            # Clip to reasonable range
            percentile_99 = torch.quantile(torch.abs(sanitized), 0.99)
            sanitized = torch.clamp(sanitized, min=-percentile_99, max=percentile_99)
        
        return sanitized
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'total_validations': self.validation_count,
            'failed_validations': self.failed_validations,
            'success_rate': (self.validation_count - self.failed_validations) / max(1, self.validation_count),
            'active_validators': [v.__class__.__name__ for v in self.validators],
            'validator_count': len(self.validators)
        }


def create_neural_operator_input_validator(
    model_type: str = "quantum_spectral_attention",
    security_level: str = "standard"  # "basic", "standard", "strict"
) -> ComprehensiveInputValidator:
    """Factory function to create input validator for specific neural operator types.
    
    Args:
        model_type: Type of neural operator ("fno", "tno", "pno", "qisa", etc.)
        security_level: Security validation level
        
    Returns:
        Configured ComprehensiveInputValidator
    """
    if security_level == "basic":
        return ComprehensiveInputValidator(
            enable_adversarial_detection=False,
            enable_numerical_stability=True,
            enable_geometric_validation=False,
            enable_quantum_validation=False
        )
    elif security_level == "standard":
        enable_quantum = model_type.lower() in ["qisa", "quantum_spectral_attention", "quantum"]
        return ComprehensiveInputValidator(
            enable_adversarial_detection=True,
            enable_numerical_stability=True,
            enable_geometric_validation=True,
            enable_quantum_validation=enable_quantum
        )
    elif security_level == "strict":
        return ComprehensiveInputValidator(
            enable_adversarial_detection=True,
            enable_numerical_stability=True,
            enable_geometric_validation=True,
            enable_quantum_validation=True,
            custom_validators=[
                AdversarialDetector(
                    max_value_threshold=1e4,  # Stricter thresholds
                    std_multiplier_threshold=5.0
                )
            ]
        )
    else:
        raise ValueError(f"Unknown security level: {security_level}")
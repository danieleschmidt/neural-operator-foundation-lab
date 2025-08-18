"""Multi-Resolution Adaptive Neural Operator implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import math

from ..base import NeuralOperatorBase


class AdaptivePooling(nn.Module):
    """Adaptive pooling for multi-resolution processing."""
    
    def __init__(self, base_resolution: int = 64, max_resolution: int = 1024):
        super().__init__()
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        
        # Adaptive pooling layers for different resolutions
        self.pooling_layers = nn.ModuleDict({
            f'pool_{res}': nn.AdaptiveAvgPool3d((res, res, res))
            for res in [32, 64, 128, 256, 512, 1024]
            if res <= max_resolution
        })
        
    def forward(self, x: torch.Tensor, target_resolution: int) -> torch.Tensor:
        """Adaptively pool to target resolution."""
        if target_resolution in [32, 64, 128, 256, 512, 1024]:
            pool_layer = self.pooling_layers[f'pool_{target_resolution}']
            return pool_layer(x)
        else:
            # Use interpolation for arbitrary resolutions
            return F.interpolate(x, size=(target_resolution,) * 3, mode='trilinear', align_corners=False)


class ResolutionEncoder(nn.Module):
    """Encoder that learns resolution-aware features."""
    
    def __init__(self, input_channels: int, hidden_dim: int):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # Multi-scale convolutions
        self.conv_layers = nn.ModuleList([
            nn.Conv3d(input_channels, hidden_dim // 4, kernel_size=3, padding=1),
            nn.Conv3d(input_channels, hidden_dim // 4, kernel_size=5, padding=2),
            nn.Conv3d(input_channels, hidden_dim // 4, kernel_size=7, padding=3),
            nn.Conv3d(input_channels, hidden_dim // 4, kernel_size=1, padding=0)
        ])
        
        self.activation = nn.GELU()
        self.norm = nn.BatchNorm3d(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-scale feature encoding."""
        features = []
        for conv in self.conv_layers:
            features.append(self.activation(conv(x)))
        
        # Concatenate multi-scale features
        out = torch.cat(features, dim=1)
        return self.norm(out)


class AdaptiveAttention(nn.Module):
    """Attention mechanism that adapts to different resolutions."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, max_resolution: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_resolution = max_resolution
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Resolution-adaptive positional encoding
        self.pos_encoding = nn.ModuleDict({
            f'pos_{res}': nn.Parameter(torch.randn(1, res**3, embed_dim))
            for res in [32, 64, 128, 256, 512]
            if res <= max_resolution
        })
        
    def get_positional_encoding(self, resolution: int) -> torch.Tensor:
        """Get positional encoding for given resolution."""
        if f'pos_{resolution}' in self.pos_encoding:
            return self.pos_encoding[f'pos_{resolution}']
        else:
            # Interpolate from nearest available resolution
            available_resolutions = [int(k.split('_')[1]) for k in self.pos_encoding.keys()]
            nearest_res = min(available_resolutions, key=lambda x: abs(x - resolution))
            pos_enc = self.pos_encoding[f'pos_{nearest_res}']
            
            # Interpolate to target resolution
            current_length = pos_enc.shape[1]
            target_length = resolution**3
            if current_length != target_length:
                pos_enc = F.interpolate(
                    pos_enc.transpose(1, 2), 
                    size=target_length, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            
            return pos_enc
    
    def forward(self, x: torch.Tensor, resolution: int) -> torch.Tensor:
        """Apply resolution-adaptive attention."""
        B, N, C = x.shape
        
        # Add positional encoding
        pos_enc = self.get_positional_encoding(resolution)
        if pos_enc.shape[1] == N:
            x = x + pos_enc
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class MultiResolutionAdaptiveNeuralOperator(NeuralOperatorBase):
    """Neural Operator that adapts to multiple resolutions."""
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 256,
        output_dim: int = 4,
        num_layers: int = 6,
        num_heads: int = 8,
        base_resolution: int = 64,
        max_resolution: int = 1024,
        adaptive_levels: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.adaptive_levels = adaptive_levels
        
        # Resolution-aware components
        self.adaptive_pooling = AdaptivePooling(base_resolution, max_resolution)
        self.resolution_encoder = ResolutionEncoder(input_dim, hidden_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-resolution adaptive layers
        self.adaptive_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': AdaptiveAttention(hidden_dim, num_heads, max_resolution),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, 4 * hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'resolution_gate': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Sigmoid()
                )
            })
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Resolution predictor
        self.resolution_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, adaptive_levels),
            nn.Softmax(dim=-1)
        )
        
    def predict_optimal_resolution(self, x: torch.Tensor) -> int:
        """Predict optimal resolution for the input."""
        # Global average pooling
        x_global = x.mean(dim=1)
        
        # Predict resolution level
        resolution_probs = self.resolution_predictor(x_global)
        resolution_level = torch.argmax(resolution_probs, dim=-1)
        
        # Map to actual resolution
        resolution_map = {0: 64, 1: 128, 2: 256}
        if self.adaptive_levels > 3:
            resolution_map.update({3: 512, 4: 1024})
        
        return resolution_map.get(resolution_level.item(), self.base_resolution)
    
    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None, target_resolution: Optional[int] = None, **kwargs) -> torch.Tensor:
        """Forward pass with adaptive resolution."""
        B, N, C = x.shape
        
        # Determine target resolution
        if target_resolution is None:
            target_resolution = self.predict_optimal_resolution(x)
        
        # Project input
        x = self.input_proj(x)
        
        # Multi-resolution processing
        for i, layer in enumerate(self.adaptive_layers):
            # Adaptive attention
            attn_out = layer['attention'](x, target_resolution)
            x = layer['norm1'](x + attn_out)
            
            # Resolution gating
            gate = layer['resolution_gate'](x)
            
            # MLP with resolution gating
            mlp_out = layer['mlp'](x)
            x = layer['norm2'](x + gate * mlp_out)
        
        # Output projection
        output = self.output_proj(x)
        
        return output
    
    def forward_multi_resolution(self, x: torch.Tensor, resolutions: List[int]) -> Dict[int, torch.Tensor]:
        """Forward pass at multiple resolutions."""
        results = {}
        
        for resolution in resolutions:
            # Adapt input to target resolution if needed
            x_adapted = x
            if hasattr(x, 'shape') and len(x.shape) == 5:  # Assume (B, C, D, H, W)
                x_adapted = self.adaptive_pooling(x, resolution)
                # Flatten spatial dimensions
                B, C, D, H, W = x_adapted.shape
                x_adapted = x_adapted.view(B, D*H*W, C)
            
            # Forward pass
            output = self.forward(x_adapted, target_resolution=resolution)
            results[resolution] = output
        
        return results
    
    def compute_adaptive_loss(self, predictions: Dict[int, torch.Tensor], targets: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Compute loss across multiple resolutions."""
        total_loss = 0.0
        weight_sum = 0.0
        
        for resolution in predictions.keys():
            if resolution in targets:
                # Weight higher resolutions more
                weight = math.log2(resolution / self.base_resolution) + 1
                loss = F.mse_loss(predictions[resolution], targets[resolution])
                total_loss += weight * loss
                weight_sum += weight
        
        return total_loss / weight_sum if weight_sum > 0 else total_loss
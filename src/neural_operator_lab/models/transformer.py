"""Transformer Neural Operator (TNO) implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import numpy as np

from ..base import NeuralOperatorBase
from ..utils import FourierFeatures


class MultiHeadAttention(nn.Module):
    """Multi-head attention with spatial awareness."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional bias
        self.relative_position_bias = nn.Parameter(
            torch.zeros(2 * 64 - 1, 2 * 64 - 1, num_heads)
        )
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, pos_encoding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Multi-head attention forward pass."""
        B, N, C = x.shape
        
        # Add positional encoding if provided
        if pos_encoding is not None:
            x = x + pos_encoding
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias (simplified)
        if hasattr(self, 'relative_position_bias'):
            # Truncate bias to match sequence length
            bias_size = min(self.relative_position_bias.size(0), int(math.sqrt(N)))
            if bias_size > 1:
                rel_bias = self.relative_position_bias[:bias_size, :bias_size, :]
                rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, H, W)
                
                # Reshape and add to attention
                if rel_bias.size(-1) == int(math.sqrt(N)):
                    rel_bias = rel_bias.flatten(-2).unsqueeze(0)  # (1, 1, num_heads, N)
                    attn = attn + rel_bias[:, :, :, :N]
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TNOBlock(nn.Module):
    """Transformer Neural Operator block."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = FeedForward(embed_dim, hidden_dim, dropout)
        
        # Stochastic depth
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, pos_encoding: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x), pos_encoding))
        
        # Feed-forward with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class StochasticDepth(nn.Module):
    """Stochastic depth (drop path) regularization."""
    
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        
        return x.div(keep_prob) * random_tensor


class CrossAttention(nn.Module):
    """Cross-attention for function-to-function mapping."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Cross-attention between query and key-value functions."""
        B, N_q, C = query.shape
        B, N_kv, C = key_value.shape
        
        # Project query
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (B, num_heads, N_q, head_dim)
        
        # Project key and value
        kv = self.kv_proj(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, num_heads, N_kv, head_dim)
        k, v = kv[0], kv[1]
        
        # Cross-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)
        
        return out


class TransformerNeuralOperator(NeuralOperatorBase):
    """Transformer Neural Operator for learning operators between function spaces."""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 512,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 drop_path: float = 0.1,
                 fourier_features: bool = True,
                 max_resolution: int = 1024,
                 **kwargs):
        """Initialize TNO.
        
        Args:
            input_dim: Input function dimension
            output_dim: Output function dimension
            hidden_dim: Hidden dimension of transformer
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout rate
            drop_path: Stochastic depth rate
            fourier_features: Whether to use Fourier positional encoding
            max_resolution: Maximum supported resolution
        """
        super().__init__(input_dim, output_dim, **kwargs)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.fourier_features = fourier_features
        self.max_resolution = max_resolution
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        if fourier_features:
            self.pos_encoder = FourierFeatures(hidden_dim, scale=10.0)
            self.pos_projection = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.pos_encoder = None
            # Learnable positional embeddings
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, max_resolution**2, hidden_dim)
            )
        
        # Transformer blocks
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.blocks = nn.ModuleList([
            TNOBlock(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=drop_path_rates[i]
            )
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Cross-attention for multi-resolution handling
        self.cross_attn = CrossAttention(hidden_dim, num_heads, dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract features using transformer encoder."""
        B, *spatial_dims, C = x.shape
        
        # Flatten spatial dimensions
        x = x.view(B, -1, C)  # (B, N, C)
        N = x.size(1)
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Positional encoding
        if self.fourier_features and grid is not None:
            # Use Fourier features for positional encoding
            grid_flat = grid.view(B, -1, grid.size(-1))
            pos_encoding = self.pos_encoder(grid_flat)
            pos_encoding = self.pos_projection(pos_encoding)
        elif hasattr(self, 'pos_embedding'):
            # Use learned positional embeddings
            pos_encoding = self.pos_embedding[:, :N, :]
        else:
            pos_encoding = None
        
        return x, pos_encoding
    
    def forward_operator(self, features: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Apply transformer neural operator."""
        x, pos_encoding = features
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, pos_encoding)
        
        # Output projection
        output = self.output_projection(x)
        
        return output
    
    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through TNO."""
        # Get original spatial shape
        original_shape = x.shape
        
        # Extract features
        features = self.forward_features(x, grid)
        
        # Apply operator
        output = self.forward_operator(features)
        
        # Reshape to original spatial dimensions
        B = original_shape[0]
        spatial_dims = original_shape[1:-1]
        output_dim = output.size(-1)
        
        output = output.view(B, *spatial_dims, output_dim)
        
        return output
    
    def forward_with_attention_maps(self, x: torch.Tensor, 
                                  grid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """Forward pass returning attention maps for visualization."""
        original_shape = x.shape
        features = self.forward_features(x, grid)
        x, pos_encoding = features
        
        attention_maps = []
        
        # Store attention maps from each block
        for block in self.blocks:
            # Manually compute attention for visualization
            normed_x = block.norm1(x)
            if pos_encoding is not None:
                normed_x = normed_x + pos_encoding
            
            B, N, C = normed_x.shape
            qkv = block.attn.qkv(normed_x).reshape(B, N, 3, block.attn.num_heads, block.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = F.softmax(attn, dim=-1)
            
            attention_maps.append(attn.detach())
            
            # Continue with block forward
            x = block(x, pos_encoding)
        
        output = self.output_projection(x)
        
        # Reshape output
        B = original_shape[0]
        spatial_dims = original_shape[1:-1]
        output_dim = output.size(-1)
        output = output.view(B, *spatial_dims, output_dim)
        
        return output, attention_maps
    
    def compute_pde_residual(self, pred: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed loss."""
        # Example: minimize curl for potential flow
        if pred.dim() >= 4 and pred.size(-1) >= 2:  # Vector field
            # Compute approximate curl using finite differences
            u = pred[..., 0]  # x-component
            v = pred[..., 1]  # y-component
            
            # Finite difference approximation
            du_dy = torch.diff(u, dim=-2)
            dv_dx = torch.diff(v, dim=-1)
            
            # Pad to match dimensions
            if du_dy.shape != dv_dx.shape:
                min_h = min(du_dy.size(-2), dv_dx.size(-2))
                min_w = min(du_dy.size(-1), dv_dx.size(-1))
                du_dy = du_dy[..., :min_h, :min_w]
                dv_dx = dv_dx[..., :min_h, :min_w]
            
            curl = dv_dx - du_dy
            physics_loss = torch.mean(curl**2)
            
            return physics_loss
        
        return torch.tensor(0.0, device=pred.device)


def create_tno_2d(input_dim: int = 3, output_dim: int = 1, 
                  hidden_dim: int = 256, **kwargs) -> TransformerNeuralOperator:
    """Create 2D TNO for problems like Darcy flow."""
    return TransformerNeuralOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=8,
        num_heads=8,
        **kwargs
    )


def create_tno_3d(input_dim: int = 4, output_dim: int = 4,
                  hidden_dim: int = 512, **kwargs) -> TransformerNeuralOperator:
    """Create 3D TNO for problems like Navier-Stokes."""
    return TransformerNeuralOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=12,
        num_heads=8,
        max_resolution=256,  # Lower for 3D due to memory
        **kwargs
    )
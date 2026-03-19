"""Deep Operator Network (DeepONet) — Lu et al., 2021 (arXiv:1910.03193).

Architecture:
  - Branch net: encodes the input function evaluated at sensor points → embedding vector.
  - Trunk net:  encodes the query/output locations → embedding vector.
  - Output:     dot product of branch and trunk embeddings (+ optional bias).

The branch and trunk embeddings must have the same dimension ``p`` (the width of the
inner product space).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def _build_mlp(layer_sizes: List[int], activation: str = "tanh") -> nn.Sequential:
    """Build a fully-connected MLP with the given layer sizes."""
    act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
    layers: List[nn.Module] = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(act_fn())
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """Deep Operator Network.

    Args:
        branch_input_dim:   number of sensor points (input function is evaluated here).
        trunk_input_dim:    dimension of query/output locations (1 for 1-D PDE).
        branch_hidden:      list of hidden layer sizes for the branch net.
        trunk_hidden:       list of hidden layer sizes for the trunk net.
        p:                  inner-product / embedding dimension.
        out_channels:       number of output quantities per location (default 1).
        activation:         activation function name.
        use_bias:           add a learnable scalar bias to the output.

    Forward call:
        u  — input function values at sensors: (batch, branch_input_dim)
        y  — query locations:                  (batch, n_out, trunk_input_dim)
               OR (n_out, trunk_input_dim) — broadcast over batch.
    Returns:
        (batch, n_out, out_channels)
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int = 1,
        branch_hidden: Optional[List[int]] = None,
        trunk_hidden: Optional[List[int]] = None,
        p: int = 64,
        out_channels: int = 1,
        activation: str = "tanh",
        use_bias: bool = True,
    ):
        super().__init__()
        self.p = p
        self.out_channels = out_channels

        # Default MLP architectures
        branch_hidden = branch_hidden or [128, 128]
        trunk_hidden = trunk_hidden or [128, 128]

        branch_sizes = [branch_input_dim] + branch_hidden + [p * out_channels]
        trunk_sizes = [trunk_input_dim] + trunk_hidden + [p * out_channels]

        self.branch = _build_mlp(branch_sizes, activation)
        self.trunk = _build_mlp(trunk_sizes, activation)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, branch_input_dim) — input function at sensor locations.
            y: (batch, n_out, trunk_input_dim)  or  (n_out, trunk_input_dim)
        Returns:
            s: (batch, n_out, out_channels) — output function values at query points.
        """
        # Branch: (B, p * out_channels)
        b = self.branch(u)  # (B, p*out_channels)
        B = b.shape[0]

        # Trunk: handle unbatched y
        if y.dim() == 2:
            y = y.unsqueeze(0).expand(B, -1, -1)  # (B, n_out, trunk_dim)

        n_out = y.shape[1]
        # Flatten query batch for trunk net
        y_flat = y.reshape(B * n_out, -1)
        t = self.trunk(y_flat)  # (B*n_out, p*out_channels)
        t = t.view(B, n_out, self.p, self.out_channels)

        # Reshape branch for dot product: (B, p, out_channels)
        b = b.view(B, self.p, self.out_channels)

        # Dot product over p dimension: sum_i b[i] * t[i]
        # b: (B, 1, p, out_channels), t: (B, n_out, p, out_channels)
        s = (b.unsqueeze(1) * t).sum(dim=2)  # (B, n_out, out_channels)

        if self.bias is not None:
            s = s + self.bias

        return s

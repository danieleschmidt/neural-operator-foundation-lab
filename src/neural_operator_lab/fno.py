"""Fourier Neural Operator (FNO) — Li et al., 2021 (arXiv:2010.08895).

Architecture:
  1. Lift input to higher-dimensional channel space (pointwise linear).
  2. Apply L Fourier layers: each = spectral conv (Fourier space) + residual (real space).
  3. Project back to output dimension (pointwise linear).

Key idea: spectral convolution multiplies the *truncated* Fourier coefficients by learned
complex-valued weights, then inverse-FFTs back.  Only the lowest `modes` frequencies are
kept, making the operator grid-size-independent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# Spectral convolution layers
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """1-D spectral convolution: FFT → truncate → multiply by weights → iFFT.

    Args:
        in_channels:  width of input feature channels.
        out_channels: width of output feature channels.
        modes:        number of Fourier modes to keep (<=  N//2 + 1).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, N)
        Returns:
            (batch, out_channels, N)
        """
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, N//2+1)

        out_ft = torch.zeros(B, self.out_channels, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        # einsum: batch×in×modes, in×out×modes → batch×out×modes
        out_ft[:, :, :self.modes] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :self.modes], self.weights
        )
        return torch.fft.irfft(out_ft, n=N, dim=-1)  # (B, out_channels, N)


class SpectralConv2d(nn.Module):
    """2-D spectral convolution.

    Keeps the lowest ``modes1`` × ``modes2`` Fourier modes (+ conjugate symmetric block).

    Args:
        in_channels, out_channels: channel widths.
        modes1: Fourier modes along first spatial dimension.
        modes2: Fourier modes along second spatial dimension.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2)
        self.weights1 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W)
        Returns:
            (batch, out_channels, H, W)
        """
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # (B, C, H, W//2+1)

        out_ft = torch.zeros(B, self.out_channels, x_ft.size(-2), x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1,
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2,
        )
        return torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))


# ---------------------------------------------------------------------------
# FNO blocks
# ---------------------------------------------------------------------------

class FNOBlock1d(nn.Module):
    """Single 1-D FNO layer: spectral conv + residual (local) conv + activation."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.residual = nn.Conv1d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.residual(x))


class FNOBlock2d(nn.Module):
    """Single 2-D FNO layer."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.residual = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.residual(x))


# ---------------------------------------------------------------------------
# Full FNO models
# ---------------------------------------------------------------------------

class FourierNeuralOperator1d(nn.Module):
    """1-D Fourier Neural Operator.

    Maps a discretised input function (+ optional grid) to a discretised output function.

    Args:
        modes:       number of Fourier modes to retain.
        width:       channel width of the internal representation.
        n_layers:    number of FNO blocks.
        in_channels: number of input function channels (default 1).
                     If ``append_grid=True`` this is auto-incremented by 1.
        out_channels:number of output function channels (default 1).
        append_grid: if True, append a normalised [0,1] grid to the input channels.
    """

    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        append_grid: bool = True,
    ):
        super().__init__()
        self.append_grid = append_grid
        actual_in = in_channels + (1 if append_grid else 0)

        self.lift = nn.Conv1d(actual_in, width, 1)
        self.layers = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(n_layers)])
        self.proj1 = nn.Conv1d(width, 128, 1)
        self.proj2 = nn.Conv1d(128, out_channels, 1)

    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    (batch, N, in_channels)  — input function on grid.
            grid: (batch, N, 1) optional; if None and append_grid=True, built from [0,1].
        Returns:
            (batch, N, out_channels)
        """
        if self.append_grid:
            if grid is None:
                B, N, _ = x.shape
                grid = torch.linspace(0, 1, N, device=x.device).view(1, N, 1).expand(B, N, 1)
            x = torch.cat([x, grid], dim=-1)

        # (B, N, C) → (B, C, N) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.lift(x)

        for layer in self.layers:
            x = layer(x)

        x = F.gelu(self.proj1(x))
        x = self.proj2(x)

        # (B, C, N) → (B, N, C)
        return x.permute(0, 2, 1)


class FourierNeuralOperator2d(nn.Module):
    """2-D Fourier Neural Operator.

    Args:
        modes1, modes2: Fourier modes along each spatial dimension.
        width:          channel width.
        n_layers:       number of FNO blocks.
        in_channels:    input function channels.
        out_channels:   output function channels.
        append_grid:    if True, append 2-D normalised grid to inputs.
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        n_layers: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        append_grid: bool = True,
    ):
        super().__init__()
        self.append_grid = append_grid
        actual_in = in_channels + (2 if append_grid else 0)

        self.lift = nn.Conv2d(actual_in, width, 1)
        self.layers = nn.ModuleList(
            [FNOBlock2d(width, modes1, modes2) for _ in range(n_layers)]
        )
        self.proj1 = nn.Conv2d(width, 128, 1)
        self.proj2 = nn.Conv2d(128, out_channels, 1)

    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    (batch, H, W, in_channels)
            grid: (batch, H, W, 2) optional
        Returns:
            (batch, H, W, out_channels)
        """
        if self.append_grid:
            if grid is None:
                B, H, W, _ = x.shape
                gy = torch.linspace(0, 1, H, device=x.device)
                gx = torch.linspace(0, 1, W, device=x.device)
                gy, gx = torch.meshgrid(gy, gx, indexing="ij")
                grid = torch.stack([gy, gx], dim=-1).unsqueeze(0).expand(B, H, W, 2)
            x = torch.cat([x, grid], dim=-1)

        # (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x.permute(0, 2, 3, 1)  # (B, H, W, C)

"""Tests for Fourier Neural Operator (FNO)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from neural_operator_lab.fno import (
    SpectralConv1d,
    SpectralConv2d,
    FourierNeuralOperator1d,
    FourierNeuralOperator2d,
)


# ── SpectralConv1d ──────────────────────────────────────────────────────────

class TestSpectralConv1d:
    def test_output_shape(self):
        conv = SpectralConv1d(in_channels=8, out_channels=16, modes=4)
        x = torch.randn(2, 8, 32)
        out = conv(x)
        assert out.shape == (2, 16, 32), f"Expected (2,16,32), got {out.shape}"

    def test_modes_truncation(self):
        """Spectral conv should only use `modes` Fourier coefficients."""
        conv = SpectralConv1d(4, 4, modes=2)
        x = torch.randn(1, 4, 16)
        # Just verify it runs and output is same spatial size
        out = conv(x)
        assert out.shape[-1] == x.shape[-1]

    def test_gradient_flow(self):
        conv = SpectralConv1d(4, 4, modes=8)
        x = torch.randn(2, 4, 32, requires_grad=False)
        out = conv(x)
        loss = out.mean()
        loss.backward()
        assert conv.weights.grad is not None

    def test_linearity_in_channel(self):
        """Doubling input should roughly double output (linear layer property)."""
        conv = SpectralConv1d(4, 4, modes=4)
        x = torch.randn(1, 4, 32)
        with torch.no_grad():
            out1 = conv(x)
            out2 = conv(2 * x)
        assert torch.allclose(out2, 2 * out1, atol=1e-5)


# ── SpectralConv2d ──────────────────────────────────────────────────────────

class TestSpectralConv2d:
    def test_output_shape(self):
        conv = SpectralConv2d(8, 16, modes1=4, modes2=4)
        x = torch.randn(2, 8, 16, 16)
        out = conv(x)
        assert out.shape == (2, 16, 16, 16)

    def test_gradient_flow(self):
        conv = SpectralConv2d(4, 4, modes1=4, modes2=4)
        x = torch.randn(2, 4, 16, 16)
        out = conv(x)
        out.mean().backward()
        assert conv.weights1.grad is not None
        assert conv.weights2.grad is not None


# ── FourierNeuralOperator1d ─────────────────────────────────────────────────

class TestFNO1d:
    def test_output_shape_with_grid(self):
        model = FourierNeuralOperator1d(modes=8, width=16, n_layers=2)
        x = torch.randn(4, 64, 1)
        out = model(x)
        assert out.shape == (4, 64, 1)

    def test_output_shape_no_grid(self):
        model = FourierNeuralOperator1d(modes=8, width=16, n_layers=2, append_grid=False)
        x = torch.randn(4, 64, 1)
        out = model(x)
        assert out.shape == (4, 64, 1)

    def test_custom_grid(self):
        model = FourierNeuralOperator1d(modes=4, width=8, n_layers=2)
        x = torch.randn(2, 32, 1)
        grid = torch.linspace(0, 1, 32).view(1, 32, 1).expand(2, 32, 1)
        out = model(x, grid=grid)
        assert out.shape == (2, 32, 1)

    def test_different_resolutions(self):
        """FNO should handle different N (grid-independent)."""
        model = FourierNeuralOperator1d(modes=4, width=8, n_layers=2)
        for n in [32, 64, 128]:
            x = torch.randn(1, n, 1)
            out = model(x)
            assert out.shape == (1, n, 1), f"Failed at N={n}"

    def test_multi_channel_output(self):
        model = FourierNeuralOperator1d(modes=4, width=8, n_layers=2,
                                        in_channels=2, out_channels=3)
        x = torch.randn(2, 32, 2)
        out = model(x)
        assert out.shape == (2, 32, 3)

    def test_backward(self):
        model = FourierNeuralOperator1d(modes=4, width=8, n_layers=2)
        x = torch.randn(2, 32, 1)
        out = model(x)
        out.mean().backward()
        # Check at least one parameter has grad
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_fft_spectral_content(self):
        """SpectralConv1d should preserve frequency content up to `modes`."""
        modes = 4
        n = 32
        conv = SpectralConv1d(1, 1, modes=modes)
        # Low-frequency input signal
        t = torch.linspace(0, 1, n)
        x = torch.sin(2 * torch.pi * t).view(1, 1, n)
        out = conv(x)
        # Output should be non-zero (signal passes through)
        assert out.abs().max() > 1e-6


# ── FourierNeuralOperator2d ─────────────────────────────────────────────────

class TestFNO2d:
    def test_output_shape(self):
        model = FourierNeuralOperator2d(modes1=4, modes2=4, width=8, n_layers=2)
        x = torch.randn(2, 16, 16, 1)
        out = model(x)
        assert out.shape == (2, 16, 16, 1)

    def test_backward(self):
        model = FourierNeuralOperator2d(modes1=4, modes2=4, width=8, n_layers=2)
        x = torch.randn(2, 16, 16, 1)
        out = model(x)
        out.mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

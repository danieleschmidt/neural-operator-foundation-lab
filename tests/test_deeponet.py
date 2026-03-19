"""Tests for DeepONet."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from neural_operator_lab.deeponet import DeepONet


class TestDeepONet:
    def test_output_shape_basic(self):
        model = DeepONet(branch_input_dim=64, trunk_input_dim=1, p=32)
        u = torch.randn(4, 64)   # 4 samples, 64 sensors
        y = torch.linspace(0, 1, 100).view(100, 1)  # 100 query points
        out = model(u, y)
        assert out.shape == (4, 100, 1), f"Expected (4,100,1), got {out.shape}"

    def test_output_shape_batched_y(self):
        model = DeepONet(branch_input_dim=32, trunk_input_dim=1, p=16)
        u = torch.randn(3, 32)
        y = torch.randn(3, 50, 1)   # batched y
        out = model(u, y)
        assert out.shape == (3, 50, 1)

    def test_multi_output(self):
        model = DeepONet(branch_input_dim=64, trunk_input_dim=1, p=32, out_channels=2)
        u = torch.randn(4, 64)
        y = torch.randn(50, 1)
        out = model(u, y)
        assert out.shape == (4, 50, 2)

    def test_2d_trunk_input(self):
        """Trunk can take 2-D spatial locations."""
        model = DeepONet(branch_input_dim=64, trunk_input_dim=2, p=32)
        u = torch.randn(2, 64)
        y = torch.randn(25, 2)   # 2-D points
        out = model(u, y)
        assert out.shape == (2, 25, 1)

    def test_backward(self):
        model = DeepONet(branch_input_dim=32, trunk_input_dim=1, p=16)
        u = torch.randn(2, 32)
        y = torch.randn(20, 1)
        out = model(u, y)
        out.mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_no_bias(self):
        model = DeepONet(branch_input_dim=32, trunk_input_dim=1, p=16, use_bias=False)
        assert model.bias is None
        u = torch.randn(2, 32)
        y = torch.randn(10, 1)
        out = model(u, y)
        assert out.shape == (2, 10, 1)

    def test_branch_trunk_dot_product_structure(self):
        """When branch output is zero, the dot-product output should be zero (sans bias)."""
        model = DeepONet(branch_input_dim=16, trunk_input_dim=1, p=8, use_bias=False)
        # Zero out branch weights
        with torch.no_grad():
            for p in model.branch.parameters():
                p.zero_()
        u = torch.randn(2, 16)
        y = torch.randn(10, 1)
        out = model(u, y)
        assert out.abs().max().item() < 1e-6

    def test_custom_hidden_sizes(self):
        model = DeepONet(
            branch_input_dim=64,
            trunk_input_dim=1,
            branch_hidden=[256, 256, 128],
            trunk_hidden=[64, 64],
            p=48,
        )
        u = torch.randn(3, 64)
        y = torch.randn(30, 1)
        out = model(u, y)
        assert out.shape == (3, 30, 1)

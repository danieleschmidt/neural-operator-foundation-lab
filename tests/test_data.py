"""Tests for PDEDataGenerator."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from neural_operator_lab.data import PDEDataGenerator


class TestPDEDataGenerator:
    def test_burgers_shapes(self):
        gen = PDEDataGenerator(pde="burgers", n_x=64, T=0.5, n_t=100, seed=1)
        u0, uT = gen.generate(20)
        assert u0.shape == (20, 64, 1)
        assert uT.shape == (20, 64, 1)

    def test_heat_shapes(self):
        gen = PDEDataGenerator(pde="heat", n_x=32, T=0.1, seed=2)
        u0, uT = gen.generate(10)
        assert u0.shape == (10, 32, 1)
        assert uT.shape == (10, 32, 1)

    def test_advection_shapes(self):
        gen = PDEDataGenerator(pde="advection", n_x=64, T=0.5, seed=3)
        u0, uT = gen.generate(10)
        assert u0.shape == (10, 64, 1)
        assert uT.shape == (10, 64, 1)

    def test_heat_energy_conservation(self):
        """Heat equation: total energy (L2 norm) does not increase."""
        gen = PDEDataGenerator(pde="heat", n_x=128, T=1.0, alpha=0.005, seed=7)
        u0, uT = gen.generate(50)
        energy_before = u0.squeeze(-1).norm(dim=1)
        energy_after  = uT.squeeze(-1).norm(dim=1)
        # Energy should be non-increasing
        assert (energy_after <= energy_before + 1e-4).all(), \
            "Heat equation violated energy decay"

    def test_advection_l2_norm_preserved(self):
        """Advection is conservative: L2 norm should be exactly preserved."""
        gen = PDEDataGenerator(pde="advection", n_x=64, T=0.25, c=1.0, seed=5)
        u0, uT = gen.generate(20)
        norm0 = u0.squeeze(-1).norm(dim=1)
        normT = uT.squeeze(-1).norm(dim=1)
        assert torch.allclose(norm0, normT, atol=1e-5)

    def test_invalid_pde(self):
        with pytest.raises(ValueError, match="pde must be"):
            PDEDataGenerator(pde="wave")

    def test_dataloader_splits(self):
        gen = PDEDataGenerator(pde="burgers", n_x=32, n_t=50, T=0.5, seed=9)
        train_loader, test_loader, x_grid = gen.get_dataloader(n_train=80, n_test=20, batch_size=16)
        assert x_grid.shape == (32, 1)
        # Count samples
        n_train = sum(b[0].shape[0] for b in train_loader)
        n_test  = sum(b[0].shape[0] for b in test_loader)
        assert n_train == 80
        assert n_test  == 20

    def test_reproducibility(self):
        gen1 = PDEDataGenerator(pde="burgers", n_x=32, n_t=50, T=0.5, seed=42)
        gen2 = PDEDataGenerator(pde="burgers", n_x=32, n_t=50, T=0.5, seed=42)
        u0_1, _ = gen1.generate(5)
        u0_2, _ = gen2.generate(5)
        assert torch.allclose(u0_1, u0_2)

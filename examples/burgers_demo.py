#!/usr/bin/env python3
"""Demo: train FNO and DeepONet on 1-D Burgers equation, compare test error.

Usage:
    ~/anaconda3/bin/python3 examples/burgers_demo.py
"""

import sys
import torch
sys.path.insert(0, "src")

from neural_operator_lab import (
    FourierNeuralOperator1d,
    DeepONet,
    PDEDataGenerator,
    OperatorTrainer,
)

# ── Config ─────────────────────────────────────────────────────────────────
N_X       = 64      # spatial resolution
N_TRAIN   = 800
N_TEST    = 200
BATCH     = 32
N_EPOCHS  = 30
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Spatial grid: {N_X} points | Train: {N_TRAIN} | Test: {N_TEST}\n")

# ── Data ────────────────────────────────────────────────────────────────────
gen = PDEDataGenerator(pde="burgers", n_x=N_X, T=1.0, n_t=400, nu=0.01, seed=0)
train_loader, test_loader, x_grid = gen.get_dataloader(N_TRAIN, N_TEST, BATCH)
print(f"Data generated — u0 shape: {next(iter(train_loader))[0].shape}\n")

# ── FNO ─────────────────────────────────────────────────────────────────────
print("=" * 56)
print("  Fourier Neural Operator (FNO-1D)")
print("=" * 56)

fno = FourierNeuralOperator1d(modes=16, width=32, n_layers=4, in_channels=1, out_channels=1)
print(f"FNO parameters: {sum(p.numel() for p in fno.parameters() if p.requires_grad):,}\n")

fno_trainer = OperatorTrainer(fno, device=DEVICE)
fno_history = fno_trainer.fit(train_loader, test_loader, n_epochs=N_EPOCHS, print_every=5)
fno_test_err = fno_history["test_loss"][-1]
print(f"\nFNO final test L2-rel error: {fno_test_err:.4f}\n")


# ── DeepONet ─────────────────────────────────────────────────────────────────
print("=" * 56)
print("  Deep Operator Network (DeepONet)")
print("=" * 56)

deeponet = DeepONet(
    branch_input_dim=N_X,
    trunk_input_dim=1,
    branch_hidden=[128, 128],
    trunk_hidden=[128, 128],
    p=64,
)
print(f"DeepONet parameters: {sum(p.numel() for p in deeponet.parameters() if p.requires_grad):,}\n")

# DeepONet needs a custom forward_fn: branch gets u0 (flattened), trunk gets x_grid.
x_grid_batch = x_grid.unsqueeze(0)  # (1, n_x, 1) — will broadcast

def deeponet_forward(model, batch, device):
    u0, uT = batch
    u0 = u0.to(device)                                # (B, n_x, 1)
    uT = uT.to(device)                                # (B, n_x, 1)
    B  = u0.shape[0]
    branch_in = u0.squeeze(-1)                         # (B, n_x)
    y = x_grid_batch.expand(B, -1, -1).to(device)     # (B, n_x, 1)
    pred = model(branch_in, y)                         # (B, n_x, 1)
    return pred, uT

don_trainer = OperatorTrainer(deeponet, device=DEVICE, forward_fn=deeponet_forward)
don_history = don_trainer.fit(train_loader, test_loader, n_epochs=N_EPOCHS, print_every=5)
don_test_err = don_history["test_loss"][-1]
print(f"\nDeepONet final test L2-rel error: {don_test_err:.4f}\n")


# ── Summary ─────────────────────────────────────────────────────────────────
print("=" * 56)
print("  Results — 1-D Burgers (T=1.0, nu=0.01)")
print("=" * 56)
print(f"  FNO     test L2-rel: {fno_test_err:.4f}")
print(f"  DeepONet test L2-rel: {don_test_err:.4f}")
winner = "FNO" if fno_test_err < don_test_err else "DeepONet"
print(f"  Winner (lower is better): {winner}")
print("=" * 56)

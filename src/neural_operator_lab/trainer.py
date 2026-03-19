"""OperatorTrainer — training loop with L2 relative error metric.

L2 relative error is the standard metric in neural operator literature:
    E_rel = ||u_pred - u_true||_2 / ||u_true||_2

averaged over the batch.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
import time


def l2_relative_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute mean L2 relative error over a batch.

    Args:
        pred:   (batch, ...) predicted values.
        target: (batch, ...) ground truth values.
    Returns:
        Scalar tensor with mean relative L2 error.
    """
    diff = (pred - target).flatten(1)
    tgt  = target.flatten(1)
    return (diff.norm(dim=1) / (tgt.norm(dim=1) + eps)).mean()


class OperatorTrainer:
    """Training loop for neural operators.

    Handles both FNO-style models (forward(u0)) and DeepONet-style models
    (forward(u, y)) via a user-supplied ``forward_fn``.

    Args:
        model:      nn.Module to train.
        optimizer:  torch optimiser (default: Adam, lr=1e-3).
        scheduler:  optional LR scheduler.
        device:     "cpu", "cuda", or "auto".
        forward_fn: callable(model, batch) → (pred, target).  If None, assumes
                    the batch is (input, target) and calls model(input).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        device: str = "auto",
        forward_fn: Optional[Callable] = None,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        self.forward_fn = forward_fn or self._default_forward

        self.history: Dict[str, List[float]] = {"train_loss": [], "test_loss": []}

    @staticmethod
    def _default_forward(model, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        pred = model(x)
        return pred, y

    def _fno_forward(self, model, batch, device):
        return self._default_forward(model, batch, device)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            pred, target = self.forward_fn(self.model, batch, self.device)
            loss = l2_relative_error(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            pred, target = self.forward_fn(self.model, batch, self.device)
            loss = l2_relative_error(pred, target)
            total_loss += loss.item()
        return total_loss / len(loader)

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        n_epochs: int = 50,
        print_every: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Returns:
            history dict with "train_loss" and "test_loss" per epoch.
        """
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            test_loss  = self.evaluate(test_loader)

            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            if verbose and (epoch % print_every == 0 or epoch == 1):
                elapsed = time.time() - t0
                print(f"Epoch {epoch:4d}/{n_epochs} | "
                      f"train L2-rel: {train_loss:.4f} | "
                      f"test  L2-rel: {test_loss:.4f} | "
                      f"{elapsed:.1f}s")

        return self.history

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

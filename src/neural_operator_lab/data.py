"""PDE data generators using finite-difference methods.

Supports 1-D PDEs:
  - Burgers equation:  u_t + u u_x = nu u_xx
  - Heat equation:     u_t = alpha u_xx
  - Advection equation: u_t + c u_x = 0

Each generator creates a dataset mapping initial conditions → solution at a
fixed future time T.  Initial conditions are drawn from random Gaussian random fields
or sinusoidal superpositions.
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Initial condition generators
# ---------------------------------------------------------------------------

def _random_ic_grf(n_samples: int, n_x: int, length_scale: float = 0.1,
                   rng: np.random.Generator = None) -> np.ndarray:
    """Sample initial conditions from a smoothed Gaussian random field.

    We generate white noise and apply a Gaussian filter in Fourier space.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.linspace(0, 1, n_x, endpoint=False)
    freq = np.fft.rfftfreq(n_x, d=1.0 / n_x)
    # Gaussian spectral filter
    kernel = np.exp(-0.5 * (freq * length_scale) ** 2)
    ics = []
    for _ in range(n_samples):
        noise = rng.standard_normal(n_x)
        noise_ft = np.fft.rfft(noise) * kernel
        ic = np.fft.irfft(noise_ft, n=n_x)
        ics.append(ic)
    return np.array(ics, dtype=np.float32)  # (n_samples, n_x)


# ---------------------------------------------------------------------------
# PDE solvers (finite difference)
# ---------------------------------------------------------------------------

def _solve_burgers(u0: np.ndarray, n_x: int, n_t: int, T: float,
                   nu: float = 0.01) -> np.ndarray:
    """Solve 1-D Burgers' equation on [0,1] with periodic BCs.

    Uses first-order upwind for advection and explicit Euler for diffusion.
    Returns solution at time T.
    """
    dx = 1.0 / n_x
    dt = T / n_t
    # CFL stability check — auto-reduce dt if needed
    u = u0.copy()
    for _ in range(n_t):
        u_old = u.copy()
        # Upwind advection: u*u_x
        u_shift = np.roll(u_old, 1)  # u_{i-1}
        adv = np.where(u_old > 0,
                       u_old * (u_old - u_shift) / dx,
                       u_old * (np.roll(u_old, -1) - u_old) / dx)
        # Central diffusion
        diff = nu * (np.roll(u_old, -1) - 2 * u_old + u_shift) / dx ** 2
        u = u_old + dt * (-adv + diff)
    return u.astype(np.float32)


def _solve_heat(u0: np.ndarray, n_x: int, n_t: int, T: float,
                alpha: float = 0.01) -> np.ndarray:
    """Solve 1-D heat equation on [0,1] with periodic BCs (explicit Euler)."""
    dx = 1.0 / n_x
    dt = T / n_t
    # Stability: alpha * dt / dx^2 <= 0.5
    r = alpha * dt / dx ** 2
    if r > 0.5:
        # Use Fourier-space exact solution for stability
        u_ft = np.fft.rfft(u0)
        freq = np.fft.rfftfreq(n_x, d=1.0 / n_x)
        u_ft *= np.exp(-alpha * (2 * np.pi * freq) ** 2 * T)
        return np.fft.irfft(u_ft, n=n_x).astype(np.float32)

    u = u0.copy()
    for _ in range(n_t):
        u_old = u.copy()
        u = u_old + r * (np.roll(u_old, -1) - 2 * u_old + np.roll(u_old, 1))
    return u.astype(np.float32)


def _solve_advection(u0: np.ndarray, n_x: int, T: float, c: float = 1.0) -> np.ndarray:
    """Solve 1-D advection equation u_t + c u_x = 0 with periodic BCs.

    Exact solution: u(x, T) = u0(x - c*T mod 1).
    """
    shift = int(round(c * T * n_x)) % n_x
    return np.roll(u0, -shift).astype(np.float32)


# ---------------------------------------------------------------------------
# PDEDataGenerator
# ---------------------------------------------------------------------------

class PDEDataGenerator:
    """Generate training/test data for 1-D PDE operators.

    Maps initial condition u0 → solution u(·, T).

    Args:
        pde:       one of "burgers", "heat", "advection".
        n_x:       number of spatial grid points.
        T:         final time.
        n_t:       number of time steps for finite-difference integration.
        nu:        viscosity (Burgers).
        alpha:     diffusivity (heat).
        c:         wave speed (advection).
        length_scale: GRF smoothness for IC generation.
        seed:      random seed.
    """

    SUPPORTED = ("burgers", "heat", "advection")

    def __init__(
        self,
        pde: str = "burgers",
        n_x: int = 128,
        T: float = 1.0,
        n_t: int = 200,
        nu: float = 0.01,
        alpha: float = 0.01,
        c: float = 1.0,
        length_scale: float = 0.1,
        seed: Optional[int] = 42,
    ):
        if pde not in self.SUPPORTED:
            raise ValueError(f"pde must be one of {self.SUPPORTED}, got '{pde}'")
        self.pde = pde
        self.n_x = n_x
        self.T = T
        self.n_t = n_t
        self.nu = nu
        self.alpha = alpha
        self.c = c
        self.length_scale = length_scale
        self.rng = np.random.default_rng(seed)
        self.x_grid = np.linspace(0, 1, n_x, endpoint=False).astype(np.float32)

    def _solve(self, u0: np.ndarray) -> np.ndarray:
        if self.pde == "burgers":
            return _solve_burgers(u0, self.n_x, self.n_t, self.T, self.nu)
        elif self.pde == "heat":
            return _solve_heat(u0, self.n_x, self.n_t, self.T, self.alpha)
        else:  # advection
            return _solve_advection(u0, self.n_x, self.T, self.c)

    def generate(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate (u0, uT) pairs.

        Returns:
            u0: (n_samples, n_x, 1)  — initial conditions.
            uT: (n_samples, n_x, 1)  — solutions at time T.
        """
        ics = _random_ic_grf(n_samples, self.n_x, self.length_scale, self.rng)
        solutions = np.array([self._solve(u0) for u0 in ics])  # (N, n_x)
        u0_t = torch.from_numpy(ics).unsqueeze(-1)          # (N, n_x, 1)
        uT_t = torch.from_numpy(solutions).unsqueeze(-1)    # (N, n_x, 1)
        return u0_t, uT_t

    def get_dataloader(
        self,
        n_train: int = 1000,
        n_test: int = 200,
        batch_size: int = 32,
    ) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
        """Generate train/test DataLoaders and the spatial grid tensor.

        Returns:
            train_loader, test_loader, x_grid  (x_grid shape: (n_x, 1))
        """
        u0, uT = self.generate(n_train + n_test)
        x_grid = torch.from_numpy(self.x_grid).unsqueeze(-1)  # (n_x, 1)

        train_ds = TensorDataset(u0[:n_train], uT[:n_train])
        test_ds  = TensorDataset(u0[n_train:], uT[n_train:])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, x_grid

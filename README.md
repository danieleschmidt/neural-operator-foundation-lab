# Neural Operator Foundation Lab

A focused toolkit for learning **mappings between function spaces** — the core primitive for physics-informed ML and PDE solving.

Neural operators, unlike standard neural networks, learn operators that map entire functions to functions (e.g. initial condition → solution), not just finite-dimensional vectors to vectors. They are resolution-independent: train at one grid resolution, evaluate at any other.

## Implemented Models

### Fourier Neural Operator (FNO)
Li et al., [*Fourier Neural Operator for Parametric Partial Differential Equations*](https://arxiv.org/abs/2010.08895), 2021.

Architecture per layer:
1. **FFT** — lift to Fourier space
2. **Spectral convolution** — multiply the lowest `modes` Fourier coefficients by learned complex weights
3. **iFFT** — return to physical space
4. **Residual** — add a pointwise (local) convolution, apply GELU

The truncation to `modes` frequencies is what makes FNO grid-independent.

Supports 1-D (`FourierNeuralOperator1d`) and 2-D (`FourierNeuralOperator2d`).

### Deep Operator Network (DeepONet)
Lu et al., [*Learning Nonlinear Operators via DeepONet*](https://arxiv.org/abs/1910.03193), 2021.

Architecture:
- **Branch net** — encodes input function values at fixed sensor locations → embedding vector
- **Trunk net** — encodes query output locations → embedding vector
- **Output** — inner product of branch and trunk embeddings

### PDE Data Generator
Finite-difference solvers for three 1-D PDEs, mapping initial conditions → solution at time T:
- **Burgers** — `u_t + u u_x = ν u_xx` (nonlinear, viscous)
- **Heat** — `u_t = α u_xx`
- **Advection** — `u_t + c u_x = 0`

Initial conditions are sampled from a Gaussian random field for diversity.

### OperatorTrainer
Training loop with **L2 relative error** — the standard metric in the neural operator literature:

```
E_rel = ‖u_pred − u_true‖₂ / ‖u_true‖₂
```

## Quick Start

```python
from neural_operator_lab import (
    FourierNeuralOperator1d,
    DeepONet,
    PDEDataGenerator,
    OperatorTrainer,
)

# Generate Burgers equation data
gen = PDEDataGenerator(pde="burgers", n_x=128, T=1.0, nu=0.01)
train_loader, test_loader, x_grid = gen.get_dataloader(n_train=1000, n_test=200)

# Train FNO
fno = FourierNeuralOperator1d(modes=16, width=64, n_layers=4)
trainer = OperatorTrainer(fno)
history = trainer.fit(train_loader, test_loader, n_epochs=50)
```

## Running the Demo

Trains both FNO and DeepONet on 1-D Burgers, then compares test error:

```bash
~/anaconda3/bin/python3 examples/burgers_demo.py
```

## Running Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

31 tests covering:
- FFT operations and spectral truncation
- Output shapes at multiple resolutions (grid-independence)
- Gradient flow through both models
- Physics constraints (heat energy decay, advection L2 preservation)

## Structure

```
src/neural_operator_lab/
  fno.py       — SpectralConv1d/2d, FourierNeuralOperator1d/2d
  deeponet.py  — DeepONet (branch + trunk nets)
  data.py      — PDEDataGenerator (Burgers, heat, advection)
  trainer.py   — OperatorTrainer with L2-relative error

tests/
  test_fno.py
  test_deeponet.py
  test_data.py

examples/
  burgers_demo.py  — FNO vs DeepONet on 1-D Burgers
```

## References

- Li et al. (2021) — [FNO paper](https://arxiv.org/abs/2010.08895)
- Lu et al. (2021) — [DeepONet paper](https://arxiv.org/abs/1910.03193)
- Kovachki et al. (2023) — [Neural Operator survey](https://arxiv.org/abs/2108.08481)

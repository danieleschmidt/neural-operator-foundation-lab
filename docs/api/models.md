# Models API Reference

This document provides comprehensive API documentation for the neural operator models implemented in the Neural Operator Foundation Lab.

## Base Classes

### NeuralOperatorBase

The abstract base class for all neural operators in the framework.

```python
class NeuralOperatorBase(nn.Module, ABC)
```

#### Constructor Parameters

- `input_dim` (int): Dimension of input functions
- `output_dim` (int): Dimension of output functions
- `**kwargs`: Additional configuration parameters

#### Abstract Methods

##### `forward_features(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor`

Extract features from the input function.

**Parameters:**
- `x` (torch.Tensor): Input function data
- `grid` (Optional[torch.Tensor]): Coordinate grid information

**Returns:**
- torch.Tensor: Extracted features

##### `forward_operator(self, features: torch.Tensor) -> torch.Tensor`

Apply the neural operator transformation.

**Parameters:**
- `features` (torch.Tensor): Feature representation of input function

**Returns:**
- torch.Tensor: Output function

#### Methods

##### `forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor`

Complete forward pass through the neural operator.

**Parameters:**
- `x` (torch.Tensor): Input function tensor
- `grid` (Optional[torch.Tensor]): Optional coordinate grid

**Returns:**
- torch.Tensor: Output function tensor

##### `get_loss(self, pred: torch.Tensor, target: torch.Tensor, inputs: Optional[torch.Tensor] = None) -> torch.Tensor`

Compute loss with optional physics constraints.

**Parameters:**
- `pred` (torch.Tensor): Model predictions
- `target` (torch.Tensor): Ground truth targets
- `inputs` (Optional[torch.Tensor]): Input data for physics-informed loss

**Returns:**
- torch.Tensor: Computed loss value

##### `count_parameters(self) -> int`

Count the total number of trainable parameters.

**Returns:**
- int: Number of trainable parameters

##### `get_model_summary(self) -> Dict[str, Any]`

Get comprehensive model summary statistics.

**Returns:**
- Dict[str, Any]: Model summary including parameters, dimensions, and type

## Fourier Neural Operators

### FourierNeuralOperator

Implementation of Fourier Neural Operators for learning mappings between function spaces.

```python
class FourierNeuralOperator(NeuralOperatorBase)
```

#### Constructor Parameters

- `input_dim` (int): Input function dimension
- `output_dim` (int): Output function dimension  
- `modes` (List[int]): Number of Fourier modes in each dimension
- `width` (int): Width of the network channels
- `num_layers` (int): Number of Fourier layers
- `activation` (str): Activation function ('relu', 'gelu', 'silu')
- `padding` (int): Padding for periodic extension

#### Example Usage

```python
# 2D Fourier Neural Operator
fno = FourierNeuralOperator(
    input_dim=3,      # 3 input channels
    output_dim=1,     # 1 output channel
    modes=[32, 32],   # 32 modes in each spatial dimension
    width=64,         # 64 hidden channels
    num_layers=4,     # 4 Fourier layers
    activation='gelu'
)

# Forward pass
input_data = torch.randn(8, 3, 64, 64)  # Batch, channels, height, width
output = fno(input_data)
print(output.shape)  # torch.Size([8, 1, 64, 64])
```

### SpectralConv1d, SpectralConv2d, SpectralConv3d

Core spectral convolution layers for Fourier Neural Operators.

```python
class SpectralConv2d(nn.Module)
```

#### Constructor Parameters

- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels
- `modes1` (int): Number of Fourier modes in first dimension
- `modes2` (int): Number of Fourier modes in second dimension

#### Example Usage

```python
# 2D Spectral Convolution
spectral_conv = SpectralConv2d(
    in_channels=64,
    out_channels=64, 
    modes1=16,
    modes2=16
)

x = torch.randn(4, 64, 32, 32)
output = spectral_conv(x)
```

## Transformer Neural Operators

### TransformerNeuralOperator

Implementation combining transformer attention mechanisms with neural operators.

```python
class TransformerNeuralOperator(NeuralOperatorBase)
```

#### Constructor Parameters

- `input_dim` (int): Input function dimension
- `output_dim` (int): Output function dimension
- `hidden_dim` (int): Hidden dimension size
- `num_layers` (int): Number of transformer layers
- `num_heads` (int): Number of attention heads
- `dropout` (float): Dropout rate
- `max_seq_length` (int): Maximum sequence length
- `use_positional_encoding` (bool): Whether to use positional encoding
- `use_physics_embedding` (bool): Whether to include physics-aware embeddings

#### Example Usage

```python
# Transformer Neural Operator
tno = TransformerNeuralOperator(
    input_dim=4,
    output_dim=3,
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    dropout=0.1,
    use_positional_encoding=True,
    use_physics_embedding=True
)

# Forward pass with grid information
input_data = torch.randn(4, 4, 64, 64)
grid = torch.randn(4, 2, 64, 64)  # x, y coordinates
output = tno(input_data, grid=grid)
```

### TNOBlock

Individual transformer block used in Transformer Neural Operators.

```python
class TNOBlock(nn.Module)
```

#### Constructor Parameters

- `hidden_dim` (int): Hidden dimension
- `num_heads` (int): Number of attention heads
- `mlp_ratio` (float): Expansion ratio for MLP layers
- `dropout` (float): Dropout rate
- `drop_path` (float): Stochastic depth rate

## Probabilistic Neural Operators

### ProbabilisticNeuralOperator

Neural operator with built-in uncertainty quantification.

```python
class ProbabilisticNeuralOperator(NeuralOperatorBase)
```

#### Constructor Parameters

- `trunk_layers` (List[int]): Architecture of trunk network
- `branch_layers` (List[int]): Architecture of branch network
- `num_basis` (int): Number of basis functions
- `uncertainty_type` (str): Type of uncertainty ('epistemic', 'aleatoric', 'both')
- `dropout_rate` (float): Dropout rate for uncertainty
- `num_samples` (int): Number of samples for Monte Carlo

#### Methods

##### `forward(self, x: torch.Tensor, return_uncertainty: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`

Forward pass with optional uncertainty estimation.

**Parameters:**
- `x` (torch.Tensor): Input function data
- `return_uncertainty` (bool): Whether to return uncertainty estimates

**Returns:**
- If `return_uncertainty=False`: torch.Tensor (predictions)
- If `return_uncertainty=True`: Tuple[torch.Tensor, torch.Tensor] (mean, uncertainty)

#### Example Usage

```python
# Probabilistic Neural Operator
pno = ProbabilisticNeuralOperator(
    trunk_layers=[512, 512, 512],
    branch_layers=[256, 256],
    num_basis=128,
    uncertainty_type="epistemic",
    dropout_rate=0.1
)

# Forward pass with uncertainty
input_data = torch.randn(8, 3, 32, 32)
mean, uncertainty = pno(input_data, return_uncertainty=True)

print(f"Mean shape: {mean.shape}")
print(f"Uncertainty shape: {uncertainty.shape}")
```

## Universal Models

### UniversalPhysicsTransformer

Pre-trained universal model for multiple physics domains.

```python
class UniversalPhysicsTransformer(TransformerNeuralOperator)
```

#### Class Methods

##### `from_pretrained(model_name: str, cache_dir: Optional[str] = None) -> 'UniversalPhysicsTransformer'`

Load pre-trained universal physics transformer.

**Parameters:**
- `model_name` (str): Name of pre-trained model
- `cache_dir` (Optional[str]): Cache directory for model files

**Returns:**
- UniversalPhysicsTransformer: Loaded pre-trained model

#### Methods

##### `freeze_backbone(self)`

Freeze the backbone transformer layers for fine-tuning.

##### `add_pde_head(self, pde_type: str, output_channels: int, **kwargs)`

Add a specialized head for a specific PDE type.

**Parameters:**
- `pde_type` (str): Type of PDE ('navier_stokes', 'maxwell', etc.)
- `output_channels` (int): Number of output channels
- `**kwargs`: Additional head configuration

#### Example Usage

```python
# Load pre-trained universal model
upt = UniversalPhysicsTransformer.from_pretrained(
    "neurips2025/universal-physics-transformer-large"
)

# Fine-tune for specific PDE
upt.freeze_backbone()
upt.add_pde_head(
    pde_type="navier_stokes_3d",
    output_channels=4
)
```

## Utility Functions

### create_model

Factory function for creating models from configuration.

```python
def create_model(model_type: str, config: Dict[str, Any]) -> NeuralOperatorBase
```

**Parameters:**
- `model_type` (str): Type of model to create
- `config` (Dict[str, Any]): Model configuration

**Returns:**
- NeuralOperatorBase: Instantiated model

**Example:**

```python
from neural_operator_lab.models import create_model

config = {
    'input_dim': 3,
    'output_dim': 1,
    'hidden_dim': 256,
    'num_layers': 6
}

model = create_model('transformer_neural_operator', config)
```

### load_checkpoint

Load model from checkpoint file.

```python
def load_checkpoint(checkpoint_path: str, model: Optional[NeuralOperatorBase] = None) -> NeuralOperatorBase
```

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint file
- `model` (Optional[NeuralOperatorBase]): Model instance to load into

**Returns:**
- NeuralOperatorBase: Model loaded from checkpoint

## Model Configuration Examples

### Navier-Stokes Configuration

```python
navier_stokes_config = {
    'model_type': 'fourier_neural_operator',
    'input_dim': 4,  # u, v, w, p
    'output_dim': 4,
    'modes': [32, 32, 32],
    'width': 256,
    'num_layers': 6,
    'activation': 'gelu',
    'physics_informed': True,
    'conservation_loss': True
}
```

### Maxwell Equations Configuration

```python
maxwell_config = {
    'model_type': 'transformer_neural_operator',
    'input_dim': 6,  # Ex, Ey, Ez, Bx, By, Bz
    'output_dim': 6,
    'hidden_dim': 512,
    'num_layers': 8,
    'num_heads': 16,
    'use_positional_encoding': True,
    'max_seq_length': 4096
}
```

### Uncertainty Quantification Configuration

```python
uncertainty_config = {
    'model_type': 'probabilistic_neural_operator',
    'trunk_layers': [512, 512, 512],
    'branch_layers': [256, 256],
    'num_basis': 64,
    'uncertainty_type': 'both',
    'num_samples': 50,
    'dropout_rate': 0.15
}
```
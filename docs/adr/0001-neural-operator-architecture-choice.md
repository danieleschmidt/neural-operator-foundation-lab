# 0001: Neural Operator Architecture Choice

Date: 2025-08-01

## Status
Accepted

## Context

We need to establish the foundational neural operator architectures for the Neural Operator Foundation Lab. The field has several competing approaches:

1. **Fourier Neural Operators (FNO)**: Use FFT for efficient convolution in frequency domain
2. **Probabilistic Neural Operators (PNO)**: Incorporate uncertainty quantification
3. **Transformer Neural Operators (TNO)**: Leverage attention mechanisms for long-range dependencies
4. **Graph Neural Operators**: Handle irregular geometries with graph structures

Each approach has different strengths for various PDE types and problem scales. We need to decide which architectures to prioritize and how to structure the codebase to support multiple approaches.

## Decision

We will implement a multi-architecture approach with the following priority order:

1. **Primary Support**: FNO, PNO, and TNO as core architectures
2. **Secondary Support**: Graph Neural Operators for specialized use cases
3. **Extensible Framework**: Abstract base classes to enable community contributions
4. **Unified Interface**: Common API across all operator types

### Architecture Rationale

**Fourier Neural Operators (FNO)**:
- Proven effectiveness on regular grid PDEs
- Excellent performance on fluid dynamics problems
- Efficient FFT-based implementation
- Strong theoretical foundations

**Probabilistic Neural Operators (PNO)**:
- Essential for uncertainty quantification in scientific applications
- Enables Bayesian inference in PDE solutions
- Critical for safety-critical applications
- Complements deterministic approaches

**Transformer Neural Operators (TNO)**:
- Superior performance on complex, multi-scale problems
- Excellent at capturing long-range dependencies
- State-of-the-art results on 3D turbulence
- Scalable to large model sizes

### Implementation Strategy

```python
# Unified base class
class NeuralOperatorBase(nn.Module):
    def forward(self, x, grid=None):
        raise NotImplementedError
    
    def get_loss(self, pred, target, inputs=None):
        raise NotImplementedError

# Concrete implementations
class FourierNeuralOperator(NeuralOperatorBase):
    # FFT-based implementation
    pass

class ProbabilisticNeuralOperator(NeuralOperatorBase):
    # Uncertainty quantification
    pass

class TransformerNeuralOperator(NeuralOperatorBase):
    # Attention-based implementation
    pass
```

## Consequences

### Positive
- **Comprehensive Coverage**: Support for diverse PDE types and use cases
- **Research Flexibility**: Researchers can compare architectures on same problems
- **Future-Proof**: Extensible design allows for new architectures
- **Performance Optimization**: Each architecture optimized for its strengths
- **Community Adoption**: Multiple entry points for different research communities

### Negative
- **Increased Complexity**: More code to maintain and test
- **Performance Overhead**: Abstraction layer may impact performance
- **Documentation Burden**: Need comprehensive docs for multiple approaches
- **Resource Requirements**: Testing across architectures increases compute needs

### Neutral
- **Code Duplication**: Some shared components across architectures
- **API Consistency**: Need to balance flexibility with unified interface
- **Migration Path**: Users may need to switch architectures for optimal performance

## References
- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- [Probabilistic Neural Operators](https://arxiv.org/abs/2106.09178)
- [Transformer Neural Operators](https://arxiv.org/abs/2205.13671)
- [Neural Operator: Graph Kernel Network for Partial Differential Equations](https://arxiv.org/abs/2003.03485)

## Notes
- Performance benchmarking across architectures will be critical for users
- Consider hybrid approaches that combine multiple architectures
- Community contributions of new architectures should follow the established patterns
- Regular review of architecture performance and relevance to research trends
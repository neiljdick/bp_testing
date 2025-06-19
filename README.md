# Relay-BP: Improved Belief Propagation for Quantum Error Correction

This repository contains a Python implementation of the **Relay-BP** algorithm described in the paper:

> ["Improved belief propagation is sufficient for real-time decoding of quantum memory"](https://arxiv.org/html/2506.01779v1)  
> by Tristan Müller et al., arXiv:2506.01779v1

## Overview

Relay-BP is a heuristic decoder designed for real-time quantum circuit decoding on large-scale quantum computers. It achieves high accuracy across circuit-noise decoding problems while being suitable for FPGA/ASIC implementation due to its lightweight message-passing structure.

### Key Features

- **Flexible**: Decodes a wide range of quantum LDPC (qLDPC) circuits
- **Accurate**: Achieves low logical error rates comparable to or better than existing methods
- **Compact**: Small footprint suitable for FPGA implementation
- **Fast**: Parallel processing enables real-time decoding

### Algorithm Components

1. **DMem-BP (Disordered Memory Belief Propagation)**: Core BP algorithm enhanced with:
   - Memory terms that dampen oscillations
   - Disordered memory strengths (including negative values)
   - Bias term updates using previous marginals

2. **Relay-BP**: Chains multiple DMem-BP runs where:
   - Each "leg" uses different memory strengths
   - Final marginals from one leg initialize the next
   - Multiple valid solutions can be found and compared

## Installation

### Requirements

- Python 3.6+
- NumPy

### Install Dependencies

```bash
pip install numpy
```

Or if using system Python:

```bash
# Ubuntu/Debian
sudo apt install python3-numpy

# Or with pip (may require --break-system-packages flag)
python3 -m pip install numpy
```

## Usage

### Basic Example

```python
import numpy as np
from relay_bp import RelayBP, create_example_problem, generate_random_syndrome

# Create a synthetic quantum error correction problem
H, A, p = create_example_problem(
    n_checks=20,     # Number of check nodes (syndrome bits)
    n_errors=40,     # Number of error locations
    density=0.15,    # Sparsity of check matrix
    error_prob=0.01  # Base error probability
)

# Generate a random syndrome (in practice, this comes from measurements)
syndrome, true_error = generate_random_syndrome(H, p)

# Initialize decoder
decoder = RelayBP(H, A, p)

# Run decoding
result = decoder.decode(
    syndrome,
    S=3,        # Find up to 3 solutions
    R=50,       # Use up to 50 relay legs
    T_r=30,     # 30 iterations per leg
    gamma_range=(-0.25, 0.85)  # Memory strength range
)

# Check results
print(f"Decoding successful: {result.success}")
print(f"Solutions found: {result.solutions_found}")
print(f"Total iterations: {result.iterations}")
print(f"Error estimate weight: {np.sum(result.error_estimate)}")

# Verify solution
predicted_syndrome = (H @ result.error_estimate) % 2
valid = np.array_equal(predicted_syndrome, syndrome)
print(f"Solution is valid: {valid}")
```

### Advanced Usage

```python
# Custom memory strength configurations
gamma_configs = [
    np.full(decoder.N, 0.125),  # First leg: uniform memory
    np.random.uniform(-0.3, 0.8, decoder.N),  # Second leg: random
    np.random.uniform(-0.5, 1.0, decoder.N),  # Third leg: wider range
]

result = decoder.decode(
    syndrome,
    S=5,
    R=100,
    T_r=60,
    gamma_configs=gamma_configs
)
```

## Algorithm Details

### DMem-BP Message Updates

The algorithm follows these key equations from the paper:

**Check-to-error messages** (Equation 1):
```
μ_{i→j}(t) = κ_{i,j}(t) × (-1)^{σ_i} × min_{j'∈N(i)\{j}} |ν_{j'→i}(t-1)|
```
where `κ_{i,j}(t) = sgn{∏_{j'∈N(i)\{j}} ν_{j'→i}(t-1)}`

**Error-to-check messages** (Equation 2):
```
ν_{j→i}(t) = Λ_j(t) + ∑_{i'∈N(j)\{i}} μ_{i'→j}(t)
```

**Marginals** (Equation 3):
```
M_j(t) = Λ_j(t) + ∑_{i'∈N(j)} μ_{i'→j}(t)
```

**Memory update**:
```
Λ_j(t) = (1-γ_j)Λ_j(0) + γ_j M_j(t-1)
```

### Key Parameters

- **S**: Number of solutions to find (default: 1)
- **R**: Maximum number of relay legs (default: 301)
- **T_r**: Maximum iterations per leg (default: 60)
- **γ range**: Memory strength range, typically `(-0.25, 0.85)`
- **H**: Parity-check matrix (M × N)
- **A**: Action matrix for logical operations (K × N)
- **p**: Error probabilities (N,)

## API Reference

### RelayBP Class

```python
class RelayBP:
    def __init__(self, H: np.ndarray, A: np.ndarray, p: np.ndarray)
    def decode(self, syndrome: np.ndarray, S: int = 1, R: int = 301, 
               T_r: int = 60, gamma_configs: Optional[List[np.ndarray]] = None,
               gamma_range: Tuple[float, float] = (-0.25, 0.85)) -> DecodingResult
```

### DecodingResult

```python
@dataclass
class DecodingResult:
    success: bool                # Whether decoding succeeded
    error_estimate: np.ndarray   # Estimated error pattern
    weight: float                # Log-likelihood weight of solution
    iterations: int              # Total iterations across all legs
    solutions_found: int         # Number of valid solutions found
    final_marginals: np.ndarray  # Final marginals for initialization
```

### Helper Functions

```python
create_example_problem(n_checks: int, n_errors: int, density: float, 
                      error_prob: float, seed: Optional[int] = None)
generate_random_syndrome(H: np.ndarray, p: np.ndarray, 
                        seed: Optional[int] = None)
```

## Testing

Run the test suite to verify the implementation:

```bash
python3 test_relay_bp.py
```

The tests cover:
- Basic algorithm functionality
- Message passing correctness
- Edge cases and error conditions
- Reproducibility with seeds
- Helper function validation

## Performance Characteristics

### Advantages over Standard BP

- **Convergence**: Memory terms and negative γ values help escape oscillations
- **Accuracy**: Multiple relay legs can find better solutions
- **Flexibility**: Works with any qLDPC code structure
- **Parallelizable**: Message passing structure suits FPGA implementation

### Typical Performance

- **Iterations**: Often converges in 10-100 total iterations
- **Success Rate**: High for problems within code distance
- **Memory**: O(edges) space complexity for message storage
- **Time**: O(iterations × edges) time complexity

## Implementation Notes

### Pseudocode Fidelity

This implementation follows the pseudocode from the paper exactly:
- Line-by-line correspondence with Algorithm 1
- Proper initialization of bias terms and marginals
- Correct relay chaining with marginal passing
- Faithful implementation of message update equations

### Numerical Considerations

- Uses `float64` precision for numerical stability
- Handles edge cases (zero/single neighbors gracefully)
- Proper modulo arithmetic for syndrome checking
- Robust sign function for hard decisions

### FPGA Considerations

While this is a Python implementation, the algorithm structure is designed for hardware:
- No dynamic memory allocation during decoding
- Regular message passing patterns
- Local computations without global coordination
- Fixed-point arithmetic could replace floating-point

## References

1. Müller, T., et al. "Improved belief propagation is sufficient for real-time decoding of quantum memory." arXiv:2506.01779v1 (2025).

2. Related work on quantum LDPC codes:
   - Panteleev & Kalachev, "Degenerate quantum LDPC codes with good finite length performance" (2021)
   - Roffe et al., "Decoding across the quantum low-density parity-check code landscape" (2020)

## License

This implementation is provided for research and educational purposes. Please cite the original paper when using this code.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional test cases for specific qLDPC codes
- Performance optimizations
- FPGA/hardware implementation variants
- Integration with quantum circuit simulators

---

*Note: This implementation is based on the theoretical description in the paper. For production use with real quantum hardware, additional considerations for noise models, timing constraints, and hardware-specific optimizations may be required.*

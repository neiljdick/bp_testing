#!/usr/bin/env python3
"""
GF(2)-Optimized Relay-BP Implementation

This version implements the GF(2) optimizations mentioned in the paper:
"adopting a GF(2) version of EWAInit-BP"

Optimizations:
1. Efficient binary arithmetic for syndrome computations
2. Fast parity checking using XOR operations
3. Optimized matrix-vector products in GF(2)
4. Streamlined convergence checking
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class DecodingResult:
    """Result of a decoding operation."""
    success: bool
    error_estimate: np.ndarray
    weight: float
    iterations: int
    solutions_found: int
    final_marginals: np.ndarray


class GF2OptimizedRelayBP:
    """
    GF(2)-optimized Relay-BP decoder.
    
    This implementation uses binary arithmetic (GF(2)) optimizations
    while keeping the core belief propagation in the continuous domain.
    """
    
    def __init__(self, H: np.ndarray, A: np.ndarray, p: np.ndarray):
        """
        Initialize GF(2)-optimized Relay-BP decoder.
        
        Args:
            H: Parity-check matrix (M x N)
            A: Action matrix (K x N)
            p: Error probabilities (N,)
        """
        # Store matrices in both float and binary formats
        self.H_float = H.astype(np.float64)
        self.A_float = A.astype(np.float64)
        self.H_binary = H.astype(np.uint8)  # GF(2) optimized
        self.A_binary = A.astype(np.uint8)  # GF(2) optimized
        self.p = p.astype(np.float64)
        
        self.M, self.N = H.shape
        self.K = A.shape[0] if A.size > 0 else 0
        
        # Precompute log-likelihood ratios
        self.lambda_j = np.log((1 - p) / p)
        
        # Build neighbor lists
        self._build_neighbor_lists()
        
        # Precompute binary masks for efficiency
        self._precompute_binary_masks()
    
    def _build_neighbor_lists(self):
        """Build neighbor lists for message passing."""
        self.check_neighbors = []
        for i in range(self.M):
            neighbors = np.where(self.H_binary[i, :] == 1)[0]
            self.check_neighbors.append(neighbors)
        
        self.error_neighbors = []
        for j in range(self.N):
            neighbors = np.where(self.H_binary[:, j] == 1)[0]
            self.error_neighbors.append(neighbors)
    
    def _precompute_binary_masks(self):
        """Precompute binary masks for efficient GF(2) operations."""
        # Create sparse representations for faster binary ops
        self.H_sparse_rows = []
        for i in range(self.M):
            nonzero_cols = np.where(self.H_binary[i, :] == 1)[0]
            self.H_sparse_rows.append(nonzero_cols)
        
        self.A_sparse_rows = []
        for k in range(self.K):
            nonzero_cols = np.where(self.A_binary[k, :] == 1)[0]
            self.A_sparse_rows.append(nonzero_cols)
    
    def _gf2_syndrome_check(self, error_estimate: np.ndarray, syndrome: np.ndarray) -> bool:
        """Fast GF(2) syndrome validation using XOR operations."""
        # Compute H @ e mod 2 efficiently
        computed_syndrome = np.zeros(self.M, dtype=np.uint8)
        
        for i, nonzero_cols in enumerate(self.H_sparse_rows):
            # XOR operation in GF(2): sum of error_estimate[nonzero_cols] mod 2
            computed_syndrome[i] = np.sum(error_estimate[nonzero_cols]) % 2
        
        # Fast binary comparison
        return np.array_equal(computed_syndrome, syndrome.astype(np.uint8))
    
    def _gf2_logical_outcome(self, error_estimate: np.ndarray) -> np.ndarray:
        """Fast GF(2) logical outcome computation."""
        logical_outcome = np.zeros(self.K, dtype=np.uint8)
        
        for k, nonzero_cols in enumerate(self.A_sparse_rows):
            logical_outcome[k] = np.sum(error_estimate[nonzero_cols]) % 2
        
        return logical_outcome
    
    def _hard_decision(self, marginals: np.ndarray) -> np.ndarray:
        """Optimized hard decision with direct binary output."""
        return (marginals <= 0).astype(np.uint8)  # Direct binary conversion
    
    def _compute_mu_messages(self, nu: np.ndarray, syndrome: np.ndarray, t: int) -> np.ndarray:
        """Compute check-to-error messages with GF(2) optimizations."""
        mu = np.zeros((self.M, self.N), dtype=np.float64)
        
        for i in range(self.M):
            neighbors = self.check_neighbors[i]
            if len(neighbors) <= 1:
                continue
                
            for j in neighbors:
                other_neighbors = neighbors[neighbors != j]
                if len(other_neighbors) == 0:
                    continue
                
                other_messages = nu[other_neighbors, i]
                
                # Fast sign computation using GF(2) logic
                product = np.prod(other_messages)
                kappa = np.sign(product) if product != 0 else 1
                
                # Efficient syndrome factor: (-1)^syndrome[i] in GF(2)
                syndrome_factor = 1 if syndrome[i] == 0 else -1
                
                min_abs_message = np.min(np.abs(other_messages))
                mu[i, j] = kappa * syndrome_factor * min_abs_message
        
        return mu
    
    def _compute_nu_messages(self, mu: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
        """Compute error-to-check messages."""
        nu = np.zeros((self.N, self.M), dtype=np.float64)
        
        for j in range(self.N):
            neighbors = self.error_neighbors[j]
            
            for i in neighbors:
                other_neighbors = neighbors[neighbors != i]
                message_sum = np.sum(mu[other_neighbors, j]) if len(other_neighbors) > 0 else 0.0
                nu[j, i] = Lambda[j] + message_sum
        
        return nu
    
    def _compute_marginals(self, mu: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
        """Compute marginals with optimized summation."""
        marginals = Lambda.copy()  # Start with bias terms
        
        # Vectorized computation where possible
        for j in range(self.N):
            neighbors = self.error_neighbors[j]
            if len(neighbors) > 0:
                marginals[j] += np.sum(mu[neighbors, j])
        
        return marginals
    
    def _dmem_bp_leg(self, syndrome: np.ndarray, gamma: np.ndarray, 
                     initial_marginals: np.ndarray, max_iterations: int) -> Tuple[bool, np.ndarray, float, np.ndarray, int]:
        """Run one DMem-BP leg with GF(2) optimizations."""
        # Convert syndrome to binary for GF(2) operations
        syndrome_binary = syndrome.astype(np.uint8)
        
        # Initialize
        Lambda = self.lambda_j.copy()
        nu = np.zeros((self.N, self.M), dtype=np.float64)
        
        # Initialize messages
        for j in range(self.N):
            for i in self.error_neighbors[j]:
                nu[j, i] = self.lambda_j[j]
        
        marginals = initial_marginals.copy()
        
        for t in range(max_iterations):
            # Memory update
            if t > 0:
                Lambda = (1 - gamma) * self.lambda_j + gamma * marginals
            
            # Message passing
            mu = self._compute_mu_messages(nu, syndrome_binary, t)
            nu = self._compute_nu_messages(mu, Lambda)
            marginals = self._compute_marginals(mu, Lambda)
            
            # Hard decision with binary output
            error_estimate = self._hard_decision(marginals)
            
            # Fast GF(2) convergence check
            if self._gf2_syndrome_check(error_estimate, syndrome_binary):
                weight = np.sum(error_estimate * self.lambda_j)
                return True, error_estimate, weight, marginals, t + 1
        
        # No convergence
        error_estimate = self._hard_decision(marginals)
        weight = np.sum(error_estimate * self.lambda_j)
        return False, error_estimate, weight, marginals, max_iterations
    
    def decode(self, syndrome: np.ndarray, 
               S: int = 1, R: int = 301, T_r: int = 60,
               gamma_configs: Optional[List[np.ndarray]] = None,
               gamma_range: Tuple[float, float] = (-0.25, 0.85)) -> DecodingResult:
        """Run GF(2)-optimized Relay-BP decoding."""
        syndrome = syndrome.astype(np.uint8)  # Ensure binary syndrome
        
        # Initialize
        initial_marginals = self.lambda_j.copy()
        r = 0
        s = 0
        best_error = None
        best_weight = np.inf
        total_iterations = 0
        
        # Generate gamma configs if needed
        if gamma_configs is None:
            gamma_configs = self._generate_gamma_configs(R, gamma_range)
        
        # Relay loop
        for r in range(R):
            gamma = gamma_configs[r] if r < len(gamma_configs) else np.random.uniform(gamma_range[0], gamma_range[1], size=self.N)
            
            # Run DMem-BP leg
            converged, error_estimate, weight, final_marginals, iterations = self._dmem_bp_leg(
                syndrome, gamma, initial_marginals, T_r
            )
            
            total_iterations += iterations
            
            if converged:
                s += 1
                if weight < best_weight:
                    best_error = error_estimate.copy()
                    best_weight = weight
                
                if s >= S:
                    break
            
            # Pass marginals to next leg
            initial_marginals = final_marginals.copy()
        
        # Return results
        success = s > 0
        if best_error is None:
            best_error = error_estimate
            best_weight = weight
        
        return DecodingResult(
            success=success,
            error_estimate=best_error,
            weight=best_weight,
            iterations=total_iterations,
            solutions_found=s,
            final_marginals=initial_marginals
        )
    
    def _generate_gamma_configs(self, R: int, gamma_range: Tuple[float, float]) -> List[np.ndarray]:
        """Generate memory strength configurations."""
        configs = []
        
        # First leg: uniform
        first_gamma = np.full(self.N, 0.125)
        configs.append(first_gamma)
        
        # Subsequent legs: random
        for _ in range(1, R):
            gamma = np.random.uniform(gamma_range[0], gamma_range[1], size=self.N)
            configs.append(gamma)
        
        return configs
    
    def get_logical_outcome(self, error_estimate: np.ndarray) -> np.ndarray:
        """Get logical outcome using GF(2) optimization."""
        return self._gf2_logical_outcome(error_estimate)


def compare_implementations():
    """Compare original vs GF(2)-optimized implementations."""
    print("Comparing Original vs GF(2)-Optimized Relay-BP")
    print("=" * 50)
    
    # Import original implementation
    from relay_bp import RelayBP
    
    # Create test problem
    np.random.seed(42)
    H = np.random.binomial(1, 0.15, size=(20, 40))
    A = np.eye(10, 40)
    p = np.random.uniform(0.005, 0.015, size=40)
    
    # Ensure matrices are valid
    for i in range(H.shape[0]):
        if np.sum(H[i, :]) == 0:
            H[i, 0] = 1
    
    # Generate test syndrome
    true_error = np.random.binomial(1, p)
    syndrome = (H @ true_error) % 2
    
    print(f"Test problem: {H.shape[0]} checks, {H.shape[1]} errors")
    print(f"Syndrome weight: {np.sum(syndrome)}")
    
    # Test original implementation
    print("\nTesting Original Relay-BP...")
    original_decoder = RelayBP(H, A, p)
    
    import time
    start_time = time.time()
    original_result = original_decoder.decode(
        syndrome, S=2, R=20, T_r=15, gamma_range=(-0.2, 0.8)
    )
    original_time = time.time() - start_time
    
    print(f"  Success: {original_result.success}")
    print(f"  Iterations: {original_result.iterations}")
    print(f"  Time: {original_time*1000:.2f} ms")
    
    # Test GF(2)-optimized implementation
    print("\nTesting GF(2)-Optimized Relay-BP...")
    gf2_decoder = GF2OptimizedRelayBP(H, A, p)
    
    start_time = time.time()
    gf2_result = gf2_decoder.decode(
        syndrome, S=2, R=20, T_r=15, gamma_range=(-0.2, 0.8)
    )
    gf2_time = time.time() - start_time
    
    print(f"  Success: {gf2_result.success}")
    print(f"  Iterations: {gf2_result.iterations}")
    print(f"  Time: {gf2_time*1000:.2f} ms")
    
    # Compare results
    print(f"\nComparison:")
    if original_time > 0:
        speedup = original_time / gf2_time
        print(f"  Speedup: {speedup:.2f}x")
    
    # Verify correctness
    if original_result.success and gf2_result.success:
        original_syndrome_check = (H @ original_result.error_estimate) % 2
        gf2_syndrome_check = (H @ gf2_result.error_estimate) % 2
        
        original_valid = np.array_equal(original_syndrome_check, syndrome)
        gf2_valid = np.array_equal(gf2_syndrome_check, syndrome)
        
        print(f"  Original valid: {original_valid}")
        print(f"  GF(2) valid: {gf2_valid}")
        print(f"  Both solutions correct: {original_valid and gf2_valid}")
    
    return original_result, gf2_result


if __name__ == "__main__":
    try:
        compare_implementations()
        print("\n✅ GF(2) optimization testing completed!")
    except Exception as e:
        print(f"\nError: {e}")
        raise

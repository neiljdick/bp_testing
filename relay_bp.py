#!/usr/bin/env python3
"""
Relay-BP: Improved Belief Propagation for Quantum Error Correction

Implementation of the Relay-BP algorithm from the paper:
"Improved belief propagation is sufficient for real-time decoding of quantum memory"
https://arxiv.org/html/2506.01779v1

This implementation follows the pseudocode provided in the paper exactly.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import warnings
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


class RelayBP:
    """
    Relay-BP decoder implementing the algorithm from the paper.
    
    This follows the pseudocode exactly as provided in Algorithm 1.
    """
    
    def __init__(self, H: np.ndarray, A: np.ndarray, p: np.ndarray):
        """
        Initialize Relay-BP decoder.
        
        Args:
            H: Parity-check matrix (M x N) - rows are check nodes, columns are error nodes
            A: Action matrix (K x N) - for logical operations  
            p: Error probabilities (N,) - probability that each error occurs
        """
        self.H = H.astype(np.int32)
        self.A = A.astype(np.int32) 
        self.p = p.astype(np.float64)
        
        self.M, self.N = H.shape  # M check nodes, N error nodes
        self.K = A.shape[0] if A.size > 0 else 0  # Number of logical operations
        
        # Precompute λj = log((1-pj)/pj) as in line 1 of pseudocode
        self.lambda_j = np.log((1 - p) / p)
        
        # Build neighbor lists for efficient message passing
        self._build_neighbor_lists()
    
    def _build_neighbor_lists(self):
        """Build neighbor lists N(i) and N(j) for message passing."""
        # N(i): neighbors of check node i (error nodes connected to check i)
        self.check_neighbors = []
        for i in range(self.M):
            neighbors = np.where(self.H[i, :] == 1)[0]
            self.check_neighbors.append(neighbors)
        
        # N(j): neighbors of error node j (check nodes connected to error j)
        self.error_neighbors = []
        for j in range(self.N):
            neighbors = np.where(self.H[:, j] == 1)[0]
            self.error_neighbors.append(neighbors)
    
    def _hard_decision(self, marginals: np.ndarray) -> np.ndarray:
        """Hard decision function: HD(x) = (1 - sgn(x))/2.
        
        Note: We treat 0 as negative (sgn(0) = -1) for hard decisions.
        """
        # Use np.where to ensure sgn(0) = -1
        sign_vals = np.where(marginals > 0, 1, -1)
        return ((1 - sign_vals) / 2).astype(np.int32)
    
    def _compute_mu_messages(self, nu: np.ndarray, syndrome: np.ndarray, t: int) -> np.ndarray:
        """
        Compute check-to-error messages μ_{i→j}(t) via Equation (1).
        
        μ_{i→j}(t) = κ_{i,j}(t) * (-1)^{σ_i} * min_{j'∈N(i)\\{j}} |ν_{j'→i}(t-1)|
        where κ_{i,j}(t) = sgn{∏_{j'∈N(i)\\{j}} ν_{j'→i}(t-1)}
        """
        mu = np.zeros((self.M, self.N), dtype=np.float64)
        
        for i in range(self.M):
            neighbors = self.check_neighbors[i]
            if len(neighbors) <= 1:
                continue  # Need at least 2 neighbors for message passing
                
            for j in neighbors:
                # Get other neighbors: N(i) \ {j}
                other_neighbors = neighbors[neighbors != j]
                if len(other_neighbors) == 0:
                    continue
                
                # Get messages from other neighbors
                other_messages = nu[other_neighbors, i]
                
                # Compute κ_{i,j} = sgn{∏_{j'∈N(i)\{j}} ν_{j'→i}(t-1)}
                product = np.prod(other_messages)
                kappa = np.sign(product) if product != 0 else 1
                
                # Compute μ_{i→j} = κ_{i,j} * (-1)^{σ_i} * min |ν_{j'→i}|
                syndrome_factor = (-1) ** syndrome[i]
                min_abs_message = np.min(np.abs(other_messages))
                
                mu[i, j] = kappa * syndrome_factor * min_abs_message
        
        return mu
    
    def _compute_nu_messages(self, mu: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
        """
        Compute error-to-check messages ν_{j→i}(t) via Equation (2).
        
        ν_{j→i}(t) = Λ_j(t) + ∑_{i'∈N(j)\\{i}} μ_{i'→j}(t)
        """
        nu = np.zeros((self.N, self.M), dtype=np.float64)
        
        for j in range(self.N):
            neighbors = self.error_neighbors[j]
            
            for i in neighbors:
                # Get other neighbors: N(j) \ {i}
                other_neighbors = neighbors[neighbors != i]
                
                # Sum messages from other neighbors
                message_sum = np.sum(mu[other_neighbors, j]) if len(other_neighbors) > 0 else 0.0
                
                nu[j, i] = Lambda[j] + message_sum
        
        return nu
    
    def _compute_marginals(self, mu: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
        """
        Compute marginals M_j(t) via Equation (3).
        
        M_j(t) = Λ_j(t) + ∑_{i'∈N(j)} μ_{i'→j}(t)
        """
        marginals = np.zeros(self.N, dtype=np.float64)
        
        for j in range(self.N):
            neighbors = self.error_neighbors[j]
            
            # Sum all messages from neighboring check nodes
            message_sum = np.sum(mu[neighbors, j]) if len(neighbors) > 0 else 0.0
            marginals[j] = Lambda[j] + message_sum
        
        return marginals
    
    def _dmem_bp_leg(self, syndrome: np.ndarray, gamma: np.ndarray, 
                     initial_marginals: np.ndarray, max_iterations: int) -> Tuple[bool, np.ndarray, float, np.ndarray, int]:
        """
        Run one leg of DMem-BP following the pseudocode lines 3-23.
        
        Returns:
            (converged, error_estimate, weight, final_marginals, iterations)
        """
        # Line 3: Initialize Λ_j(0) ← ν_{j→i}(0) ← λ_j
        Lambda = self.lambda_j.copy()
        nu = np.zeros((self.N, self.M), dtype=np.float64)
        
        # Initialize ν_{j→i}(0) = λ_j for all edges
        for j in range(self.N):
            for i in self.error_neighbors[j]:
                nu[j, i] = self.lambda_j[j]
        
        marginals = initial_marginals.copy()
        
        # Line 4: for t ≤ T_r do
        for t in range(max_iterations):
            # Line 5: Λ_j(t) ← (1-γ_j(r))⋅Λ_j(0) + γ_j(r)⋅M_j(t-1)
            if t > 0:
                Lambda = (1 - gamma) * self.lambda_j + gamma * marginals
            
            # Line 6: Compute μ_{i→j}(t) via Eq. (1)
            mu = self._compute_mu_messages(nu, syndrome, t)
            
            # Line 7: Compute ν_{j→i}(t) via Eq. (2) 
            nu = self._compute_nu_messages(mu, Lambda)
            
            # Line 8: Compute M_j(t) via Eq. (3)
            marginals = self._compute_marginals(mu, Lambda)
            
            # Line 9: e^_j(t) ← HD(M_j(t))
            error_estimate = self._hard_decision(marginals)
            
            # Line 10: if H⋅e^(t) = σ then
            syndrome_check = (self.H @ error_estimate) % 2
            if np.array_equal(syndrome_check, syndrome):
                # Line 11: BP converged
                # Line 12: ω_r ← w(e^) = ∑_j e^_j λ_j
                weight = np.sum(error_estimate * self.lambda_j)
                return True, error_estimate, weight, marginals, t + 1
        
        # Did not converge
        error_estimate = self._hard_decision(marginals)
        weight = np.sum(error_estimate * self.lambda_j)
        return False, error_estimate, weight, marginals, max_iterations
    
    def decode(self, syndrome: np.ndarray, 
               S: int = 1,  # number of solutions to find
               R: int = 301,  # maximum number of legs
               T_r: int = 60,  # max iterations per leg
               gamma_configs: Optional[List[np.ndarray]] = None,
               gamma_range: Tuple[float, float] = (-0.25, 0.85)) -> DecodingResult:
        """
        Run Relay-BP decoding following the pseudocode exactly.
        
        Args:
            syndrome: Observed syndrome σ (M,)
            S: Number of solutions to find
            R: Maximum number of relay legs  
            T_r: Maximum iterations per leg
            gamma_configs: Memory strengths for each leg. If None, generates random configs
            gamma_range: Range for random memory strengths
            
        Returns:
            DecodingResult with best solution found
        """
        syndrome = syndrome.astype(np.int32)
        
        # Line 1: λ_j, M_j(0) ← log((1-p_j)/p_j), r←0, s←0, e^←∅, ω_e^←∞
        # λ_j already computed in __init__
        initial_marginals = self.lambda_j.copy()  # M_j(0)
        r = 0  # leg counter
        s = 0  # solutions found counter
        best_error = None  # e^
        best_weight = np.inf  # ω_e^
        
        total_iterations = 0
        
        # Generate gamma configs if not provided
        if gamma_configs is None:
            gamma_configs = self._generate_gamma_configs(R, gamma_range)
        
        # Line 2: for r ≤ R do
        for r in range(R):
            # Get memory strengths for this leg
            if r < len(gamma_configs):
                gamma = gamma_configs[r]
            else:
                # Generate random gamma if we run out of configs
                gamma = np.random.uniform(gamma_range[0], gamma_range[1], size=self.N)
            
            # Run DMem-BP leg (lines 3-23)
            converged, error_estimate, weight, final_marginals, iterations = self._dmem_bp_leg(
                syndrome, gamma, initial_marginals, T_r
            )
            
            total_iterations += iterations
            
            if converged:
                # Line 13: s ← s + 1
                s += 1
                
                # Line 14: if ω_r < ω_e^ then
                if weight < best_weight:
                    # Line 15: e^ ← e^(t)
                    best_error = error_estimate.copy()
                    # Line 16: ω_e^ ← ω_r  
                    best_weight = weight
                
                # Line 24: if s = S then
                if s >= S:
                    # Line 25: break (found enough solutions)
                    break
            
            # Line 27: M_j(0) ← M_j(t) (reuse final marginals for next leg)
            initial_marginals = final_marginals.copy()
        
        # Line 31: return (s > 0), e^
        success = s > 0
        if best_error is None:
            # If no solution found, return last error estimate
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
        """
        Generate memory strength configurations for relay legs.
        
        Based on the paper, first leg uses uniform memory strength,
        subsequent legs use random strengths from the given range.
        """
        configs = []
        
        # First leg: uniform memory strength (mentioned in paper)
        first_gamma = np.full(self.N, 0.125)  # Default value from paper
        configs.append(first_gamma)
        
        # Subsequent legs: random memory strengths from range
        for _ in range(1, R):
            gamma = np.random.uniform(gamma_range[0], gamma_range[1], size=self.N)
            configs.append(gamma)
        
        return configs


def create_example_problem(n_checks: int = 50, n_errors: int = 100, 
                          density: float = 0.1, error_prob: float = 0.01,
                          seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a synthetic quantum error correction problem for testing.
    
    Args:
        n_checks: Number of check nodes (M)
        n_errors: Number of error nodes (N)
        density: Density of connections in check matrix
        error_prob: Base error probability
        seed: Random seed for reproducibility
        
    Returns:
        (H, A, p) - Check matrix, action matrix, probability vector
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random sparse check matrix
    H = np.random.binomial(1, density, size=(n_checks, n_errors))
    
    # Ensure each check node has at least one connection
    for i in range(n_checks):
        if np.sum(H[i, :]) == 0:
            j = np.random.randint(n_errors)
            H[i, j] = 1
    
    # Ensure each error node has at least one connection
    for j in range(n_errors):
        if np.sum(H[:, j]) == 0:
            i = np.random.randint(n_checks)
            H[i, j] = 1
    
    # Generate simple action matrix (identity for first k errors)
    k = min(10, n_errors // 2)  # Number of logical operations
    A = np.zeros((k, n_errors))
    for i in range(k):
        A[i, i] = 1
    
    # Generate error probabilities (slightly random around base probability)
    p = np.random.uniform(error_prob * 0.5, error_prob * 1.5, size=n_errors)
    
    return H, A, p


def generate_random_syndrome(H: np.ndarray, p: np.ndarray, 
                           seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random syndrome by sampling errors according to probabilities.
    
    Args:
        H: Check matrix
        p: Error probabilities
        seed: Random seed for reproducibility
        
    Returns:
        (syndrome, true_error) - The observed syndrome and true error pattern
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample errors according to probabilities
    true_error = np.random.binomial(1, p)
    
    # Compute syndrome
    syndrome = (H @ true_error) % 2
    
    return syndrome, true_error


if __name__ == "__main__":
    # Example usage
    print("Relay-BP Quantum Error Correction Decoder")
    print("==========================================\n")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create a test problem
    print("Creating synthetic problem...")
    H, A, p = create_example_problem(n_checks=20, n_errors=40, density=0.15, seed=42)
    print(f"Check matrix shape: {H.shape}")
    print(f"Action matrix shape: {A.shape}") 
    print(f"Error probabilities shape: {p.shape}")
    print(f"Check matrix density: {np.mean(H):.3f}")
    
    # Generate a syndrome
    syndrome, true_error = generate_random_syndrome(H, p, seed=123)
    print(f"\nTrue error weight: {np.sum(true_error)}")
    print(f"Syndrome weight: {np.sum(syndrome)}")
    print(f"Syndrome: {syndrome[:10]}..." if len(syndrome) > 10 else f"Syndrome: {syndrome}")
    
    # Test Relay-BP
    print("\nTesting Relay-BP...")
    decoder = RelayBP(H, A, p)
    
    # Test with different parameters
    result = decoder.decode(
        syndrome, 
        S=3,  # Find 3 solutions
        R=50,  # Up to 50 legs
        T_r=30,  # 30 iterations per leg
        gamma_range=(-0.2, 0.8)
    )
    
    print(f"\nResults:")
    print(f"Success: {result.success}")
    print(f"Total iterations: {result.iterations}")
    print(f"Solutions found: {result.solutions_found}")
    print(f"Error estimate weight: {np.sum(result.error_estimate)}")
    print(f"Solution weight (log-likelihood): {result.weight:.3f}")
    
    # Verify solution
    predicted_syndrome = (H @ result.error_estimate) % 2
    is_valid = np.array_equal(predicted_syndrome, syndrome)
    print(f"Solution validates syndrome: {is_valid}")
    
    if is_valid:
        # Compare to true error
        error_diff = np.sum(result.error_estimate != true_error)
        print(f"Differences from true error: {error_diff}/{len(true_error)}")
        
        # Note: The decoded error doesn't need to match the true error exactly,
        # as long as it produces the same syndrome and logical outcome
        logical_diff = (A @ result.error_estimate) % 2
        true_logical = (A @ true_error) % 2
        logical_error = not np.array_equal(logical_diff, true_logical)
        print(f"Logical error occurred: {logical_error}")
    
    print("\nDemo completed successfully!")

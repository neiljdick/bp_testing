#!/usr/bin/env python3
"""
Extract matrices from surface code for Relay-BP testing.

This script extracts the check matrix H, action matrix A, and probability vector p
from a surface code's detector error model for use with our Relay-BP implementation.
"""

import stim
import numpy as np
from typing import Tuple
import scipy.sparse


def extract_matrices_from_dem(dem: stim.DetectorErrorModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract H, A, and p matrices from a detector error model.
    
    Args:
        dem: Detector error model from stim
        
    Returns:
        Tuple of (H, A, p) where:
        - H: Check matrix (num_detectors x num_errors)
        - A: Action matrix (num_observables x num_errors) 
        - p: Error probabilities (num_errors,)
    """
    num_detectors = dem.num_detectors
    num_observables = dem.num_observables
    
    # Collect error information
    error_data = []
    
    for instruction in dem:
        if instruction.type == "error":
            probability = instruction.args_copy()[0]
            targets = instruction.targets_copy()
            
            # Parse targets - detectors are D0, D1, ..., observables are L0, L1, ...
            detector_targets = []
            observable_targets = []
            
            for target in targets:
                if target.is_relative_detector_id():
                    detector_targets.append(target.val)
                elif target.is_logical_observable_id():
                    observable_targets.append(target.val)
                # Note: separators (^) are ignored for now
            
            error_data.append({
                'probability': probability,
                'detectors': detector_targets,
                'observables': observable_targets
            })
    
    num_errors = len(error_data)
    
    # Build matrices
    H = np.zeros((num_detectors, num_errors), dtype=np.int32)
    A = np.zeros((num_observables, num_errors), dtype=np.int32)
    p = np.zeros(num_errors, dtype=np.float64)
    
    for i, error in enumerate(error_data):
        # Set probabilities
        p[i] = error['probability']
        
        # Set detector connections (check matrix)
        for detector_id in error['detectors']:
            H[detector_id, i] = 1
        
        # Set observable connections (action matrix)
        for obs_id in error['observables']:
            A[obs_id, i] = 1
    
    return H, A, p


def test_matrix_extraction():
    """Test matrix extraction from a simple surface code."""
    print("Testing matrix extraction from surface code...")
    
    # Create a small surface code
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=2,
        distance=3,
        after_clifford_depolarization=0.001
    )
    
    # Extract detector error model
    dem = circuit.detector_error_model(decompose_errors=True)
    
    print(f"DEM: {dem.num_detectors} detectors, {dem.num_observables} observables")
    
    # Extract matrices
    H, A, p = extract_matrices_from_dem(dem)
    
    print(f"\nExtracted matrices:")
    print(f"  H shape: {H.shape} (detectors x errors)")
    print(f"  A shape: {A.shape} (observables x errors)")
    print(f"  p shape: {p.shape} (error probabilities)")
    
    print(f"\nMatrix properties:")
    print(f"  H density: {np.mean(H):.4f}")
    print(f"  A density: {np.mean(A):.4f}")
    print(f"  p range: [{np.min(p):.6f}, {np.max(p):.6f}]")
    print(f"  p mean: {np.mean(p):.6f}")
    
    # Check matrix properties
    print(f"\nMatrix validation:")
    print(f"  H max degree (detector): {np.max(np.sum(H, axis=1))}")
    print(f"  H max degree (error): {np.max(np.sum(H, axis=0))}")
    print(f"  A max degree (observable): {np.max(np.sum(A, axis=1))}")
    print(f"  A max degree (error): {np.max(np.sum(A, axis=0))}")
    
    # Verify that probabilities are valid
    assert np.all(p >= 0) and np.all(p <= 1), "Invalid probabilities"
    print(f"  All probabilities valid: ✓")
    
    return H, A, p, dem


def generate_syndrome_from_matrices(H: np.ndarray, p: np.ndarray, 
                                   seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a syndrome using the extracted matrices."""
    if seed is not None:
        np.random.seed(seed)
    
    # Sample errors according to probabilities
    errors = np.random.rand(len(p)) < p
    
    # Compute syndrome
    syndrome = (H @ errors.astype(np.int32)) % 2
    
    return syndrome, errors.astype(np.int32)


def compare_with_stim_sampling(circuit: stim.Circuit, H: np.ndarray, p: np.ndarray, 
                              num_samples: int = 1000, seed: int = 42):
    """Compare our matrix-based syndrome generation with stim's sampling."""
    print(f"\nComparing syndrome generation methods ({num_samples} samples)...")
    
    # Sample from stim
    np.random.seed(seed)
    sampler = circuit.compile_detector_sampler()
    stim_syndromes, _ = sampler.sample(shots=num_samples, separate_observables=True)
    
    # Sample from our matrices
    np.random.seed(seed)
    our_syndromes = []
    for _ in range(num_samples):
        syndrome, _ = generate_syndrome_from_matrices(H, p)
        our_syndromes.append(syndrome)
    our_syndromes = np.array(our_syndromes)
    
    # Compare syndrome statistics
    stim_weight = np.mean(np.sum(stim_syndromes, axis=1))
    our_weight = np.mean(np.sum(our_syndromes, axis=1))
    
    print(f"  Stim syndrome weight: {stim_weight:.4f}")
    print(f"  Our syndrome weight: {our_weight:.4f}")
    print(f"  Relative difference: {abs(stim_weight - our_weight) / stim_weight:.4f}")
    
    # Note: Exact matching isn't expected because:
    # 1. Stim uses a more sophisticated error model
    # 2. Our extraction may not capture all correlations
    # 3. Different random number usage
    
    return stim_syndromes, our_syndromes


if __name__ == "__main__":
    print("Surface Code Matrix Extraction")
    print("=" * 50)
    
    try:
        # Test matrix extraction
        H, A, p, dem = test_matrix_extraction()
        
        # Generate some test syndromes
        print(f"\nGenerating test syndromes...")
        for i in range(3):
            syndrome, errors = generate_syndrome_from_matrices(H, p, seed=100+i)
            print(f"  Sample {i+1}: {np.sum(errors)} errors → syndrome weight {np.sum(syndrome)}")
        
        # Compare with stim sampling
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            rounds=2,
            distance=3,
            after_clifford_depolarization=0.001
        )
        
        stim_syndromes, our_syndromes = compare_with_stim_sampling(circuit, H, p)
        
        print(f"\n✓ Matrix extraction completed successfully!")
        print(f"  Ready for Relay-BP testing with {H.shape[1]} error mechanisms")
        
    except Exception as e:
        print(f"\nError during matrix extraction: {e}")
        raise

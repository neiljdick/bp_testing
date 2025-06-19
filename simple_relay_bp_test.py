#!/usr/bin/env python3
"""
Simple test of Relay-BP on surface code syndromes.

Quick comparison between Relay-BP and PyMatching on a few examples.
"""

import stim
import pymatching
import numpy as np
import time

# Import our implementations
from relay_bp import RelayBP
from surface_code_matrices import extract_matrices_from_dem


def quick_test():
    """Quick test of Relay-BP vs PyMatching."""
    print("Quick Relay-BP vs PyMatching Test")
    print("=" * 40)
    
    # Create small surface code
    print("\n1. Creating surface code...")
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=2,
        distance=3,
        after_clifford_depolarization=0.002  # 0.2% error rate
    )
    
    dem = circuit.detector_error_model(decompose_errors=True)
    print(f"   {dem.num_detectors} detectors, {dem.num_errors} error mechanisms")
    
    # Extract matrices
    print("\n2. Extracting matrices...")
    H, A, p = extract_matrices_from_dem(dem)
    print(f"   H: {H.shape}, A: {A.shape}, p: {p.shape}")
    
    # Create decoders
    print("\n3. Creating decoders...")
    pymatching_decoder = pymatching.Matching.from_detector_error_model(dem)
    relay_bp_decoder = RelayBP(H, A, p)
    
    # Sample a few syndromes
    print("\n4. Sampling syndromes...")
    sampler = circuit.compile_detector_sampler()
    syndromes, true_logicals = sampler.sample(shots=10, separate_observables=True)
    
    print(f"   Generated {len(syndromes)} syndromes")
    
    # Test each syndrome
    print("\n5. Testing decoders...")
    print(f"{'Case':<6} {'Syn Wt':<8} {'True Log':<10} {'PyMatch':<10} {'Relay-BP':<10} {'PM Time':<10} {'RBP Time':<10}")
    print("-" * 70)
    
    for i, (syndrome, true_logical) in enumerate(zip(syndromes, true_logicals)):
        syndrome_weight = np.sum(syndrome)
        
        # Skip zero syndromes
        if syndrome_weight == 0:
            print(f"{i+1:<6} {syndrome_weight:<8} {str(true_logical):<10} {'SKIP':<10} {'SKIP':<10} {'-':<10} {'-':<10}")
            continue
        
        # PyMatching
        start_time = time.time()
        pm_prediction = pymatching_decoder.decode(syndrome)
        pm_time = time.time() - start_time
        pm_error = not np.array_equal(pm_prediction, true_logical)
        
        # Relay-BP
        start_time = time.time()
        rbp_result = relay_bp_decoder.decode(
            syndrome,
            S=2,       # Find 2 solutions
            R=20,      # Up to 20 legs
            T_r=15,    # 15 iterations per leg
            gamma_range=(-0.2, 0.8)
        )
        rbp_time = time.time() - start_time
        
        if rbp_result.success:
            rbp_logical = (A @ rbp_result.error_estimate) % 2
            rbp_error = not np.array_equal(rbp_logical, true_logical)
            rbp_status = "ERROR" if rbp_error else "OK"
        else:
            rbp_logical = np.zeros_like(true_logical)
            rbp_error = not np.array_equal(rbp_logical, true_logical)
            rbp_status = "FAIL"
        
        pm_status = "ERROR" if pm_error else "OK"
        
        print(f"{i+1:<6} {syndrome_weight:<8} {str(true_logical):<10} {pm_status:<10} {rbp_status:<10} "
              f"{pm_time*1000:<10.2f} {rbp_time*1000:<10.2f}")
        
        # Show details for interesting cases
        if syndrome_weight > 0 and (pm_error or rbp_error or not rbp_result.success):
            print(f"       Details - PM: {pm_prediction}, RBP: {rbp_logical} (iter: {rbp_result.iterations})")
    
    print("\n✓ Quick test completed!")


def focused_test():
    """Test on a specific challenging syndrome."""
    print("\n" + "=" * 40)
    print("Focused Test on Challenging Case")
    print("=" * 40)
    
    # Create surface code with higher noise
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.005  # 0.5% error rate
    )
    
    dem = circuit.detector_error_model(decompose_errors=True)
    H, A, p = extract_matrices_from_dem(dem)
    
    # Sample until we get a non-zero syndrome
    sampler = circuit.compile_detector_sampler()
    
    for attempt in range(50):
        syndromes, true_logicals = sampler.sample(shots=1, separate_observables=True)
        syndrome = syndromes[0]
        true_logical = true_logicals[0]
        
        if np.sum(syndrome) > 2:  # Want a challenging case
            break
    
    print(f"\nFound syndrome with weight {np.sum(syndrome)}")
    print(f"True logical: {true_logical}")
    print(f"Syndrome: {syndrome[:10]}..." if len(syndrome) > 10 else f"Syndrome: {syndrome}")
    
    # Test PyMatching
    pymatching_decoder = pymatching.Matching.from_detector_error_model(dem)
    pm_prediction = pymatching_decoder.decode(syndrome)
    pm_error = not np.array_equal(pm_prediction, true_logical)
    
    print(f"\nPyMatching:")
    print(f"  Prediction: {pm_prediction}")
    print(f"  Logical error: {pm_error}")
    
    # Test Relay-BP with different configurations
    relay_decoder = RelayBP(H, A, p)
    
    configs = [
        {"name": "Conservative", "S": 1, "R": 10, "T_r": 20, "gamma": (-0.1, 0.5)},
        {"name": "Balanced", "S": 3, "R": 30, "T_r": 25, "gamma": (-0.25, 0.85)},
        {"name": "Aggressive", "S": 5, "R": 50, "T_r": 40, "gamma": (-0.5, 1.2)},
    ]
    
    print(f"\nRelay-BP configurations:")
    
    for config in configs:
        print(f"\n{config['name']} config:")
        
        start_time = time.time()
        result = relay_decoder.decode(
            syndrome,
            S=config['S'],
            R=config['R'],
            T_r=config['T_r'],
            gamma_range=config['gamma']
        )
        decode_time = time.time() - start_time
        
        print(f"  Success: {result.success}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Solutions found: {result.solutions_found}")
        print(f"  Time: {decode_time*1000:.2f} ms")
        
        if result.success:
            rbp_logical = (A @ result.error_estimate) % 2
            rbp_error = not np.array_equal(rbp_logical, true_logical)
            print(f"  Prediction: {rbp_logical}")
            print(f"  Logical error: {rbp_error}")
            print(f"  Error estimate weight: {np.sum(result.error_estimate)}")
            print(f"  Solution weight: {result.weight:.3f}")
        else:
            print(f"  Failed to converge")


if __name__ == "__main__":
    try:
        quick_test()
        focused_test()
    except Exception as e:
        print(f"\nError: {e}")
        raise

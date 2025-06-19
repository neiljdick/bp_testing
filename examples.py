#!/usr/bin/env python3
"""
Example scripts demonstrating Relay-BP usage for quantum error correction.

This file contains several examples showing different aspects of the Relay-BP
algorithm and how to use it effectively.
"""

import numpy as np
import time
from relay_bp import RelayBP, create_example_problem, generate_random_syndrome


def example_basic_usage():
    """Basic example of using Relay-BP for quantum error correction."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Create a moderate-sized problem
    print("Creating quantum error correction problem...")
    H, A, p = create_example_problem(
        n_checks=30,
        n_errors=60, 
        density=0.12,
        error_prob=0.02,
        seed=42
    )
    
    print(f"Problem size: {H.shape[0]} checks, {H.shape[1]} error locations")
    print(f"Check matrix density: {np.mean(H):.3f}")
    print(f"Average error probability: {np.mean(p):.4f}")
    
    # Generate syndrome with some errors
    syndrome, true_error = generate_random_syndrome(H, p, seed=123)
    true_weight = np.sum(true_error)
    syndrome_weight = np.sum(syndrome)
    
    print(f"\nGenerated syndrome:")
    print(f"  True error weight: {true_weight}")
    print(f"  Syndrome weight: {syndrome_weight}")
    
    if syndrome_weight == 0:
        print("  (No errors occurred - trivial case)")
    
    # Initialize decoder
    decoder = RelayBP(H, A, p)
    
    # Run decoding
    print("\nRunning Relay-BP decoding...")
    start_time = time.time()
    
    result = decoder.decode(
        syndrome,
        S=3,          # Find up to 3 solutions
        R=50,         # Use up to 50 relay legs
        T_r=40,       # 40 iterations per leg
        gamma_range=(-0.3, 0.9)
    )
    
    decode_time = time.time() - start_time
    
    # Display results
    print(f"\nDecoding Results:")
    print(f"  Success: {result.success}")
    print(f"  Solutions found: {result.solutions_found}")
    print(f"  Total iterations: {result.iterations}")
    print(f"  Estimated error weight: {np.sum(result.error_estimate)}")
    print(f"  Solution log-likelihood: {result.weight:.3f}")
    print(f"  Decoding time: {decode_time:.3f} seconds")
    
    # Verify solution correctness
    predicted_syndrome = (H @ result.error_estimate) % 2
    is_valid = np.array_equal(predicted_syndrome, syndrome)
    print(f"  Solution validates syndrome: {is_valid}")
    
    if is_valid and true_weight > 0:
        # Compare to true error (they don't need to match exactly)
        differences = np.sum(result.error_estimate != true_error)
        print(f"  Differences from true error: {differences}/{len(true_error)}")
        
        # Check logical error
        logical_diff = (A @ result.error_estimate) % 2
        true_logical = (A @ true_error) % 2
        logical_error = not np.array_equal(logical_diff, true_logical)
        print(f"  Logical error occurred: {logical_error}")


def example_memory_strength_analysis():
    """Example showing the effect of different memory strength ranges."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Memory Strength Analysis")
    print("="*60)
    
    # Create test problem
    H, A, p = create_example_problem(
        n_checks=25, n_errors=50, density=0.15, error_prob=0.015, seed=42
    )
    syndrome, _ = generate_random_syndrome(H, p, seed=456)
    
    if np.sum(syndrome) == 0:
        print("Zero syndrome - skipping memory strength analysis")
        return
    
    decoder = RelayBP(H, A, p)
    
    # Test different memory strength ranges
    ranges_to_test = [
        (-0.1, 0.3),   # Conservative range
        (-0.25, 0.85), # Paper default
        (-0.5, 1.2),   # Aggressive range
        (0.0, 0.5),    # Positive only
    ]
    
    print(f"Testing different memory strength ranges...")
    print(f"Syndrome weight: {np.sum(syndrome)}")
    
    results = []
    
    for i, gamma_range in enumerate(ranges_to_test):
        print(f"\nRange {i+1}: [{gamma_range[0]:.2f}, {gamma_range[1]:.2f}]")
        
        # Set seed for fair comparison
        np.random.seed(789)
        
        start_time = time.time()
        result = decoder.decode(
            syndrome,
            S=2,
            R=30,
            T_r=25,
            gamma_range=gamma_range
        )
        decode_time = time.time() - start_time
        
        results.append((gamma_range, result, decode_time))
        
        print(f"  Success: {result.success}")
        print(f"  Solutions: {result.solutions_found}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Weight: {result.weight:.3f}")
        print(f"  Time: {decode_time:.3f}s")
    
    # Find best result
    successful_results = [(r, res, t) for r, res, t in results if res.success]
    
    if successful_results:
        best_range, best_result, best_time = min(
            successful_results, key=lambda x: x[1].weight
        )
        print(f"\nBest result: range {best_range} with weight {best_result.weight:.3f}")
    else:
        print("\nNo successful decodings found.")


def example_custom_gamma_configs():
    """Example using custom memory strength configurations."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Memory Configurations")
    print("="*60)
    
    # Create problem
    H, A, p = create_example_problem(
        n_checks=20, n_errors=40, density=0.18, error_prob=0.012, seed=42
    )
    syndrome, _ = generate_random_syndrome(H, p, seed=789)
    
    decoder = RelayBP(H, A, p)
    
    print(f"Problem: {H.shape[0]} checks, {H.shape[1]} errors")
    print(f"Syndrome weight: {np.sum(syndrome)}")
    
    # Design custom memory strength configurations
    gamma_configs = [
        # Leg 1: Uniform small positive memory
        np.full(decoder.N, 0.1),
        
        # Leg 2: Small random values around 0
        np.random.normal(0.0, 0.2, decoder.N),
        
        # Leg 3: Bimodal distribution (some negative, some positive)
        np.concatenate([
            np.random.uniform(-0.4, -0.1, decoder.N // 2),
            np.random.uniform(0.3, 0.8, decoder.N - decoder.N // 2)
        ]),
        
        # Leg 4: High variance random
        np.random.uniform(-0.6, 1.0, decoder.N),
        
        # Leg 5: Structured based on error probabilities
        0.5 * (np.log(p) - np.mean(np.log(p))) / np.std(np.log(p)),
    ]
    
    print(f"\nCustom configurations designed:")
    for i, gamma in enumerate(gamma_configs):
        print(f"  Leg {i+1}: mean={np.mean(gamma):.3f}, std={np.std(gamma):.3f}, "
              f"range=[{np.min(gamma):.3f}, {np.max(gamma):.3f}]")
        neg_frac = np.mean(gamma < 0)
        print(f"         negative fraction: {neg_frac:.2f}")
    
    # Run with custom configurations
    print("\nRunning with custom configurations...")
    result = decoder.decode(
        syndrome,
        S=3,
        R=len(gamma_configs),
        T_r=35,
        gamma_configs=gamma_configs
    )
    
    print(f"\nResults:")
    print(f"  Success: {result.success}")
    print(f"  Solutions found: {result.solutions_found}")
    print(f"  Total iterations: {result.iterations}")
    print(f"  Final weight: {result.weight:.3f}")
    
    # Verify
    predicted_syndrome = (H @ result.error_estimate) % 2
    is_valid = np.array_equal(predicted_syndrome, syndrome)
    print(f"  Valid solution: {is_valid}")


def example_performance_comparison():
    """Example comparing different Relay-BP parameter settings."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Performance Comparison")
    print("="*60)
    
    # Create challenging problem
    H, A, p = create_example_problem(
        n_checks=40, n_errors=80, density=0.1, error_prob=0.025, seed=42
    )
    syndrome, true_error = generate_random_syndrome(H, p, seed=101)
    
    if np.sum(syndrome) == 0:
        print("Zero syndrome - creating artificial syndrome for comparison")
        # Create a syndrome with a few errors for testing
        artificial_error = np.zeros(len(p), dtype=int)
        artificial_error[:3] = 1  # First 3 errors
        syndrome = (H @ artificial_error) % 2
    
    decoder = RelayBP(H, A, p)
    
    print(f"Problem size: {H.shape}")
    print(f"Syndrome weight: {np.sum(syndrome)}")
    
    # Test different parameter combinations
    test_configs = [
        {"name": "Fast (low accuracy)", "S": 1, "R": 10, "T_r": 15},
        {"name": "Balanced", "S": 2, "R": 30, "T_r": 25},
        {"name": "Thorough (high accuracy)", "S": 5, "R": 100, "T_r": 50},
        {"name": "Single leg (standard BP)", "S": 1, "R": 1, "T_r": 100},
    ]
    
    print("\nTesting different configurations:")
    
    for config in test_configs:
        print(f"\n{config['name']}:")
        print(f"  Parameters: S={config['S']}, R={config['R']}, T_r={config['T_r']}")
        
        # Run decoding
        np.random.seed(202)  # Consistent seed for fair comparison
        start_time = time.time()
        
        result = decoder.decode(
            syndrome,
            S=config['S'],
            R=config['R'], 
            T_r=config['T_r'],
            gamma_range=(-0.25, 0.85)
        )
        
        decode_time = time.time() - start_time
        
        # Results
        print(f"  Success: {result.success}")
        print(f"  Solutions: {result.solutions_found}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Time: {decode_time:.3f}s")
        print(f"  Iterations/second: {result.iterations/decode_time:.0f}")
        
        if result.success:
            print(f"  Weight: {result.weight:.3f}")
            error_weight = np.sum(result.error_estimate)
            print(f"  Error weight: {error_weight}")


def example_edge_cases():
    """Example demonstrating edge cases and robustness."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Edge Cases and Robustness")
    print("="*60)
    
    # Test with very small problem
    print("\nTesting with minimal problem size...")
    H_small = np.array([[1, 1, 0], [0, 1, 1]])
    A_small = np.eye(3)
    p_small = np.array([0.1, 0.1, 0.1])
    
    decoder_small = RelayBP(H_small, A_small, p_small)
    syndrome_small = np.array([1, 0])
    
    result_small = decoder_small.decode(syndrome_small, S=1, R=5, T_r=10)
    print(f"  Small problem success: {result_small.success}")
    print(f"  Solution: {result_small.error_estimate}")
    
    # Test with extreme memory ranges
    print("\nTesting with extreme memory strength ranges...")
    H, A, p = create_example_problem(n_checks=15, n_errors=30, seed=42)
    syndrome, _ = generate_random_syndrome(H, p, seed=303)
    decoder = RelayBP(H, A, p)
    
    extreme_ranges = [
        (-2.0, -1.0),  # All negative
        (1.0, 2.0),    # All high positive
        (-0.01, 0.01), # Very small range around zero
    ]
    
    for i, gamma_range in enumerate(extreme_ranges):
        print(f"  Range {gamma_range}: ", end="")
        result = decoder.decode(
            syndrome, S=1, R=10, T_r=20, gamma_range=gamma_range
        )
        print(f"success={result.success}, iterations={result.iterations}")
    
    # Test with single iteration per leg
    print("\nTesting with single iteration per leg...")
    result_single = decoder.decode(syndrome, S=1, R=20, T_r=1)
    print(f"  Single iteration: success={result_single.success}")
    
    print("\nEdge case testing completed.")


def main():
    """Run all examples."""
    print("Relay-BP Examples")
    print("=" * 60)
    print("This script demonstrates various aspects of the Relay-BP algorithm.")
    
    # Set global seed for reproducibility
    np.random.seed(42)
    
    try:
        example_basic_usage()
        example_memory_strength_analysis() 
        example_custom_gamma_configs()
        example_performance_comparison()
        example_edge_cases()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during examples: {e}")
        raise


if __name__ == "__main__":
    main()

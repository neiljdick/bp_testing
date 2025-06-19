#!/usr/bin/env python3
"""
Surface Code Test with Stim and PyMatching

This script creates a surface code using stim, generates detector error models,
samples syndromes with errors, and decodes them using PyMatching.
"""

import stim
import pymatching
import numpy as np
from typing import Tuple, List


def create_surface_code_circuit(distance: int, rounds: int, noise_level: float) -> stim.Circuit:
    """
    Create a surface code circuit with the given parameters.
    
    Args:
        distance: Code distance (must be odd)
        rounds: Number of syndrome measurement rounds
        noise_level: Physical error rate (e.g., 0.001 for 0.1%)
        
    Returns:
        stim.Circuit: The surface code circuit
    """
    if distance % 2 == 0:
        raise ValueError("Distance must be odd")
    
    # Create the circuit using stim's surface code generator
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=noise_level,
        after_reset_flip_probability=noise_level,
        before_measure_flip_probability=noise_level,
        before_round_data_depolarization=noise_level
    )
    
    return circuit


def extract_detector_error_model(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """
    Extract the detector error model from the circuit.
    
    Args:
        circuit: The quantum circuit
        
    Returns:
        stim.DetectorErrorModel: The detector error model
    """
    return circuit.detector_error_model(decompose_errors=True)


def sample_detection_events(circuit: stim.Circuit, num_shots: int, 
                          seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample detection events (syndromes) and logical observables from the circuit.
    
    Args:
        circuit: The quantum circuit
        num_shots: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (detection_events, logical_observables)
        - detection_events: Binary array (num_shots, num_detectors)
        - logical_observables: Binary array (num_shots, num_observables)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create a sampler
    sampler = circuit.compile_detector_sampler()
    
    # Sample detection events and logical observables
    detection_events, logical_observables = sampler.sample(
        shots=num_shots, 
        separate_observables=True
    )
    
    return detection_events, logical_observables


def create_pymatching_decoder(dem: stim.DetectorErrorModel) -> pymatching.Matching:
    """
    Create a PyMatching decoder from the detector error model.
    
    Args:
        dem: The detector error model
        
    Returns:
        pymatching.Matching: The decoder
    """
    return pymatching.Matching.from_detector_error_model(dem)


def decode_syndromes(decoder: pymatching.Matching, 
                    detection_events: np.ndarray) -> np.ndarray:
    """
    Decode detection events (syndromes) using PyMatching.
    
    Args:
        decoder: The PyMatching decoder
        detection_events: Binary array of detection events (num_shots, num_detectors)
        
    Returns:
        np.ndarray: Predicted logical observables (num_shots, num_observables)
    """
    return decoder.decode_batch(detection_events)


def calculate_logical_error_rate(true_observables: np.ndarray, 
                               predicted_observables: np.ndarray) -> float:
    """
    Calculate the logical error rate.
    
    Args:
        true_observables: True logical observable outcomes
        predicted_observables: Decoder predictions
        
    Returns:
        float: Logical error rate (fraction of incorrect predictions)
    """
    errors = np.any(true_observables != predicted_observables, axis=1)
    return np.mean(errors)


def analyze_surface_code_performance(distance: int, rounds: int, 
                                   noise_level: float, num_shots: int,
                                   seed: int = None) -> dict:
    """
    Complete analysis of surface code performance.
    
    Args:
        distance: Code distance
        rounds: Number of syndrome measurement rounds
        noise_level: Physical error rate
        num_shots: Number of error samples
        seed: Random seed
        
    Returns:
        dict: Performance analysis results
    """
    print(f"Analyzing surface code performance:")
    print(f"  Distance: {distance}")
    print(f"  Rounds: {rounds}")
    print(f"  Noise level: {noise_level}")
    print(f"  Shots: {num_shots}")
    
    # Create circuit
    print("\n1. Creating surface code circuit...")
    circuit = create_surface_code_circuit(distance, rounds, noise_level)
    print(f"   Circuit has {len(circuit)} instructions")
    
    # Extract detector error model
    print("\n2. Extracting detector error model...")
    dem = extract_detector_error_model(circuit)
    print(f"   DEM has {dem.num_detectors} detectors")
    print(f"   DEM has {dem.num_observables} logical observables")
    print(f"   DEM has {dem.num_errors} error mechanisms")
    
    # Sample detection events
    print(f"\n3. Sampling {num_shots} detection events...")
    detection_events, true_observables = sample_detection_events(
        circuit, num_shots, seed
    )
    
    syndrome_weight = np.mean(np.sum(detection_events, axis=1))
    logical_flip_rate = np.mean(np.any(true_observables, axis=1))
    
    print(f"   Average syndrome weight: {syndrome_weight:.2f}")
    print(f"   Logical flip rate (before decoding): {logical_flip_rate:.4f}")
    
    # Create decoder
    print("\n4. Creating PyMatching decoder...")
    decoder = create_pymatching_decoder(dem)
    
    # Decode syndromes
    print("\n5. Decoding syndromes...")
    predicted_observables = decode_syndromes(decoder, detection_events)
    
    # Calculate error rates
    logical_error_rate = calculate_logical_error_rate(
        true_observables, predicted_observables
    )
    
    print(f"\n6. Results:")
    print(f"   Logical error rate: {logical_error_rate:.6f}")
    
    # Analysis breakdown
    total_errors = np.sum(np.any(true_observables, axis=1))
    corrected_errors = np.sum(
        np.any(true_observables, axis=1) & 
        ~np.any(true_observables != predicted_observables, axis=1)
    )
    
    print(f"   Total logical errors: {total_errors}")
    print(f"   Successfully corrected: {corrected_errors}")
    print(f"   Correction rate: {corrected_errors/total_errors:.4f}" if total_errors > 0 else "   (No errors to correct)")
    
    return {
        'distance': distance,
        'rounds': rounds,
        'noise_level': noise_level,
        'num_shots': num_shots,
        'num_detectors': dem.num_detectors,
        'num_observables': dem.num_observables,
        'syndrome_weight': syndrome_weight,
        'logical_flip_rate': logical_flip_rate,
        'logical_error_rate': logical_error_rate,
        'total_errors': total_errors,
        'corrected_errors': corrected_errors,
        'circuit': circuit,
        'dem': dem,
        'decoder': decoder
    }


def demo_basic_surface_code():
    """Basic demonstration of surface code simulation and decoding."""
    print("=" * 60)
    print("BASIC SURFACE CODE DEMO")
    print("=" * 60)
    
    # Test with a small surface code
    results = analyze_surface_code_performance(
        distance=3,
        rounds=3,
        noise_level=0.001,  # 0.1% error rate
        num_shots=10000,
        seed=42
    )
    
    return results


def demo_distance_comparison():
    """Compare performance across different code distances."""
    print("\n" + "=" * 60)
    print("DISTANCE COMPARISON DEMO")
    print("=" * 60)
    
    distances = [3, 5]
    noise_level = 0.002  # 0.2% error rate
    rounds = 3
    num_shots = 5000
    
    results = []
    
    for distance in distances:
        print(f"\n{'='*20} Distance {distance} {'='*20}")
        result = analyze_surface_code_performance(
            distance=distance,
            rounds=rounds,
            noise_level=noise_level,
            num_shots=num_shots,
            seed=123
        )
        results.append(result)
    
    # Summary comparison
    print(f"\n{'='*20} COMPARISON SUMMARY {'='*20}")
    print(f"{'Distance':<10} {'Detectors':<12} {'Log Error Rate':<15} {'Syndrome Wt':<12}")
    print("-" * 55)
    
    for result in results:
        print(f"{result['distance']:<10} {result['num_detectors']:<12} "
              f"{result['logical_error_rate']:<15.6f} {result['syndrome_weight']:<12.2f}")
    
    return results


def inspect_detector_error_model(dem: stim.DetectorErrorModel):
    """Inspect the structure of a detector error model."""
    print(f"\nDetector Error Model Structure:")
    print(f"  Detectors: {dem.num_detectors}")
    print(f"  Observables: {dem.num_observables}")
    print(f"  Error mechanisms: {dem.num_errors}")
    
    # Show first few error mechanisms
    print(f"\nFirst few error mechanisms:")
    error_count = 0
    for instruction in dem:
        if error_count >= 5:  # Show first 5
            break
        if instruction.type == "error":
            targets = [str(t) for t in instruction.targets_copy()]
            prob = instruction.args_copy()[0] if instruction.args_copy() else "?"
            print(f"  Error {error_count}: probability={prob}, targets={targets[:5]}{'...' if len(targets) > 5 else ''}")
            error_count += 1


if __name__ == "__main__":
    print("Surface Code Testing with Stim and PyMatching")
    print("=" * 60)
    
    try:
        # Basic demo
        basic_results = demo_basic_surface_code()
        
        # Inspect the detector error model
        inspect_detector_error_model(basic_results['dem'])
        
        # Distance comparison
        comparison_results = demo_distance_comparison()
        
        print("\n" + "=" * 60)
        print("SURFACE CODE TESTING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during surface code testing: {e}")
        raise

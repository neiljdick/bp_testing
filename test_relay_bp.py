#!/usr/bin/env python3
"""
Unit tests for Relay-BP implementation.
"""

import numpy as np
import unittest
from relay_bp import RelayBP, create_example_problem, generate_random_syndrome


class TestRelayBP(unittest.TestCase):
    """Test cases for Relay-BP decoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.H, self.A, self.p = create_example_problem(
            n_checks=10, n_errors=20, density=0.2, seed=42
        )
        self.decoder = RelayBP(self.H, self.A, self.p)
    
    def test_initialization(self):
        """Test decoder initialization."""
        self.assertEqual(self.decoder.H.shape, (10, 20))
        self.assertEqual(self.decoder.A.shape, (10, 20))
        self.assertEqual(self.decoder.p.shape, (20,))
        self.assertEqual(self.decoder.M, 10)
        self.assertEqual(self.decoder.N, 20)
        
        # Check lambda_j computation
        expected_lambda = np.log((1 - self.p) / self.p)
        np.testing.assert_array_almost_equal(self.decoder.lambda_j, expected_lambda)
    
    def test_neighbor_lists(self):
        """Test neighbor list construction."""
        # Check that neighbor lists are consistent with H matrix
        for i in range(self.decoder.M):
            expected_neighbors = np.where(self.H[i, :] == 1)[0]
            np.testing.assert_array_equal(self.decoder.check_neighbors[i], expected_neighbors)
        
        for j in range(self.decoder.N):
            expected_neighbors = np.where(self.H[:, j] == 1)[0]
            np.testing.assert_array_equal(self.decoder.error_neighbors[j], expected_neighbors)
    
    def test_hard_decision(self):
        """Test hard decision function."""
        marginals = np.array([1.5, -0.5, 0.0, -2.0, 3.0])
        expected = np.array([0, 1, 1, 1, 0])
        result = self.decoder._hard_decision(marginals)
        np.testing.assert_array_equal(result, expected)
    
    def test_zero_syndrome_decoding(self):
        """Test decoding with zero syndrome (no errors)."""
        syndrome = np.zeros(self.decoder.M, dtype=np.int32)
        
        result = self.decoder.decode(syndrome, S=1, R=10, T_r=20)
        
        # Should find the all-zero solution quickly
        self.assertTrue(result.success)
        self.assertEqual(np.sum(result.error_estimate), 0)
        
        # Verify syndrome match
        predicted_syndrome = (self.H @ result.error_estimate) % 2
        np.testing.assert_array_equal(predicted_syndrome, syndrome)
    
    def test_simple_syndrome_decoding(self):
        """Test decoding with a simple syndrome."""
        # Create a known error pattern
        true_error = np.zeros(self.decoder.N, dtype=np.int32)
        true_error[0] = 1  # Single error
        
        syndrome = (self.H @ true_error) % 2
        
        result = self.decoder.decode(
            syndrome, S=1, R=50, T_r=30, 
            gamma_range=(-0.3, 0.9)
        )
        
        # Should find a valid solution
        predicted_syndrome = (self.H @ result.error_estimate) % 2
        np.testing.assert_array_equal(predicted_syndrome, syndrome)
    
    def test_memory_strengths(self):
        """Test that different memory strengths are used."""
        syndrome, _ = generate_random_syndrome(self.H, self.p, seed=123)
        
        # Test with custom gamma configurations
        gamma_configs = [
            np.full(self.decoder.N, 0.1),  # First leg: uniform
            np.random.uniform(-0.2, 0.8, self.decoder.N),  # Second leg: random
            np.random.uniform(-0.5, 1.0, self.decoder.N),  # Third leg: different range
        ]
        
        result = self.decoder.decode(
            syndrome, S=1, R=3, T_r=20, 
            gamma_configs=gamma_configs
        )
        
        # Should complete without errors
        predicted_syndrome = (self.H @ result.error_estimate) % 2
        np.testing.assert_array_equal(predicted_syndrome, syndrome)
    
    def test_multiple_solutions(self):
        """Test finding multiple solutions."""
        syndrome, _ = generate_random_syndrome(self.H, self.p, seed=456)
        
        if np.sum(syndrome) == 0:
            # Skip if syndrome is zero (trivial case)
            return
        
        result = self.decoder.decode(
            syndrome, S=3, R=100, T_r=25
        )
        
        # Should find at least one solution
        predicted_syndrome = (self.H @ result.error_estimate) % 2
        np.testing.assert_array_equal(predicted_syndrome, syndrome)
        
        # Solutions found should be <= requested
        self.assertLessEqual(result.solutions_found, 3)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        syndrome = np.zeros(self.decoder.M, dtype=np.int32)
        
        # Test with S=0 (should still run)
        result = self.decoder.decode(syndrome, S=0, R=5, T_r=10)
        self.assertIsNotNone(result.error_estimate)
        
        # Test with R=1 (single leg)
        result = self.decoder.decode(syndrome, S=1, R=1, T_r=10)
        self.assertIsNotNone(result.error_estimate)
        
        # Test with T_r=1 (single iteration per leg)
        result = self.decoder.decode(syndrome, S=1, R=5, T_r=1)
        self.assertIsNotNone(result.error_estimate)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        syndrome, _ = generate_random_syndrome(self.H, self.p, seed=789)
        
        # Run decoder twice with same random seed
        np.random.seed(100)
        result1 = self.decoder.decode(syndrome, S=1, R=10, T_r=20)
        
        np.random.seed(100)
        result2 = self.decoder.decode(syndrome, S=1, R=10, T_r=20)
        
        # Results should be identical
        np.testing.assert_array_equal(result1.error_estimate, result2.error_estimate)
        self.assertEqual(result1.success, result2.success)
        self.assertEqual(result1.weight, result2.weight)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""
    
    def test_create_example_problem(self):
        """Test example problem creation."""
        H, A, p = create_example_problem(
            n_checks=5, n_errors=10, density=0.3, seed=42
        )
        
        self.assertEqual(H.shape, (5, 10))
        self.assertEqual(A.shape, (5, 10))  # min(10, 10//2) = 5
        self.assertEqual(p.shape, (10,))
        
        # Check that each check node has at least one connection
        for i in range(5):
            self.assertGreater(np.sum(H[i, :]), 0, f"Check node {i} has no connections")
        
        # Check that each error node has at least one connection
        for j in range(10):
            self.assertGreater(np.sum(H[:, j]), 0, f"Error node {j} has no connections")
        
        # Check probability bounds
        self.assertTrue(np.all(p > 0))
        self.assertTrue(np.all(p < 1))
    
    def test_generate_random_syndrome(self):
        """Test syndrome generation."""
        H, _, p = create_example_problem(n_checks=8, n_errors=15, seed=42)
        
        syndrome, true_error = generate_random_syndrome(H, p, seed=123)
        
        self.assertEqual(syndrome.shape, (8,))
        self.assertEqual(true_error.shape, (15,))
        
        # Verify syndrome calculation
        expected_syndrome = (H @ true_error) % 2
        np.testing.assert_array_equal(syndrome, expected_syndrome)
        
        # Check that error values are binary
        self.assertTrue(np.all(np.isin(true_error, [0, 1])))
    
    def test_syndrome_reproducibility(self):
        """Test that syndrome generation is reproducible."""
        H, _, p = create_example_problem(n_checks=6, n_errors=12, seed=42)
        
        syndrome1, error1 = generate_random_syndrome(H, p, seed=999)
        syndrome2, error2 = generate_random_syndrome(H, p, seed=999)
        
        np.testing.assert_array_equal(syndrome1, syndrome2)
        np.testing.assert_array_equal(error1, error2)


class TestMessagePassing(unittest.TestCase):
    """Test message passing components."""
    
    def setUp(self):
        """Set up simple test case."""
        # Create a simple 3x3 check matrix for testing
        self.H = np.array([
            [1, 1, 0],
            [1, 0, 1], 
            [0, 1, 1]
        ])
        self.A = np.eye(3)  # Simple identity action matrix
        self.p = np.array([0.1, 0.1, 0.1])
        self.decoder = RelayBP(self.H, self.A, self.p)
    
    def test_message_computation_shapes(self):
        """Test that message computations produce correct shapes."""
        # Initialize messages
        nu = np.random.randn(3, 3)  # error-to-check messages
        Lambda = np.random.randn(3)  # bias terms
        syndrome = np.array([1, 0, 1])
        
        # Compute messages
        mu = self.decoder._compute_mu_messages(nu, syndrome, 0)
        nu_new = self.decoder._compute_nu_messages(mu, Lambda)
        marginals = self.decoder._compute_marginals(mu, Lambda)
        
        # Check shapes
        self.assertEqual(mu.shape, (3, 3))
        self.assertEqual(nu_new.shape, (3, 3))
        self.assertEqual(marginals.shape, (3,))
    
    def test_dmem_bp_leg(self):
        """Test single DMem-BP leg execution."""
        syndrome = np.array([1, 0, 1])
        gamma = np.array([0.2, 0.3, 0.1])
        initial_marginals = self.decoder.lambda_j.copy()
        
        converged, error_est, weight, final_marginals, iterations = self.decoder._dmem_bp_leg(
            syndrome, gamma, initial_marginals, max_iterations=10
        )
        
        # Check return types and shapes
        self.assertIsInstance(converged, bool)
        self.assertEqual(error_est.shape, (3,))
        self.assertIsInstance(weight, float)
        self.assertEqual(final_marginals.shape, (3,))
        self.assertIsInstance(iterations, int)
        self.assertLessEqual(iterations, 10)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)

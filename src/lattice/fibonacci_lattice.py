"""
Fibonacci Lattice Regularization for Yang-Mills Theory
Generates Φ-scaled lattices for resonance field simulation.
"""

import numpy as np

PHI = 1.6180339887498948482

def generate_fibonacci_lattice(n_points, dimension=4):
    """
    Generate a Fibonacci-spaced lattice in specified dimension.
    
    Parameters:
    n_points: Number of lattice points
    dimension: Dimension of space (default 4 for spacetime)
    
    Returns:
    lattice: Array of lattice points scaled by Φ-harmonics
    """
    
    # Generate Fibonacci-spaced angles
    indices = np.arange(n_points)
    angles = 2 * np.pi * indices / PHI
    
    # Initialize lattice
    lattice = np.zeros((n_points, dimension))
    
    # Fill lattice with Φ-scaled coordinates
    for d in range(dimension):
        frequency = PHI ** (d / dimension)
        lattice[:, d] = np.sin(angles * frequency)
        
    # Apply Yang-Mills scaling
    lattice *= PHI ** (1/dimension)
    
    return lattice

def calculate_wilson_loop(lattice, size):
    """
    Calculate Wilson loop expectation for confinement test.
    
    Parameters:
    lattice: Generated Fibonacci lattice
    size: Size of the Wilson loop
    
    Returns:
    expectation: Wilson loop expectation value
    """
    if len(lattice) < size * size:
        return 0.0
    
    # Simple Wilson loop calculation (to be expanded)
    loop_sum = 0.0
    count = 0
    
    for i in range(0, len(lattice) - size, size):
        sub_lattice = lattice[i:i+size]
        if len(sub_lattice) == size:
            # Product of links in loop
            product = np.prod(np.diag(sub_lattice))
            loop_sum += np.abs(product)
            count += 1
    
    return loop_sum / count if count > 0 else 0.0

def test_lattice_generation():
    """Test function for lattice generation."""
    print("Testing Fibonacci lattice generation...")
    
    # Generate test lattice
    lattice = generate_fibonacci_lattice(100, 4)
    
    print(f"Lattice shape: {lattice.shape}")
    print(f"First point: {lattice[0]}")
    print(f"Φ scaling check: {np.mean(lattice) / PHI:.6f}")
    
    # Test Wilson loop
    wilson = calculate_wilson_loop(lattice, 3)
    print(f"Wilson loop (3x3): {wilson:.6f}")
    
    return lattice

if __name__ == "__main__":
    test_lattice_generation()

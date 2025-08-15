import numpy as np


def solve_system_high_precision(A, b, n):
    """
    Solves the equation A^n * x = b with higher precision.

    Args:
        A (np.ndarray): The matrix A (dtype=np.float64 for higher precision).
        b (np.ndarray): The vector b (dtype=np.float64 for higher precision).
        n (int): The power to which the matrix A is raised.

    Returns:
        np.ndarray: The solution vector x (dtype=np.float64 for higher precision).
    """
    # Raise the matrix A to the power of n
    A_power_n = np.linalg.matrix_power(A, n)

    # Solve the system A^n * x = b
    x = np.linalg.solve(A_power_n, b)

    return x


# Define a larger matrix A and vector b with higher precision
A = np.array([[10, 2, 3],
              [4, 20, 5],
              [6, 7, 30]], dtype=np.float64)

b = np.array([1, 2, 3], dtype=np.float64)

n = 3  # Test with n = 3 to check higher powers and precision

# Solve A^n * x = b
x = solve_system_high_precision(A, b, n)

# Output the solution with higher precision
print("Solution vector x with higher precision:", x)

# Verification: A^n * x should give b
verification = np.dot(np.linalg.matrix_power(A, n), x)

print("Verification A^n * x:", verification)
print("Original vector b:", b)

# Check if the solution is close to the original vector b
print("Solution matches b:", np.allclose(verification, b, atol=1e-8))

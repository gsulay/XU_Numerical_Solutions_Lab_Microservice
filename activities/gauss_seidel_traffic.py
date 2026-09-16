import numpy as np
import hashlib

def generate_params(student_id: str) -> dict:
    """
    Generates deterministic parameters for a 4x4 traffic network SLE.
    The system must be strictly diagonally dominant to guarantee convergence.
    It is also internally validated to ensure Gauss-Seidel converges faster than Jacobi.
    """
    # 1. Deterministic Seeding
    seed_str = f"gauss_seidel_mmda_{student_id}"
    seed = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
    np.random.seed(seed)
    
    N = 4
    
    # Internal function to simulate the notebook's iteration counts
    def simulate_convergence(A, b):
        # Simulate Jacobi
        x_j = np.zeros(N)
        iters_j = 0
        for _ in range(100):
            iters_j += 1
            x_new = np.zeros(N)
            for i in range(N):
                sigma = sum(A[i, j] * x_j[j] for j in range(N) if i != j)
                x_new[i] = (b[i] - sigma) / A[i, i]
            if np.max(np.abs(x_new - x_j)) < 1e-6:
                break
            x_j = x_new.copy()
            
        # Simulate Gauss-Seidel
        x_g = np.zeros(N)
        iters_g = 0
        for _ in range(100):
            iters_g += 1
            x_old = x_g.copy()
            for i in range(N):
                sigma = sum(A[i, j] * x_g[j] for j in range(N) if i != j)
                x_g[i] = (b[i] - sigma) / A[i, i]
            if np.max(np.abs(x_g - x_old)) < 1e-6:
                break
                
        return iters_j, iters_g

    # 2. Generate Mathematically Sound Matrices
    while True:
        # Off-diagonals represent traffic dispersing (negative coefficients)
        # We include zeros to simulate a sparse interconnected network
        A = np.random.choice([0.0, -1.0, -2.0, -3.0], size=(N, N))
        np.fill_diagonal(A, 0.0) 
        
        # Enforce strict diagonal dominance (positive diagonals representing capacities)
        for i in range(N):
            row_sum = np.sum(np.abs(A[i, :]))
            # Make the diagonal safely larger than the row sum to ensure fast convergence
            A[i, i] = row_sum + np.random.uniform(1.0, 5.0)
            
        # Generate reasonable traffic volumes (vehicles per hour)
        b = np.random.uniform(800, 2500, size=N)
        
        # 3. Verify the notebook's hidden test constraints
        iters_j, iters_g = simulate_convergence(A, b)
        
        # Must converge within 100 iterations AND Gauss-Seidel must be faster
        if iters_j < 100 and iters_g < 100 and iters_g < iters_j:
            break

    # 4. JSON Serialization
    # Using np.round for cleaner JSON representations of traffic flow parameters
    return {
        "A_matrix": np.round(A, 1).tolist(),
        "b_vector": np.round(b, 1).tolist()
    }
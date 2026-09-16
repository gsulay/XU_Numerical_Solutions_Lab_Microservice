import numpy as np
import hashlib

def generate_params(student_id: str) -> dict:
    """
    Generates deterministic parameters for a 5x5 static truss system.
    Requires the resulting A_matrix to be non-singular for Cramer's Rule.
    """
    # 1. Deterministic Seeding based on student_id
    seed_str = f"cramer_truss_{student_id}"
    seed = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
    np.random.seed(seed)
    
    # Typical geometric coefficients in planar trusses (sin/cos of 0, 30, 45, 60, 90 deg)
    geom_coeffs = [0.0, 1.0, -1.0, 0.5, -0.5, 0.707, -0.707, 0.866, -0.866]
    
    # 2. Enforce Mathematical Integrity (Non-Singularity)
    while True:
        # Generate a 5x5 matrix using the geometric coefficients
        A = np.random.choice(geom_coeffs, size=(5, 5))
        
        # Introduce sparsity (truss joint equilibrium equations are typically sparse)
        # Randomly zero out ~40% of the elements
        mask = np.random.choice([True, False], size=(5, 5), p=[0.4, 0.6])
        A[mask] = 0.0
        
        # Check determinant. Must be non-zero (and strictly well-conditioned enough)
        det_A = np.linalg.det(A)
        if abs(det_A) > 0.1:
            break
            
    # 3. Generate Load Vector b (Typical point loads in kN)
    load_values = [0.0, 5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0]
    b = np.random.choice(load_values, size=5)
    
    # 4. JSON Serialization
    return {
        "A_matrix": A.tolist(),
        "b_vector": b.tolist()
    }
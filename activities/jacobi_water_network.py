import numpy as np
import hashlib

def generate_params(student_id: str) -> dict:
    """
    Generates deterministic parameters for a water network SLE.
    Generates a strictly diagonally dominant matrix, then explicitly shuffles 
    the rows to force the student to implement a row-swapping algorithm.
    """
    # 1. Deterministic Seeding
    seed_str = f"jacobi_dpwh_{student_id}"
    seed = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
    np.random.seed(seed)
    
    N = 4  # 4x4 matrix for the DPWH network
    
    # 2. Base Matrix Generation (Enforcing row & column constraints)
    A = np.random.randint(-5, 6, size=(N, N)).astype(float)
    np.fill_diagonal(A, 0) # Clear diagonals first
    
    for i in range(N):
        row_sum = np.sum(np.abs(A[i, :]))
        col_max = np.max(np.abs(A[:, i]))
        
        # The diagonal must be dominant in its row to guarantee Jacobi convergence.
        # It must also be the maximum in its column so the student's `argmax` logic finds it.
        min_required = max(row_sum, col_max)
        
        # Set diagonal to a value safely above the required minimum
        A[i, i] = (min_required + np.random.randint(1, 5)) * np.random.choice([1, -1])
        
    b = np.random.randint(10, 50, size=N).astype(float)
    
    # 3. Explicit Row Shuffling (Derangement)
    # We must shuffle the rows so that diagonal dominance is broken.
    while True:
        idx = np.random.permutation(N)
        A_shuffled = A[idx]
        b_shuffled = b[idx]
        
        # Check if the shuffled matrix is no longer diagonally dominant
        is_dominant = True
        for i in range(N):
            diag = abs(A_shuffled[i, i])
            off_diag = np.sum(np.abs(A_shuffled[i, :])) - diag
            if diag <= off_diag:
                is_dominant = False
                break
                
        # Break the loop if we successfully broke the diagonal dominance
        if not is_dominant:
            break
            
    # 4. JSON Serialization
    return {
        "A_matrix": A_shuffled.tolist(),
        "b_vector": b_shuffled.tolist()
    }
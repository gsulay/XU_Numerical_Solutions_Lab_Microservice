import numpy as np

from activities.ie15_common import (clean_id, make_rng, rho_gauss_seidel, rho_jacobi,
                                    row_orders, strictly_dominant)

NUTRIENTS = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]
MATERIALS = ["Blend X", "Blend Y", "Blend Z"]


def generate_params(student_id: str) -> dict:
    """
    IE 15 LA5 Scenario 2: fertilizer blending, 3 nutrient equations in 3 unknown blend amounts.
    The equations come in N, P, K order, which is never diagonally dominant; exactly one row
    order is. Both methods diverge on the given order.
    """
    student_id = clean_id(student_id)
    rng = make_rng("fertilizer", student_id)
    orders = row_orders(3)

    for _ in range(5000):
        sigma = rng.permutation(3)              # nutrient i is rich in blend sigma[i]
        if np.all(sigma == np.arange(3)):
            continue
        A = np.round(rng.uniform(0.02, 0.15, (3, 3)), 2)
        for i in range(3):
            A[i, sigma[i]] = round(float(rng.uniform(0.28, 0.52)), 2)
        x_true = rng.uniform(200.0, 1200.0, 3)
        b = np.round(A @ x_true, 1)

        dominant = [o for o in orders if strictly_dominant(A[o])]
        if len(dominant) != 1 or strictly_dominant(A):
            continue
        if rho_jacobi(A) < 1.05 or rho_gauss_seidel(A) < 1.05:
            continue
        Af = A[dominant[0]]
        if rho_gauss_seidel(Af) > 0.6:
            continue
        x = np.linalg.solve(A, b)
        if np.any(x < 150):
            continue
        return {
            "student_id": student_id,
            "nutrients": NUTRIENTS,
            "materials": MATERIALS,
            "composition": A.tolist(),
            "targets": b.tolist(),
        }
    raise RuntimeError("Fertilizer dataset generation failed.")

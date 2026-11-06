import numpy as np

from activities.ie15_common import clean_id, make_rng, rho_gauss_seidel, row_orders, strictly_dominant

WAREHOUSES = ["W1 Cebu", "W2 Iligan", "W3 Davao", "W4 Cagayan de Oro"]


def generate_params(student_id: str) -> dict:
    """
    IE 15 LA5 Scenario 3: weekly 4x4 balance for a warehouse network.
    Each student secretly gets one of three cases (never returned):
      A - some row order is strictly diagonally dominant (the given order is not);
      B - no row order is dominant, but Gauss-Seidel on the given order converges (rho < 0.85);
      C - no row order is dominant and Gauss-Seidel diverges for every row order (rho >= 1.05).
    The warehouse with the largest flow is also spread across students.
    """
    student_id = clean_id(student_id)
    rng = make_rng("warehouse", student_id)
    case = int(rng.integers(3))
    target = int(rng.integers(4))
    orders = row_orders(4)

    for _ in range(40000):
        scale = rng.uniform(1.0, 6.0)
        A = rng.uniform(-1.0, 1.0, (4, 4)) * scale
        for i in range(4):
            A[i, i] = abs(A[i, i]) + rng.uniform(0.5, 4.0)
        A = np.round(A, 1)
        if np.any(A == 0):
            continue
        dom = [o for o in orders if strictly_dominant(A[o])]
        if case == 0:
            if not dom:
                continue
            for _ in range(20):               # scramble so the given order is not a dominant one
                A_try = A[rng.permutation(4)]
                if not strictly_dominant(A_try):
                    break
            else:
                continue
            A = A_try
        elif case == 1:
            if dom or rho_gauss_seidel(A) > 0.85:
                continue
        else:
            if dom or min(rho_gauss_seidel(A[o]) for o in orders) < 1.05:
                continue
        if np.linalg.cond(A) > 200:
            continue

        for _ in range(50):
            x_true = rng.uniform(40.0, 300.0, 4)
            x_true[target] = x_true.max() + rng.uniform(20.0, 100.0)
            x_true = np.round(x_true)
            b = np.round(A @ x_true, 1)
            x = np.linalg.solve(A, b)
            if int(np.argmax(x)) == target and np.all(x > 0):
                break
        else:
            continue
        return {
            "student_id": student_id,
            "warehouses": WAREHOUSES,
            "A_balance": A.tolist(),
            "b_balance": b.tolist(),
        }
    raise RuntimeError("Warehouse dataset generation failed.")

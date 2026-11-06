import numpy as np

from activities.ie15_common import clean_id, make_rng, strictly_dominant

DEPARTMENTS = ["Fabrication", "Assembly", "Packaging"]


def _jacobi_iters(M, b, x0, tol=1e-4, max_iter=500):
    D = np.diag(M)
    x = np.array(x0, float)
    for k in range(1, max_iter + 1):
        x_new = (b - (M @ x - D * x)) / D
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k
        x = x_new
    raise RuntimeError("Jacobi did not converge.")


def generate_params(student_id: str) -> dict:
    """
    IE 15 LA5 Scenario 1: a three-department Leontief input-output plant.
    (I - A) is strictly diagonally dominant for every student. The department with the
    largest indirect requirement (x - d) is spread evenly across students and is never returned.
    """
    student_id = clean_id(student_id)
    rng = make_rng("leontief", student_id)
    target = int(rng.integers(3))

    for _ in range(5000):
        A = np.round(rng.uniform(0.04, 0.32, (3, 3)), 2)
        rows = A.sum(axis=1)
        if np.any(rows < 0.40) or np.any(rows > 0.85):
            continue
        d = np.round(rng.uniform(20.0, 80.0, 3), 1)
        d_next = np.round(d * (1 + rng.uniform(-0.06, 0.06, 3)), 1)
        M = np.eye(3) - A
        if not strictly_dominant(M):
            continue
        x = np.linalg.solve(M, d)
        indirect = x - d
        order = np.sort(indirect)
        if int(np.argmax(indirect)) != target or order[-1] - order[-2] < 1.0:
            continue
        x_cold, _ = _jacobi_iters(M, d, np.zeros(3))
        _, n_next_cold = _jacobi_iters(M, d_next, np.zeros(3))
        _, n_next_warm = _jacobi_iters(M, d_next, x_cold)
        if n_next_warm >= n_next_cold - 2:
            continue
        return {
            "student_id": student_id,
            "departments": DEPARTMENTS,
            "tech_coeff": A.tolist(),
            "final_demand": d.tolist(),
            "final_demand_next": d_next.tolist(),
        }
    raise RuntimeError("Leontief dataset generation failed.")

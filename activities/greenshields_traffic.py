import hashlib

import numpy as np


def _clean_id(student_id: str) -> str:
    student_id = student_id.strip()
    if not student_id:
        raise ValueError("Student ID cannot be empty.")
    if len(student_id) > 64:
        raise ValueError("Student ID is too long.")
    return student_id


def _fit_line(x, y):
    n = len(x)
    a1 = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
    return np.mean(y) - a1 * np.mean(x), a1


def generate_params(student_id: str) -> dict:
    """
    Generates deterministic speed-density readings for the LA4 linear regression scenario.
    The query density falls in one of three cases, which is never returned:
    inside the data, beyond the data but before jam density, or beyond jam density.
    """
    student_id = _clean_id(student_id)
    seed = int(hashlib.sha256(f"CE17-LA4-traffic-{student_id}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    case = int(rng.integers(3))

    n = 12
    for _ in range(2000):
        vf = float(rng.uniform(60.0, 100.0))
        kj = float(rng.uniform(100.0, 160.0))
        k = np.round(np.sort(rng.uniform(0.08 * kj, 0.70 * kj, n)), 1)
        v = np.round(vf * (1 - k / kj) + rng.normal(0, 2.5, n), 1)
        if np.any(v <= 0):
            continue

        a0, a1 = _fit_line(k, v)
        if a1 >= 0:
            continue
        r2 = 1 - np.sum((v - (a0 + a1 * k)) ** 2) / np.sum((v - v.mean()) ** 2)
        if r2 < 0.90:
            continue

        jam = -a0 / a1
        if case == 0:
            query = float(rng.uniform(k[2], k[-3]))
        elif case == 1:
            if 1.08 * k[-1] >= 0.92 * jam:
                continue
            query = float(rng.uniform(1.08 * k[-1], 0.92 * jam))
        else:
            query = float(rng.uniform(1.10 * jam, 1.30 * jam))
        query = round(query, 1)

        return {
            "student_id": student_id,
            "density": k.tolist(),
            "speed": v.tolist(),
            "query_density": query,
        }
    raise RuntimeError("Traffic dataset generation failed.")

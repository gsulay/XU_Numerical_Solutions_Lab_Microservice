import hashlib

import numpy as np

from activities.greenshields_traffic import _clean_id


def _fit_poly(x, y, m):
    A = np.array([[np.sum(x ** (i + j)) for j in range(m + 1)] for i in range(m + 1)])
    b = np.array([np.sum((x ** i) * y) for i in range(m + 1)])
    return np.linalg.solve(A, b)


def _peval(a, x):
    return sum(a[k] * x ** k for k in range(len(a)))


def generate_params(student_id: str) -> dict:
    """
    Generates deterministic gauge readings for the LA4 polynomial regression scenario.
    A simply supported beam under a uniform load is measured on 5%-65% of its span
    and the moment is needed at 85%, so high-degree fits extrapolate badly.
    """
    student_id = _clean_id(student_id)
    seed = int(hashlib.sha256(f"CE17-LA4-beam-{student_id}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)

    n = 10
    for _ in range(5000):
        L = round(float(rng.uniform(8.0, 14.0)), 1)
        w = round(float(rng.uniform(15.0, 40.0)), 1)
        x = np.linspace(0.05 * L, 0.65 * L, n) + rng.uniform(-0.01, 0.01, n) * L
        x = np.round(np.sort(x), 2)
        M = np.round(w * x * (L - x) / 2 + rng.normal(0, 0.03 * w * L ** 2 / 8, n), 2)
        x_pred = round(0.85 * L, 2)
        theory = w * x_pred * (L - x_pred) / 2

        errors, r2s = {}, []
        for m in range(1, 7):
            a = _fit_poly(x, M, m)
            errors[m] = abs(_peval(a, x_pred) - theory) / theory
            r2s.append(1 - np.sum((M - _peval(a, x)) ** 2) / np.sum((M - M.mean()) ** 2))

        if errors[2] >= 0.08 or errors[6] <= 0.25 or errors[6] <= 4 * errors[2]:
            continue
        if any(r2s[i + 1] < r2s[i] for i in range(5)):
            continue
        if any(0.09 <= e <= 0.11 for e in errors.values()):
            continue

        return {
            "student_id": student_id,
            "span": L,
            "w": w,
            "position": x.tolist(),
            "moment": M.tolist(),
            "x_predict": x_pred,
        }
    raise RuntimeError("Beam dataset generation failed.")

import hashlib

import numpy as np

from activities.greenshields_traffic import _clean_id

# Learning rates listed in the notebook's raw-feature experiment.
LEARNING_RATES = (2e-7, 5e-7, 1e-6, 2e-6)
AGES = np.array([3, 7, 14, 21, 28, 56], float)


def generate_params(student_id: str) -> dict:
    """
    Generates deterministic column test data for the LA4 gradient descent scenario.
    Rebar area is thousands of times larger than age, so unscaled gradient descent
    diverges at ordinary learning rates. The stable limit varies per student.
    """
    student_id = _clean_id(student_id)
    seed = int(hashlib.sha256(f"CE17-LA4-concrete-{student_id}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)

    n = 24
    options = np.array(LEARNING_RATES)
    for _ in range(5000):
        rebar_max = float(np.exp(rng.uniform(np.log(800.0), np.log(3000.0))))
        age = rng.choice(AGES, n)
        rebar = np.round(rng.uniform(0.25 * rebar_max, rebar_max, n), -1)
        capacity = np.round(
            rng.uniform(400, 900) + rng.uniform(2, 8) * age + rng.uniform(0.275, 0.415) * rebar
            + rng.normal(0, 25, n), 1
        )

        if abs(np.corrcoef(age, rebar)[0, 1]) >= 0.5:
            continue
        # The new column's age must stay inside the tested range.
        if age.min() > 10 or age.max() < 45:
            continue

        X = np.column_stack([np.ones(n), age, rebar])
        limit = 1.0 / np.max(np.linalg.eigvalsh(X.T @ X / n))
        if limit <= options[0] or np.any(np.abs(limit - options) / options < 0.15):
            continue

        Z = (X[:, 1:] - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0)
        Z1 = np.column_stack([np.ones(n), Z])
        b = np.linalg.solve(Z1.T @ Z1, Z1.T @ capacity)
        strengths = np.abs(b[1:])
        if max(strengths) / min(strengths) <= 1.10:
            continue

        return {
            "student_id": student_id,
            "age": age.tolist(),
            "rebar_area": rebar.tolist(),
            "capacity": capacity.tolist(),
            "new_column": {
                "age": int(rng.choice(np.array([10, 18, 35, 45]))),
                "rebar_area": float(np.round(rng.uniform(0.4 * rebar_max, 0.9 * rebar_max), -1)),
            },
        }
    raise RuntimeError("Concrete dataset generation failed.")

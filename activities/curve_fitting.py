"""
CE 17 · Lab 04 (Curve Fitting): per-group dataset generator.

Backs the FastAPI route that returns exactly the JSON body the notebook expects from

    GET {API_BASE_URL}/api/v1/curvefit/datasets/{group_id}

Every dataset is deterministic per group_id (seeded with hashlib, never the
built-in hash(), which changes every time the server restarts), and is
rejection-sampled so the edge cases the notebook relies on are guaranteed.
"""
import hashlib
import numpy as np


# ---------------------------------------------------------------- helpers
def _seed(group_id: str) -> int:
    return int(hashlib.sha256(f"CE17-LAB04-{group_id}".encode()).hexdigest()[:16], 16)


def _fit_poly_ne(x, y, m):
    A = np.array([[np.sum(x ** (i + j)) for j in range(m + 1)] for i in range(m + 1)])
    b = np.array([np.sum((x ** i) * y) for i in range(m + 1)])
    return np.linalg.solve(A, b)


def _peval(a, x):
    return sum(a[k] * x ** k for k in range(len(a)))


def _fit_line(x, y):
    n = len(x)
    a1 = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
    return np.mean(y) - a1 * np.mean(x), a1


def _flag(k, v, a0, a1, c=2.5):
    e = v - (a0 + a1 * k)
    s = np.sqrt(np.sum(e ** 2) / (len(k) - 2))
    return np.abs(e) > c * s


def _qmax(a0, a1):
    return -a0 ** 2 / (4 * a1)


# ---------------------------------------------------------------- datasets
def _beam(rng):
    """Simply supported girder, UDL proof load, gauges on 5%-65% of span only."""
    for _ in range(2000):
        L = round(float(rng.uniform(8.0, 14.0)), 1)          # m
        w = round(float(rng.uniform(15.0, 40.0)), 1)         # kN/m
        x = np.linspace(0.05 * L, 0.65 * L, 10) + rng.uniform(-0.01, 0.01, 10) * L
        x = np.round(np.sort(x), 2)
        M_true = w * x * (L - x) / 2
        M = np.round(M_true + rng.normal(0, 0.03 * w * L ** 2 / 8, 10), 2)
        x_pred = round(0.85 * L, 2)
        th = w * x_pred * (L - x_pred) / 2
        p2 = _peval(_fit_poly_ne(x, M, 2), x_pred)
        p6 = _peval(_fit_poly_ne(x, M, 6), x_pred)
        # Guarantee: degree 2 is accurate, degree 6 (best R2) extrapolates badly
        if abs(p2 - th) < 0.08 * th and abs(p6 - th) > 4 * abs(p2 - th) and abs(p6 - th) > 0.25 * th:
            return {"span": L, "w": w, "x": x.tolist(), "M": M.tolist(), "x_predict": x_pred}
    raise RuntimeError("beam generator failed")


def _traffic(rng):
    """
    Loop-detector speed-density data with 2-4 faulty readings. The fault type varies by group:
      "stuck_zero"  : detector reports v = 0 in free-flowing traffic (low density)
                      -> the raw fit UNDERestimates capacity
      "frozen_high" : detector keeps repeating a free-flow speed in heavy traffic (high density)
                      -> the raw fit OVERestimates capacity
    The fault type is NOT returned; students must discover it from the data.
    """
    fault = "stuck_zero" if rng.random() < 0.5 else "frozen_high"
    for _ in range(4000):
        vf = round(float(rng.uniform(60.0, 100.0)), 1)       # km/h
        kj = round(float(rng.uniform(100.0, 160.0)), 1)      # veh/km
        kg = rng.uniform(5.0, 0.9 * kj, 36)
        vg = np.clip(vf * (1 - kg / kj) + rng.normal(0, 3.0, 36), 1.0, None)
        nb = int(rng.integers(2, 5))
        if fault == "stuck_zero":
            kb = rng.uniform(5.0, 0.35 * kj, nb)
            vb = np.zeros(nb)
        else:
            kb = rng.uniform(0.6 * kj, 0.9 * kj, nb)
            vb = np.full(nb, round(float(rng.uniform(0.9, 1.0)) * vf, 1))
        k = np.round(np.concatenate([kg, kb]), 1)
        v = np.round(np.concatenate([vg, vb]), 1)
        bad = np.concatenate([np.zeros(36, bool), np.ones(nb, bool)])
        order = np.argsort(k, kind="stable")
        k, v, bad = k[order], v[order], bad[order]
        a0r, a1r = _fit_line(k, v)
        if a1r >= 0:
            continue
        mask = _flag(k, v, a0r, a1r)
        a0c, a1c = _fit_line(k[~mask], v[~mask])
        shift = abs(_qmax(a0r, a1r) - _qmax(a0c, a1c)) / _qmax(a0c, a1c)
        # Guarantee: the 2.5*s rule recovers exactly the faulty readings, and they matter
        if np.array_equal(mask, bad) and shift > 0.03:
            return {"k": k.tolist(), "v": v.tolist()}
    raise RuntimeError("traffic generator failed")


def _ffs(rng):
    """Free-flow speed vs lane width (m) and access-point density (points/km)."""
    n = 30
    options = np.array([0.0020, 0.0030, 0.0035, 0.0040])   # learning rates listed in checkpoint cp3_3
    for _ in range(2000):
        lw = np.round(rng.uniform(2.8, 3.7, n), 2)
        ad = np.round(rng.uniform(0.0, 30.0, n), 1)
        b0, b1, b2 = rng.uniform(45, 55), rng.uniform(8, 12), rng.uniform(-0.6, -0.3)
        ffs = np.round(b0 + b1 * lw + b2 * ad + rng.normal(0, 1.5, n), 1)
        X1 = np.column_stack([np.ones(n), lw, ad])
        a_th = 1.0 / np.max(np.linalg.eigvalsh(X1.T @ X1 / n))
        # Guarantee: cp3_3 has an unambiguous answer (stability limit not within 2% of a listed rate)
        if a_th > 0.00205 and np.all(np.abs(a_th - options) / options > 0.02):
            break
    else:
        raise RuntimeError("ffs generator failed")
    new = {"lane_width": round(float(rng.uniform(2.9, 3.6)), 2),
           "access_density": round(float(rng.uniform(5.0, 28.0)), 1)}
    return {"lane_width": lw.tolist(), "access_density": ad.tolist(),
            "ffs": ffs.tolist(), "new_segment": new}


def generate_params(group_id: str) -> dict:
    group_id = group_id.strip()
    if not group_id:
        raise ValueError("Group ID cannot be empty.")
    if len(group_id) > 64:
        raise ValueError("Group ID is too long.")

    rng = np.random.default_rng(_seed(group_id))
    return {"group_id": group_id, "beam": _beam(rng), "traffic": _traffic(rng), "ffs": _ffs(rng)}

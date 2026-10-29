"""
CE 17 Lab 04 (Curve Fitting): per-group datasets for one bridge-and-approach project.

    GET /api/v1/curvefit/datasets/{group_id}

The three parts of the lab are chained:
  hydro      -> rating curve Q = C (h - h0)^beta, fitted in log space; the design flood
                level sets the approach-embankment height
  proctor    -> compaction curve of the borrow soil (quadratic), with readings above the
                zero-air-voids line; feeds the earthworks water quantity
  concrete   -> strength model f'c = a0 + a1 w/c + a2 ln(age) + a3 cement, trained with
                gradient descent; collinearity between cement and w/c varies by group
  schedule   -> curing age needed before the deck can be launched

Every dataset is deterministic per group_id (seeded with hashlib and a local Generator,
never global NumPy state) and rejection-sampled so the edge cases the notebook relies
on hold on the rounded values actually returned. Nothing that students are meant to
discover (which h0 is right, which readings are faulty, the collinearity level) is
returned.
"""
import hashlib

import numpy as np

AGES = (3, 7, 14, 28, 56)            # days, standard cylinder test ages
GD_LR, GD_TOL, GD_CAP = 0.1, 1e-3, 1500
GD_BUCKETS = (200, 600, 1500)        # iteration bounds of the cp3_3 options
WATER_SD = (30.0, 12.0, 5.0, 1.0)    # kg/m3 scatter of mixing water -> collinearity level


# ---------------------------------------------------------------- helpers
def _seed(group_id: str) -> int:
    return int(hashlib.sha256(f"CE17-LAB04-{group_id}".encode()).hexdigest()[:16], 16)


def _fit_line(x, y):
    n = len(x)
    a1 = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
    return np.mean(y) - a1 * np.mean(x), a1


def _fit_rating(h, Q, h0):
    lnC, beta = _fit_line(np.log(h - h0), np.log(Q))
    return np.exp(lnC), beta


def _rating_sr(h, Q, h0):
    C, beta = _fit_rating(h, Q, h0)
    return np.sum((Q - C * (h - h0) ** beta) ** 2)


def _zav(w_pct, Gs):
    return Gs / (1 + w_pct / 100 * Gs)          # Mg/m3, rho_w = 1


def _fit_quad(x, y):
    A = np.array([[np.sum(x ** (i + j)) for j in range(3)] for i in range(3)])
    b = np.array([np.sum(x ** i * y) for i in range(3)])
    return np.linalg.solve(A, b)


def _design_matrix(wc, age, cem):
    return np.column_stack([wc, np.log(age), cem])


def _gd_iters(X, y, cap):
    """
    Iterations the lab's (correct) trainer needs on standardised features, i.e. the first k
    (1-based) with sum|grad_k| < GD_TOL. Starting from b = 0, gradient descent on the MSE
    has the closed form grad_k = H (I - lr H)^k (0 - b*), H = (2/n) Z1^T Z1, so all
    iterations are evaluated at once through the eigen-decomposition of H.
    """
    Z = (X - X.mean(0)) / X.std(0)
    Z1 = np.column_stack([np.ones(len(y)), Z])
    H = (2 / len(y)) * Z1.T @ Z1
    b_star = np.linalg.solve(Z1.T @ Z1, Z1.T @ y)
    lam, V = np.linalg.eigh(H)
    c = V.T @ (-b_star)
    k = np.arange(cap)
    G = V @ ((lam * c)[:, None] * (1 - GD_LR * lam)[:, None] ** k[None, :])
    hit = np.flatnonzero(np.sum(np.abs(G), axis=0) < GD_TOL)
    return int(hit[0]) + 1 if hit.size else cap + 1


# ---------------------------------------------------------------- datasets
def _hydro(rng):
    """Gaugings at the bridge site; three candidate cease-to-flow stages h0."""
    for _ in range(2000):
        h0 = rng.uniform(0.4, 1.0)
        idx = int(rng.integers(0, 3))                    # which candidate is the true one
        cands = np.round(h0 + 0.25 * (np.arange(3) - idx), 2)
        C, beta = rng.uniform(8.0, 25.0), rng.uniform(1.5, 2.2)
        depth = np.sort(0.75 + 3.75 * rng.uniform(0, 1, 12) ** 1.5)
        h = np.round(h0 + depth, 2)
        Q = np.round(C * depth ** beta * np.exp(rng.normal(0, 0.05, 12)), 2)
        Q_design = float(np.round(C * depth[-1] ** beta * rng.uniform(1.4, 2.0), -1))
        datum = round(float(rng.uniform(20.0, 60.0)), 2)
        freeboard = 1.5
        h_true = h0 + (Q_design / C) ** (1 / beta)
        ground = round(float(datum + h_true + freeboard - rng.uniform(3.0, 5.0)), 2)

        if h.min() - cands.max() < 0.15 or Q_design < 1.2 * Q.max():
            continue
        sr = np.array([_rating_sr(h, Q, c) for c in cands])
        order = np.argsort(sr)
        if order[0] != idx or sr[order[0]] > 0.8 * sr[order[1]]:
            continue
        Cf, bf = _fit_rating(h, Q, cands[idx])
        H_emb = datum + cands[idx] + (Q_design / Cf) ** (1 / bf) + freeboard - ground
        if not 2.0 <= H_emb <= 7.0:
            continue
        return {"h": h.tolist(), "Q": Q.tolist(), "h0_candidates": cands.tolist(),
                "Q_design": Q_design, "gauge_datum_elev": datum,
                "ground_elev": ground, "freeboard": freeboard}
    raise RuntimeError("hydro generator failed")


def _proctor(rng):
    """Standard Proctor test of the borrow soil; 0-2 readings plot above the ZAV line."""
    n_bad = int(rng.integers(0, 3))
    for _ in range(4000):
        Gs = round(float(rng.uniform(2.60, 2.75)), 2)
        omc = rng.uniform(12.0, 18.0)
        S_opt = rng.uniform(0.80, 0.88)
        mdd = Gs / (1 + omc / 100 * Gs / S_opt)
        k = rng.uniform(0.004, 0.007)
        t = int(rng.integers(1, 5))                      # intended points wet of optimum
        w = omc - 1.5 * (7 - t) + rng.uniform(0.3, 1.2) + 1.5 * np.arange(7) + rng.uniform(-0.2, 0.2, 7)
        w = np.round(w, 1)
        rho = mdd - k * (w - omc) ** 2 + rng.normal(0, 0.006, 7)
        bad = np.zeros(7, bool)
        if n_bad:
            pick = rng.choice(np.flatnonzero(w > omc - 3.5), size=n_bad, replace=False) \
                if np.sum(w > omc - 3.5) >= n_bad else None
            if pick is None:
                continue
            bad[pick] = True
            rho[pick] = _zav(w[pick], Gs) * rng.uniform(1.02, 1.05, n_bad)
        rho = np.round(rho, 3)

        zav = _zav(w, Gs)
        if np.any(zav[~bad] - rho[~bad] < 0.015) or np.any(rho[bad] - zav[bad] < 0.015):
            continue
        wv, rv = w[~bad], rho[~bad]
        a = _fit_quad(wv, rv)
        if a[2] >= 0:
            continue
        omc_f = -a[1] / (2 * a[2])
        n_wet = int(np.sum(wv > omc_f))
        n_dry = int(np.sum(wv < omc_f))
        if not (1 <= n_wet <= 4 and n_dry >= 2) or np.min(np.abs(wv - omc_f)) < 0.3:
            continue
        if n_bad:
            ar = _fit_quad(w, rho)
            if ar[2] < 0:
                omc_r = -ar[1] / (2 * ar[2])
                shift = abs(omc_r - omc_f) >= 0.5 or \
                    abs((ar[0] - ar[1] ** 2 / (4 * ar[2])) - (a[0] - a[1] ** 2 / (4 * a[2]))) >= 0.01
                if not shift:
                    continue
        w_borrow = round(float(omc_f - rng.uniform(2.0, 6.0)), 1)
        return {"Gs": Gs, "w": w.tolist(), "rho_d": rho.tolist()}, w_borrow
    raise RuntimeError("proctor generator failed")


def _earthworks(rng, w_borrow):
    return {"length": round(float(rng.uniform(80.0, 160.0)), 1),
            "crest_width": round(float(rng.uniform(8.0, 12.0)), 1),
            "side_slope": 2.0,
            "w_borrow": w_borrow,
            "relative_compaction": 0.95,
            "tank_volume": 10.0}


def _concrete(rng):
    """Batching-plant cylinder records and the deck-launch schedule."""
    level = int(rng.integers(0, 4))                      # collinearity level -> cp3_3 bucket
    lo = (0,) + GD_BUCKETS[:-1]
    for _ in range(2000):
        n = 48
        wc = np.round(rng.uniform(0.38, 0.62, n), 2)
        age = rng.choice(AGES, n).astype(float)
        W0 = rng.uniform(175.0, 195.0)
        cem = np.round((W0 + rng.normal(0, WATER_SD[level], n)) / wc)
        A, B, Cg, D = rng.uniform(36, 44), rng.uniform(50, 70), rng.uniform(5, 7), rng.uniform(0.5, 2.0)
        fc = np.round(A - B * wc + Cg * np.log(age) + D * (cem - 350) / 100 + rng.normal(0, 1.5, n), 1)

        X = _design_matrix(wc, age, cem)
        it = _gd_iters(X, fc, 3000)
        # 15 % margin from every bucket edge, plus 2 iterations for round-off between the
        # closed form above and a student's iterative loop
        if level < 3:
            ok = lo[level] * 1.15 + 2 < it < GD_BUCKETS[level] * 0.85 - 2
        else:
            ok = it > GD_BUCKETS[-1] * 1.2 + 2
        if not ok:
            continue
        a = np.linalg.lstsq(np.column_stack([np.ones(n), X]), fc, rcond=None)[0]
        if a[1] >= 0 or a[2] <= 0:
            continue

        wc_mix = round(float(rng.uniform(0.42, 0.50)), 2)
        cem_mix = float(round(W0 / wc_mix))
        f_req = round(float(a[0] + a[1] * wc_mix + a[2] * np.log(rng.uniform(7, 40)) + a[3] * cem_mix), 1)
        t_req = np.exp((f_req - a[0] - a[1] * wc_mix - a[3] * cem_mix) / a[2])
        if not 5.0 <= t_req <= 50.0:
            continue
        enough = rng.random() < 0.5
        days = int(round(t_req * (rng.uniform(1.2, 1.6) if enough else rng.uniform(0.55, 0.85))))
        if abs(days - t_req) / t_req < 0.15:
            continue
        return ({"wc": wc.tolist(), "age": age.tolist(), "cement": cem.tolist(), "fc": fc.tolist()},
                {"f_req": f_req, "wc_mix": wc_mix, "cement_mix": cem_mix, "days_available": days})
    raise RuntimeError("concrete generator failed")


def generate_params(group_id: str) -> dict:
    group_id = group_id.strip()
    if not group_id:
        raise ValueError("Group ID cannot be empty.")
    if len(group_id) > 64:
        raise ValueError("Group ID is too long.")
    rng = np.random.default_rng(_seed(group_id))
    hydro = _hydro(rng)
    proctor, w_borrow = _proctor(rng)
    earthworks = _earthworks(rng, w_borrow)
    concrete, schedule = _concrete(rng)
    return {"group_id": group_id, "hydro": hydro, "proctor": proctor,
            "earthworks": earthworks, "concrete": concrete, "schedule": schedule}

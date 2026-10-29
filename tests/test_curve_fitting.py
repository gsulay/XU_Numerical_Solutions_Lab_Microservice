"""
Tests for the Lab 04 (Curve Fitting) dataset endpoint.

The invariant checks are re-implemented here on purpose (no private helpers are imported
from the module), so a bug in a generator helper cannot hide itself.
"""
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from activities.curve_fitting import generate_params
from main import app

client = TestClient(app)
IDS = [f"{i:06d}" for i in range(1000)]


# ---------------------------------------------------------------- independent references
def ref_line(x, y):
    a1, a0 = np.polyfit(x, y, 1)
    return a0, a1


def ref_rating(h, Q, h0):
    lnC, beta = ref_line(np.log(h - h0), np.log(Q))
    return np.exp(lnC), beta


def ref_sr(h, Q, h0):
    C, beta = ref_rating(h, Q, h0)
    return float(np.sum((Q - C * (h - h0) ** beta) ** 2))


def ref_zav(w_pct, Gs):
    return Gs / (1 + w_pct / 100 * Gs)


def ref_quad(x, y):
    return np.polyfit(x, y, 2)[::-1]           # ascending [a0, a1, a2]


def ref_gd_iters(X, y, lr=0.1, tol=1e-3, cap=3000):
    """Plain iterative gradient descent, exactly as the notebook's correct trainer."""
    Z = (X - X.mean(0)) / X.std(0)
    Z1 = np.column_stack([np.ones(len(y)), Z])
    b = np.zeros(4)
    for i in range(cap):
        g = (2 / len(y)) * Z1.T @ (Z1 @ b - y)
        if np.sum(np.abs(g)) < tol:
            return i + 1
        b = b - lr * g
    return cap + 1


def answers(p):
    """Every data-dependent quantity the notebook asks for, from the raw response."""
    out = {}
    hy = p["hydro"]
    h, Q, cands = np.array(hy["h"]), np.array(hy["Q"]), hy["h0_candidates"]
    sr = [ref_sr(h, Q, c) for c in cands]
    j = int(np.argmin(sr))
    out["sr"], out["best"] = sorted(sr), j
    C, beta = ref_rating(h, Q, cands[j])
    out["H_emb"] = hy["gauge_datum_elev"] + cands[j] + (hy["Q_design"] / C) ** (1 / beta) \
        + hy["freeboard"] - hy["ground_elev"]

    pr = p["proctor"]
    w, r = np.array(pr["w"]), np.array(pr["rho_d"])
    zav = ref_zav(w, pr["Gs"])
    bad = r > zav
    a = ref_quad(w[~bad], r[~bad])
    omc = -a[1] / (2 * a[2])
    out.update(zav_gap=zav - r, bad=bad, a2=a[2], omc=omc,
               n_bad=int(bad.sum()), n_wet=int(np.sum(w[~bad] > omc)), n_dry=int(np.sum(w[~bad] < omc)),
               min_gap_omc=float(np.min(np.abs(w[~bad] - omc))))
    if bad.any():
        ar = ref_quad(w, r)
        omc_r = -ar[1] / (2 * ar[2])
        mdd = a[0] + a[1] * omc + a[2] * omc ** 2
        mdd_r = ar[0] + ar[1] * omc_r + ar[2] * omc_r ** 2
        out["raw_shift"] = (abs(omc_r - omc), abs(mdd_r - mdd), ar[2])

    co, s = p["concrete"], p["schedule"]
    X = np.column_stack([co["wc"], np.log(co["age"]), co["cement"]])
    y = np.array(co["fc"])
    A = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
    out["a_normal"] = A
    out["iters"] = ref_gd_iters(X, y)
    out["t_req"] = float(np.exp((s["f_req"] - A[0] - A[1] * s["wc_mix"] - A[3] * s["cement_mix"]) / A[2]))
    return out


def bucket(iters):
    if iters > 1500:
        return "D"
    return "A" if iters < 200 else ("B" if iters < 600 else "C")


@pytest.fixture(scope="module")
def all_params():
    return {gid: generate_params(gid) for gid in IDS}


@pytest.fixture(scope="module")
def all_answers(all_params):
    return {gid: answers(p) for gid, p in all_params.items()}


# ---------------------------------------------------------------- route and status codes
def test_route_ok():
    r = client.get("/api/v1/curvefit/datasets/012983")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"group_id", "hydro", "proctor", "earthworks", "concrete", "schedule"}
    assert body["group_id"] == "012983"


def test_blank_id_rejected():
    assert client.get("/api/v1/curvefit/datasets/%20").status_code == 400


def test_long_id_rejected():
    assert client.get("/api/v1/curvefit/datasets/" + "x" * 65).status_code == 400


def test_id_is_stripped():
    assert generate_params("  012983 ") == generate_params("012983")


# ---------------------------------------------------------------- determinism
def test_repeat_calls_identical():
    a = client.get("/api/v1/curvefit/datasets/012983").content
    b = client.get("/api/v1/curvefit/datasets/012983").content
    assert a == b


def test_threads_match_serial():
    ids = [f"{i:06d}" for i in range(0, 400, 7)]
    serial = [json.dumps(generate_params(g), sort_keys=True) for g in ids]
    with ThreadPoolExecutor(max_workers=8) as ex:
        threaded = list(ex.map(lambda g: json.dumps(generate_params(g), sort_keys=True), ids))
    assert threaded == serial


def test_different_groups_differ():
    assert generate_params("000001")["hydro"]["Q"] != generate_params("000002")["hydro"]["Q"]


# ---------------------------------------------------------------- schema, types, ranges
def _floats(xs):
    return all(type(v) is float for v in xs)


def test_schema(all_params):
    for gid, p in list(all_params.items())[:200]:
        hy, pr, ew, co, s = p["hydro"], p["proctor"], p["earthworks"], p["concrete"], p["schedule"]
        assert len(hy["h"]) == len(hy["Q"]) == 12 and _floats(hy["h"]) and _floats(hy["Q"])
        assert hy["h"] == sorted(hy["h"]) and min(hy["Q"]) > 0
        assert len(hy["h0_candidates"]) == 3 and hy["h0_candidates"] == sorted(hy["h0_candidates"])
        assert _floats([hy["Q_design"], hy["gauge_datum_elev"], hy["ground_elev"], hy["freeboard"]])

        assert len(pr["w"]) == len(pr["rho_d"]) == 7 and _floats(pr["w"]) and _floats(pr["rho_d"])
        assert pr["w"] == sorted(pr["w"]) and 2.6 <= pr["Gs"] <= 2.75 and min(pr["w"]) > 0

        assert set(ew) == {"length", "crest_width", "side_slope", "w_borrow", "relative_compaction", "tank_volume"}
        assert _floats(ew.values()) and 80 <= ew["length"] <= 160 and ew["w_borrow"] > 0

        assert all(len(co[k]) == 48 and _floats(co[k]) for k in ("wc", "age", "cement", "fc"))
        assert set(co["age"]) <= {3.0, 7.0, 14.0, 28.0, 56.0}
        assert 0.38 <= min(co["wc"]) and max(co["wc"]) <= 0.62

        assert _floats([s["f_req"], s["wc_mix"], s["cement_mix"]]) and type(s["days_available"]) is int


def test_no_secrets_returned():
    text = json.dumps(generate_params("012983")).lower()
    for word in ("fault", "bad", "true", "level", "idx", "h0_true"):
        assert word not in text


# ---------------------------------------------------------------- dataset invariants (1000 IDs)
def test_hydro_invariants(all_params, all_answers):
    for gid in IDS:
        hy, a = all_params[gid]["hydro"], all_answers[gid]
        assert min(hy["h"]) - max(hy["h0_candidates"]) >= 0.15 - 1e-9, gid          # H1
        assert a["sr"][0] <= 0.8 * a["sr"][1], gid                                  # H2
        assert hy["Q_design"] >= 1.2 * max(hy["Q"]), gid                            # H3
        assert 2.0 <= a["H_emb"] <= 7.0, gid                                        # H4


def test_proctor_invariants(all_params, all_answers):
    for gid in IDS:
        a, ew = all_answers[gid], all_params[gid]["earthworks"]
        assert np.all(a["zav_gap"][~a["bad"]] >= 0.015 - 1e-9), gid                 # P1
        assert np.all(-a["zav_gap"][a["bad"]] >= 0.015 - 1e-9), gid
        assert a["a2"] < 0 and 1 <= a["n_wet"] <= 4 and a["n_dry"] >= 2, gid        # P2
        assert a["min_gap_omc"] >= 0.3 - 1e-6, gid
        if a["n_bad"]:                                                              # P3
            d_omc, d_mdd, a2r = a["raw_shift"]
            assert a2r >= 0 or d_omc >= 0.5 - 1e-6 or d_mdd >= 0.01 - 1e-6, gid
        assert ew["w_borrow"] <= a["omc"] - 2.0 + 0.05, gid                         # P4


def test_concrete_invariants(all_params, all_answers):
    for gid in IDS:
        a, s = all_answers[gid], all_params[gid]["schedule"]
        it = a["iters"]                                                             # C1
        assert (it < 170) or (230 < it < 510) or (690 < it < 1275) or (it > 1800), (gid, it)
        assert a["a_normal"][1] < 0 and a["a_normal"][2] > 0, gid                   # C2
        assert 5.0 <= a["t_req"] <= 50.0, gid                                       # C3
        assert abs(s["days_available"] - a["t_req"]) >= 0.15 * a["t_req"] - 0.5, gid


# ---------------------------------------------------------------- answer spread
def test_answer_spread(all_params, all_answers):
    n = len(IDS)
    cp12 = Counter("ABC"[all_answers[g]["best"]] for g in IDS)
    cp21 = Counter(all_answers[g]["n_bad"] for g in IDS)
    cp22 = Counter(all_answers[g]["n_wet"] for g in IDS)
    cp33 = Counter(bucket(all_answers[g]["iters"]) for g in IDS)
    cp34 = Counter("A" if all_answers[g]["t_req"] <= all_params[g]["schedule"]["days_available"] else "B"
                   for g in IDS)
    for name, counts, options in (("cp1_2", cp12, "ABC"), ("cp2_1", cp21, (0, 1, 2)),
                                  ("cp2_2", cp22, (1, 2, 3, 4)), ("cp3_3", cp33, "ABCD"),
                                  ("cp3_4", cp34, "AB")):
        for opt in options:
            assert counts[opt] >= 0.10 * n, f"{name} option {opt!r} too rare: {dict(counts)}"


# ---------------------------------------------------------------- performance
def test_runtime():
    t0 = time.perf_counter()
    for i in range(1000):
        generate_params(f"p{i:05d}")
    assert time.perf_counter() - t0 < 5.0

"""
Tests for the Lab 04 (Curve Fitting) dataset endpoint.

The invariant checks are re-implemented here on purpose instead of importing the
module's private helpers, so a bug in a helper cannot hide itself.
"""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from activities import curve_fitting
from main import app

client = TestClient(app)

GROUP_IDS = [f"{i:06d}" for i in range(1000)]
CP3_3_OPTIONS = np.array([0.0020, 0.0030, 0.0035, 0.0040])

# Appendix B golden values for group "012983"
GOLDEN_BEAM = {"span": 12.2, "w": 30.7,
               "x": [0.59, 1.47, 2.23, 3.03, 3.9, 4.65, 5.52, 6.36, 7.11, 7.95],
               "M": [120.72, 240.25, 349.21, 435.73, 489.73, 530.37, 565.95, 566.81, 567.79, 509.42],
               "x_predict": 10.37}
GOLDEN_TRAFFIC_LEN = 39
GOLDEN_TRAFFIC_K5 = [5.2, 10.4, 13.0, 16.1, 19.0]
GOLDEN_TRAFFIC_V5 = [62.3, 66.2, 65.6, 56.3, 64.8]
GOLDEN_FFS_LW5 = [3.14, 2.86, 2.97, 3.63, 3.42]
GOLDEN_FFS_AD5 = [29.4, 29.9, 26.3, 13.0, 14.6]
GOLDEN_FFS_Y5 = [59.0, 60.8, 60.2, 72.0, 72.1]
GOLDEN_NEW_SEGMENT = {"lane_width": 2.98, "access_density": 25.8}
GOLDEN_SHA256 = "4e3261197b8c50b826fada16a1bbe2e99600621d0dfa37defb44c9ac95b01d36"


# ---------------------------------------------------------------- independent reference math
def poly_fit(x, y, m):
    x, y = np.asarray(x), np.asarray(y)
    A = np.empty((m + 1, m + 1))
    b = np.empty(m + 1)
    for i in range(m + 1):
        b[i] = np.sum(x ** i * y)
        for j in range(m + 1):
            A[i, j] = np.sum(x ** (i + j))
    return np.linalg.solve(A, b)


def poly_eval(coeffs, x):
    return float(np.polyval(np.asarray(coeffs)[::-1], x))


def line_fit(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    slope = (n * np.dot(x, y) - x.sum() * y.sum()) / (n * np.dot(x, x) - x.sum() ** 2)
    return y.mean() - slope * x.mean(), slope


def capacity(a0, a1):
    return -a0 ** 2 / (4 * a1)


def traffic_analysis(traffic):
    k, v = np.asarray(traffic["k"]), np.asarray(traffic["v"])
    a0r, a1r = line_fit(k, v)
    e = v - (a0r + a1r * k)
    s = np.sqrt(np.sum(e ** 2) / (len(k) - 2))
    flagged = np.abs(e) > 2.5 * s
    a0c, a1c = line_fit(k[~flagged], v[~flagged])
    return a1r, flagged, capacity(a0r, a1r), capacity(a0c, a1c)


def stability_limit(ffs):
    lw, ad = np.asarray(ffs["lane_width"]), np.asarray(ffs["access_density"])
    X1 = np.column_stack([np.ones(len(lw)), lw, ad])
    return 1.0 / np.linalg.eigvalsh(X1.T @ X1 / len(lw)).max()


def cp3_3_answer(alpha_th):
    """Largest listed learning rate that is still below the stability limit."""
    return "ABCD"[int(np.sum(CP3_3_OPTIONS < alpha_th)) - 1]


@pytest.fixture(scope="module")
def all_datasets():
    start = time.perf_counter()
    data = [curve_fitting.generate_params(gid) for gid in GROUP_IDS]
    return data, time.perf_counter() - start


# ---------------------------------------------------------------- 1. route and status codes
def test_route_ok():
    r = client.get("/api/v1/curvefit/datasets/012983")
    assert r.status_code == 200
    assert set(r.json()) == {"group_id", "beam", "traffic", "ffs"}


def test_route_blank_id_is_400():
    r = client.get("/api/v1/curvefit/datasets/%20")
    assert r.status_code == 400
    assert r.json()["detail"] == "Group ID cannot be empty."


def test_route_long_id_is_400():
    r = client.get("/api/v1/curvefit/datasets/" + "1" * 65)
    assert r.status_code == 400
    assert r.json()["detail"] == "Group ID is too long."


def test_id_is_stripped():
    assert curve_fitting.generate_params("  012983 ") == curve_fitting.generate_params("012983")
    assert curve_fitting.generate_params("  012983 ")["group_id"] == "012983"


# ---------------------------------------------------------------- 2. determinism
def test_same_id_same_bytes():
    a = client.get("/api/v1/curvefit/datasets/012983")
    b = client.get("/api/v1/curvefit/datasets/012983")
    assert a.content == b.content


def test_thread_safety():
    ids = [f"{i:06d}" for i in range(0, 400, 10)] + ["012983", "abc", "G-07"]
    serial = [curve_fitting.generate_params(gid) for gid in ids]
    with ThreadPoolExecutor(max_workers=8) as pool:
        threaded = list(pool.map(curve_fitting.generate_params, ids * 2))
    assert threaded == serial * 2


def test_golden_values():
    d = curve_fitting.generate_params("012983")
    assert d["group_id"] == "012983"
    assert d["beam"] == GOLDEN_BEAM
    assert len(d["traffic"]["k"]) == GOLDEN_TRAFFIC_LEN
    assert d["traffic"]["k"][:5] == GOLDEN_TRAFFIC_K5
    assert d["traffic"]["v"][:5] == GOLDEN_TRAFFIC_V5
    assert d["ffs"]["lane_width"][:5] == GOLDEN_FFS_LW5
    assert d["ffs"]["access_density"][:5] == GOLDEN_FFS_AD5
    assert d["ffs"]["ffs"][:5] == GOLDEN_FFS_Y5
    assert d["ffs"]["new_segment"] == GOLDEN_NEW_SEGMENT
    assert hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest() == GOLDEN_SHA256


# ---------------------------------------------------------------- 3. schema and types
def _all_floats(values):
    return all(type(val) is float for val in values)


def test_schema(all_datasets):
    for d in all_datasets[0]:
        assert isinstance(d["group_id"], str)

        beam = d["beam"]
        L = beam["span"]
        assert set(beam) == {"span", "w", "x", "M", "x_predict"}
        assert _all_floats([L, beam["w"], beam["x_predict"]] + beam["x"] + beam["M"])
        assert 8 <= L <= 14 and 15 <= beam["w"] <= 40
        assert len(beam["x"]) == len(beam["M"]) == 10
        assert beam["x"] == sorted(beam["x"])
        assert 0.04 * L - 0.01 <= beam["x"][0] and beam["x"][-1] <= 0.66 * L + 0.01
        assert beam["x_predict"] == round(0.85 * L, 2)

        traffic = d["traffic"]
        assert set(traffic) == {"k", "v"}
        assert 38 <= len(traffic["k"]) <= 40
        assert len(traffic["k"]) == len(traffic["v"])
        assert traffic["k"] == sorted(traffic["k"])
        assert _all_floats(traffic["k"] + traffic["v"])

        ffs = d["ffs"]
        assert set(ffs) == {"lane_width", "access_density", "ffs", "new_segment"}
        assert len(ffs["lane_width"]) == len(ffs["access_density"]) == len(ffs["ffs"]) == 30
        assert _all_floats(ffs["lane_width"] + ffs["access_density"] + ffs["ffs"])
        assert all(2.8 <= val <= 3.7 for val in ffs["lane_width"])
        assert all(0 <= val <= 30 for val in ffs["access_density"])
        new = ffs["new_segment"]
        assert set(new) == {"lane_width", "access_density"}
        assert _all_floats(new.values())


# ---------------------------------------------------------------- 4. invariants
def test_invariants(all_datasets):
    for d in all_datasets[0]:
        gid = d["group_id"]

        beam = d["beam"]
        L, w, xp = beam["span"], beam["w"], beam["x_predict"]
        th = w * xp * (L - xp) / 2
        err2 = abs(poly_eval(poly_fit(beam["x"], beam["M"], 2), xp) - th)
        err6 = abs(poly_eval(poly_fit(beam["x"], beam["M"], 6), xp) - th)
        assert err2 < 0.08 * th, f"B1 failed for {gid}"
        assert err6 > 4 * err2 and err6 > 0.25 * th, f"B2 failed for {gid}"

        a1r, flagged, q_raw, q_clean = traffic_analysis(d["traffic"])
        n_bad = len(d["traffic"]["k"]) - 36
        assert a1r < 0, f"T1 failed for {gid}"
        assert flagged.sum() == n_bad, f"T2 failed for {gid}"
        assert abs(q_raw - q_clean) / q_clean > 0.03, f"T3 failed for {gid}"

        a_th = stability_limit(d["ffs"])
        assert a_th > 0.00205, f"F1 failed for {gid}"
        assert np.all(np.abs(a_th - CP3_3_OPTIONS) / CP3_3_OPTIONS > 0.02), f"F2 failed for {gid}"


def test_flagged_readings_are_the_faulty_ones(all_datasets):
    """T2 stronger form: flagged readings are all v == 0 or all one repeated speed."""
    for d in all_datasets[0]:
        _, flagged, _, _ = traffic_analysis(d["traffic"])
        v_flagged = np.asarray(d["traffic"]["v"])[flagged]
        assert len(set(v_flagged.tolist())) == 1, d["group_id"]


# ---------------------------------------------------------------- 5. spread
def test_spread(all_datasets):
    data = all_datasets[0]
    over = sum(1 for d in data if traffic_analysis(d["traffic"])[2] > traffic_analysis(d["traffic"])[3])
    assert 0.30 <= over / len(data) <= 0.70

    answers = [cp3_3_answer(stability_limit(d["ffs"])) for d in data]
    for option in "ABCD":
        assert answers.count(option) / len(data) >= 0.05, option


# ---------------------------------------------------------------- 6. performance
def test_performance(all_datasets):
    assert all_datasets[1] < 10.0

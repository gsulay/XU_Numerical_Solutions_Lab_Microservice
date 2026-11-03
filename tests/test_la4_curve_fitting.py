import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from activities import beam_polynomial, concrete_regression, greenshields_traffic
from main import app

client = TestClient(app)

ROUTES = {
    "traffic_flow": greenshields_traffic,
    "beam_moment": beam_polynomial,
    "concrete_capacity": concrete_regression,
}
IDS = [f"{i:06d}" for i in range(1000)]
LEARNING_RATES = (2e-7, 5e-7, 1e-6, 2e-6)


def line(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    return np.linalg.solve(X.T @ X, X.T @ y)


def poly(x, y, m):
    V = np.vander(x, m + 1, increasing=True)
    return np.linalg.solve(V.T @ V, V.T @ y)


def peval(a, x):
    return sum(c * x ** k for k, c in enumerate(a))


def query_case(d):
    k, v = np.array(d["density"]), np.array(d["speed"])
    a0, a1 = line(k, v)
    q, jam = d["query_density"], -a0 / a1
    if q <= k.max():
        return "A"
    return "B" if q < jam else "C"


def beam_answers(d):
    x, M = np.array(d["position"]), np.array(d["moment"])
    xp = d["x_predict"]
    th = d["w"] * xp * (d["span"] - xp) / 2
    preds = {m: peval(poly(x, M, m), xp) for m in range(1, 7)}
    count = sum(abs(p - th) <= 0.10 * th for p in preds.values())
    return count, ("A" if preds[6] > th else "B"), preds, th


def concrete_answers(d):
    age, rebar, y = np.array(d["age"]), np.array(d["rebar_area"]), np.array(d["capacity"])
    X = np.column_stack([np.ones(len(y)), age, rebar])
    limit = 1 / np.max(np.linalg.eigvalsh(X.T @ X / len(y)))
    stable = [lr for lr in LEARNING_RATES if lr < limit]
    Z = np.column_stack([np.ones(len(y)), (age - age.mean()) / age.std(), (rebar - rebar.mean()) / rebar.std()])
    b = np.linalg.solve(Z.T @ Z, Z.T @ y)
    return max(stable), ("A" if abs(b[1]) > abs(b[2]) else "B"), limit, b, Z, X


@pytest.mark.parametrize("route", ROUTES)
def test_status_codes(route):
    assert client.get(f"/api/v1/la4/{route}", params={"id": "012983"}).status_code == 200
    assert client.get(f"/api/v1/la4/{route}", params={"id": "   "}).status_code == 400
    assert client.get(f"/api/v1/la4/{route}", params={"id": "x" * 65}).status_code == 400
    assert client.get(f"/api/v1/la4/{route}").status_code == 422


@pytest.mark.parametrize("route", ROUTES)
def test_determinism_and_thread_safety(route):
    module = ROUTES[route]
    serial = [module.generate_params(i) for i in IDS[:40]]
    assert serial == [module.generate_params(i) for i in IDS[:40]]
    with ThreadPoolExecutor(8) as pool:
        assert serial == list(pool.map(module.generate_params, IDS[:40]))
    assert serial[0] != serial[1]


def test_traffic_invariants():
    for i in IDS:
        d = greenshields_traffic.generate_params(i)
        k, v = np.array(d["density"]), np.array(d["speed"])
        a0, a1 = line(k, v)
        assert len(k) == len(v) == 12 and a1 < 0 and np.all(v > 0)
        assert 1 - np.sum((v - a0 - a1 * k) ** 2) / np.sum((v - v.mean()) ** 2) >= 0.90
        q, jam = d["query_density"], -a0 / a1
        if q > k.max():
            assert q >= 1.05 * k.max() and abs(q - jam) >= 0.05 * jam


def test_beam_invariants():
    for i in IDS:
        d = beam_polynomial.generate_params(i)
        x, M = np.array(d["position"]), np.array(d["moment"])
        count, _, preds, th = beam_answers(d)
        errs = {m: abs(p - th) / th for m, p in preds.items()}
        assert x.max() <= 0.67 * d["span"] and d["x_predict"] > x.max()
        assert errs[2] < 0.08 and errs[6] > 0.25 and errs[6] > 4 * errs[2]
        assert not any(0.09 <= e <= 0.11 for e in errs.values())
        r2 = [1 - np.sum((M - peval(poly(x, M, m), x)) ** 2) / np.sum((M - M.mean()) ** 2) for m in range(1, 7)]
        assert all(r2[j + 1] >= r2[j] - 1e-9 for j in range(5))


def test_concrete_invariants():
    for i in IDS:
        d = concrete_regression.generate_params(i)
        _, _, limit, b, Z, X = concrete_answers(d)
        y = np.array(d["capacity"])
        assert len(y) == 24 and abs(np.corrcoef(X[:, 1], X[:, 2])[0, 1]) < 0.5
        assert limit > LEARNING_RATES[0]
        assert all(abs(limit - lr) / lr >= 0.15 for lr in LEARNING_RATES)
        assert max(abs(b[1:])) / min(abs(b[1:])) > 1.10
        a = np.zeros(3)
        for _ in range(2000):
            a = a - 0.1 * (2 / len(y)) * Z.T @ (Z @ a - y)
        np.testing.assert_allclose(a, b, rtol=1e-6)
        assert X[:, 1].min() <= d["new_column"]["age"] <= X[:, 1].max()


def test_simulated_stability_matches_limit():
    for i in IDS[:100]:
        d = concrete_regression.generate_params(i)
        _, _, limit, _, _, X = concrete_answers(d)
        y = np.array(d["capacity"])
        for lr in LEARNING_RATES:
            a = np.zeros(3)
            first = np.mean(y ** 2)
            with np.errstate(all="ignore"):
                for _ in range(3000):
                    a = a - lr * (2 / len(y)) * X.T @ (X @ a - y)
                last = np.mean((X @ a - y) ** 2)
            assert bool(np.isfinite(last) and last < first) == (lr < limit)


def share(values):
    counts = Counter(values)
    return {k: c / len(values) for k, c in counts.items()}


def test_answer_spread():
    traffic = [greenshields_traffic.generate_params(i) for i in IDS]
    beam = [beam_polynomial.generate_params(i) for i in IDS]
    concrete = [concrete_regression.generate_params(i) for i in IDS]

    cp1 = share([query_case(d) for d in traffic])
    cp2_1 = share([beam_answers(d)[0] for d in beam])
    cp2_2 = share([beam_answers(d)[1] for d in beam])
    cp3_1 = share([concrete_answers(d)[0] for d in concrete])
    cp3_2 = share([concrete_answers(d)[1] for d in concrete])
    print({"cp1_1": cp1, "cp2_1": cp2_1, "cp2_2": cp2_2, "cp3_1": cp3_1, "cp3_2": cp3_2})

    assert set(cp1) == {"A", "B", "C"} and min(cp1.values()) >= 0.10
    assert sum(v >= 0.10 for v in cp2_1.values()) >= 2
    assert set(cp2_2) == {"A", "B"} and min(cp2_2.values()) >= 0.10
    assert len(cp3_1) == 4 and min(cp3_1.values()) >= 0.10
    assert set(cp3_2) == {"A", "B"} and min(cp3_2.values()) >= 0.10


def test_runtime():
    t0 = time.time()
    for i in IDS:
        greenshields_traffic.generate_params(i)
        beam_polynomial.generate_params(i)
        concrete_regression.generate_params(i)
    assert time.time() - t0 < 30

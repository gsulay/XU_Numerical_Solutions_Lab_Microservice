"""Tests for the IE 15 LA5 (Gauss-Jacobi / Gauss-Seidel) dataset endpoints."""
import itertools
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from activities import ie15_fertilizer_blend, ie15_leontief, ie15_warehouse_network
from main import app

client = TestClient(app)

ROUTES = {
    "leontief": ie15_leontief,
    "fertilizer_blend": ie15_fertilizer_blend,
    "warehouse_network": ie15_warehouse_network,
}
IDS = [f"{i:06d}" for i in range(1000)]


# ---------- independent reference implementations ----------
def dominant(A):
    d = np.abs(np.diag(A))
    return bool(np.all(d > np.abs(A).sum(axis=1) - d))


def rho(T):
    return float(np.max(np.abs(np.linalg.eigvals(T))))


def rho_gs(A):
    return rho(-np.linalg.inv(np.tril(A)) @ np.triu(A, 1))


def rho_j(A):
    D = np.diag(np.diag(A))
    return rho(-np.linalg.inv(D) @ (A - D))


def jacobi(A, b, x0, tol=1e-4, max_iter=1000):
    x = np.array(x0, float)
    for k in range(1, max_iter + 1):
        x_new = np.array([(b[i] - sum(A[i, j] * x[j] for j in range(len(b)) if j != i)) / A[i, i]
                          for i in range(len(b))])
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k
        x = x_new
    return x, None


def gauss_seidel(A, b, x0, tol=1e-4, max_iter=1000):
    x = np.array(x0, float)
    for k in range(1, max_iter + 1):
        old = x.copy()
        for i in range(len(b)):
            x[i] = (b[i] - sum(A[i, j] * x[j] for j in range(len(b)) if j != i)) / A[i, i]
        if np.max(np.abs(x - old)) < tol:
            return x, k
    return x, None


pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def data():
    return {r: [m.generate_params(i) for i in IDS] for r, m in ROUTES.items()}


# ---------- API behaviour ----------
@pytest.mark.parametrize("route", ROUTES)
def test_status_codes(route):
    url = f"/api/v1/ie15/la5/{route}"
    assert client.get(url, params={"id": "012983"}).status_code == 200
    assert client.get(url, params={"id": "   "}).status_code == 400
    assert client.get(url, params={"id": "x" * 65}).status_code == 400
    assert client.get(url).status_code == 422


@pytest.mark.parametrize("route", ROUTES)
def test_id_is_stripped_and_returned(route):
    r = client.get(f"/api/v1/ie15/la5/{route}", params={"id": "  012983 "}).json()
    assert r["student_id"] == "012983"
    assert r == ROUTES[route].generate_params("012983")


@pytest.mark.parametrize("route", ROUTES)
def test_determinism_and_thread_safety(route):
    module = ROUTES[route]
    serial = [module.generate_params(i) for i in IDS[:40]]
    assert serial == [module.generate_params(i) for i in IDS[:40]]
    with ThreadPoolExecutor(8) as pool:
        assert serial == list(pool.map(module.generate_params, IDS[:40]))
    assert serial[0] != serial[1]


@pytest.mark.parametrize("route", ROUTES)
def test_runtime(route):
    t0 = time.time()
    for i in IDS:
        ROUTES[route].generate_params(i)
    assert time.time() - t0 < 10


# ---------- Scenario 1: Leontief ----------
def test_leontief_schema_and_invariants(data):
    picks = Counter()
    for d in data["leontief"]:
        assert set(d) == {"student_id", "departments", "tech_coeff", "final_demand", "final_demand_next"}
        A = np.array(d["tech_coeff"])
        dem = np.array(d["final_demand"])
        dem2 = np.array(d["final_demand_next"])
        assert A.shape == (3, 3) and dem.shape == (3,) and dem2.shape == (3,)
        assert np.all((A > 0) & (A < 0.5))
        assert np.all(A.sum(axis=1) < 1)
        M = np.eye(3) - A
        assert dominant(M)
        x, n_cold = jacobi(M, dem, np.zeros(3))
        assert n_cold is not None
        np.testing.assert_allclose(x, np.linalg.solve(M, dem), atol=1e-3)
        assert np.all(x > dem)
        _, n2_cold = jacobi(M, dem2, np.zeros(3))
        _, n2_warm = jacobi(M, dem2, x)
        assert n2_warm < n2_cold - 2
        picks["ABC"[int(np.argmax(np.linalg.solve(M, dem) - dem))]] += 1
    assert min(picks.values()) >= 250, picks          # cp1_1 spread


# ---------- Scenario 2: fertilizer blend ----------
def test_fertilizer_schema_and_invariants(data):
    first_rows = Counter()
    for d in data["fertilizer_blend"]:
        assert set(d) == {"student_id", "nutrients", "materials", "composition", "targets"}
        A = np.array(d["composition"])
        b = np.array(d["targets"])
        assert A.shape == (3, 3) and b.shape == (3,)
        assert np.all((A > 0) & (A < 1))
        assert not dominant(A)
        orders = [list(p) for p in itertools.permutations(range(3)) if dominant(A[list(p)])]
        assert len(orders) == 1
        o = orders[0]
        # the given order makes both methods diverge
        assert rho_j(A) > 1 and rho_gs(A) > 1
        assert jacobi(A, b, np.zeros(3), max_iter=60)[1] is None
        assert gauss_seidel(A, b, np.zeros(3), max_iter=60)[1] is None
        # the dominant order converges, and Gauss-Seidel is faster
        xj, nj = jacobi(A[o], b[o], np.zeros(3))
        xg, ng = gauss_seidel(A[o], b[o], np.zeros(3))
        assert nj is not None and ng is not None and ng < nj
        assert np.all(np.linalg.solve(A, b) > 100)
        # exactly one row order makes Gauss-Seidel converge (cp2_2 is static)
        assert sum(rho_gs(A[list(p)]) < 1 for p in itertools.permutations(range(3))) == 1
        first_rows["ABC"[o[0]]] += 1
    assert min(first_rows.values()) >= 100, first_rows   # cp2_1 spread


# ---------- Scenario 3: warehouse network ----------
def warehouse_case(A):
    if any(dominant(A[list(p)]) for p in itertools.permutations(range(4))):
        return "A"
    return "B" if rho_gs(A) < 1 else "C"


def test_warehouse_schema_and_invariants(data):
    cases, biggest = Counter(), Counter()
    for d in data["warehouse_network"]:
        assert set(d) == {"student_id", "warehouses", "A_balance", "b_balance"}
        A = np.array(d["A_balance"])
        b = np.array(d["b_balance"])
        assert A.shape == (4, 4) and b.shape == (4,)
        assert np.all(np.diag(A) != 0)
        assert not dominant(A)
        x = np.linalg.solve(A, b)
        assert np.all(x > 0)
        c = warehouse_case(A)
        cases[c] += 1
        if c == "A":
            o = [list(p) for p in itertools.permutations(range(4)) if dominant(A[list(p)])][0]
            assert gauss_seidel(A[o], b[o], np.zeros(4))[1] is not None
        elif c == "B":
            assert rho_gs(A) < 0.9
            xg, n = gauss_seidel(A, b, np.zeros(4))
            assert n is not None
            np.testing.assert_allclose(xg, x, atol=1e-2)
        else:
            assert all(rho_gs(A[list(p)]) > 1 for p in itertools.permutations(range(4)))
            assert gauss_seidel(A, b, np.zeros(4), max_iter=200)[1] is None
        biggest["ABCD"[int(np.argmax(x))]] += 1
    assert min(cases.values()) >= 250, cases          # cp3_1 spread
    assert min(biggest.values()) >= 180, biggest      # cp3_2 spread

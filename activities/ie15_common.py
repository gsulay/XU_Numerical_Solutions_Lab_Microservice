"""Shared helpers for the IE 15 Lab 05 (Gauss-Jacobi / Gauss-Seidel) activities."""
import hashlib
import itertools

import numpy as np


def clean_id(student_id: str) -> str:
    student_id = student_id.strip()
    if not student_id:
        raise ValueError("Student ID cannot be empty.")
    if len(student_id) > 64:
        raise ValueError("Student ID is too long.")
    return student_id


def make_rng(salt: str, student_id: str) -> np.random.Generator:
    seed = int(hashlib.sha256(f"IE15-LA5-{salt}-{student_id}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed)


def strictly_dominant(A) -> bool:
    A = np.asarray(A, float)
    d = np.abs(np.diag(A))
    return bool(np.all(d > np.abs(A).sum(axis=1) - d))


def rho_jacobi(A) -> float:
    A = np.asarray(A, float)
    D = np.diag(np.diag(A))
    return float(np.max(np.abs(np.linalg.eigvals(-np.linalg.solve(D, A - D)))))


def rho_gauss_seidel(A) -> float:
    A = np.asarray(A, float)
    DL = np.tril(A)
    U = np.triu(A, 1)
    return float(np.max(np.abs(np.linalg.eigvals(-np.linalg.solve(DL, U)))))


def row_orders(n):
    return [list(p) for p in itertools.permutations(range(n))]

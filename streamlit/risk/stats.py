"""Numerical and statistical helpers for the Monte Carlo risk engine."""

from __future__ import annotations

import numpy as np

def safe_cholesky(cov: np.ndarray) -> tuple[np.ndarray, float]:
    
    jitter = 0.0
    n = cov.shape[0]

    for i in range(12):
        try:
            L = np.linalg.cholesky(cov + jitter * np.eye(n))
            return L, jitter
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0.0 else jitter * 10
    
    raise np.linalg.LinAlgError("Covariance matrix isn't positive semi-definite.")

def moments(x: np.ndarray) -> dict[str, float]:

    mean = float(x.mean())
    std = float(x.std())

    if std == 0.0:
        return {"mean": mean, "std" : 0.0, "skew": 0.0, "kurt": 0.0}
    
    diff = x - mean
    skew = float((diff**3).mean() / std**3)
    kurt = float((diff**4).mean() / std**4 - 3.0)

    return {"mean": mean, "std" : std, "skew": skew, "kurt": kurt}

def is_positive_definite(matrix: np.ndarray) -> bool:

    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
    
def eigenvalue_clip(corr: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:

    a = (corr + corr.T) / 2.0
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, epsilon, None)

    rebuilt = (vecs * vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(rebuilt), epsilon, None))
    rebuilt = rebuilt / np.outer(d, d)
    
    np.fill_diagonal(rebuilt, 1.0)

    return rebuilt

def nearest_correlation(corr: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:

    a = (corr + corr.T) / 2.0
    n = a.shape[0]
    y = a.copy()
    delta_s = np.zeros_like(a)

    for i in range(max_iter):
        r = y - delta_s
        vals, vecs = np.linalg.eigh(r)
        x = (vecs * np.clip(vals, 0, None)) @ vecs.T
        delta_s = x - r
        
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        denom = np.linalg.norm(y, "fro")

        if denom > 0 and np.linalg.norm(y - x, "fro") / denom < tol:
            break

    vals, vecs = np.linalg.eigh((y + y.T) / 2.0)
    y = (vecs * np.clip(vals, 1e-12, None)) @ vecs.T
    d = np.sqrt(np.diag(y))
    y = y / np.outer(d, d)
    
    np.fill_diagonal(y, 1.0)
    return y

def repair_correlation(corr: np.ndarray, method: str = "nearest") -> np.ndarray:

    if method == "none" or is_positive_definite(corr):
        return corr
    elif method == "clip":
        return eigenvalue_clip(corr)
    elif method == "nearest":
        return nearest_correlation(corr)
    raise ValueError (f"Unknown repair method: {method!r}")
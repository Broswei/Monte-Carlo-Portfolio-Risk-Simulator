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
    std = fload(x.std())
    
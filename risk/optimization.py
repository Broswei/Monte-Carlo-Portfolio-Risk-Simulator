"""Mean variance portfolio optimization"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class Portfolio:

    weights: np.ndarray
    ret: float
    vol: float
    sharpe: float
    long_only_applied: bool = False

def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0): 

    weight = np.asarray(weights, dtype = float)
    ret = float(weight @ mu)
    vol = float(np.sqrt(weight @ cov @ weight))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    
    return ret, vol, sharpe

def _try_scipy():

    try:
        from scipy.optimize import minimize
        return minimize
    
    except ImportError:
        return None
    
def min_variance_weights(cov: np.ndarray, long_only: bool = False) -> Portfolio:

    n = cov.shape[0]
    ones = np.ones(n)
    inv = np.linalg.pinv(cov)

    weight = inv @ ones
    weight = weight / weight.sum()
    applied = False

    if long_only:

        minimize = _try_scipy()
        if minimize is not None:
            res = minimize(
                lambda x: x @ cov @ x,
                x0 = np.full(n, 1 / n),
                method = "SLSQP",
                bounds = [(0.0, 1.0)] * n,
                constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1.0}],
            )
            if res.success:
                w = res.x
                applied = True
            
    ret, vol, sharpe = portfolio_stats(w, np.zeros(n), cov)
    return Portfolio(weights = w, ret = ret, vol = vol, sharpe = sharpe, long_only_applied = applied)

def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, long_only: bool = False) -> Portfolio:

    n = cov.shape[0]
    inverse = np.linalg.pinv(cov)
    excess = mu
    w = inverse @ excess
    sum = w.sum()
    w = w / sum if sum != 0 else np.full(n, 1 / n)
    applied = False

    if long_only:
        
        minimize = _try_scipy()
        if minimize is not None:
            def neg_sharpe(x):
                ret = x @ mu
                vol = np.sqrt(x @ cov @ x)
                return -(ret - rf) / vol if vol > 0 else 0.0
            
            res = minimize(
                neg_sharpe,
                x0 = np.full(n, 1 / n),
                method = "SLSQP",
                bounds=[(0.0, 1.0)] * n,
                constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0}]
            )
            if res.success:
                w = res.x
                applied = True

    ret, vol, sharpe = portfolio_stats(w, mu, cov, rf)
    return Portfolio(weights = w, ret = ret, vol = vol, sharpe = sharpe, long_only_applied = applied)

def efficient_frontier(mu: np.ndarray, cov: np.ndarray, points: int = 40):

    inverse = np.linalg.pinv(cov)
    ones = np.ones(len(mu))
    A = float(ones @ inverse @ ones)
    B = float(ones @ inverse @ mu)
    C = float(mu @ inverse @ mu)
    denom = A * C - B * B

    low, high = float(mu.min()), float(mu.max())
    pad = 0.2 * (high - low) if high > low else abs(high) + 1e-6

    returns = np.linspace(low - pad, high + pad, points)

    if denom <= 0: 
        vols = np.full_like(returns, np.sqrt(1.0 / A) if A > 0 else 0.0)
    else:
        var = (A * returns**2 - 2*B*returns + C) / denom
        vols = np.sqrt(np.clip(var, 0, None))

    return returns, vols


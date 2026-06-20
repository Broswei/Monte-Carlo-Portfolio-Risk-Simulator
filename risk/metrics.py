"""Risk-adjusted performance metrics"""

from __future__ import annotations

import numpy as np

def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, periods_per_year: float | None = None) -> float:

    sd = float(returns.std())
    if sd == 0:
        return 0.0
    
    sharpe = (float(returns.mean()) - rf) / sd
    if periods_per_year: 
        sharpe *= np.sqrt(periods_per_year)

    return float(sharpe)

def sortino_ratio(returns: np.ndarray, target: float = 0.0, periods_per_year: float | None = None) -> float:

    downside = np.minimum(returns - target, 0.0)
    downside_dev = float(np.sqrt((downside**2).mean()))

    if downside_dev == 0.0:
        return float("inf")
    
    sortino = (float(returns.mean()) - target) / downside_dev
    if periods_per_year:
        sortino *= np.sqrt(periods_per_year)
    
    return float(sortino)

def max_drawdown(equity: np.ndarray) -> float:

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(-drawdown.min())

def path_max_drawdowns(equity_mat: np.ndarray) -> np.ndarray:

    running_max = np.maximum.accumulate(equity_mat, axis = 1)
    drawdown = equity_mat / running_max - 1.0
    return -drawdown.min(axis = 1)

def calmar_ratio(annual_return: float, max_dd: float) -> float:

    if max_dd == 0.0:
        return float("inf")
    return float(annual_return / max_dd)

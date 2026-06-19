"""Mean variance portfolio optimization"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class Portfolio:

    weights: np.ndarray
    returns: float
    vol: float
    sharpe: float
    long_applied_only: bool = False

def portfolio_stats(weights: np.ndarray, mu)
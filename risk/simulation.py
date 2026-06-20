"""Monte Carlo portfolio risk engine"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import safe_cholesky, moments, repair_correlation
from .metrics import path_max_drawdowns

TRADING_DAYS = 252
CONFIDENCE_LVLS = (0.90, 0.95, 0.99, 0.999)
HIST_BINS = 64

@dataclass
class RiskLvl:

    confidence: float
    var_return: float
    var_money: float
    es_return: float
    es_money: float

@dataclass
class RiskRes:

    paths: int 
    horizon_days: int
    portfolio_val: float
    jitter: float
    risk: list[RiskLvl]
    moments: dict[str, float]
    vol_horizon: float
    vol_annual: float
    prob_loss: float
    best: float
    worst: float
    returns: np.ndarray

    def level(self, confidence: float) -> RiskLvl:

        return next(r for r in self.risk if abs(r.confidence - confidence) < 1e-9)
    
def run_simulation(
    returns_pct: np.ndarray, 
    vols_pct: np.ndarray, 
    weights: np.ndarray, 
    corr: np.ndarray, 
    paths: int = 50000, 
    horizon_days: int = 10, 
    portfolio_val: float = 1000000.0, 
    seed: int | None = None, 
    drift: str = "log", 
    repair: str = "nearest"
) -> RiskRes:
    
    n = len(weights)
    weights = np.asarray(weights, dtype= float)
    w_norm = weights / (weights.sum() or 1.0)

    mu_ann = np.asarray(returns_pct, dtype = float) / 100
    vol_ann = np.asarray(vols_pct, dtype = float) / 100
    corr = repair_correlation(np.asarray(corr, dtype = float), method = repair)

    cov = corr * np.outer(vol_ann, vol_ann)
    ell, jitter = safe_cholesky(cov)

    horizon = horizon_days / TRADING_DAYS
    sqrt_hor = np.sqrt(horizon)
    log_mu_ann = mu_ann - 0.5 * np.diag(cov) if drift == "arithmetic" else mu_ann
    mu_hor = log_mu_ann * horizon

    n_paths = int(np.clip(paths, 1000, 200000))
    rng = np.random.default_rng(seed)

    z = rng.standard_normal((n_paths, n))
    g = mu_hor + sqrt_hor * (z @ ell.T)
    simple = np.expm1(g)
    port = simple @ w_norm

    sorted_port = np.sort(port)
    moment = moments(port)

    risk: list[RiskLvl] = []
    for lvl in CONFIDENCE_LVLS:

        alpha = 1.0 - lvl
        quant = float(np.quantile(sorted_port, alpha))
        worst_case = max(1, int(alpha * n_paths))
        est_ret = float(sorted_port[:worst_case].mean()) 

        risk.append(
            RiskLvl(
                confidence = lvl,
                var_return = quant,
                var_money = -portfolio_val * quant,
                es_return = -est_ret,
                es_money = -portfolio_val * est_ret
            )
        )

    prob_loss = float((port < 0).mean())
    vol_hor = moment["std"]
    vol_ann_output = vol_hor * np.sqrt(1 / horizon)

    return RiskRes(
        paths = n_paths,
        horizon_days = horizon_days,
        portfolio_val = portfolio_val,
        jitter = jitter,
        risk = risk,
        moments = moment,
        vol_horizon = vol_hor,
        vol_annual = float(vol_ann_output),
        prob_loss = prob_loss,
        best = float(sorted_port[-1]),
        worst = float(sorted_port[0]),
        returns = port
    )

def histogram (returns: np.ndarray, bins: int = HIST_BINS):

    counts, edges = np.histogram(returns, bins = bins)
    centeres = (edges[:-1] + edges[1:]) / 2.0
    return centeres, counts

@dataclass
class StepResult:

    equity: np.ndarray
    final_returns: np.ndarray
    drawdowns: np.ndarray
    steps: int
    paths: int

def simulate_paths(
        returns_pct: np.ndarray,
        vols_pct: np.ndarray,
        weights: np.ndarray,
        corr: np.ndarray,
        paths: int = 10000,
        horizon_days: int = 10,
        drift: str = "log",
        repair: str = "nearest",
        seed: int | None = None,
        max_paths: int = 20000 
) -> StepResult:
    
    n = len(weights)
    weights = np.asarray(weights, dtype= float)
    w_norm = weights / (weights.sum() or 1.0)

    mu_ann = np.asarray(returns_pct, dtype = float) / 100
    vol_ann = np.asarray(vols_pct, dtype = float) / 100
    corr = repair_correlation(np.asarray(corr, dtype = float), method = repair)

    cov_ann = corr * np.outer(vol_ann, vol_ann)
    cov_daily = cov_ann / TRADING_DAYS
    ell, jitter = safe_cholesky(cov_daily)

    mu_daily = mu_ann / TRADING_DAYS
    if drift == "arithmetic":
        mu_daily = mu_daily - 0.5 * np.diag(cov_daily)

    n_paths = int(np.clip(paths, 1000, max_paths))
    steps = int(horizon_days)
    rng = np.random.default_rng(seed)

    equity = np.ones((n_paths, steps + 1))
    for t in range(steps):
        z = rng.standard_normal((n_paths, n))
        g = mu_daily + (z @ ell.T)
        port_ret = np.expm1(g) @ w_norm
        equity[:, t+1] = equity[:, t] * (1.0 + port_ret)

    final_returns = equity[:, -1] - 1.0
    drawdowns = path_max_drawdowns(equity)

    return StepResult(
        equity = equity,
        final_returns = final_returns,
        drawdowns = drawdowns,
        steps = steps,
        paths = n_paths
    )




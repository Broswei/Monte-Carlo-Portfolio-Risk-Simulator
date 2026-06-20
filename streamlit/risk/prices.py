"""Estimate model inputs from historical prices"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .simulation import TRADING_DAYS

@dataclass
class Estimate:

    names: list[str]
    annual_ret_pct: np.ndarray
    annual_vol_pct: np.ndarray
    corr: np.ndarray
    observations: int

def estimate_from_prices(df: pd.DataFrame):

    work = df.copy()
    first = df.columns[0]
    col = work[first]

    if pd.api.types.is_datetime64_any_dtype(col):
        work = work.drop(columns = [first])
    elif not pd.api.types.is_numeric_dtype:
        force_work = pd.to_numeric(col, errors = "force")
        if force_work.isna().mean() > 0.5:
            work = work.drop(columns = [first])

    numeric = work.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all").dropna()
    numeric = numeric.loc[:, (numeric > 0).all()]

    if numeric.shape[1] < 2:
        raise ValueError("Need at least two numeric asset columns.")
    if numeric.shape[0] < 3:
        raise ValueError("Need at least two rows of prices.")
    
    names = list(numeric.columns.astype(str))
    prices = numeric.to_numpy(dtype=float)

    log_return = np.diff(np.log(prices), axis = 0)
    obs = log_return.shape[0]

    mean_daily = log_return.mean(axis = 0)
    cov_daily = np.cov(log_return, rowvar = False)
    std_daily = np.sqrt(np.diag(cov_daily))

    denom = np.outer(std_daily, std_daily)
    with np.errstate(invalid = "ignore", divide = "ignore"):
        corr = np.where(denom > 0, cov_daily / denom, 0.0)
    np.fill_diagonal(corr, 1.0)

    annual_ret_pct = mean_daily * TRADING_DAYS * 100.0
    annual_vol_pct = std_daily * np.sqrt(TRADING_DAYS) * 100.0

    return Estimate(
        names=names,
        annual_ret_pct=annual_ret_pct,
        annual_vol_pct=annual_vol_pct,
        corr=corr,
        observations=obs,
    )
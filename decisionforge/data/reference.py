"""Independent pandas implementation of period comparison.

The critic uses this to re-derive numbers the SQL tools produced, through a
completely different code path. Agreement = strong evidence the number is real.
"""
from __future__ import annotations

import pandas as pd


def window_bounds(df: pd.DataFrame, window: int):
    mx = df["week_start"].max()
    cur_start = mx - pd.Timedelta(days=7 * (window - 1))
    prior_start = cur_start - pd.Timedelta(days=7 * window)
    return prior_start, cur_start, mx


def _agg(d: pd.DataFrame, metric: str, dimension: str) -> pd.Series:
    g = d.groupby(dimension)
    if metric == "revenue":
        return g["revenue"].sum()
    if metric == "units":
        return g["units"].sum()
    if metric == "stockout_rate":
        return g["stockout_flag"].mean()
    if metric == "avg_price":
        return g["revenue"].sum() / g["units"].sum()
    raise ValueError(metric)


def period_delta(df, metric="revenue", dimension="region", window=13, filters=None) -> pd.DataFrame:
    d = df
    for k, v in (filters or {}).items():
        d = d[d[k] == v]
    prior_start, cur_start, _ = window_bounds(df, window)
    cur = d[d["week_start"] >= cur_start]
    prior = d[(d["week_start"] >= prior_start) & (d["week_start"] < cur_start)]
    out = pd.DataFrame({"cur": _agg(cur, metric, dimension), "prior": _agg(prior, metric, dimension)}).dropna()
    out["delta"] = out["cur"] - out["prior"]
    return out.sort_values("delta")

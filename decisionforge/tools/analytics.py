from __future__ import annotations

from datetime import timedelta
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from decisionforge.data.store import DIMENSIONS, SQLGuardError
from decisionforge.tools.base import ToolError, tool
from decisionforge.util import num

Metric = Literal["revenue", "units", "stockout_rate", "avg_price"]
Dim = Literal["region", "store_id", "category", "sku_id"]


def _metric_expr(metric: str, cond: str) -> str:
    if metric == "revenue":
        return f"SUM(revenue) FILTER (WHERE {cond})"
    if metric == "units":
        return f"SUM(units) FILTER (WHERE {cond})"
    if metric == "stockout_rate":
        return f"AVG(stockout_flag) FILTER (WHERE {cond})"
    return f"SUM(revenue) FILTER (WHERE {cond}) / NULLIF(SUM(units) FILTER (WHERE {cond}), 0)"


def _filters_sql(filters: dict) -> tuple[list[str], dict]:
    conds, params = [], {}
    for i, (k, v) in enumerate(filters.items()):
        if k not in DIMENSIONS:  # identifiers are whitelisted; values are bound
            raise ToolError(f"cannot filter on '{k}'")
        conds.append(f"{k} = $f{i}")
        params[f"f{i}"] = v
    return conds, params


def _scope(filters: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in filters.items()) or "all data"


# --------------------------------------------------------------------- profile
class ProfileArgs(BaseModel):
    pass


@tool("profile_table", "Data-quality profile: rows, coverage, nulls, duplicates, anomalies in the raw table.", ProfileArgs)
def profile_table(store, a: ProfileArgs):
    df = store.df
    null_pct = {c: round(float(p), 2) for c, p in (df.isna().mean() * 100).items() if p > 0}
    dups = int(df.duplicated(["week_start", "store_id", "sku_id"]).sum())
    weeks = int(df["week_start"].nunique())
    expected = weeks * df["store_id"].nunique() * df["sku_id"].nunique()
    coverage = round(len(df) / expected, 4)
    flags = [f"{c}: {p}% null" for c, p in null_pct.items()]
    if dups:
        flags.append(f"{dups} duplicate (week, store, sku) keys")
    if int((df["units"] < 0).sum()):
        flags.append("negative units present")
    if coverage < 0.99:
        flags.append(f"panel coverage {coverage:.1%}")
    data = {
        "rows": int(len(df)),
        "weeks": weeks,
        "stores": int(df["store_id"].nunique()),
        "skus": int(df["sku_id"].nunique()),
        "coverage": coverage,
        "duplicate_keys": dups,
        "null_pct": null_pct,
        "flags": flags,
    }
    return f"{len(df):,} rows, {weeks} weeks; flags: {flags or 'none'}", data


# ------------------------------------------------------------------------- sql
class SQLArgs(BaseModel):
    sql: str = Field(..., description="A single read-only SELECT/WITH query against table `sales`.")


@tool("run_sql", "Run an ad-hoc READ-ONLY SQL query on table `sales` (guarded, row-capped).", SQLArgs)
def run_sql(store, a: SQLArgs):
    try:
        df = store.safe_query(a.sql, max_rows=50)
    except SQLGuardError as e:
        raise ToolError(f"blocked by SQL guard: {e}") from e
    except Exception as e:  # DuckDB binder/parser errors are useful feedback for repair
        raise ToolError(f"SQL error: {e}") from e
    rows = [[None if (isinstance(v, float) and np.isnan(v)) else (v if isinstance(v, (int, float, str)) else str(v)) for v in r] for r in df.itertuples(index=False)]
    return f"{len(rows)} rows returned", {"columns": list(df.columns), "rows": rows, "row_count": len(rows)}


# --------------------------------------------------------------------- compare
class CompareArgs(BaseModel):
    metric: Metric = "revenue"
    dimension: Dim = "region"
    window_weeks: int = Field(13, ge=2, le=26)
    filters: dict[Dim, str] = Field(default_factory=dict)


@tool(
    "compare_periods",
    "Compare the latest N weeks with the N weeks before, broken down by a dimension. Returns deltas sorted worst-first.",
    CompareArgs,
)
def compare_periods(store, a: CompareArgs):
    mx = store.scalar("SELECT max(week_start) FROM sales")
    cur_start = mx - timedelta(days=7 * (a.window_weeks - 1))
    prior_start = cur_start - timedelta(days=7 * a.window_weeks)
    conds, params = _filters_sql(a.filters)
    params.update(cs=cur_start, ps=prior_start)
    where = " AND ".join(["week_start >= $ps"] + conds)
    cur = _metric_expr(a.metric, "week_start >= $cs")
    pri = _metric_expr(a.metric, "week_start >= $ps AND week_start < $cs")
    grp = store.query(f"SELECT {a.dimension} AS grp, {cur} AS cur, {pri} AS prior FROM sales WHERE {where} GROUP BY 1 ORDER BY 1", params)
    tot = store.query(f"SELECT {cur} AS cur, {pri} AS prior FROM sales WHERE {where}", params)
    if tot.empty or tot.iloc[0].isna().any():
        raise ToolError(f"no data for scope: {_scope(a.filters)}")
    t_cur, t_prior = float(tot.iloc[0]["cur"]), float(tot.iloc[0]["prior"])
    t_delta = t_cur - t_prior
    additive = a.metric in ("revenue", "units")
    rows = []
    for _, r in grp.dropna(subset=["cur", "prior"]).iterrows():
        d = float(r["cur"]) - float(r["prior"])
        rows.append(
            {
                "group": str(r["grp"]),
                "cur": num(r["cur"]),
                "prior": num(r["prior"]),
                "delta": num(d),
                "pct": num(d / r["prior"]) if r["prior"] else None,
                "share_of_change": num(d / t_delta) if (additive and t_delta) else None,
            }
        )
    rows.sort(key=lambda x: x["delta"])
    # "worst" = most adverse direction: a fall for revenue/units/price, a RISE for stockout_rate
    adverse_is_rise = a.metric == "stockout_rate"
    worst = (rows[-1] if adverse_is_rise else rows[0]) if rows else None
    best = (rows[0] if adverse_is_rise else rows[-1]) if rows else None
    data = {
        "metric": a.metric,
        "dimension": a.dimension,
        "scope": _scope(a.filters),
        "as_of": str(mx),
        "window_weeks": a.window_weeks,
        "total": {"cur": num(t_cur), "prior": num(t_prior), "delta": num(t_delta), "pct": num(t_delta / t_prior) if t_prior else None},
        "groups": rows,
        "worst": worst,
        "best": best,
    }
    w = data["worst"]
    pct = data["total"]["pct"]
    if not w or pct is None:
        return "no comparable groups", data
    chg = f"{t_delta * 100:+.1f} pts" if adverse_is_rise else f"{pct:+.1%}"
    return f"{a.metric} {chg} vs prior {a.window_weeks}w ({_scope(a.filters)}); worst {a.dimension}={w['group']} ({w['delta']:+,.2f})", data


# ------------------------------------------------------------------- anomalies
class AnomalyArgs(BaseModel):
    metric: Literal["revenue", "units"] = "revenue"
    dimension: Dim = "store_id"
    recent_weeks: int = Field(13, ge=4, le=26)
    filters: dict[Dim, str] = Field(default_factory=dict)


@tool(
    "detect_anomalies",
    "Find groups whose SHARE of a metric dropped abnormally (robust z<=-3) in recent weeks. Share-based, so common seasonality/trend cancels out.",
    AnomalyArgs,
)
def detect_anomalies(store, a: AnomalyArgs):
    conds, params = _filters_sql(a.filters)
    where = " AND ".join(["TRUE"] + conds)
    d = store.query(f"SELECT week_start, {a.dimension} AS grp, SUM({a.metric}) AS v FROM sales WHERE {where} GROUP BY 1, 2 ORDER BY 1, 2", params)
    if d.empty:
        raise ToolError(f"no data for scope: {_scope(a.filters)}")
    pv = d.pivot(index="week_start", columns="grp", values="v").fillna(0).sort_index()
    share = pv.div(pv.sum(axis=1), axis=0)
    n, r = len(share), a.recent_weeks
    if n - r < 20:
        raise ToolError("not enough baseline history for anomaly detection")
    base, rec = share.iloc[: n - r], share.iloc[n - r :]
    med = base.median()
    mad = base.sub(med).abs().median() * 1.4826
    floor = 0.02 * med
    mad = mad.where(mad > floor, floor).replace(0, 1e-6)
    z = (rec - med) / mad
    out = []
    for g in share.columns:
        flagged = int((z[g] <= -3).sum())
        if flagged >= 3:
            out.append(
                {
                    "group": str(g),
                    "weeks_flagged": flagged,
                    "min_z": num(z[g].min(), 2),
                    "mean_share_change_pct": num(rec[g].mean() / med[g] - 1),
                }
            )
    out.sort(key=lambda x: x["mean_share_change_pct"])
    data = {"metric": a.metric, "dimension": a.dimension, "scope": _scope(a.filters), "recent_weeks": r, "checked_groups": int(share.shape[1]), "groups": out}
    return (f"{len(out)} anomalous {a.dimension}(s) in {_scope(a.filters)}" + (f"; worst {out[0]['group']} ({out[0]['mean_share_change_pct']:+.0%} share)" if out else "")), data


# -------------------------------------------------------------------- forecast
class ForecastArgs(BaseModel):
    metric: Literal["revenue", "units"] = "revenue"
    horizon_weeks: int = Field(8, ge=1, le=13)
    filters: dict[Dim, str] = Field(default_factory=dict)


BAND_K = 2.0


def _predict(hist: np.ndarray, k: int) -> np.ndarray:
    m = len(hist)
    g = float(np.clip(hist[-26:].sum() / hist[-78:-52].sum(), 0.7, 1.3))
    return np.array([hist[m + i - 52] * g for i in range(k)])


@tool(
    "forecast",
    "Forecast a metric N weeks ahead (seasonal-naive x YoY drift) with a prediction band calibrated on rolling-origin backtests.",
    ForecastArgs,
)
def forecast(store, a: ForecastArgs):
    conds, params = _filters_sql(a.filters)
    where = " AND ".join(["TRUE"] + conds)
    s = store.query(f"SELECT week_start, SUM({a.metric}) AS v FROM sales WHERE {where} GROUP BY 1 ORDER BY 1", params)
    y = s["v"].to_numpy(float)
    n, h = len(y), a.horizon_weeks
    if n < 91:
        raise ToolError(f"need >= 91 weeks of history, have {n}")
    # Rolling-origin backtest: replay the exact method from many past origins and
    # measure how wrong it was over h weeks. Intervals come from those real errors.
    wk_err, tot_err, ape = [], [], []
    for o in range(78, n - h + 1):
        pred, act = _predict(y[:o], h), y[o : o + h]
        wk_err.extend(np.log(np.maximum(act, 1) / np.maximum(pred, 1)))
        tot_err.append(np.log(max(act.sum(), 1) / max(pred.sum(), 1)))
        ape.extend(np.abs(act - pred) / np.maximum(act, 1))
    wk_err, tot_err = np.array(wk_err), np.array(tot_err)
    raw = _predict(y, h)
    wb, tb = float(np.median(wk_err)), float(np.median(tot_err))
    # Band half-width = 2 x RMS backtest error. Chosen empirically: plain 10-90% quantiles of the
    # (heavily overlapping) rolling errors covered only ~43% of true holdouts; this covers ~70%.
    whw = BAND_K * float(np.sqrt(np.mean(wk_err**2)))
    thw = BAND_K * float(np.sqrt(np.mean(tot_err**2)))
    wlo, whi, tlo, thi = wb - whw, wb + whw, tb - thw, tb + thw
    last = s["week_start"].iloc[-1]
    weekly = [
        {
            "week": str((last + timedelta(days=7 * (i + 1))).date()),
            "point": num(raw[i] * np.exp(wb), 2),
            "lo": num(raw[i] * np.exp(wlo), 2),
            "hi": num(raw[i] * np.exp(whi), 2),
        }
        for i in range(h)
    ]
    data = {
        "metric": a.metric,
        "scope": _scope(a.filters),
        "horizon_weeks": h,
        "method": "seasonal-naive x YoY drift, bias-corrected; band = 2 x RMS rolling-origin backtest error",
        "calibration_note": "band is approximate (empirical holdout coverage ~70% on synthetic data, below an 80% nominal target)",
        "history_weeks": n,
        "backtest": {"origins": int(len(tot_err)), "mape": num(float(np.mean(ape)))},
        "weekly": weekly,
        "total": {"point": num(raw.sum() * np.exp(tb), 2), "lo": num(raw.sum() * np.exp(tlo), 2), "hi": num(raw.sum() * np.exp(thi), 2)},
    }
    t = data["total"]
    return f"{a.metric} next {h}w ({_scope(a.filters)}): {t['point']:,.0f} [{t['lo']:,.0f}, {t['hi']:,.0f}], backtest MAPE {data['backtest']['mape']:.1%}", data


# ---------------------------------------------------------------------- effect
class EffectArgs(BaseModel):
    group_by: Literal["category", "region", "store_id"] = "category"
    filters: dict[Dim, str] = Field(default_factory=dict)
    n_boot: int = Field(1000, ge=200, le=5000)


@tool(
    "effect_estimate",
    "Estimate promotion unit-lift and REVENUE-lift (net of discount) per group, with bootstrap 95% CIs over store-SKU pairs. Observational.",
    EffectArgs,
)
def effect_estimate(store, a: EffectArgs):
    conds, params = _filters_sql(a.filters)
    where = " AND ".join(["TRUE"] + conds)
    d = store.query(
        f"""SELECT store_id, sku_id, {a.group_by} AS grp,
            AVG(units) FILTER (WHERE promo_flag = 1) AS p,
            AVG(units) FILTER (WHERE promo_flag = 0) AS b,
            SUM(revenue) FILTER (WHERE promo_flag = 1) / NULLIF(SUM(units) FILTER (WHERE promo_flag = 1), 0) AS pp,
            SUM(revenue) FILTER (WHERE promo_flag = 0) / NULLIF(SUM(units) FILTER (WHERE promo_flag = 0), 0) AS bp
            FROM sales WHERE {where} GROUP BY 1, 2, 3 ORDER BY 3, 1, 2""",
        params,
    ).dropna()
    d = d[d["b"] > 0]
    if d.empty:
        raise ToolError("no promo/non-promo pairs for scope")
    d = d.assign(pr=d["pp"] / d["bp"])
    rng = np.random.default_rng(0)
    groups = []
    for g, sub in d.groupby("grp"):
        P, B, R = sub["p"].to_numpy(), sub["b"].to_numpy(), sub["pr"].to_numpy()
        m = len(sub)
        idx = rng.integers(0, m, size=(a.n_boot, m))
        lifts = P[idx].sum(axis=1) / B[idx].sum(axis=1)
        ratios = R[idx].mean(axis=1)
        rev = lifts * ratios
        lift, ratio = P.sum() / B.sum(), R.mean()
        groups.append(
            {
                "group": str(g),
                "n_pairs": int(m),
                "lift": num(lift),
                "ci_low": num(np.quantile(lifts, 0.025)),
                "ci_high": num(np.quantile(lifts, 0.975)),
                "price_ratio": num(ratio),
                "breakeven_lift": num(1 / ratio),
                "rev_lift": num(lift * ratio),
                "rev_ci_low": num(np.quantile(rev, 0.025)),
                "rev_ci_high": num(np.quantile(rev, 0.975)),
            }
        )
    groups.sort(key=lambda x: -x["lift"])
    data = {"group_by": a.group_by, "scope": _scope(a.filters), "groups": groups, "note": "observational; promo weeks are not randomised in real data"}
    top = groups[0]
    return f"top {a.group_by}: {top['group']} unit-lift {top['lift']:.2f}x (95% CI {top['ci_low']:.2f}-{top['ci_high']:.2f})", data

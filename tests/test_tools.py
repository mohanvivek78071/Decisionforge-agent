import numpy as np
import pytest

from decisionforge.data.reference import period_delta
from decisionforge.tools import ToolError, run_tool


def test_compare_matches_independent_pandas(store):
    _, _, d = run_tool(store, "compare_periods", {"metric": "revenue", "dimension": "region"})
    ref = period_delta(store.df, "revenue", "region", 13)
    assert d["worst"]["group"] == ref.index[0]
    assert d["worst"]["delta"] == pytest.approx(ref["delta"].iloc[0], rel=1e-4)


def test_stockout_worst_is_largest_rise(store):
    _, _, d = run_tool(store, "compare_periods", {"metric": "stockout_rate", "dimension": "store_id", "filters": {"region": "West", "category": "Beverages"}})
    assert d["worst"]["group"] in ("W02", "W03") and d["worst"]["delta"] > 0.4


def test_anomaly_detector_flags_planted_stores(store):
    _, _, d = run_tool(store, "detect_anomalies", {"filters": {"region": "West", "category": "Beverages"}})
    assert {g["group"] for g in d["groups"]} == {"W02", "W03"}


def test_no_false_alarm_on_healthy_scope(store):
    _, _, d = run_tool(store, "detect_anomalies", {"filters": {"region": "North", "category": "Snacks"}})
    assert d["groups"] == []


def test_forecast_shape_and_band(store):
    _, _, d = run_tool(store, "forecast", {"metric": "units", "horizon_weeks": 8, "filters": {"category": "Snacks"}})
    t = d["total"]
    assert len(d["weekly"]) == 8 and t["lo"] < t["point"] < t["hi"]
    assert all(w["lo"] < w["point"] < w["hi"] for w in d["weekly"])


def test_effect_recovers_planted_lift_and_breakeven(store):
    _, _, d = run_tool(store, "effect_estimate", {})
    g = {x["group"]: x for x in d["groups"]}
    assert g["Snacks"]["ci_low"] <= 1.35 <= g["Snacks"]["ci_high"]
    assert g["Beverages"]["rev_ci_high"] < 1.0 < g["Beverages"]["lift"]  # units up, revenue DOWN


def test_profile_finds_nulls(store):
    _, _, d = run_tool(store, "profile_table", {})
    assert "on_hand" in d["null_pct"]


def test_bad_args_and_unknown_tool(store):
    with pytest.raises(ToolError):
        run_tool(store, "compare_periods", {"dimension": "planet"})
    with pytest.raises(ToolError):
        run_tool(store, "compare_periods", {"window_weeks": 999})
    with pytest.raises(ToolError):
        run_tool(store, "does_not_exist", {})
    with pytest.raises(ToolError):
        run_tool(store, "run_sql", {"sql": "DROP TABLE sales"})
    assert store.table_exists()

from __future__ import annotations

import json
import time

import pandas as pd

from decisionforge.data.store import DataStore
from decisionforge.data.synth import generate
from decisionforge.evals.cases import CASES, truth
from decisionforge.orchestrator import Orchestrator


def evaluate(llm) -> dict:
    df = generate()
    ctx_truth = truth(df)
    rows, grounded, findings, calls = [], 0, 0, []
    ctx = {"truth": ctx_truth}
    for c in CASES:
        data = df
        if c.holdout_weeks:
            cut = df["week_start"].max() - pd.Timedelta(days=7 * c.holdout_weeks)
            data = df[df["week_start"] <= cut]
            ctx["holdout"] = df[df["week_start"] > cut]
        store = DataStore(data)
        ctx["store"] = store
        t0 = time.time()
        res = Orchestrator(store, llm).run(c.question)
        passed, detail = c.check(res, ctx)
        findings += len(res.memo.findings)
        grounded += sum(v.status == "verified" for v in res.verdicts)
        calls.append(len(res.ledger))
        rows.append({"case": c.name, "passed": bool(passed), "detail": detail, "tool_calls": len(res.ledger), "ms": round((time.time() - t0) * 1000)})
    cov = ctx.get("forecast_cov", [])
    return {
        "provider": llm.name,
        "cases": rows,
        "pass_rate": sum(r["passed"] for r in rows) / len(rows),
        "grounding_rate": grounded / findings if findings else 1.0,
        "avg_tool_calls": sum(calls) / len(calls),
        "forecast_band_hits": f"{sum(cov)}/{len(cov)}" if cov else "n/a",
        "truth": ctx_truth,
    }


def run_eval(llm, as_json: bool = False) -> int:
    r = evaluate(llm)
    if as_json:
        print(json.dumps(r, indent=2))
    else:
        print(f"\nDecisionForge eval  |  provider: {r['provider']}  |  planted truth: {r['truth']}\n")
        for x in r["cases"]:
            print(f"  {'PASS' if x['passed'] else 'FAIL'}  {x['case']:<26} {x['tool_calls']:>2} calls  {x['ms']:>5} ms  {x['detail']}")
        print(f"\n  pass rate      : {r['pass_rate']:.0%}")
        print(f"  grounding rate : {r['grounding_rate']:.0%}  (findings verified by the critic)")
        print(f"  avg tool calls : {r['avg_tool_calls']:.1f}")
        print(f"  forecast band  : {r['forecast_band_hits']} holdout(s) inside the prediction band\n")
    return 0 if r["pass_rate"] == 1.0 else 1

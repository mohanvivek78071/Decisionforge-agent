"""Deterministic, key-free provider.

It implements the SAME interface as the Claude provider, so the whole agent loop
(plan -> investigate -> narrate -> audit) runs and is testable in CI without network
or API spend. It is intentionally rule-based: the interesting engineering is the
harness around the model (tools, ledger, critic, evals), not the prompt.
"""
from __future__ import annotations

import re
from typing import Any

from decisionforge.schemas import Evidence, Finding, Memo, NextStep, Plan, Step
from decisionforge.util import get_path

DOMAIN = re.compile(
    r"revenue|sales|units|stock-?out|price|promo|discount|region|store|categor|sku|demand|inventory|forecast|data|quality|lift|campaign",
    re.I,
)


def _abstain(q: str, why: str) -> dict:
    return Memo(
        question=q,
        answer=f"I can't answer this from the available data. {why}",
        findings=[],
        recommendation="Rephrase the question in terms of sales, units, revenue, stockouts, promotions, regions, stores or categories.",
        confidence="low",
        risks=["No analysis was run."],
        abstained=True,
    ).model_dump()


def _entities(q: str, schema: dict) -> dict[str, str]:
    f: dict[str, str] = {}
    for dim in ("region", "category"):
        for v in schema["values"][dim]:
            if re.search(rf"\b{re.escape(v.lower())}\b", q.lower()):
                f[dim] = v
    for m in re.findall(r"\b([nsewNSEW]\d{2})\b", q):
        if m.upper() in schema["values"]["store_id"]:
            f["store_id"] = m.upper()
    return f


def _find(ledger: list[Evidence], tool: str, **match) -> Evidence | None:
    for e in ledger:
        if e.tool == tool and all(e.args.get(k) == v for k, v in match.items()):
            return e
    return None


def _fmt_scope(filters: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in filters.items()) or "all data"


class OfflineLLM:
    name = "offline-rules"

    def structured(self, task: str, payload: dict, model):
        fn = {"plan": self._plan, "next_step": self._next_step, "narrate": self._narrate}.get(task)
        if fn is None:
            raise ValueError(f"unknown task {task}")
        return model.model_validate(fn(payload))

    # ------------------------------------------------------------------ plan
    def _plan(self, p: dict) -> dict:
        q, schema = p["question"], p["schema"]
        ql = q.lower()
        if not DOMAIN.search(ql):
            return Plan(intent="unsupported", rationale="No sales/inventory concept found in the question.").model_dump()
        metric = "units" if ("units" in ql or "demand" in ql) else "revenue"
        filters = _entities(q, schema)
        n = re.search(r"(\d+)\s*[- ]?week", ql)
        window = int(n.group(1)) if n else (4 if "month" in ql else 13)
        window = max(2, min(window, 26))
        if re.search(r"forecast|predict|projection|project(ed)?\b|next \d+", ql):
            h = max(1, min(int(n.group(1)) if n else 8, 13))
            return Plan(
                intent="forecast",
                rationale="Forward-looking question -> profile data quality, then forecast with backtested band.",
                steps=[
                    Step(tool="profile_table", purpose="check data quality before modelling"),
                    Step(tool="forecast", args={"metric": metric, "horizon_weeks": h, "filters": filters}, purpose="forecast with calibrated band"),
                ],
            ).model_dump()
        if re.search(r"promo|discount|lift|campaign", ql):
            return Plan(
                intent="effect",
                rationale="Promotion effectiveness -> estimate unit-lift and net revenue-lift with bootstrap CIs.",
                steps=[
                    Step(tool="profile_table", purpose="check data quality"),
                    Step(tool="effect_estimate", args={"group_by": "category", "filters": filters}, purpose="lift vs break-even"),
                ],
            ).model_dump()
        if re.search(r"quality|profile|missing|null|clean", ql):
            return Plan(intent="profile", rationale="Data-quality request.", steps=[Step(tool="profile_table")]).model_dump()
        if re.search(r"why|declin|drop|fall|fell|hurt|worst|down|decreas|underperform|where|driver|cause", ql):
            dim = "region" if "region" not in filters else ("category" if "category" not in filters else "store_id")
            return Plan(
                intent="diagnose",
                rationale="Diagnostic question -> quantify the change, then drill down the hierarchy and test a hypothesis.",
                steps=[
                    Step(tool="profile_table", purpose="rule out data-quality artefacts"),
                    Step(tool="compare_periods", args={"metric": metric, "dimension": dim, "window_weeks": window, "filters": filters}, purpose="locate the change"),
                ],
            ).model_dump()
        return Plan(intent="unsupported", rationale="Could not map the question to a supported analysis.").model_dump()

    # ------------------------------------------------------------- next step
    def _next_step(self, p: dict) -> dict:
        plan = Plan.model_validate(p["plan"])
        ledger = [Evidence.model_validate(e) for e in p["ledger"]]
        if plan.intent != "diagnose":
            return NextStep(action="stop", reason="single-shot analysis complete").model_dump()
        init = next((s for s in plan.steps if s.tool == "compare_periods"), None)
        if init is None:
            return NextStep(action="stop", reason="no comparison in plan").model_dump()
        M, w = init.args.get("metric", "revenue"), init.args.get("window_weeks", 13)
        scope = dict(init.args.get("filters", {}))

        def ask(tool: str, reason: str, **args):
            args.setdefault("filters", dict(scope))
            return NextStep(action="tool", step=Step(tool=tool, args=args, purpose=reason), reason=reason).model_dump()

        def done(tool: str, **args) -> Evidence | None:
            args.setdefault("filters", dict(scope))
            return _find(ledger, tool, **args)

        for dim in ("region", "category"):  # narrow down the hierarchy following the worst contributor
            if dim in scope:
                continue
            ev = done("compare_periods", metric=M, dimension=dim, window_weeks=w)
            if ev is None:
                return ask("compare_periods", f"drill into {dim}", metric=M, dimension=dim, window_weeks=w)
            worst = ev.data.get("worst") if ev.ok else None
            if not worst or worst["delta"] >= 0:
                return NextStep(action="stop", reason=f"no negative contributor at {dim} level").model_dump()
            scope[dim] = worst["group"]
        if "store_id" not in scope:
            if done("compare_periods", metric=M, dimension="store_id", window_weeks=w) is None:
                return ask("compare_periods", "which stores drive it", metric=M, dimension="store_id", window_weeks=w)
            if done("compare_periods", metric="stockout_rate", dimension="store_id", window_weeks=w) is None:
                return ask("compare_periods", "hypothesis: availability (stockouts) caused the drop", metric="stockout_rate", dimension="store_id", window_weeks=w)
            if done("detect_anomalies", metric=M if M in ("revenue", "units") else "revenue", dimension="store_id") is None:
                return ask("detect_anomalies", "confirm the store drop is abnormal vs peers (seasonality-free)", metric=M if M in ("revenue", "units") else "revenue", dimension="store_id")
        return NextStep(action="stop", reason="hypothesis tested; enough evidence").model_dump()

    # --------------------------------------------------------------- narrate
    def _narrate(self, p: dict) -> dict:
        q = p["question"]
        plan = Plan.model_validate(p["plan"])
        ledger = [Evidence.model_validate(e) for e in p["ledger"] if e["ok"]]
        if plan.intent == "unsupported" or not ledger:
            return _abstain(q, plan.rationale or "No successful analysis steps.")
        fn = {"diagnose": self._n_diag, "forecast": self._n_forecast, "effect": self._n_effect, "profile": self._n_profile}[plan.intent]
        memo = fn(q, plan, ledger)
        memo["question"] = q
        return memo

    def _n_diag(self, q, plan, ledger) -> dict:
        init = next(s for s in plan.steps if s.tool == "compare_periods")
        M = init.args.get("metric", "revenue")
        cmps = [e for e in ledger if e.tool == "compare_periods" and e.args["metric"] == M]
        if not cmps:
            return _abstain(q, "The comparison step failed.")
        root = cmps[0]
        w = root.args["window_weeks"]
        F: list[Finding] = []
        tp = root.data["total"]["pct"]
        if tp is not None:
            F.append(Finding(text=f"Total {M} moved {tp:+.1%} vs the prior {w} weeks (as of {root.data['as_of']}).", evidence_id=root.id, value_path="total.pct", value=tp))
        chain: list[str] = []
        culprit_stores: list[str] = []
        for e in cmps:
            worst = e.data.get("worst")
            dim = e.args["dimension"]
            if not worst or worst["delta"] >= 0:
                continue
            if dim == "store_id":
                for i, g in enumerate(e.data["groups"][:3]):
                    if g["delta"] < 0:
                        culprit_stores.append(g["group"])
                        F.append(Finding(text=f"Store {g['group']} ({e.data['scope']}): {M} {g['delta']:+,.0f} ({g['pct']:+.0%}).", evidence_id=e.id, value_path=f"groups.{i}.delta", value=g["delta"]))
            else:
                chain.append(f"{dim} '{worst['group']}'")
                F.append(Finding(text=f"Largest drag by {dim} ({e.data['scope']}): {worst['group']} at {worst['delta']:+,.0f} ({worst['pct']:+.1%}).", evidence_id=e.id, value_path="worst.delta", value=worst["delta"]))
        so = next((e for e in ledger if e.tool == "compare_periods" and e.args["metric"] == "stockout_rate"), None)
        so_hit: list[str] = []
        if so:
            for j, g in enumerate(reversed(so.data["groups"])):
                if g["delta"] >= 0.10 and j < 3:
                    idx = len(so.data["groups"]) - 1 - j
                    so_hit.append(g["group"])
                    F.append(Finding(text=f"Stockout rate at {g['group']} rose {g['delta'] * 100:+.0f} pts ({g['prior']:.0%} -> {g['cur']:.0%}).", evidence_id=so.id, value_path=f"groups.{idx}.delta", value=g["delta"]))
        an = next((e for e in ledger if e.tool == "detect_anomalies"), None)
        if an:
            for i, g in enumerate(an.data["groups"][:3]):
                F.append(Finding(text=f"{g['group']}: share of {M} fell {abs(g['mean_share_change_pct']):.0%} vs its baseline, abnormal in {g['weeks_flagged']} of the last {an.data['recent_weeks']} weeks.", evidence_id=an.id, value_path=f"groups.{i}.mean_share_change_pct", value=g["mean_share_change_pct"]))
        if not chain:
            return Memo(
                question=q,
                answer=f"{M.title()} did not decline in a way that localises to a region: total moved {tp:+.1%} vs the prior {w} weeks.",
                findings=F,
                recommendation="No corrective action indicated; keep monitoring.",
                confidence="medium",
                risks=["Only the latest-vs-prior window was compared; a longer view may differ."],
            ).model_dump()
        overlap = sorted(set(culprit_stores) & set(so_hit))
        where = " -> ".join(chain) + (f" -> stores {', '.join(culprit_stores)}" if culprit_stores else "")
        if overlap:
            cause = f" Stockout rates spiked at the same stores ({', '.join(overlap)}), pointing to an availability problem rather than lost demand."
            conf = "high"
            reco = (
                f"Prioritise replenishment for {chain[-1].split(' ', 1)[1].strip(chr(39))} at {', '.join(overlap)}: expedite stock or rebalance from nearby stores, "
                "and add an alert on weekly stockout rate for these store-category pairs. Confirm supplier fill-rate/lead-time before committing spend."
            )
        else:
            cause = " No matching availability signal was found, so the cause is unconfirmed."
            conf = "medium"
            reco = "Investigate the flagged locations (pricing, local competition, assortment) before acting; availability does not explain the drop."
        answer = f"{M.title()} moved {tp:+.1%} overall, but the decline localises to {where}.{cause}"
        return Memo(
            question=q,
            answer=answer,
            findings=F,
            recommendation=reco,
            confidence=conf,
            risks=[
                "Observational: the stockout link is co-movement, not proof of causality - confirm with supplier/DC data.",
                "Latest-vs-prior windows can be distorted by seasonality; the share-based anomaly check controls for common seasonality.",
                "stockout_flag is a weekly availability proxy, not measured lost sales.",
            ],
        ).model_dump()

    def _n_forecast(self, q, plan, ledger) -> dict:
        fc = next((e for e in ledger if e.tool == "forecast"), None)
        if not fc:
            return _abstain(q, "The forecast step failed.")
        d, t = fc.data, fc.data["total"]
        mape = d["backtest"]["mape"]
        F = [
            Finding(text=f"Expected {d['metric']} over the next {d['horizon_weeks']} weeks ({d['scope']}): {t['point']:,.0f}.", evidence_id=fc.id, value_path="total.point", value=t["point"]),
            Finding(text=f"Lower edge of the prediction band: {t['lo']:,.0f}.", evidence_id=fc.id, value_path="total.lo", value=t["lo"]),
            Finding(text=f"Upper edge of the prediction band: {t['hi']:,.0f}.", evidence_id=fc.id, value_path="total.hi", value=t["hi"]),
            Finding(text=f"Rolling-origin backtest weekly error (MAPE): {mape:.1%}.", evidence_id=fc.id, value_path="backtest.mape", value=mape),
        ]
        conf = "high" if mape < 0.05 else "medium" if mape < 0.10 else "low"
        return Memo(
            question=q,
            answer=f"{d['metric'].title()} for {d['scope']} over the next {d['horizon_weeks']} weeks is expected around {t['point']:,.0f} (band {t['lo']:,.0f} to {t['hi']:,.0f}).",
            findings=F,
            recommendation=f"Plan inventory/capacity to the point estimate and hold contingency up to the upper band ({t['hi']:,.0f}) for high-service-level items; re-run weekly.",
            confidence=conf,
            risks=[
                "Promotions, price changes and one-off events are not modelled.",
                "Band is approximate: measured holdout coverage is below the nominal level (see tool calibration_note).",
                "Seasonal-naive needs >=91 weeks; structural breaks (like a stockout incident) inflate error.",
            ],
        ).model_dump()

    def _n_effect(self, q, plan, ledger) -> dict:
        ef = next((e for e in ledger if e.tool == "effect_estimate"), None)
        if not ef:
            return _abstain(q, "The effect-estimation step failed.")
        F: list[Finding] = []
        keep, drop = [], []
        for i, g in enumerate(ef.data["groups"]):
            F.append(Finding(text=f"{g['group']}: promo weeks sell {g['lift']:.2f}x baseline units (n={g['n_pairs']} store-SKU pairs).", evidence_id=ef.id, value_path=f"groups.{i}.lift", value=g["lift"]))
            F.append(Finding(text=f"{g['group']}: revenue-lift net of discount is {g['rev_lift']:.2f}x (break-even unit-lift is {g['breakeven_lift']:.2f}x).", evidence_id=ef.id, value_path=f"groups.{i}.rev_lift", value=g["rev_lift"]))
            (keep if g["rev_ci_low"] > 1.0 else drop).append(g["group"])
        answer = "Promotions lift units in every category, but only " + (", ".join(keep) if keep else "no category") + " gains revenue after the discount" + (f"; {', '.join(drop)} lose revenue on promo weeks." if drop else ".")
        return Memo(
            question=q,
            answer=answer,
            findings=F,
            recommendation=(f"Keep promotions in {', '.join(keep) or 'no category'}; stop or redesign (shallower discount, targeted) in {', '.join(drop) or 'none'}. "
                            "A discount only pays when unit-lift exceeds 1/(1-discount)."),
            confidence="medium",
            risks=[
                "Observational: promo timing is not randomised in real data, so lift can be confounded by targeting.",
                "Revenue-lift ignores margin, cannibalisation and post-promo dips; true profit impact is likely worse than revenue-lift.",
                "Bootstrap CIs resample store-SKU pairs and assume independence between them.",
            ],
        ).model_dump()

    def _n_profile(self, q, plan, ledger) -> dict:
        pr = next((e for e in ledger if e.tool == "profile_table"), None)
        if not pr:
            return _abstain(q, "Profiling failed.")
        F = [Finding(text=f"The table has {pr.data['rows']:,} rows.", evidence_id=pr.id, value_path="rows", value=float(pr.data["rows"]))]
        for col, pct in pr.data["null_pct"].items():
            F.append(Finding(text=f"Column '{col}' is {pct}% null.", evidence_id=pr.id, value_path=f"null_pct.{col}", value=pct))
        flags = pr.data["flags"]
        return Memo(
            question=q,
            answer="Data quality is good with minor gaps: " + "; ".join(flags) if flags else "No data-quality issues detected.",
            findings=F,
            recommendation="Impute or exclude the null columns before using them in models; add a null-rate check to the ingestion pipeline.",
            confidence="high",
            risks=["Profile checks structure, not business correctness."],
        ).model_dump()

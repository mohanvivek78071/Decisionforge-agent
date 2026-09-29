"""The critic: a non-LLM auditor that decides whether the memo is allowed to stand.

Check 1  grounding   - every finding points to a real number in the ledger and matches it
Check 2  recompute   - compare_periods evidence is re-derived through an independent pandas path
"""
from __future__ import annotations

from decisionforge.data.reference import period_delta
from decisionforge.schemas import Evidence, Memo, Verdict
from decisionforge.util import get_path


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= max(tol * abs(b), 1e-6)


class Critic:
    def __init__(self, store, tracer):
        self.store, self.tracer = store, tracer
        self._recompute_cache: dict[str, tuple[bool, str]] = {}

    def _recompute(self, ev: Evidence) -> tuple[bool, str]:
        if ev.id in self._recompute_cache:
            return self._recompute_cache[ev.id]
        ok, detail = True, "n/a"
        if ev.tool == "compare_periods" and ev.data.get("groups"):
            a = ev.args
            ref = period_delta(self.store.df, a["metric"], a["dimension"], a["window_weeks"], a.get("filters") or {})
            if ref.empty:
                ok, detail = False, "independent recompute returned no rows"
            else:
                adverse_rise = a["metric"] == "stockout_rate"
                r_worst = ref.index[-1] if adverse_rise else ref.index[0]
                r_delta = float(ref["delta"].iloc[-1] if adverse_rise else ref["delta"].iloc[0])
                w = ev.data["worst"]
                if str(r_worst) != w["group"] or not _close(w["delta"], r_delta, 1e-3):
                    ok, detail = False, f"independent recompute disagrees: worst={r_worst} delta={r_delta:.4f} vs ledger {w['group']} {w['delta']}"
                else:
                    detail = "independent pandas recompute agrees"
        self._recompute_cache[ev.id] = (ok, detail)
        return ok, detail

    def verify(self, memo: Memo, ledger: list[Evidence]) -> list[Verdict]:
        by_id = {e.id: e for e in ledger}
        out: list[Verdict] = []
        for i, f in enumerate(memo.findings):
            ev = by_id.get(f.evidence_id)
            if ev is None or not ev.ok:
                out.append(Verdict(finding_index=i, status="missing_evidence", detail=f"evidence '{f.evidence_id}' does not exist in the ledger"))
                continue
            actual = get_path(ev.data, f.value_path)
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                out.append(Verdict(finding_index=i, status="missing_evidence", detail=f"path '{f.value_path}' is not a number in {ev.id}"))
                continue
            if not _close(float(f.value), float(actual), f.tolerance):
                out.append(Verdict(finding_index=i, status="mismatch", detail=f"claimed {f.value} but {ev.id}.{f.value_path} = {actual}"))
                continue
            ok, why = self._recompute(ev)
            if not ok:
                out.append(Verdict(finding_index=i, status="mismatch", detail=why))
                continue
            out.append(Verdict(finding_index=i, status="verified", detail=f"{ev.id}.{f.value_path} = {actual}; {why}"))
        self.tracer.log("audit", verdicts=[v.model_dump() for v in out])
        return out

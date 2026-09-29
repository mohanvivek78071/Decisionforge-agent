from __future__ import annotations

import json
import time

from decisionforge.config import Budget
from decisionforge.schemas import Evidence, NextStep, Plan, Step
from decisionforge.tools import ToolError, run_tool


class Analyst:
    """Executes the plan, then keeps investigating adaptively until the budget or the question runs out."""

    def __init__(self, store, llm, budget: Budget, tracer):
        self.store, self.llm, self.budget, self.tracer = store, llm, budget, tracer
        self.ledger: list[Evidence] = []

    def _key(self, step: Step) -> str:
        return step.tool + json.dumps(step.args, sort_keys=True, default=str)

    def run_step(self, step: Step) -> Evidence:
        eid = f"E{len(self.ledger) + 1}"
        t0 = time.time()
        try:
            args, summary, data = run_tool(self.store, step.tool, step.args)
            ev = Evidence(id=eid, tool=step.tool, args=args, ok=True, summary=summary, data=data)
        except ToolError as e:  # tool failures are evidence too; the loop can react to them
            ev = Evidence(id=eid, tool=step.tool, args=step.args, ok=False, error=str(e), summary=f"FAILED: {e}")
        except Exception as e:  # noqa: BLE001
            ev = Evidence(id=eid, tool=step.tool, args=step.args, ok=False, error=f"{type(e).__name__}: {e}", summary=f"FAILED: {e}")
        ev.ms = round((time.time() - t0) * 1000, 1)
        self.ledger.append(ev)
        self.tracer.log("tool_call", id=eid, tool=ev.tool, args=ev.args, ok=ev.ok, summary=ev.summary, ms=ev.ms)
        return ev

    def execute(self, question: str, plan: Plan) -> list[Evidence]:
        seen: set[str] = set()
        for step in plan.steps:
            if len(self.ledger) >= self.budget.max_tool_calls:
                self.tracer.log("budget_exhausted", where="plan")
                return self.ledger
            seen.add(self._key(step))
            self.run_step(step)
        for _ in range(self.budget.max_followups):
            if len(self.ledger) >= self.budget.max_tool_calls:
                self.tracer.log("budget_exhausted", where="followups")
                break
            ns = self.llm.structured(
                "next_step",
                {"question": question, "plan": plan.model_dump(), "ledger": [e.model_dump() for e in self.ledger], "remaining_calls": self.budget.max_tool_calls - len(self.ledger)},
                NextStep,
            )
            self.tracer.log("next_step", action=ns.action, reason=ns.reason, step=ns.step.model_dump() if ns.step else None)
            if ns.action == "stop" or ns.step is None:
                break
            k = self._key(ns.step)
            if k in seen:  # loop guard: never repeat an identical call
                self.tracer.log("loop_guard", key=k)
                break
            seen.add(k)
            self.run_step(ns.step)
        return self.ledger

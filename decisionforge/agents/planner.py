from __future__ import annotations

from decisionforge.schemas import Plan
from decisionforge.tools import REGISTRY, ToolError, catalogue, run_tool  # noqa: F401
from pydantic import ValidationError


class Planner:
    def __init__(self, llm, tracer):
        self.llm, self.tracer = llm, tracer

    def plan(self, question: str, schema: dict) -> Plan:
        plan = self.llm.structured("plan", {"question": question, "schema": schema, "tools": catalogue()}, Plan)
        valid = []
        for s in plan.steps:  # never trust the model: drop unknown tools / invalid args
            t = REGISTRY.get(s.tool)
            if t is None:
                self.tracer.log("plan_step_dropped", tool=s.tool, reason="unknown tool")
                continue
            try:
                t.args_model(**s.args)
            except ValidationError as e:
                self.tracer.log("plan_step_dropped", tool=s.tool, reason=str(e)[:200])
                continue
            valid.append(s)
        plan.steps = valid
        if plan.intent != "unsupported" and not valid:
            plan.intent = "unsupported"
        self.tracer.log("plan", intent=plan.intent, rationale=plan.rationale, steps=[s.model_dump() for s in plan.steps])
        return plan

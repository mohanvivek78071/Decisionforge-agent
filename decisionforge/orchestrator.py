from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from decisionforge.agents.analyst import Analyst
from decisionforge.agents.critic import Critic
from decisionforge.agents.narrator import Narrator
from decisionforge.agents.planner import Planner
from decisionforge.config import Budget
from decisionforge.report import render_markdown
from decisionforge.schemas import Evidence, Memo, Plan, Verdict
from decisionforge.tracing import Tracer


@dataclass
class RunResult:
    question: str
    plan: Plan
    ledger: list[Evidence]
    memo: Memo
    verdicts: list[Verdict]
    trace: list[dict] = field(default_factory=list)
    run_dir: Path | None = None
    repaired: bool = False
    redacted: int = 0

    @property
    def grounding_rate(self) -> float:
        if not self.memo.findings:
            return 1.0
        return sum(v.status == "verified" for v in self.verdicts) / len(self.memo.findings)


class Orchestrator:
    """plan -> investigate (adaptive) -> narrate -> audit -> repair | redact -> report."""

    def __init__(self, store, llm, budget: Budget | None = None, runs_root: str | Path | None = None):
        self.store, self.llm = store, llm
        self.budget = budget or Budget()
        self.runs_root = Path(runs_root) if runs_root else None

    def run(self, question: str) -> RunResult:
        run_dir = None
        if self.runs_root:
            slug = re.sub(r"[^a-z0-9]+", "-", question.lower())[:40].strip("-") or "run"
            run_dir = self.runs_root / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"
        tracer = Tracer(run_dir)
        tracer.log("start", question=question, llm=self.llm.name)

        plan = Planner(self.llm, tracer).plan(question, self.store.schema_summary())
        analyst = Analyst(self.store, self.llm, self.budget, tracer)
        ledger = analyst.execute(question, plan) if plan.intent != "unsupported" else []
        narrator, critic = Narrator(self.llm, tracer), Critic(self.store, tracer)

        memo = narrator.write(question, plan, ledger)
        verdicts = critic.verify(memo, ledger)
        repaired, redacted = False, 0
        for attempt in range(self.budget.max_repair_rounds + 1):
            bad = [v for v in verdicts if v.status != "verified"]
            if not bad:
                break
            if attempt < self.budget.max_repair_rounds:
                feedback = [f"finding {v.finding_index} ({memo.findings[v.finding_index].text!r}): {v.detail}" for v in bad]
                tracer.log("repair", round=attempt + 1, feedback=feedback)
                memo = narrator.write(question, plan, ledger, feedback=feedback)
                verdicts = critic.verify(memo, ledger)
                repaired = True
            else:  # still failing: the unverified claims are removed, never shipped
                keep = [i for i, v in enumerate(verdicts) if v.status == "verified"]
                redacted = len(bad)
                memo.findings = [memo.findings[i] for i in keep]
                memo.confidence = "low"
                memo.risks = [f"{redacted} unverifiable claim(s) were removed by the auditor; treat this memo as incomplete."] + memo.risks
                verdicts = critic.verify(memo, ledger)
                tracer.log("redact", removed=redacted)

        result = RunResult(question, plan, ledger, memo, verdicts, tracer.events, run_dir, repaired, redacted)
        tracer.log("done", grounding=result.grounding_rate, tool_calls=len(ledger), confidence=memo.confidence)
        if run_dir:
            (run_dir / "memo.md").write_text(render_markdown(result), encoding="utf-8")
            (run_dir / "ledger.json").write_text(json.dumps([e.model_dump() for e in ledger], indent=2, default=str), encoding="utf-8")
            (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        return result

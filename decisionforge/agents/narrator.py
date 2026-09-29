from __future__ import annotations

from decisionforge.schemas import Evidence, Memo, Plan


class Narrator:
    def __init__(self, llm, tracer):
        self.llm, self.tracer = llm, tracer

    def write(self, question: str, plan: Plan, ledger: list[Evidence], feedback: list[str] | None = None) -> Memo:
        memo = self.llm.structured(
            "narrate",
            {"question": question, "plan": plan.model_dump(), "ledger": [e.model_dump() for e in ledger], "feedback": feedback or []},
            Memo,
        )
        memo.question = question
        self.tracer.log("memo_draft", findings=len(memo.findings), confidence=memo.confidence, abstained=memo.abstained, feedback=feedback or [])
        return memo

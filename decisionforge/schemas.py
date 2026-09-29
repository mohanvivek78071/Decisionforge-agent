from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Step(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    purpose: str = ""


class Plan(BaseModel):
    intent: Literal["diagnose", "forecast", "effect", "profile", "unsupported"]
    rationale: str = ""
    steps: list[Step] = Field(default_factory=list)


class NextStep(BaseModel):
    action: Literal["tool", "stop"]
    step: Step | None = None
    reason: str = ""


class Evidence(BaseModel):
    """One immutable entry in the evidence ledger (one tool call)."""

    id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    ms: float = 0.0


class Finding(BaseModel):
    """A claim that must point at a number inside the ledger."""

    text: str
    evidence_id: str
    value_path: str
    value: float
    tolerance: float = 0.01


class Memo(BaseModel):
    question: str = ""
    answer: str
    findings: list[Finding] = Field(default_factory=list)
    recommendation: str = ""
    confidence: Literal["high", "medium", "low"] = "low"
    risks: list[str] = Field(default_factory=list)
    abstained: bool = False


class Verdict(BaseModel):
    finding_index: int
    status: Literal["verified", "mismatch", "missing_evidence"]
    detail: str = ""

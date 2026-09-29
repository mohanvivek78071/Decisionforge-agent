from __future__ import annotations

ICON = {"verified": "✅", "mismatch": "❌", "missing_evidence": "⚠️"}


def render_markdown(r) -> str:
    m = r.memo
    L = [f"# Decision memo", "", f"**Question:** {r.question}", ""]
    if m.abstained:
        L += ["## Answer", m.answer, "", f"_{m.recommendation}_"]
        return "\n".join(L) + "\n"
    L += ["## Answer", m.answer, "", "## Evidence-backed findings"]
    status = {v.finding_index: v.status for v in r.verdicts}
    for i, f in enumerate(m.findings):
        L.append(f"- {ICON.get(status.get(i, ''), '•')} {f.text} `[{f.evidence_id}]`")
    L += ["", f"## Recommendation  _(confidence: {m.confidence})_", m.recommendation, "", "## Risks & assumptions"]
    L += [f"- {x}" for x in m.risks]
    ok = sum(v.status == "verified" for v in r.verdicts)
    L += ["", "## Audit trail", f"- Intent: `{r.plan.intent}` · tool calls: {len(r.ledger)} · findings verified: {ok}/{len(m.findings)}"]
    if r.repaired:
        L.append("- Memo was sent back to the narrator once after failing audit.")
    if r.redacted:
        L.append(f"- {r.redacted} unverifiable claim(s) were redacted.")
    for e in r.ledger:
        L.append(f"- `{e.id}` **{e.tool}** — {e.summary}")
    return "\n".join(L) + "\n"

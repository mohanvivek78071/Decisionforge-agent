# Interview guide: DecisionForge ↔ MathCo "AI Analyst" JD

> Read the code until you can explain every file in your own words. The best answers below are the ones where you say
> "I tried X, measured it, and changed it": the forecast-calibration story is a real one from this build.

## 1. JD requirement → where it lives in the repo

| JD line | Evidence in DecisionForge |
|---|---|
| Translate business problems into AI-solvable problem statements | `agents/planner.py`: question → typed plan (`diagnose / forecast / effect / profile / unsupported`) |
| Communicate to technical **and** non-technical audiences | Decision memo: plain-English answer up top, cited findings, recommendation, risks, full audit trail below |
| Agentic workflows and multi-step AI orchestration | `orchestrator.py` + `analyst.py`: plan → adaptive next-step loop → narrate → audit → repair/redact |
| Build/deploy ML models (forecasting, statistics) | `forecast` (rolling-origin backtest), `effect_estimate` (bootstrap CIs), `detect_anomalies` (robust z on shares) |
| Generative AI / LangChain-style tooling | Tool registry with pydantic schemas, provider-agnostic LLM interface, Claude client with schema validation + repair retry |
| Data pipelines: quality, availability, integrity | `profile_table` runs first on every analysis; null/duplicate/coverage checks |
| Feature engineering & model evaluation | Share-of-total features for anomalies; rolling-origin evaluation; eval suite vs planted truth |
| Production-ready, scalable systems | Budgets, loop guard, tracing, CI gate, Docker, typed interfaces, guarded SQL |
| Own end-to-end delivery, user-facing prototypes | CLI → Streamlit app → Docker; run artefacts on disk |
| Iterate on stakeholder feedback | Critic feedback loop is the same pattern at machine scale |
| Curiosity, "ask the question others don't" | Headline says +3.6% growth; the agent asks *where* and finds the incident hidden inside |
| Python, Spark, TensorFlow, LangChain | Python/DuckDB/pandas/pydantic here; be honest that Spark/LangChain are next (DuckDB → Spark adapter, tool registry → LangChain tools) |

## 2. Your 60-second pitch
"I noticed most AI-analyst demos produce fluent answers you have to trust. I built DecisionForge around the opposite rule:
no claim ships unless it's traceable. Agents plan an investigation, drill down adaptively, and write a memo where every number
cites a tool result. A non-LLM critic re-derives those numbers with a second implementation. If a claim fails, it goes back
once, and if it still fails it's removed. I evaluate it against synthetic data with planted truth. For example a hidden stockout
incident inside +3.6% overall growth, and it finds the exact region, category and stores. It runs offline in CI and with Claude."

## 3. Questions you should expect (and the honest answer)

1. **"Why not just let the LLM write SQL?"** Free-form SQL is one tool (`run_sql`), guarded and row-capped. The main path uses typed tools, so identifiers are whitelisted and values are bound. That removes a class of injection and makes results comparable and testable.
2. **"How do you know the LLM isn't hallucinating numbers?"** Structural, not hopeful: findings must cite `evidence_id + value_path`; the critic checks the number exists and matches, then recomputes it independently. Test: `test_critic_catches_hallucination_and_redacts`.
3. **"What if the critic and the tool share a bug?"** That's why they're different implementations (DuckDB SQL vs pandas). A shared *conceptual* error (e.g. wrong window definition) would still pass. Mitigation: eval against planted truth computed from the raw generator.
4. **"Your forecast intervals?"** Measured them: quantile bands covered ~43% of holdouts, so I widened to 2×RMS backtest error (~70%). Still under 80%, so the tool flags it. Next step is conformal prediction.
5. **"Correlation vs causation on stockouts?"** The memo says co-movement, lists it under risks, and states how to confirm (supplier/DC data). Confidence is `high` only when the hypothesis test *and* the anomaly check agree on the same stores.
6. **"How does it handle prompt injection?"** Untrusted text is tagged `<data>`; but the real defence is architectural: the model can't execute anything except typed tools, SQL is read-only and locked at engine level, and there's a test where the question itself says DROP TABLE.
7. **"How would you productionise it?"** Semantic layer for metric definitions, multi-table support, auth per tool, cost/latency budgets in the trace, human approval for expensive queries, eval-on-every-prompt-change in CI, monitoring grounding rate as a health metric.
8. **"What's the promo insight?"** A 15% discount needs a 1/0.85 = 1.18× unit lift to break even on revenue. Beverages lift 1.15× → units up, revenue down. Stats alone would call it "significant".

## 4. Make it yours (do at least two before you push)
- Point `--data` at data from one of your existing analytics projects and add a tool for it.
- Add one new tool (e.g. `price_elasticity`) and one eval case with planted truth.
- Run with `--provider anthropic`, compare its traces to the offline ones, and note in the README what differed.
- Add a screenshot/GIF of the Streamlit app to the README.

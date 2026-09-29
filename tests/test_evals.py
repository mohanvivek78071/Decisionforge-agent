from decisionforge.evals.run import evaluate


def test_offline_eval_suite_passes_completely():
    r = evaluate(__import__("decisionforge.llm", fromlist=["OfflineLLM"]).OfflineLLM())
    failed = [c for c in r["cases"] if not c["passed"]]
    assert not failed, failed
    assert r["grounding_rate"] == 1.0

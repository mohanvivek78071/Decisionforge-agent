from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from decisionforge.data.store import DataStore
from decisionforge.data.synth import generate
from decisionforge.llm import get_llm
from decisionforge.orchestrator import Orchestrator
from decisionforge.report import render_markdown


def load_df(path: str | None) -> pd.DataFrame:
    if path and Path(path).exists():
        return pd.read_csv(path, parse_dates=["week_start"])
    return generate()


def main(argv=None) -> int:
    default_provider = os.getenv("DECISIONFORGE_PROVIDER", "offline")
    p = argparse.ArgumentParser(prog="decisionforge", description="Verifiable multi-agent decision intelligence")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="ask a business question")
    a.add_argument("question")
    a.add_argument("--provider", default=default_provider, choices=["offline", "anthropic"])
    a.add_argument("--data", help="CSV with the same schema (default: built-in synthetic data)")
    a.add_argument("--runs", default="runs", help="directory for memo/ledger/trace artefacts ('' to disable)")
    a.add_argument("--json", action="store_true", help="print the memo as JSON instead of markdown")

    e = sub.add_parser("eval", help="run the evaluation suite against planted ground truth")
    e.add_argument("--provider", default=default_provider, choices=["offline", "anthropic"])
    e.add_argument("--json", action="store_true")

    d = sub.add_parser("gen-data", help="write the synthetic dataset to CSV")
    d.add_argument("--out", default="data/sales.csv")

    args = p.parse_args(argv)
    if args.cmd == "gen-data":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(args.out, index=False)
        print(f"wrote {args.out}")
        return 0
    if args.cmd == "eval":
        from decisionforge.evals.run import run_eval

        return run_eval(get_llm(args.provider), as_json=args.json)
    store = DataStore(load_df(args.data))
    res = Orchestrator(store, get_llm(args.provider), runs_root=args.runs or None).run(args.question)
    if args.json:
        print(json.dumps({"memo": res.memo.model_dump(), "verdicts": [v.model_dump() for v in res.verdicts]}, indent=2))
    else:
        print(render_markdown(res))
        if res.run_dir:
            print(f"artefacts: {res.run_dir}")
    return 0

"""DuckDB-backed data store with a read-only SQL guard.

Defence in depth for agent-written SQL:
  1. static guard (single SELECT/WITH statement, keyword + function deny-list)
  2. DuckDB external access disabled and configuration locked
  3. hard row cap on every result
"""
from __future__ import annotations

import re
from typing import Any

import duckdb
import pandas as pd

DIMENSIONS = ["region", "store_id", "category", "sku_id"]

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|pragma|install|load|"
    r"export|import|call|set|truncate|replace|vacuum|checkpoint|use)\b",
    re.I,
)
_EXTERNAL = re.compile(
    r"\b(read_\w+|glob|httpfs|parquet_scan|csv_scan|sniff_csv|getenv|current_setting|duckdb_\w+|pragma_\w+)\b",
    re.I,
)


class SQLGuardError(ValueError):
    pass


def guard_sql(sql: str) -> str:
    s = re.sub(r"--[^\n]*", " ", sql)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = s.strip().rstrip(";").strip()
    if not s:
        raise SQLGuardError("empty query")
    probe = re.sub(r"'[^']*'", "''", s)  # ignore string literals when scanning
    if ";" in probe:
        raise SQLGuardError("multiple statements are not allowed")
    if not re.match(r"(select|with)\b", probe, re.I):
        raise SQLGuardError("only SELECT / WITH queries are allowed")
    m = _FORBIDDEN.search(probe)
    if m:
        raise SQLGuardError(f"forbidden keyword: {m.group(0).lower()}")
    m = _EXTERNAL.search(probe)
    if m:
        raise SQLGuardError(f"forbidden function: {m.group(0).lower()}")
    return s


class DataStore:
    TABLE = "sales"

    def __init__(self, df: pd.DataFrame):
        d = df.copy()
        d["week_start"] = pd.to_datetime(d["week_start"])
        self.df = d
        try:
            self.con = duckdb.connect(":memory:", config={"enable_external_access": False, "lock_configuration": True})
        except Exception:
            self.con = duckdb.connect(":memory:")
        self.con.register("_src", d.assign(week_start=d["week_start"].dt.date))
        self.con.execute(f"CREATE TABLE {self.TABLE} AS SELECT * FROM _src")
        self.con.unregister("_src")
        try:
            self.con.execute("SET enable_external_access=false")
            self.con.execute("SET lock_configuration=true")
        except Exception:  # older DuckDB builds
            pass

    def query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Trusted internal query (built by our own tools with bound parameters)."""
        return self.con.execute(sql, params if params else None).df()

    def scalar(self, sql: str) -> Any:
        return self.con.execute(sql).fetchone()[0]

    def safe_query(self, sql: str, max_rows: int = 200) -> pd.DataFrame:
        """Untrusted query (LLM-written). Guarded and row-capped."""
        s = guard_sql(sql)
        return self.con.execute(f"SELECT * FROM ({s}) LIMIT {int(max_rows)}").df()

    def table_exists(self) -> bool:
        n = self.con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [self.TABLE]
        ).fetchone()[0]
        return n == 1

    def schema_summary(self) -> dict[str, Any]:
        cols = self.con.execute(f"DESCRIBE {self.TABLE}").fetchall()
        values = {d: sorted(self.df[d].unique().tolist()) for d in DIMENSIONS if d != "sku_id"}
        return {
            "table": self.TABLE,
            "columns": {c[0]: c[1] for c in cols},
            "values": values,
            "n_skus": int(self.df["sku_id"].nunique()),
            "date_range": [str(self.df["week_start"].min().date()), str(self.df["week_start"].max().date())],
            "metrics": ["revenue", "units", "stockout_rate", "avg_price"],
            "grain": "one row per week x store x sku",
        }

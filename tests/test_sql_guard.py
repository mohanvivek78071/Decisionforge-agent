import pytest

from decisionforge.data.store import SQLGuardError, guard_sql


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales",
        "select 1; drop table sales",
        "SELECT * FROM sales; DELETE FROM sales",
        "update sales set units = 0",
        "create table x as select * from sales",
        "COPY sales TO '/tmp/x.csv'",
        "ATTACH '/tmp/evil.db'",
        "select * from read_csv('/etc/passwd')",
        "select * from glob('/*')",
        "PRAGMA database_list",
        "with x as (select 1) insert into sales select * from x",
        "SET enable_external_access=true",
        "",
    ],
)
def test_blocks_dangerous_sql(sql):
    with pytest.raises(SQLGuardError):
        guard_sql(sql)


def test_allows_select_and_cte_and_string_literals():
    assert guard_sql("SELECT region, SUM(revenue) FROM sales GROUP BY 1;").startswith("SELECT")
    assert guard_sql("WITH a AS (SELECT * FROM sales) SELECT count(*) FROM a").startswith("WITH")
    assert guard_sql("select * from sales where category = 'drop; set'")  # keywords inside literals are fine
    assert guard_sql("select 1 -- drop table sales")  # comments are stripped


def test_row_cap(store):
    assert len(store.safe_query("select * from sales", max_rows=7)) == 7


def test_engine_level_lockdown(store):
    with pytest.raises(Exception):
        store.con.execute("select * from read_csv('/etc/passwd')").fetchall()

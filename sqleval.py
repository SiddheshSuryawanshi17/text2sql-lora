"""SQL evaluation helpers: normalisation, exact match and execution accuracy.

Exact match is too strict ("SELECT a FROM t WHERE x = 'y'" vs "... WHERE \"x\" = 'y'"
are the same query). Execution accuracy runs both queries on a database and
compares the results, which is how Spider and BIRD benchmarks score models.

The dataset ships schemas only, not data. So we build a small SQLite database
per example and fill it with rows that are *designed to make queries
distinguishable*: every string and number literal that appears in the gold SQL
is planted in the data, mixed with random distractor values. A wrong WHERE
clause, column or aggregate then returns a different result. This is an
approximation of the "test-suite accuracy" idea (Zhong et al., 2020); it
can occasionally accept a wrong query, never reject a correct one.
"""

from __future__ import annotations

import random
import re
import sqlite3
from collections import Counter

import sqlglot
from sqlglot import exp

# string literals in single OR double quotes (this dataset uses "..."), and numbers
LITERAL = re.compile(r"'((?:[^']|'')*)'|\"([^\"]*)\"|\b(\d+(?:\.\d+)?)\b")
TABLE = re.compile(r"CREATE TABLE\s+(\w+)\s*\(([^)]*)\)", re.I)


def normalize(sql: str) -> str:
    """Canonical form for exact match: parse and re-print with sqlglot, lowercase."""
    sql = sql.strip().rstrip(";")
    try:
        return sqlglot.transpile(sql, read="sqlite", write="sqlite")[0].lower()
    except Exception:
        return re.sub(r"\s+", " ", sql).lower()


def extract_sql(text: str) -> str:
    """Models sometimes wrap SQL in ``` fences or add a sentence; keep the query."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1)
    m = re.search(r"\b(SELECT|WITH)\b.*", text, re.S | re.I)
    text = m.group(0) if m else text
    return text.strip().split(";")[0].strip()


def _conditions(gold_sql: str) -> list[tuple[str, str, str]]:
    """(column, operator, literal) triples from the WHERE clause of the gold query."""
    try:
        tree = sqlglot.parse_one(gold_sql, read="sqlite")
    except Exception:
        return []
    out = []
    for node in tree.find_all(exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE):
        left, right = node.left, node.right
        if isinstance(left, exp.Column) and isinstance(right, (exp.Literal, exp.Column)):
            # the dataset writes "text" in double quotes, which sqlglot parses as a Column
            out.append((left.name.lower(), node.key, right.name if isinstance(right, exp.Column) else right.this))
    return out


def _value_for(op: str, lit: str, satisfy: bool) -> str:
    try:
        num = float(lit)
    except ValueError:
        return lit if satisfy else f"{lit}_x"
    shift = {"eq": 0, "gt": 1, "gte": 0, "lt": -1, "lte": 0}[op]
    if not satisfy:
        shift = {"eq": 7, "gt": -1, "gte": -1, "lt": 1, "lte": 1}[op]
    v = num + shift
    return str(int(v)) if v == int(v) else str(v)


def build_db(schema: str, gold_sql: str, seed: int = 0, rows: int = 24) -> sqlite3.Connection:
    """Random rows + rows that satisfy all gold conditions (witnesses) + rows that
    break exactly one condition (near misses). Wrong filters then change the result."""
    rng = random.Random(seed)
    literals = [a.replace("''", "'") if a else (b or c) for a, b, c in LITERAL.findall(gold_sql)]
    pool = literals + [str(rng.randint(0, 2000)) for _ in range(10)] + [f"val_{i}" for i in range(10)]
    conds = _conditions(gold_sql)
    con = sqlite3.connect(":memory:")
    for table, cols in TABLE.findall(schema):
        names = [c.strip().split()[0] for c in cols.split(",") if c.strip()]
        con.execute(f'CREATE TABLE {table} ({", ".join(f"[{n}] TEXT" for n in names)})')
        lower = [n.lower() for n in names]

        def row(broken: int | None = None) -> list[str]:
            vals = [rng.choice(pool) if rng.random() < 0.6 else str(rng.randint(0, 50)) for _ in names]
            for i, (col, op, lit) in enumerate(conds):
                if col in lower:
                    vals[lower.index(col)] = _value_for(op, lit, satisfy=(i != broken))
            return vals

        batch = [row() for _ in range(3)]                                   # witnesses
        batch += [row(broken=i) for i in range(len(conds)) for _ in range(2)]  # near misses
        batch += [[rng.choice(pool) if rng.random() < 0.6 else str(rng.randint(0, 50)) for _ in names] for _ in range(rows)]
        con.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in names)})", batch)
    return con


def run(con: sqlite3.Connection, sql: str):
    try:
        return Counter(map(tuple, con.execute(sql).fetchall())), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def gold_is_usable(schema: str, gold: str) -> bool:
    """Some dataset rows have schemas/SQL that SQLite can't run; those are excluded."""
    try:
        g, err = run(build_db(schema, gold, 0), gold)
    except sqlite3.Error:
        return False
    return err is None


def score(schema: str, gold: str, pred: str) -> dict:
    exact = normalize(pred) == normalize(gold)
    valid, same = True, True
    for seed in (0, 1, 2):                       # three different databases
        con = build_db(schema, gold, seed)
        g, _ = run(con, gold)
        p, err = run(con, pred)
        if err:
            valid, same = False, False
            break
        if g != p:
            same = False
    return {"exact": exact, "exec": same, "valid": valid}

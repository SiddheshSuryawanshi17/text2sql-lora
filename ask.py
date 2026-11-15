"""Turn a question into SQL with the fine-tuned model, and optionally run it on a SQLite DB.

    python ask.py --schema "CREATE TABLE employees (name VARCHAR, dept VARCHAR, salary INT)" \\
                  "Who earns the most in the sales department?"

    python ask.py --db shop.sqlite "How many orders were placed in 2024?"   # schema read from the DB
"""

from __future__ import annotations

import argparse
import sqlite3

from mlx_lm import generate, load

from evaluate import BASE, prompt_for
from sqleval import extract_sql


def schema_from_db(path: str) -> str:
    con = sqlite3.connect(path)
    rows = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    return "\n".join(r[0] for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--schema", help="CREATE TABLE statement(s)")
    ap.add_argument("--db", help="SQLite file: schema is read from it and the query is executed")
    ap.add_argument("--adapter", default="adapters", help="'none' to use the base model")
    args = ap.parse_args()

    schema = schema_from_db(args.db) if args.db else args.schema
    if not schema:
        ap.error("give --schema or --db")
    model, tok = load(BASE, adapter_path=None if args.adapter == "none" else args.adapter)
    sql = extract_sql(generate(model, tok, prompt=prompt_for(tok, schema, args.question), max_tokens=160, verbose=False))
    print(sql)
    if args.db:
        for row in sqlite3.connect(args.db).execute(sql).fetchmany(20):
            print("  ", row)


if __name__ == "__main__":
    main()

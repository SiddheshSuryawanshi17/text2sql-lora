"""Compare the base model and the LoRA fine-tuned model on the held-out test set.

    python evaluate.py --adapter none          # base model
    python evaluate.py --adapter adapters      # fine-tuned
    python evaluate.py --limit 50              # quick run

Metrics
- exact match       normalised SQL text equals the reference
- execution acc.    returns the same rows as the reference on 3 synthetic databases
- valid SQL         runs without an error
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlx_lm import generate, load

from prepare_data import SYSTEM
from sqleval import extract_sql, gold_is_usable, score

ROOT = Path(__file__).parent
BASE = "Qwen/Qwen2.5-0.5B-Instruct"


def prompt_for(tokenizer, schema: str, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--adapter", default="adapters", help="adapter dir, or 'none' for the base model")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    adapter = None if args.adapter == "none" else args.adapter
    model, tokenizer = load(args.model, adapter_path=adapter)
    tests = [json.loads(l) for l in open(ROOT / "data" / "test_raw.jsonl")]
    tests = [t for t in tests if gold_is_usable(t["context"], t["answer"])]
    if args.limit:
        tests = tests[: args.limit]

    rows, t0 = [], time.time()
    for i, t in enumerate(tests, start=1):
        raw = generate(model, tokenizer, prompt=prompt_for(tokenizer, t["context"], t["question"]), max_tokens=128, verbose=False)
        pred = extract_sql(raw)
        rows.append({**t, "pred": pred, **score(t["context"], t["answer"], pred)})
        if i % 25 == 0:
            print(f"  {i}/{len(tests)}  exec acc so far {sum(r['exec'] for r in rows) / i:.1%}")

    n = len(rows)
    summary = {
        "model": args.model,
        "adapter": adapter or "none (base model)",
        "examples": n,
        "exact_match": round(100 * sum(r["exact"] for r in rows) / n, 1),
        "execution_accuracy": round(100 * sum(r["exec"] for r in rows) / n, 1),
        "valid_sql": round(100 * sum(r["valid"] for r in rows) / n, 1),
        "seconds_per_query": round((time.time() - t0) / n, 2),
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tag = "base" if adapter is None else Path(adapter).name
    (out / f"eval_{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

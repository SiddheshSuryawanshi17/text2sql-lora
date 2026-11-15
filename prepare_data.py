"""Build train/valid splits (+ raw test set) in the chat format mlx-lm expects.

Dataset: b-mc2/sql-create-context (78k examples, derived from WikiSQL + Spider).
Each example gives a natural-language question, the CREATE TABLE schema, and the
correct SQL. We hold out a fixed test set that the model never sees in training.

    python prepare_data.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

DATA = Path(__file__).parent / "data"
SYSTEM = "You translate questions into a single SQLite query. Reply with SQL only, no explanation."
SEED, N_TRAIN, N_VALID, N_TEST = 42, 6000, 300, 300


def to_chat(ex: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Schema:\n{ex['context']}\n\nQuestion: {ex['question']}"},
            {"role": "assistant", "content": ex["answer"].strip()},
        ]
    }


def main() -> None:
    ds = load_dataset("b-mc2/sql-create-context", split="train")
    rows = [r for r in ds if len(r["context"]) < 1200]          # keep prompts short
    random.Random(SEED).shuffle(rows)
    test = rows[:N_TEST]
    valid = rows[N_TEST : N_TEST + N_VALID]
    train = rows[N_TEST + N_VALID : N_TEST + N_VALID + N_TRAIN]

    DATA.mkdir(exist_ok=True)
    for name, split in [("train", train), ("valid", valid)]:
        with open(DATA / f"{name}.jsonl", "w") as f:
            for ex in split:
                f.write(json.dumps(to_chat(ex)) + "\n")
    # test keeps the raw fields so the evaluator can build databases from the schema
    with open(DATA / "test_raw.jsonl", "w") as f:
        for ex in test:
            f.write(json.dumps({"question": ex["question"], "context": ex["context"], "answer": ex["answer"]}) + "\n")
    print(f"train {len(train)}  valid {len(valid)}  test {len(test)}  (from {len(rows):,} usable rows)")


if __name__ == "__main__":
    main()

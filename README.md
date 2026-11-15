# Text-to-SQL with LoRA

**Fine-tuning a 0.5B-parameter LLM to write SQL on a MacBook, measured by actually running the queries.**
LoRA on Qwen2.5-0.5B-Instruct with Apple MLX: 6 minutes of training, 0.29% of the weights,
and execution accuracy goes from **37.3% to 83.9%** on 292 held-out questions.

## Results

| Metric (292 held-out questions) | Base model | **+ LoRA** |
|---|---|---|
| Execution accuracy | 37.3% | **83.9%** |
| Exact match (normalised) | 14.0% | **80.1%** |
| Valid SQL (runs without error) | 84.2% | **97.6%** |
| Seconds per query (M4 Pro) | 0.14 | 0.17 |

| Training | |
|---|---|
| Trainable parameters | **1.44M of 494M (0.29%)** · adapter file 5.8 MB |
| Data | 6,000 examples from `b-mc2/sql-create-context` (WikiSQL + Spider) |
| Steps | 800 × batch 8 (~1.1 epochs), loss on SQL tokens only |
| Validation loss | 2.322 → 0.051 |
| Time / peak memory | ~6 min / 4.7 GB on Apple M4 Pro |

### Example

```bash
$ python ask.py --db examples/shop.sqlite "What is the total order amount for customers from Pune?"
SELECT SUM(amount) FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE city = 'Pune')
   (70790.24,)        # correct: matches the hand-written JOIN query
```

### Where it still fails (47 of 292)
- **Missing `DISTINCT`**: "how many *kinds* of roles" → `COUNT(*)` instead of `COUNT(DISTINCT role_code)`
- **Join keys** on multi-table questions: picks the wrong column to join on
- **Comparison operators**: "later than 1997" → `= 1997` instead of `> 1997`
- **Multi-table schemas it wasn't trained on**: on the demo DB, "How many orders were placed in 2025?"
  produced `... WHERE year = 2025 AND customer_id = (SELECT id FROM customers WHERE name = "2025")` (wrong:
  invents a condition). Most training data is single-table.
- **Label noise**: a few reference answers in the dataset are themselves wrong (e.g. a broken column name
  `murder__2012__rate_per_100, 000`), which puts a ceiling on measurable accuracy.

## How the evaluation works (the part worth reading)

Exact string match undercounts correct SQL, so the main metric is **execution accuracy**: run the
predicted and reference query and compare the returned rows. The dataset has schemas but **no data**,
so `sqleval.py` generates databases that are built to separate right from wrong queries:

- **witness rows** satisfying every condition in the reference query's `WHERE` clause (parsed with `sqlglot`)
- **near-miss rows** that break exactly one condition
- random filler; each query is checked on **3** different databases

**The scorer was tested too**: deleting one `AND` condition from 68 correct queries was detected **68/68** times,
and every reference query matches itself (292/292). Rows whose reference SQL can't run in SQLite (8 of 300) are excluded.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python prepare_data.py                         # download + split the dataset
python evaluate.py --adapter none              # baseline (~1 min)
mlx_lm.lora --config lora_config.yaml          # fine-tune (~6 min on M-series)
python evaluate.py --adapter adapters          # fine-tuned
python ask.py --schema "CREATE TABLE emp (name VARCHAR, dept VARCHAR, salary INT)" "Who earns the most in sales?"
```

The trained adapter is included (`adapters/adapters.safetensors`, 5.8 MB), so `ask.py` works without training.
Requires an Apple Silicon Mac (MLX). Merge it into the base model with `mlx_lm.fuse --model Qwen/Qwen2.5-0.5B-Instruct --adapter-path adapters`.

## Files
| File | Purpose |
|---|---|
| `prepare_data.py` | chat-formatted train/valid splits + raw held-out test set (seeded) |
| `lora_config.yaml` | all training hyperparameters, with comments |
| `sqleval.py` | SQL normalisation, adversarial database builder, scoring |
| `evaluate.py` | runs a model over the test set and writes `results/` |
| `ask.py` | question → SQL, optionally executed on your SQLite file |

See **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)** for LoRA explained and likely interview questions.

## License
MIT. Base model: Qwen2.5-0.5B-Instruct (Apache-2.0). Dataset: b-mc2/sql-create-context (CC-BY-4.0).

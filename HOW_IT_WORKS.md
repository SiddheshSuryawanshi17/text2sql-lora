# How it works (and how to explain it in an interview)

## 1. The task
Text-to-SQL: given a database schema and a question in English, write the SQL that answers it.
It's a classic LLM application (analytics chatbots, BI tools) with a big advantage for learning:
**correctness is checkable**. You can run the SQL.

## 2. Why fine-tune instead of prompting?
A 0.5B model is cheap and fast enough to run anywhere, but out of the box it adds
explanations, uses functions SQLite doesn't have, and misreads which column to filter.
Fine-tuning teaches it the *format* and the *habits* of the task from a few thousand
examples. A bigger model with a long prompt might do as well, but costs far more per query.

## 3. LoRA in one picture
Full fine-tuning updates every weight matrix W (millions of numbers each).
**LoRA** (Hu et al., 2021) freezes W and learns a low-rank update:

```
W' = W + (alpha / r) · B·A        B: d×r,  A: r×d,  r ≪ d   (here r = 16, d = 896)
```

- Trainable parameters drop ~340×: **1.44M of 494M (0.29%)** in this project. Only A and B are trained.
- The base model stays untouched; the adapter is a small separate file (a few MB).
- At inference you can **merge** B·A into W (`mlx_lm.fuse`), so there is zero extra latency.
- We adapt the attention projections (q, k, v, o) in the top 16 layers. The LoRA paper found
  attention weights are where most of the benefit is.

## 4. Training details (`lora_config.yaml`)
| Setting | Value | Why |
|---|---|---|
| Base model | Qwen2.5-0.5B-Instruct | Small, strong at code for its size, Apache-2.0 |
| Framework | Apple MLX (`mlx-lm`) | Fast training on the Mac's unified memory/GPU |
| Data | 6,000 examples from `sql-create-context` | WikiSQL + Spider derived; schema + question + SQL |
| `mask_prompt: true` | loss only on the SQL tokens | The model shouldn't waste capacity learning to repeat the question |
| rank / scale / dropout | 16 / 2.0 / 0.05 | Common defaults; enough capacity for a narrow task |
| LR / batch / iters | 1e-4 / 8 / 800 (~1 epoch) | One pass is enough for a narrow task and limits overfitting |

## 5. Evaluation (the most important part)
- **Held-out test set**: 300 examples never used in training (292 after removing rows whose
  reference SQL can't run in SQLite).
- **Exact match**: normalised with `sqlglot` so formatting and quoting differences don't count as errors.
- **Execution accuracy**: run the predicted and reference SQL on databases and compare the rows.
  The dataset has schemas but **no data**, so `sqleval.py` builds databases designed to tell queries apart:
  - *witness rows* that satisfy every condition of the reference query
  - *near-miss rows* that break exactly one condition
  - random rows

  A query with a missing or wrong condition returns different rows. This was verified with a
  **mutation test**: removing one `AND` condition from 68 correct queries was caught 68/68 times.
- **Valid SQL rate**: how often the output runs at all.

## 6. Likely interview questions

**Why LoRA and not full fine-tuning?**
Memory and speed: you don't store optimiser state for every weight. Also portability: one base
model, many small adapters per task. Quality is close to full fine-tuning on narrow tasks.

**What does the rank r control?**
The capacity of the update. Too low and it can't express the change; too high and it costs more
and can overfit. For a narrow task, 8–16 is typical.

**Why execution accuracy instead of exact match?**
Many different SQL strings are equivalent. Exact match punishes harmless differences; execution
compares behaviour. Its weakness: two wrong queries can coincidentally return the same rows,
which is why the databases are built adversarially (witness and near-miss rows).

**How do you know the model didn't just memorise?**
The test set is disjoint from training and fixed by a seed, and the schemas are mostly unique tables.

**What would you do next?**
Train on Spider/BIRD (multi-table joins), add constrained decoding so output always parses,
self-consistency (sample several SQLs, pick the most common execution result), and
schema linking for very large databases.

**How would you serve it?**
Fuse the adapter (`mlx_lm.fuse`), quantize to 4-bit, and serve with `mlx_lm.server`, which offers an
OpenAI-compatible API. Or export to GGUF for llama.cpp/Ollama on other hardware.

## 7. Glossary
- **Adapter**: the small set of LoRA weights (A, B) trained on top of a frozen model.
- **Instruct model**: a base model further trained to follow chat instructions.
- **Masking the prompt**: excluding input tokens from the loss so only the answer is learned.
- **MLX**: Apple's array framework for machine learning on Apple Silicon.

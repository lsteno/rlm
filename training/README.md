# Training pipeline utilities

This folder contains lightweight utilities for two steps:

1. selecting a diverse seed set from [data/train.parquet](../data/train.parquet)
2. generating recursive trajectories that can later be converted into SFT data

## Install

We keep training dependencies out of the core package. Install the extra group with `uv`:

```bash
uv sync --group training
```

## 1. Select a diverse seed set

This creates a balanced sample across dataset/task families, then uses metadata and
length-aware greedy selection inside each family.

```bash
.venv/bin/python training/select_diverse_tasks.py \
  --input data/train.parquet \
  --output data/sft_seed_tasks_250.jsonl \
  --summary-output data/sft_seed_tasks_250_summary.json \
  --count 250
```

## 2. Generate recursive trajectories

Use the extracted task file as input. The script:

- runs each task with `max_depth=2`
- passes the large context as the RLM context payload
- passes the question as `root_prompt`
- saves the full completion tree, including nested child metadata when available
- emits flattened step-level SFT examples for every model output

```bash
export OPENROUTER_API_KEY=...

.venv/bin/python training/generate_sft_trajectories.py \
  --tasks data/sft_seed_tasks_250.jsonl \
  --output-dir data/sft_runs/openrouter_qwen3_32b \
  --backend openrouter \
  --model-name qwen/qwen3-32b \
  --max-depth 2 \
  --max-iterations 10
```

The output directory contains:

- `runs/<task_id>.json`: one fully serialized trajectory tree per task
- `run_summaries.jsonl`: one summary line per task
- `sft_examples.jsonl`: flattened step-level SFT examples across all tasks

These files are designed to let you keep the raw recursive tree while also having
an immediately usable SFT-style dataset.
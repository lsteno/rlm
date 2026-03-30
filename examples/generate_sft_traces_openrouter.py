import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.prompts import RLM_SYSTEM_PROMPT, RLM_SYSTEM_PROMPT_B


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RLM traces from a parquet SFT dataset using OpenRouter."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sft_traces.parquet",
        help="Path to input parquet dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated_sft_traces.jsonl",
        help="Path to output JSONL with enriched traces.",
    )
    parser.add_argument(
        "--raw-log-dir",
        type=str,
        default="logs/sft_traces_raw",
        help="Directory for raw per-run logger JSONL files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="minimax/minimax-m2.5",
        help="OpenRouter model name.",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        choices=["A", "B"],
        default="A",
        help="System prompt variant: A=default, B=short/dense.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to process.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Row offset to start from.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="RLM max depth.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="RLM max iterations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose RLM console output.",
    )
    return parser.parse_args()


def _stringify_answer(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list):
        if len(answer) == 1:
            return str(answer[0])
        return ", ".join(str(item) for item in answer)
    return str(answer)


def _extract_label_like_answer(text: str) -> str:
    label_match = re.search(r"label\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if label_match:
        return label_match.group(1).strip()
    return text.strip()


def _normalize_answer(text: str) -> str:
    normalized = text.strip().strip("[]\"'")
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _build_trace_record(
    row: dict[str, Any],
    completion_response: str,
    trajectory: dict[str, Any] | None,
) -> dict[str, Any]:
    correct_answer = _stringify_answer(row.get("answer"))
    model_answer = _extract_label_like_answer(completion_response)

    normalized_model_answer = _normalize_answer(model_answer)
    normalized_correct_answer = _normalize_answer(_extract_label_like_answer(correct_answer))

    return {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "task": row.get("task"),
        "prompt": row.get("prompt"),
        "context_token_count": row.get("context_token_count"),
        "answer_highlights": {
            "model_answer": model_answer,
            "correct_answer": correct_answer,
            "is_exact_match": normalized_model_answer == normalized_correct_answer,
        },
        "model_response_raw": completion_response,
        "trace": trajectory,
    }


def main() -> None:
    load_dotenv()
    args = parse_args()

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is required in environment or .env")

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    raw_log_dir = Path(args.raw_log_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_log_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(dataset_path)
    rows = table.to_pylist()

    start = max(args.offset, 0)
    end = len(rows) if args.limit is None else min(len(rows), start + args.limit)
    selected_rows = rows[start:end]

    if not selected_rows:
        raise ValueError("No rows selected. Check --offset/--limit and dataset size.")

    print(f"Loaded {len(rows)} rows from {dataset_path}")
    print(f"Processing rows [{start}:{end}] -> {len(selected_rows)} rows")
    print(f"Output JSONL: {output_path}")
    print(f"Raw logger JSONL dir: {raw_log_dir}")
    print(f"Prompt variant: {args.prompt_variant}")

    system_prompt = RLM_SYSTEM_PROMPT if args.prompt_variant == "A" else RLM_SYSTEM_PROMPT_B

    with output_path.open("w", encoding="utf-8") as outfile:
        for i, row in enumerate(selected_rows, start=start):
            logger = RLMLogger(log_dir=str(raw_log_dir), file_name=f"trace_{i:06d}")
            rlm = RLM(
                backend="openrouter",
                backend_kwargs={
                    "model_name": args.model,
                    "api_key": openrouter_api_key,
                },
                environment="local",
                environment_kwargs={},
                max_depth=args.max_depth,
                max_iterations=args.max_iterations,
                custom_system_prompt=system_prompt,
                logger=logger,
                verbose=args.verbose,
            )

            try:
                context = row.get("context")
                question = row.get("prompt")
                completion = rlm.completion(context, root_prompt=question)

                record = _build_trace_record(
                    row=row,
                    completion_response=completion.response,
                    trajectory=completion.metadata,
                )
                record["status"] = "ok"
                record["prompt_variant"] = args.prompt_variant
            except Exception as exc:
                record = {
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "task": row.get("task"),
                    "prompt": row.get("prompt"),
                    "answer_highlights": {
                        "model_answer": "",
                        "correct_answer": _stringify_answer(row.get("answer")),
                        "is_exact_match": False,
                    },
                    "status": "error",
                    "prompt_variant": args.prompt_variant,
                    "error": str(exc),
                    "trace": logger.get_trajectory(),
                }
            finally:
                rlm.close()

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{i + 1}/{end}] id={record.get('id')} status={record.get('status')} "
                f"match={record.get('answer_highlights', {}).get('is_exact_match')}"
            )

    print("Trace generation complete.")


if __name__ == "__main__":
    main()

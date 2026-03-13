from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from rlm import RLM
from rlm.logger import RLMLogger
from training.utils import append_jsonl, compare_answers, read_jsonl


def ensure_message_list(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        return prompt
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False, default=str)}]


def serialize_completion_tree(completion: Any) -> dict[str, Any]:
    if hasattr(completion, "to_dict"):
        return completion.to_dict()
    if isinstance(completion, dict):
        return completion
    raise TypeError(f"Unsupported completion type: {type(completion)}")


def flatten_sft_examples(
    completion: dict[str, Any],
    task: dict[str, Any],
    *,
    depth: int = 0,
    node_path: str = "root",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata = completion.get("metadata")

    if metadata and metadata.get("iterations"):
        for iteration_index, iteration in enumerate(metadata["iterations"], start=1):
            rows.append(
                {
                    "task_id": task["id"],
                    "dataset": task["dataset"],
                    "task_type": task["task"],
                    "depth": depth,
                    "node_path": node_path,
                    "node_kind": "rlm_iteration",
                    "iteration_index": iteration_index,
                    "root_model": completion.get("root_model"),
                    "messages": ensure_message_list(iteration.get("prompt")),
                    "assistant_response": iteration.get("response", ""),
                    "final_answer": iteration.get("final_answer"),
                }
            )

            for code_index, code_block in enumerate(iteration.get("code_blocks", []), start=1):
                repl_result = code_block.get("result", {})
                for subcall_index, subcall in enumerate(repl_result.get("rlm_calls", []), start=1):
                    child_path = (
                        f"{node_path}.iter{iteration_index}.code{code_index}.sub{subcall_index}"
                    )
                    rows.extend(
                        flatten_sft_examples(
                            subcall,
                            task,
                            depth=depth + 1,
                            node_path=child_path,
                        )
                    )
    else:
        rows.append(
            {
                "task_id": task["id"],
                "dataset": task["dataset"],
                "task_type": task["task"],
                "depth": depth,
                "node_path": node_path,
                "node_kind": "leaf_completion",
                "iteration_index": 1,
                "root_model": completion.get("root_model"),
                "messages": ensure_message_list(completion.get("prompt", "")),
                "assistant_response": completion.get("response", ""),
                "final_answer": completion.get("response", ""),
            }
        )

    return rows


def run_task(
    task: dict[str, Any],
    *,
    backend: str,
    model_name: str,
    api_key: str | None,
    max_depth: int,
    max_iterations: int,
    max_timeout: float | None,
    verbose: bool,
) -> dict[str, Any]:
    logger = RLMLogger()
    backend_kwargs: dict[str, Any] = {"model_name": model_name}
    if api_key:
        backend_kwargs["api_key"] = api_key

    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_timeout=max_timeout,
        logger=logger,
        verbose=verbose,
    )

    try:
        result = rlm.completion(task["context"], root_prompt=task["prompt"])
        completion_tree = serialize_completion_tree(result)
        evaluation = compare_answers(result.response, task["answer"])
        sft_examples = flatten_sft_examples(completion_tree, task)
        return {
            "status": "ok",
            "task": task,
            "result": completion_tree,
            "evaluation": evaluation,
            "sft_examples": sft_examples,
        }
    finally:
        rlm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate max-depth-2 recursive trajectories.")
    parser.add_argument("--tasks", type=Path, required=True, help="Input task JSONL file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--backend", default="openrouter", help="RLM backend.")
    parser.add_argument("--model-name", required=True, help="Model name for the backend.")
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum recursion depth.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iterations per node.",
    )
    parser.add_argument(
        "--max-timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds per root run.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of tasks to skip from the start (for batching).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of tasks to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks whose per-run output file already exists.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose RLM output.")
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    if args.offset:
        tasks = tasks[args.offset :]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    output_dir = args.output_dir
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    summaries_path = output_dir / "run_summaries.jsonl"
    sft_examples_path = output_dir / "sft_examples.jsonl"

    api_key = os.getenv(args.api_key_env)

    for index, task in enumerate(tasks, start=1):
        run_path = runs_dir / f"{task['id']}.json"
        if args.resume and run_path.exists():
            print(f"[{index}/{len(tasks)}] skipping {task['id']} (already exists)")
            continue

        print(f"[{index}/{len(tasks)}] running {task['id']}")
        try:
            payload = run_task(
                task,
                backend=args.backend,
                model_name=args.model_name,
                api_key=api_key,
                max_depth=args.max_depth,
                max_iterations=args.max_iterations,
                max_timeout=args.max_timeout,
                verbose=args.verbose,
            )
            run_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

            append_jsonl(
                summaries_path,
                {
                    "task_id": task["id"],
                    "dataset": task["dataset"],
                    "task_type": task["task"],
                    "status": payload["status"],
                    "response": payload["result"]["response"],
                    "evaluation": payload["evaluation"],
                    "n_sft_examples": len(payload["sft_examples"]),
                },
            )
            for example in payload["sft_examples"]:
                append_jsonl(sft_examples_path, example)
        except Exception as exc:
            error_payload = {
                "status": "error",
                "task": task,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            run_path.write_text(json.dumps(error_payload, indent=2, ensure_ascii=False))
            append_jsonl(
                summaries_path,
                {
                    "task_id": task["id"],
                    "dataset": task["dataset"],
                    "task_type": task["task"],
                    "status": "error",
                    "error": str(exc),
                },
            )


if __name__ == "__main__":
    main()
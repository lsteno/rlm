import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rlm.clients import get_client


SYSTEM_PROMPT = """You are an expert semantic answer judge.

Task:
- Compare a candidate answer (from an RLM) against a reference answer (from dataset labels).
- Judge semantic correctness, not string equality.

Judging rules:
1) Mark as CORRECT when the candidate preserves the same meaning/intention as the reference.
2) Ignore harmless formatting differences (quotes, JSON vs Python dict style, spacing, punctuation).
3) Ignore verbosity differences when the core answer is present and unambiguous.
4) For structured/code-locating answers, treat minor numeric offsets (e.g., line numbers off by 1) as fully correct if entities (function names, paths, order) are otherwise correct.
5) Mark INCORRECT only when key facts/entities are missing, contradictory, or wrong in a materially important way.

Return ONLY valid JSON with this exact schema:
{
  "is_correct": boolean,
  "score": number,
  "reason": string
}

Notes:
- score is in [0,1].
- Keep reason concise (<= 2 sentences).
- Do not include markdown, code fences, or extra keys.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge generated SFT traces semantically using OpenRouter."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/generated_sft_traces_glm5.jsonl",
        help="Path to generated trace JSONL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5-nano",
        help="OpenRouter judge model name.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="data/generated_sft_traces_glm5_judged_gpt5nano.jsonl",
        help="Path to output JSONL with judge decisions.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="data/generated_sft_traces_glm5_judged_gpt5nano.md",
        help="Path to output Markdown summary.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to judge.",
    )
    return parser.parse_args()


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Judge output is not valid JSON: {text[:300]}")

    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Judge output JSON is not an object")
    return parsed


def _escape_md(text: str, max_len: int = 240) -> str:
    s = text.replace("\r", "").replace("\n", " <br> ").replace("|", "\\|").strip()
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _to_score(value: Any, fallback: float) -> float:
    try:
        score = float(value)
    except Exception:
        score = fallback
    return max(0.0, min(1.0, score))


def main() -> None:
    load_dotenv()
    args = parse_args()

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is required in environment or .env")

    input_path = Path(args.input)
    output_jsonl_path = Path(args.output_jsonl)
    output_md_path = Path(args.output_md)

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_client(
        backend="openrouter",
        backend_kwargs={
            "model_name": args.model,
            "api_key": openrouter_api_key,
        },
    )

    judged_rows: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as infile, output_jsonl_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for row_idx, line in enumerate(infile):
            if not line.strip():
                continue
            if args.limit is not None and len(judged_rows) >= args.limit:
                break

            source = json.loads(line)
            answer_highlights = source.get("answer_highlights", {})
            candidate_answer = str(answer_highlights.get("model_answer", ""))
            reference_answer = str(answer_highlights.get("correct_answer", ""))

            user_prompt = (
                "Candidate answer:\n"
                f"{candidate_answer}\n\n"
                "Reference answer:\n"
                f"{reference_answer}\n\n"
                "Return ONLY the JSON verdict."
            )

            response = client.completion(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )

            parsed = _extract_json(response)
            is_correct = _to_bool(parsed.get("is_correct"))
            score = _to_score(parsed.get("score"), 1.0 if is_correct else 0.0)
            reason = str(parsed.get("reason", ""))

            judged = {
                "id": source.get("id"),
                "dataset": source.get("dataset"),
                "task": source.get("task"),
                "prompt": source.get("prompt"),
                "context_token_count": source.get("context_token_count"),
                "status": source.get("status"),
                "model_answer": candidate_answer,
                "expected_answer": reference_answer,
                "model_response_raw": source.get("model_response_raw"),
                "trace": source.get("trace"),
                "judge": {
                    "model": args.model,
                    "is_correct": is_correct,
                    "score": score,
                    "reason": reason,
                },
            }
            judged_rows.append(judged)
            outfile.write(json.dumps(judged, ensure_ascii=False) + "\n")

            print(
                f"[{len(judged_rows)}] id={judged.get('id')} "
                f"is_correct={is_correct} score={score:.2f}"
            )

    ok_rows = [row for row in judged_rows if row.get("status") == "ok"]
    correct_rows = [row for row in ok_rows if row.get("judge", {}).get("is_correct") is True]
    avg_score = (
        sum(float(row.get("judge", {}).get("score", 0.0)) for row in ok_rows) / len(ok_rows)
        if ok_rows
        else 0.0
    )

    md_lines = [
        "# LLM Judge Summary",
        "",
        f"- Judge model: `{args.model}`",
        f"- Input file: `{input_path}`",
        f"- Total judged rows: {len(judged_rows)}",
        f"- Correct (semantic): {len(correct_rows)}/{len(ok_rows)}",
        f"- Semantic accuracy: {(len(correct_rows) / len(ok_rows) * 100) if ok_rows else 0.0:.1f}%",
        f"- Average judge score: {avg_score:.3f}",
        "",
        "| ID | Judge Correct | Score | Model Answer | Expected Answer | Reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in judged_rows:
        judge = row.get("judge", {})
        md_lines.append(
            "| "
            + f"{_escape_md(str(row.get('id', '')))} | "
            + f"{'✅' if judge.get('is_correct') else '❌'} | "
            + f"{float(judge.get('score', 0.0)):.2f} | "
            + f"{_escape_md(str(row.get('model_answer', '')))} | "
            + f"{_escape_md(str(row.get('expected_answer', '')))} | "
            + f"{_escape_md(str(judge.get('reason', '')))} |"
        )

    output_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote judged JSONL: {output_jsonl_path}")
    print(f"Wrote summary Markdown: {output_md_path}")


if __name__ == "__main__":
    main()
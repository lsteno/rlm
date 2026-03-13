from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def load_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            f.write("\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
        f.write("\n")


def parse_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"raw": parsed}
        except json.JSONDecodeError:
            return {"raw": value}
    return {"raw": value}


def length_bin(length: int) -> str:
    if length < 10_000:
        return "xs"
    if length < 100_000:
        return "s"
    if length < 500_000:
        return "m"
    if length < 1_500_000:
        return "l"
    if length < 3_000_000:
        return "xl"
    return "xxl"


def stable_tiebreak(*parts: str) -> float:
    digest = hashlib.sha256("||".join(parts).encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def build_diversity_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = parse_metadata(row.get("metadata"))
    prompt = row.get("prompt") or ""
    context = row.get("context") or ""
    answer = row.get("answer") or ""
    dataset = row.get("dataset") or "unknown"
    task = row.get("task") or "unknown"

    if dataset == "longcodeu":
        subgroup = metadata.get("repo") or metadata.get("target_file") or "unknown"
    elif dataset == "frames":
        subgroup = metadata.get("reasoning_types") or "unknown"
    elif dataset == "oolong":
        subgroup = (
            f"{metadata.get('source_dataset', 'unknown')}::"
            f"{metadata.get('task_group', 'unknown')}"
        )
    else:
        subgroup = "unknown"

    answer_type = str(row.get("answer_type") or "unknown")
    prompt_len = len(prompt)
    context_len = len(context)
    answer_len = len(answer)
    group_key = f"{dataset}::{task}"

    enriched = dict(row)
    enriched["metadata"] = metadata
    enriched["selection_features"] = {
        "group_key": group_key,
        "subgroup": subgroup,
        "answer_type": answer_type,
        "prompt_len": prompt_len,
        "context_len": context_len,
        "answer_len": answer_len,
        "prompt_bin": length_bin(prompt_len),
        "context_bin": length_bin(context_len),
        "answer_bin": length_bin(answer_len),
    }
    return enriched


def normalize_candidate_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return normalize_candidate_value(json.loads(stripped))
        except Exception:
            return stripped
    if isinstance(value, list):
        return [normalize_candidate_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_candidate_value(value[key]) for key in sorted(value)}
    return value


def compare_answers(candidate: str, reference: str) -> dict[str, Any]:
    normalized_candidate = normalize_candidate_value(candidate)
    normalized_reference = normalize_candidate_value(reference)

    exact_match = normalized_candidate == normalized_reference
    acceptable_match = False

    if isinstance(normalized_reference, list):
        acceptable_match = normalized_candidate in normalized_reference
        if len(normalized_reference) == 1:
            only = normalized_reference[0]
            acceptable_match = acceptable_match or normalized_candidate == only
            if isinstance(only, str) and isinstance(normalized_candidate, str):
                acceptable_match = acceptable_match or normalized_candidate.endswith(only)
    elif isinstance(normalized_reference, str) and isinstance(normalized_candidate, str):
        acceptable_match = normalized_candidate == normalized_reference
        acceptable_match = acceptable_match or normalized_candidate.endswith(normalized_reference)
    else:
        acceptable_match = exact_match

    return {
        "exact_match": exact_match,
        "acceptable_match": acceptable_match,
        "normalized_candidate": normalized_candidate,
        "normalized_reference": normalized_reference,
    }


def summarize_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_counts = Counter(row["selection_features"]["group_key"] for row in rows)
    dataset_counts = Counter(row.get("dataset", "unknown") for row in rows)
    subgroup_counts = defaultdict(Counter)

    for row in rows:
        features = row["selection_features"]
        subgroup_counts[features["group_key"]][features["subgroup"]] += 1

    return {
        "num_rows": len(rows),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "top_subgroups_per_group": {
            group: counter.most_common(10) for group, counter in sorted(subgroup_counts.items())
        },
    }
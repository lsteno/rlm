from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from training.utils import (
    build_diversity_row,
    load_parquet_rows,
    stable_tiebreak,
    summarize_selection,
    write_jsonl,
)


def allocate_group_quotas(grouped_rows: dict[str, list[dict[str, Any]]], count: int) -> dict[str, int]:
    groups = sorted(grouped_rows)
    quotas = {group: 0 for group in groups}

    if count <= 0:
        return quotas

    base = count // len(groups)
    for group in groups:
        quotas[group] = min(base, len(grouped_rows[group]))

    assigned = sum(quotas.values())
    remaining = count - assigned

    while remaining > 0:
        candidates = sorted(
            groups,
            key=lambda group: (len(grouped_rows[group]) - quotas[group], len(grouped_rows[group]), group),
            reverse=True,
        )
        advanced = False
        for group in candidates:
            if quotas[group] >= len(grouped_rows[group]):
                continue
            quotas[group] += 1
            remaining -= 1
            advanced = True
            if remaining == 0:
                break
        if not advanced:
            break

    return quotas


def novelty_score(
    row: dict[str, Any],
    feature_counts: dict[str, Counter[str]],
    global_feature_counts: dict[str, Counter[str]],
) -> float:
    features = row["selection_features"]
    score = 0.0

    for feature_name in ["subgroup", "answer_type", "prompt_bin", "context_bin", "answer_bin"]:
        value = str(features[feature_name])
        score += 2.0 / (1.0 + feature_counts[feature_name][value])
        score += 0.5 / (1.0 + global_feature_counts[feature_name][value])

    score += stable_tiebreak(str(row.get("id", ""))) * 0.001
    return score


def select_group_rows(
    rows: list[dict[str, Any]],
    quota: int,
    global_feature_counts: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(rows)
    feature_counts: dict[str, Counter[str]] = defaultdict(Counter)

    while remaining and len(selected) < quota:
        best_row = max(
            remaining,
            key=lambda row: novelty_score(row, feature_counts, global_feature_counts),
        )
        remaining.remove(best_row)
        selected.append(best_row)

        features = best_row["selection_features"]
        for feature_name in ["subgroup", "answer_type", "prompt_bin", "context_bin", "answer_bin"]:
            value = str(features[feature_name])
            feature_counts[feature_name][value] += 1
            global_feature_counts[feature_name][value] += 1

    return selected


def strip_selection_features(row: dict[str, Any]) -> dict[str, Any]:
    features = row["selection_features"]
    return {
        key: value for key, value in row.items() if key != "selection_features"
    } | {
        "selection_metadata": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a diverse seed set from data/train.parquet.")
    parser.add_argument("--input", type=Path, required=True, help="Input parquet file.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )
    parser.add_argument("--count", type=int, default=250, help="Number of tasks to select.")
    args = parser.parse_args()

    enriched_rows = [build_diversity_row(row) for row in load_parquet_rows(args.input)]

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched_rows:
        grouped_rows[row["selection_features"]["group_key"]].append(row)

    quotas = allocate_group_quotas(grouped_rows, args.count)
    global_feature_counts: dict[str, Counter[str]] = defaultdict(Counter)
    selected: list[dict[str, Any]] = []

    for group in sorted(grouped_rows):
        group_rows = sorted(grouped_rows[group], key=lambda row: str(row.get("id", "")))
        selected.extend(select_group_rows(group_rows, quotas[group], global_feature_counts))

    if len(selected) != args.count:
        raise ValueError(f"Expected {args.count} rows, selected {len(selected)}")

    stripped_rows = [strip_selection_features(row) for row in selected]
    write_jsonl(args.output, stripped_rows)

    summary = summarize_selection(selected)
    summary["quotas"] = dict(sorted(quotas.items()))

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Wrote {len(stripped_rows)} rows to {args.output}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
"""Assemble base, hard-request, search, and routing planner-SFT records."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.augment_planner_sft import build_augmented, load_jsonl, write_jsonl


def assemble(
    canonical: list[dict[str, Any]],
    base_llama: list[dict[str, Any]],
    hard_metadata: list[dict[str, Any]],
    search_count: int,
    routing_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(canonical) != len(base_llama):
        raise ValueError("canonical and LLaMA records must have identical order and length")
    hard_by_id = {str(row["sample_id"]): row for row in hard_metadata}
    hard_rows = []
    categories: Counter[str] = Counter()
    for canonical_row, llama_row in zip(canonical, base_llama):
        hard = hard_by_id.get(str(canonical_row["sample_id"]))
        if hard is None:
            continue
        row = copy.deepcopy(llama_row)
        row["messages"][0]["content"] = f"<video>\n{hard['raw_user_request']}"
        hard_rows.append(row)
        categories[str(hard["category"])] += 1
    calibrated_rows, calibration_metadata = build_augmented(
        base_llama, search_count=search_count, routing_count=routing_count
    )
    calibration_rows = calibrated_rows[len(base_llama) :]
    final_rows = [*base_llama, *hard_rows, *calibration_rows]
    summary = {
        "base": len(base_llama),
        "hard": len(hard_rows),
        "search": search_count,
        "routing": routing_count,
        "total": len(final_rows),
        "hard_categories": dict(categories),
        "calibration_metadata": calibration_metadata,
    }
    return final_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--base-llama", type=Path, required=True)
    parser.add_argument("--hard-metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--search-count", type=int, default=2000)
    parser.add_argument("--routing-count", type=int, default=1000)
    args = parser.parse_args()
    rows, summary = assemble(
        load_jsonl(args.canonical),
        load_jsonl(args.base_llama),
        load_jsonl(args.hard_metadata),
        args.search_count,
        args.routing_count,
    )
    write_jsonl(args.out, rows)
    summary.pop("calibration_metadata")
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

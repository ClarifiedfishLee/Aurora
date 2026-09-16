"""Assemble base, hard-request, search, and routing planner-SFT records."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.augment_planner_sft import build_augmented, load_jsonl, write_jsonl


def _unique_rows_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, (str, int)) or not str(sample_id).strip():
            raise ValueError(f"{label} row {index} has no non-empty sample_id")
        key = str(sample_id)
        if key in indexed:
            raise ValueError(f"duplicate {label} sample_id: {key}")
        indexed[key] = row
    return indexed


def _video_matches(canonical_video: Any, llama_videos: Any) -> bool:
    if not isinstance(canonical_video, str) or not canonical_video.strip():
        return False
    if not isinstance(llama_videos, list) or len(llama_videos) != 1:
        return False
    llama_video = llama_videos[0]
    if not isinstance(llama_video, str) or not llama_video.strip():
        return False
    canonical_path = Path(canonical_video)
    llama_path = Path(llama_video)
    if canonical_path.is_absolute():
        return canonical_path == llama_path
    canonical_parts = canonical_path.parts
    return (
        len(llama_path.parts) >= len(canonical_parts)
        and llama_path.parts[-len(canonical_parts) :] == canonical_parts
    )


def _validate_base_pair(canonical_row: dict[str, Any], llama_row: dict[str, Any], index: int) -> None:
    sample_id = str(canonical_row["sample_id"])
    raw_request = canonical_row.get("raw_user_request")
    if not isinstance(raw_request, str) or not raw_request.strip():
        raise ValueError(f"canonical row {index} ({sample_id}) has no non-empty raw_user_request")
    if not isinstance(canonical_row.get("target_plan"), dict):
        raise ValueError(f"canonical row {index} ({sample_id}) has no target_plan object")
    messages = llama_row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"base row {index} ({sample_id}) must contain exactly two messages")
    if [message.get("role") for message in messages if isinstance(message, dict)] != ["user", "assistant"]:
        raise ValueError(f"base row {index} ({sample_id}) must contain user/assistant messages")
    expected_user = f"<video>\n{raw_request}"
    if messages[0].get("content") != expected_user:
        raise ValueError(f"canonical/base user request mismatch for sample_id {sample_id}")
    try:
        assistant_plan = json.loads(messages[1].get("content", ""))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"base assistant plan is invalid JSON for sample_id {sample_id}") from exc
    if assistant_plan != canonical_row.get("target_plan"):
        raise ValueError(f"canonical/base assistant plan mismatch for sample_id {sample_id}")
    if not _video_matches(canonical_row.get("video_path"), llama_row.get("videos")):
        raise ValueError(f"canonical/base video mismatch for sample_id {sample_id}")


def _source_audit(rows: list[dict[str, Any]], source_count: int) -> dict[str, Any]:
    indices = [int(row["source_index"]) for row in rows]
    quartiles = Counter(
        f"q{min(4, (source_index * 4) // source_count + 1)}" for source_index in indices
    )
    return {
        "records": len(rows),
        "unique_sources": len(set(indices)),
        "source_coverage": len(set(indices)) / source_count,
        "min_source_index": min(indices) if indices else None,
        "max_source_index": max(indices) if indices else None,
        "source_quartile_counts": {
            f"q{index}": quartiles.get(f"q{index}", 0) for index in range(1, 5)
        },
    }


def _calibration_audit(metadata: list[dict[str, Any]], source_count: int) -> dict[str, Any]:
    by_category: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in metadata}):
        category_rows = [row for row in metadata if str(row["category"]) == category]
        by_category[category] = _source_audit(category_rows, source_count)
    search_entities = Counter(
        str(row["target_plan"]["image_search"])
        for row in metadata
        if row["category"] == "under_search"
    )
    routing_subtasks = Counter(
        str(row["target_plan"]["subtask"])
        for row in metadata
        if row["category"] == "routing_calibration"
    )
    return {
        "source_population": source_count,
        **_source_audit(metadata, source_count),
        "by_category": by_category,
        "search_unique_entities": len(search_entities),
        "search_entity_counts": dict(sorted(search_entities.items())),
        "routing_subtask_counts": dict(sorted(routing_subtasks.items())),
    }


def assemble(
    canonical: list[dict[str, Any]],
    base_llama: list[dict[str, Any]],
    hard_metadata: list[dict[str, Any]],
    search_count: int,
    routing_count: int,
    require_complete_hard: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not canonical:
        raise ValueError("canonical SFT data is empty")
    if len(canonical) != len(base_llama):
        raise ValueError("canonical and LLaMA records must have identical order and length")
    canonical_by_id = _unique_rows_by_id(canonical, "canonical")
    for index, (canonical_row, llama_row) in enumerate(zip(canonical, base_llama), 1):
        _validate_base_pair(canonical_row, llama_row, index)
    hard_by_id = _unique_rows_by_id(hard_metadata, "hard")
    unknown_hard_ids = sorted(set(hard_by_id) - set(canonical_by_id))
    if unknown_hard_ids:
        raise ValueError(f"hard metadata contains unknown sample_ids: {unknown_hard_ids[:5]}")
    missing_hard_ids = sorted(set(canonical_by_id) - set(hard_by_id))
    if require_complete_hard and missing_hard_ids:
        raise ValueError(f"hard metadata is missing sample_ids: {missing_hard_ids[:5]}")
    for sample_id, hard in hard_by_id.items():
        rejected_flags = [
            flag for flag in ("generated", "accepted") if flag in hard and hard[flag] is not True
        ]
        if rejected_flags:
            raise ValueError(f"hard sample {sample_id} was rejected by flags: {rejected_flags}")
        if not isinstance(hard.get("raw_user_request"), str) or not hard["raw_user_request"].strip():
            raise ValueError(f"hard sample {sample_id} has no non-empty raw_user_request")
        if not isinstance(hard.get("category"), str) or not hard["category"].strip():
            raise ValueError(f"hard sample {sample_id} has no non-empty category")
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
    calibration_counts = Counter(str(row["category"]) for row in calibration_metadata)
    final_rows = [*base_llama, *hard_rows, *calibration_rows]
    summary = {
        "base": len(base_llama),
        "hard": len(hard_rows),
        "search": calibration_counts.get("under_search", 0),
        "routing": calibration_counts.get("routing_calibration", 0),
        "total": len(final_rows),
        "hard_categories": dict(sorted(categories.items())),
        "hard_complete": not missing_hard_ids,
        "hard_missing": len(missing_hard_ids),
        "hard_missing_examples": missing_hard_ids[:20],
        "hard_quality_flags": {
            "generated_true": sum(row.get("generated") is True for row in hard_metadata),
            "generated_missing": sum("generated" not in row for row in hard_metadata),
            "accepted_true": sum(row.get("accepted") is True for row in hard_metadata),
            "accepted_missing": sum("accepted" not in row for row in hard_metadata),
        },
        "calibration_audit": _calibration_audit(calibration_metadata, len(base_llama)),
        "calibration_metadata": calibration_metadata,
    }
    return final_rows, summary


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    public_summary = {
        key: value for key, value in summary.items() if key != "calibration_metadata"
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(public_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--base-llama", type=Path, required=True)
    parser.add_argument("--hard-metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--search-count", type=int, default=2000)
    parser.add_argument("--routing-count", type=int, default=1000)
    parser.add_argument(
        "--allow-partial-hard",
        action="store_true",
        help="Permit missing hard-case rows; rejected or unknown rows still fail validation.",
    )
    args = parser.parse_args()
    rows, summary = assemble(
        load_jsonl(args.canonical),
        load_jsonl(args.base_llama),
        load_jsonl(args.hard_metadata),
        args.search_count,
        args.routing_count,
        require_complete_hard=not args.allow_partial_hard,
    )
    write_jsonl(args.out, rows)
    write_summary(args.summary_out, summary)
    public_summary = {
        key: value for key, value in summary.items() if key != "calibration_metadata"
    }
    print(json.dumps(public_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

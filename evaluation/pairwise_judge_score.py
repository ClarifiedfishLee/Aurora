"""Aggregate independent UniEditBench scores into blinded A/B labels."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.judge_agreement import load_jsonl, score


DIMENSIONS = (
    "structural_fidelity",
    "text_video_alignment",
    "background_consistency",
    "naturalness",
    "temporal_spatial_consistency",
)
METHODS = {
    "all5": DIMENSIONS,
    "outcome4": DIMENSIONS[1:],
    "alignment": ("text_video_alignment",),
}


def parse_response(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    payload = json.loads(match.group(0) if match else raw)
    for name in DIMENSIONS:
        value = payload.get(name)
        if not isinstance(value, (int, float)) or not 0 <= value <= 5:
            raise ValueError(f"invalid {name}: {value!r}")
    return payload


def _label(left: float, right: float) -> str:
    if left > right:
        return "A"
    if right > left:
        return "B"
    return "tie"


def aggregate(
    human_rows: list[dict[str, Any]],
    key_rows: list[dict[str, Any]],
    judge_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    human = {str(row["bench_id"]): row for row in human_rows}
    key = {str(row["pair_id"]): row for row in key_rows}
    parsed: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    for result in judge_results:
        result_id = str(result["id"])
        try:
            parsed[result_id] = parse_response(str(result["response"]))
        except (ValueError, json.JSONDecodeError) as exc:
            failures.append({"id": result_id, "error": str(exc)})

    annotations: dict[str, list[dict[str, Any]]] = {name: [] for name in METHODS}
    details: list[dict[str, Any]] = []
    for pair_id, pair_key in key.items():
        if pair_id not in human:
            raise ValueError(f"missing human annotation: {pair_id}")
        left = parsed.get(f"{pair_id}_A")
        right = parsed.get(f"{pair_id}_B")
        if left is None or right is None:
            continue
        base = {
            "bench_id": pair_id,
            "axis": pair_key["axis"],
            "human_label": human[pair_id]["human_label"],
            "human_notes": human[pair_id].get("human_notes", ""),
        }
        detail = {
            "pair_id": pair_id,
            "axis": pair_key["axis"],
            "human_label": base["human_label"],
            "good_side": pair_key["good_side"],
            "scores": {"A": left, "B": right},
            "judge_labels": {},
        }
        for method, dimensions in METHODS.items():
            left_score = sum(float(left[name]) for name in dimensions) / len(dimensions)
            right_score = sum(float(right[name]) for name in dimensions) / len(dimensions)
            judge_label = _label(left_score, right_score)
            annotations[method].append(
                {
                    **base,
                    "judge_label": judge_label,
                    "judge_scores": {"A": left_score, "B": right_score},
                    "method": method,
                }
            )
            detail["judge_labels"][method] = judge_label
        details.append(detail)

    summary = {
        "num_results": len(judge_results),
        "num_parsed": len(parsed),
        "parse_failures": failures,
        "human_labels": dict(Counter(row.get("human_label") for row in human_rows)),
        "judge_label_counts": {
            method: dict(Counter(row["judge_label"] for row in rows)) for method, rows in annotations.items()
        },
        "metrics": {method: score(rows) for method, rows in annotations.items()},
    }
    return summary, details, annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--judge-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    summary, details, annotations = aggregate(
        load_jsonl(args.human),
        json.loads(args.key.read_text(encoding="utf-8")),
        json.loads(args.judge_results.read_text(encoding="utf-8")),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "judge_pair_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "judge_agreement_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for method, rows in annotations.items():
        (args.out_dir / f"judge_agreement_{method}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

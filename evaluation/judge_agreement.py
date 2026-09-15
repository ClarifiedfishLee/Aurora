"""Agreement metrics for pairwise human and video-judge annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


LABELS = ("A", "B", "tie")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(row)
    return rows


def _cohen_kappa(pairs: Iterable[tuple[str, str]]) -> float | None:
    pairs = list(pairs)
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(left_counts[label] * right_counts[label] for label in LABELS) / len(pairs) ** 2
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _score_pairs(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    directional = [(human, judge) for human, judge in pairs if human != "tie" and judge != "tie"]
    confusion = {human: {judge: 0 for judge in LABELS} for human in LABELS}
    for human, judge in pairs:
        confusion[human][judge] += 1
    return {
        "num_pairs": len(pairs),
        "exact_agreement": sum(human == judge for human, judge in pairs) / len(pairs) if pairs else None,
        "cohen_kappa": _cohen_kappa(pairs),
        "directional_num_pairs": len(directional),
        "directional_agreement": (
            sum(human == judge for human, judge in directional) / len(directional) if directional else None
        ),
        "human_tie_rate": sum(human == "tie" for human, _ in pairs) / len(pairs) if pairs else None,
        "judge_tie_rate": sum(judge == "tie" for _, judge in pairs) / len(pairs) if pairs else None,
        "confusion": confusion,
    }


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    usable: list[tuple[str, str, str]] = []
    excluded: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        bench_id = str(row.get("bench_id") or f"row_{index}")
        if bench_id in seen:
            raise ValueError(f"duplicate bench_id: {bench_id}")
        seen.add(bench_id)
        human = row.get("human_label")
        judge = row.get("judge_label")
        if human not in LABELS or judge not in LABELS:
            excluded.append({"bench_id": bench_id, "reason": "missing_or_invalid_label"})
            continue
        usable.append((str(row.get("axis", "unknown")), human, judge))

    overall_pairs = [(human, judge) for _, human, judge in usable]
    by_axis_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for axis, human, judge in usable:
        by_axis_pairs[axis].append((human, judge))
    return {
        "num_rows": len(rows),
        "num_usable": len(usable),
        "coverage": len(usable) / len(rows) if rows else 0.0,
        "overall": _score_pairs(overall_pairs),
        "by_axis": {axis: _score_pairs(pairs) for axis, pairs in sorted(by_axis_pairs.items())},
        "excluded": excluded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = score(load_jsonl(args.annotations))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

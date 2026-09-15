#!/usr/bin/env python3
"""Validate Aurora smoke-test JSONL and its referenced media files."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "smoke" / "cases.jsonl"
REQUIRED_FIELDS = {"bench_id", "prompt", "video_path"}


def main() -> int:
    errors: list[str] = []
    seen_ids: set[str] = set()
    cases: list[dict] = []

    for line_number, line in enumerate(CASES_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc}")
            continue
        missing = REQUIRED_FIELDS - case.keys()
        if missing:
            errors.append(f"line {line_number}: missing fields {sorted(missing)}")
        bench_id = str(case.get("bench_id", ""))
        if bench_id in seen_ids:
            errors.append(f"line {line_number}: duplicate bench_id {bench_id}")
        seen_ids.add(bench_id)
        video_path = ROOT / str(case.get("video_path", ""))
        if not video_path.is_file():
            errors.append(f"line {line_number}: missing video {video_path}")
        elif video_path.stat().st_size < 1024:
            errors.append(f"line {line_number}: suspiciously small video {video_path}")
        cases.append(case)

    if len(cases) != 10:
        errors.append(f"expected 10 cases, found {len(cases)}")
    if errors:
        print("Smoke-case validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Validated {len(cases)} smoke cases and all referenced videos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

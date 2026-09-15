"""Prepare teacher cases and LLaMA-Factory records for planner SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from evaluation.agent_only_score import valid_plan


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_teacher_cases(manifest_path: Path) -> list[dict[str, Any]]:
    root = manifest_path.parent
    rows = []
    for item in load_jsonl(manifest_path):
        rows.append(
            {
                "bench_id": item["sample_id"],
                "video_path": str((root / item["video_path"]).resolve()),
                "prompt": item["clean_instruction"],
                "edit_type": item.get("edit_type") or item.get("subset") or "custom",
            }
        )
    return rows


def degrade_instruction(clean: str, sample_id: str) -> str:
    """Create a deterministic, meaning-preserving colloquial request."""
    text = clean.strip().rstrip(".")
    text = re.sub(r"\s+", " ", text)
    variants = [text]
    variants.append(re.sub(r"^(Please\s+)?Remove\b", "get rid of", text, flags=re.IGNORECASE))
    variants.append(re.sub(r"^(Please\s+)?Add\b", "put", text, flags=re.IGNORECASE))
    variants.append(re.sub(r"^(Please\s+)?Replace\b", "swap", text, flags=re.IGNORECASE))
    variants.append(re.sub(r"^(Please\s+)?Change\b", "make", text, flags=re.IGNORECASE))
    index = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) % len(variants)
    result = variants[index]
    return result[:1].lower() + result[1:] if result else clean


def _expected_subtask(item: dict[str, Any], teacher_subtask: str) -> str:
    subset = str(item.get("subset", ""))
    edit_type = str(item.get("edit_type", "")).lower()
    if subset == "ditto-combined":
        return "combined_tasks"
    if "removal" in subset or "removal" in edit_type:
        return "remove_object"
    if "insertion" in subset or "insertion" in edit_type:
        return "add_object"
    return teacher_subtask


def compose(
    manifest_rows: list[dict[str, Any]], teacher_rows: list[dict[str, Any]], system_prompt: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    teachers = {str(row["bench_id"]): row for row in teacher_rows}
    canonical = []
    llama = []
    for item in manifest_rows:
        sample_id = str(item["sample_id"])
        teacher = teachers.get(sample_id)
        if teacher is None or not valid_plan(teacher.get("plan")):
            continue
        plan = dict(teacher["plan"])
        plan["refined_text_instruction"] = item["clean_instruction"]
        plan["subtask"] = _expected_subtask(item, str(plan["subtask"]))
        raw_request = degrade_instruction(item["clean_instruction"], sample_id)
        target = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
        video_path = str(Path(teacher["video_path"]).resolve())
        canonical.append(
            {
                **item,
                "raw_user_request": raw_request,
                "target_plan": plan,
                "teacher_source": "aurora_released_lora_with_clean_instruction_override",
            }
        )
        llama.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"<video>\n{raw_request}"},
                    {"role": "assistant", "content": target},
                ],
                "videos": [video_path],
            }
        )
    return canonical, llama


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cases_parser = subparsers.add_parser("teacher-cases")
    cases_parser.add_argument("--manifest", type=Path, required=True)
    cases_parser.add_argument("--out", type=Path, required=True)
    compose_parser = subparsers.add_parser("compose")
    compose_parser.add_argument("--manifest", type=Path, required=True)
    compose_parser.add_argument("--teacher-records", type=Path, required=True)
    compose_parser.add_argument("--system-prompt", type=Path, default=Path("aurora/prompts/type1_system.txt"))
    compose_parser.add_argument("--canonical-out", type=Path, required=True)
    compose_parser.add_argument("--llama-out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "teacher-cases":
        rows = build_teacher_cases(args.manifest)
        write_jsonl(args.out, rows)
        print(json.dumps({"num_cases": len(rows), "out": str(args.out)}, indent=2))
        return
    canonical, llama = compose(
        load_jsonl(args.manifest),
        load_jsonl(args.teacher_records),
        args.system_prompt.read_text(encoding="utf-8").strip(),
    )
    write_jsonl(args.canonical_out, canonical)
    write_jsonl(args.llama_out, llama)
    print(json.dumps({"num_samples": len(canonical), "canonical": str(args.canonical_out), "llama": str(args.llama_out)}, indent=2))


if __name__ == "__main__":
    main()

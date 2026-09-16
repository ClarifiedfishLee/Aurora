#!/usr/bin/env python3
"""Create and verify a reproducible Week-2 training artifact bundle.

Expected bundle layout::

    lora-final-12597/
    lora-final-12597-best-eval/
    metadata/
    day14_gate/

The manifest uses paths relative to the bundle root so the same file can be
verified on the Worker, Devbox, and local machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


EXPECTED_COUNTS = {
    "metadata/sft_final_12597_v2.jsonl": 12_597,
    "metadata/sft_final_12597_v2_train.jsonl": 12_339,
    "metadata/sft_final_12597_v2_eval.jsonl": 258,
    "day14_gate/agent_pipeline_records.jsonl": 100,
}

REQUIRED_FILES = (
    "lora-final-12597/adapter_config.json",
    "lora-final-12597/adapter_model.safetensors",
    "lora-final-12597/train.log",
    "lora-final-12597/trainer_log.jsonl",
    "lora-final-12597/trainer_state.json",
    "lora-final-12597/exit_code",
    "lora-final-12597-best-eval/adapter_config.json",
    "lora-final-12597-best-eval/adapter_model.safetensors",
    "lora-final-12597-best-eval/best_eval.json",
    "metadata/train_final_12597.yaml",
    "metadata/dataset_info.json",
    "metadata/sft_final_12597_v2_summary.json",
    "metadata/sft_final_12597_v2.jsonl",
    "metadata/sft_final_12597_v2_train.jsonl",
    "metadata/sft_final_12597_v2_eval.jsonl",
    "day14_gate/planner.log",
    "day14_gate/scorer.log",
    "day14_gate/metrics.json",
    "day14_gate/gate_summary.json",
    "day14_gate/agent_pipeline_records.jsonl",
)

REQUIRED_GATE_KEYS = (
    "num_cases",
    "json_validity",
    "strict_raw_json_validity",
    "subtask_accuracy",
    "image_search_trigger",
    "image_search_query",
    "mask_trigger",
    "constraint_retention",
    "constraint_case_accuracy",
    "source_entity_false_trigger",
    "details",
    "by_axis",
)

_MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(root: Path, raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe manifest path: {raw!r}")
    candidate = root / relative
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes bundle root: {raw!r}") from exc
    return candidate


def iter_bundle_files(root: Path, excluded: Iterable[Path] = ()) -> Iterable[Path]:
    excluded_resolved = {path.resolve() for path in excluded}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"bundle must not contain symlinks: {path}")
        if path.is_file() and path.resolve() not in excluded_resolved:
            yield path


def write_manifest(root: Path, output: Path) -> int:
    root = root.resolve()
    output = output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in iter_bundle_files(root, excluded=(output,))
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def verify_manifest(root: Path, manifest: Path) -> list[str]:
    root = root.resolve()
    errors: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            errors.append(f"manifest line {line_number}: invalid format")
            continue
        expected, relative = match.groups()
        if relative in seen:
            errors.append(f"manifest line {line_number}: duplicate path {relative!r}")
            continue
        seen.add(relative)
        try:
            path = _safe_relative_path(root, relative)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.is_file():
            errors.append(f"missing manifest file: {relative}")
        elif path.is_symlink():
            errors.append(f"manifest file is a symlink: {relative}")
        else:
            actual = sha256(path)
            if actual != expected:
                errors.append(f"checksum mismatch: {relative}")
    if not seen:
        errors.append("manifest contains no files")
    return errors


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read JSON {path.name}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"expected JSON object: {path.name}")
        return None
    return value


def _jsonl_count(path: Path, errors: list[str]) -> int:
    count = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"invalid JSONL {path.name}:{line_number}: {exc}")
                    continue
                if not isinstance(value, dict):
                    errors.append(f"expected JSON object {path.name}:{line_number}")
                count += 1
    except OSError as exc:
        errors.append(f"cannot read {path.name}: {exc}")
    return count


def audit_bundle(root: Path) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    observations: dict[str, Any] = {}

    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing required file: {relative}")
        elif path.stat().st_size == 0:
            errors.append(f"empty required file: {relative}")

    checkpoint_dirs = sorted(
        (path for path in (root / "lora-final-12597").glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    observations["checkpoint_dirs"] = [path.name for path in checkpoint_dirs if path.is_dir()]
    if not observations["checkpoint_dirs"]:
        errors.append("no checkpoint-* directories found in the complete training output")
    else:
        newest = checkpoint_dirs[-1]
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
        ):
            path = newest / name
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"latest resumable checkpoint is missing {newest.name}/{name}")

    exit_code_path = root / "lora-final-12597/exit_code"
    if exit_code_path.is_file():
        try:
            exit_code = int(exit_code_path.read_text(encoding="utf-8").strip())
            observations["training_exit_code"] = exit_code
            if exit_code != 0:
                errors.append(f"training exit code is {exit_code}, expected 0")
        except (OSError, ValueError) as exc:
            errors.append(f"invalid training exit_code: {exc}")

    for relative, expected in EXPECTED_COUNTS.items():
        path = root / relative
        if path.is_file():
            count = _jsonl_count(path, errors)
            observations[f"rows:{relative}"] = count
            if count != expected:
                errors.append(f"{relative}: found {count} rows, expected {expected}")

    best_path = root / "lora-final-12597-best-eval/best_eval.json"
    if best_path.is_file():
        best = _load_json(best_path, errors)
        if best is not None:
            observations["best_eval"] = best
            if not isinstance(best.get("best_step"), int) or not isinstance(
                best.get("eval_loss"), (int, float)
            ):
                errors.append("best_eval.json must contain numeric best_step and eval_loss")

    metrics_path = root / "day14_gate/metrics.json"
    if metrics_path.is_file():
        metrics = _load_json(metrics_path, errors)
        if metrics is not None:
            missing_keys = [key for key in REQUIRED_GATE_KEYS if key not in metrics]
            if missing_keys:
                errors.append(f"Day-14 metrics missing keys: {', '.join(missing_keys)}")
            if metrics.get("num_cases") != 100:
                errors.append(f"Day-14 metrics num_cases is {metrics.get('num_cases')!r}, expected 100")
            observations["day14_gate"] = {
                key: metrics.get(key)
                for key in (
                    "num_cases",
                    "json_validity",
                    "strict_raw_json_validity",
                    "subtask_accuracy",
                    "constraint_retention",
                    "constraint_case_accuracy",
                )
            }

    gate_summary_path = root / "day14_gate/gate_summary.json"
    if gate_summary_path.is_file():
        gate_summary = _load_json(gate_summary_path, errors)
        if gate_summary is not None:
            observations["day14_gate_passed"] = gate_summary.get("passed")
            if gate_summary.get("passed") is not True:
                errors.append("Day-14 gate_summary.json does not record a passing gate")
            if not isinstance(gate_summary.get("checks"), dict):
                errors.append("Day-14 gate_summary.json must contain a checks object")
            if not isinstance(gate_summary.get("diagnostics"), dict):
                errors.append("Day-14 gate_summary.json must contain a diagnostics object")

    train_path = root / "metadata/sft_final_12597_v2_train.jsonl"
    eval_path = root / "metadata/sft_final_12597_v2_eval.jsonl"
    if train_path.is_file() and eval_path.is_file():
        try:
            train_videos = {
                str(json.loads(line)["videos"][0])
                for line in train_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            eval_videos = {
                str(json.loads(line)["videos"][0])
                for line in eval_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            overlap = train_videos & eval_videos
            observations["train_videos"] = len(train_videos)
            observations["eval_videos"] = len(eval_videos)
            observations["video_overlap"] = len(overlap)
            if overlap:
                errors.append(f"train/eval source-video overlap: {len(overlap)}")
        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"cannot audit grouped split: {exc}")

    return {"ok": not errors, "errors": errors, "observations": observations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Week-2 bundle root")
    parser.add_argument("--write-manifest", type=Path, help="write a sorted SHA-256 manifest")
    parser.add_argument("--verify-manifest", type=Path, help="verify an existing SHA-256 manifest")
    parser.add_argument("--audit", action="store_true", help="audit the expected Week-2 layout")
    args = parser.parse_args()

    result: dict[str, Any] = {"root": str(args.root.resolve())}
    failed = False
    if args.write_manifest:
        count = write_manifest(args.root, args.write_manifest)
        result["manifest_written"] = str(args.write_manifest.resolve())
        result["manifest_files"] = count
    if args.verify_manifest:
        errors = verify_manifest(args.root, args.verify_manifest)
        result["manifest_errors"] = errors
        failed = failed or bool(errors)
    if args.audit:
        audit = audit_bundle(args.root)
        result["audit"] = audit
        failed = failed or not audit["ok"]
    if not (args.write_manifest or args.verify_manifest or args.audit):
        parser.error("select --write-manifest, --verify-manifest, or --audit")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

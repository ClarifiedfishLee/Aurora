#!/usr/bin/env python3
"""Create and verify a reproducible Week-2 training artifact bundle.

Expected bundle layout::

    lora-final-12597/
    lora-final-12597-best-eval/
    metadata/
    day14_gate/

The manifest uses paths relative to the bundle root so the same file can be
verified on the Worker, Devbox, and local machine.

The ``refresh`` audit profile expects the bounded correction run in this
portable layout::

    lora-refresh/
      checkpoint-{32,64,96,128}/
    lora-refresh-selected/
    metadata/
      refresh_{train,eval,cases,gold}.jsonl
      refresh_generation_audit.json
      train_refresh.yaml
      refresh_selection_policy.json
      refresh_selection.json
    day14_gate/
    checksums.sha256

The selected adapter is a portable copy of the checkpoint named by
``metadata/refresh_selection.json``.  Its adapter files must be byte-identical
to that checkpoint.
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

REFRESH_CHECKPOINT_STEPS = (32, 64, 96, 128)
REFRESH_EXPECTED_COUNTS = {
    "metadata/refresh_train.jsonl": 1_024,
    "metadata/refresh_eval.jsonl": 256,
    "metadata/refresh_cases.jsonl": 256,
    "metadata/refresh_gold.jsonl": 256,
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

REFRESH_REQUIRED_FILES = (
    "lora-refresh/adapter_config.json",
    "lora-refresh/adapter_model.safetensors",
    "lora-refresh/train.log",
    "lora-refresh/trainer_log.jsonl",
    "lora-refresh/trainer_state.json",
    "lora-refresh/exit_code",
    "lora-refresh-selected/adapter_config.json",
    "lora-refresh-selected/adapter_model.safetensors",
    "metadata/refresh_train.jsonl",
    "metadata/refresh_eval.jsonl",
    "metadata/refresh_cases.jsonl",
    "metadata/refresh_gold.jsonl",
    "metadata/refresh_generation_audit.json",
    "metadata/train_refresh.yaml",
    "metadata/refresh_selection_policy.json",
    "metadata/refresh_selection.json",
    "day14_gate/planner.log",
    "day14_gate/scorer.log",
    "day14_gate/metrics.json",
    "day14_gate/gate_summary.json",
    "day14_gate/agent_pipeline_records.jsonl",
    "checksums.sha256",
)

REFRESH_CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
)

REFRESH_SELECTION_POLICY = "lowest_refresh_eval_loss_then_earliest_step"

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


def _read_jsonl_objects(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
                    continue
                rows.append(value)
    except OSError as exc:
        errors.append(f"cannot read {path.name}: {exc}")
    return rows


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _audit_refresh_manifest(root: Path, manifest: Path, errors: list[str]) -> None:
    """Verify every bundle file is covered, in addition to checking hashes."""

    manifest_errors = verify_manifest(root, manifest)
    errors.extend(f"checksum manifest: {error}" for error in manifest_errors)
    listed: set[str] = set()
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            match = _MANIFEST_LINE.fullmatch(line)
            if match is not None:
                listed.add(match.group(2))
        actual = {
            path.relative_to(root).as_posix()
            for path in iter_bundle_files(root, excluded=(manifest,))
        }
    except (OSError, ValueError) as exc:
        errors.append(f"cannot audit checksum manifest coverage: {exc}")
        return
    unlisted = sorted(actual - listed)
    if unlisted:
        preview = ", ".join(unlisted[:5])
        errors.append(f"checksum manifest omits {len(unlisted)} bundle file(s): {preview}")


def _audit_refresh_config(path: Path, errors: list[str]) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read refresh training config: {exc}")
        return

    expected = {
        "dataset": "aurora_planner_refresh_train",
        "eval_dataset": "aurora_planner_refresh_eval",
        "create_new_adapter": "false",
        "max_samples": "1024",
        "gradient_accumulation_steps": "8",
        "learning_rate": "2.0e-5",
        "num_train_epochs": "1.0",
        "warmup_ratio": "0.05",
        "save_steps": "32",
        "eval_steps": "32",
        "save_total_limit": "4",
        "overwrite_output_dir": "true",
        "save_only_model": "false",
    }
    for key, value in expected.items():
        pattern = rf"(?m)^\s*{re.escape(key)}:\s*{re.escape(value)}\s*(?:#.*)?$"
        if re.search(pattern, text) is None:
            errors.append(f"refresh training config must set {key}: {value}")
    if re.search(r"(?m)^\s*resume_from_checkpoint\s*:", text):
        errors.append("refresh training config must not set resume_from_checkpoint")
    adapter_match = re.search(r"(?m)^\s*adapter_name_or_path:\s*(.+?)\s*$", text)
    if adapter_match is None or "best-eval" not in adapter_match.group(1):
        errors.append("refresh training config must load the preselected best-eval adapter")


def _audit_refresh_gate(root: Path, errors: list[str], observations: dict[str, Any]) -> None:
    metrics_path = root / "day14_gate/metrics.json"
    if metrics_path.is_file():
        metrics = _load_json(metrics_path, errors)
        if metrics is not None:
            missing_keys = [key for key in REQUIRED_GATE_KEYS if key not in metrics]
            if missing_keys:
                errors.append(f"Day-14 metrics missing keys: {', '.join(missing_keys)}")
            if metrics.get("num_cases") != 100:
                errors.append(
                    f"Day-14 metrics num_cases is {metrics.get('num_cases')!r}, expected 100"
                )
            observations["day14_gate"] = {
                key: metrics.get(key)
                for key in (
                    "num_cases",
                    "json_validity",
                    "strict_raw_json_validity",
                    "subtask_accuracy",
                    "constraint_retention",
                    "constraint_case_accuracy",
                    "source_entity_false_trigger",
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


def audit_refresh_bundle(root: Path) -> dict[str, Any]:
    """Audit the fixed Week-2 over-search correction bundle."""

    root = root.resolve()
    errors: list[str] = []
    observations: dict[str, Any] = {}

    for relative in REFRESH_REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing required file: {relative}")
        elif path.stat().st_size == 0:
            errors.append(f"empty required file: {relative}")

    output = root / "lora-refresh"
    checkpoint_names = sorted(
        path.name for path in output.glob("checkpoint-*") if path.is_dir()
    )
    expected_names = [f"checkpoint-{step}" for step in REFRESH_CHECKPOINT_STEPS]
    observations["checkpoint_dirs"] = checkpoint_names
    if set(checkpoint_names) != set(expected_names) or len(checkpoint_names) != len(expected_names):
        errors.append(
            f"refresh checkpoints are {checkpoint_names!r}, expected exactly {expected_names!r}"
        )
    for step in REFRESH_CHECKPOINT_STEPS:
        checkpoint = output / f"checkpoint-{step}"
        for name in REFRESH_CHECKPOINT_FILES:
            path = checkpoint / name
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"checkpoint-{step} is missing non-empty {name}")

    exit_code_path = output / "exit_code"
    if exit_code_path.is_file():
        try:
            exit_code = int(exit_code_path.read_text(encoding="utf-8").strip())
            observations["training_exit_code"] = exit_code
            if exit_code != 0:
                errors.append(f"training exit code is {exit_code}, expected 0")
        except (OSError, ValueError) as exc:
            errors.append(f"invalid training exit_code: {exc}")

    rows_by_relative: dict[str, list[dict[str, Any]]] = {}
    for relative, expected in REFRESH_EXPECTED_COUNTS.items():
        path = root / relative
        if path.is_file():
            rows = _read_jsonl_objects(path, errors)
            rows_by_relative[relative] = rows
            count = len(rows)
            observations[f"rows:{relative}"] = count
            if count != expected:
                errors.append(f"{relative}: found {count} rows, expected {expected}")

    train_rows = rows_by_relative.get("metadata/refresh_train.jsonl")
    eval_rows = rows_by_relative.get("metadata/refresh_eval.jsonl")
    if train_rows is not None and eval_rows is not None:
        try:
            train_videos = {str(row["videos"][0]) for row in train_rows}
            eval_videos = {str(row["videos"][0]) for row in eval_rows}
            overlap = train_videos & eval_videos
            observations["train_videos"] = len(train_videos)
            observations["eval_videos"] = len(eval_videos)
            observations["video_overlap"] = len(overlap)
            if overlap:
                errors.append(f"refresh train/eval source-video overlap: {len(overlap)}")
        except (KeyError, IndexError, TypeError) as exc:
            errors.append(f"cannot audit refresh grouped split: {exc}")

    cases = rows_by_relative.get("metadata/refresh_cases.jsonl")
    gold = rows_by_relative.get("metadata/refresh_gold.jsonl")
    if cases is not None and gold is not None:
        case_ids = [row.get("bench_id") for row in cases]
        gold_ids = [row.get("bench_id") for row in gold]
        if any(not isinstance(value, str) or not value for value in case_ids + gold_ids):
            errors.append("refresh cases/gold must contain non-empty string bench_id values")
        if len(set(case_ids)) != len(case_ids):
            errors.append("refresh cases contain duplicate bench_id values")
        if len(set(gold_ids)) != len(gold_ids):
            errors.append("refresh gold contains duplicate bench_id values")
        if case_ids != gold_ids:
            errors.append("refresh cases/gold bench_id order does not match")

    audit_path = root / "metadata/refresh_generation_audit.json"
    if audit_path.is_file():
        generation = _load_json(audit_path, errors)
        if generation is not None:
            counts = generation.get("counts")
            expected_counts = {
                "train": 1_024,
                "validation": 256,
                "validation_cases": 256,
                "validation_gold": 256,
            }
            if not isinstance(counts, dict):
                errors.append("refresh generation audit must contain a counts object")
            else:
                observations["refresh_generation_counts"] = counts
                for key, expected in expected_counts.items():
                    if counts.get(key) != expected:
                        errors.append(
                            f"refresh generation audit counts.{key} is "
                            f"{counts.get(key)!r}, expected {expected}"
                        )
            isolation = generation.get("isolation")
            isolation_zeroes = (
                "video_overlap",
                "normalized_prompt_overlap",
                "concept_overlap",
                "template_overlap",
            )
            if not isinstance(isolation, dict):
                errors.append("refresh generation audit must contain an isolation object")
            else:
                for key in isolation_zeroes:
                    if isolation.get(key) != 0:
                        errors.append(
                            f"refresh generation audit isolation.{key} is "
                            f"{isolation.get(key)!r}, expected 0"
                        )
            forbidden = generation.get("forbidden_audit")
            if not isinstance(forbidden, dict):
                errors.append("refresh generation audit must contain a forbidden_audit object")
            else:
                for key in ("exact_prompt_overlap", "phrase_hit_count"):
                    if forbidden.get(key) != 0:
                        errors.append(
                            f"refresh generation audit forbidden_audit.{key} is "
                            f"{forbidden.get(key)!r}, expected 0"
                        )

    config_path = root / "metadata/train_refresh.yaml"
    if config_path.is_file():
        _audit_refresh_config(config_path, errors)

    policy_path = root / "metadata/refresh_selection_policy.json"
    policy: dict[str, Any] | None = None
    if policy_path.is_file():
        policy = _load_json(policy_path, errors)
        if policy is not None:
            expected_policy = {
                "selection_source": "refresh_eval",
                "selection_metric": "eval_loss",
                "lower_is_better": True,
                "selection_policy": REFRESH_SELECTION_POLICY,
                "tie_breaker": "earliest_step",
                "eligible_checkpoint_steps": list(REFRESH_CHECKPOINT_STEPS),
                "external_gate_metrics_allowed": False,
                "expected_checkpoint_count": 4,
                "train_rows": 1_024,
                "eval_rows": 256,
            }
            for key, expected in expected_policy.items():
                if policy.get(key) != expected:
                    errors.append(
                        f"refresh selection policy {key} is {policy.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            for key, relative in (
                ("train_sha256", "metadata/refresh_train.jsonl"),
                ("eval_sha256", "metadata/refresh_eval.jsonl"),
            ):
                data_path = root / relative
                if data_path.is_file() and policy.get(key) != sha256(data_path):
                    errors.append(f"refresh selection policy {key} does not match {relative}")

    selection_path = root / "metadata/refresh_selection.json"
    if selection_path.is_file():
        selection = _load_json(selection_path, errors)
        if selection is not None:
            expected_selection = {
                "selection_source": "refresh_eval",
                "selection_metric": "eval_loss",
                "selection_policy": REFRESH_SELECTION_POLICY,
                "external_gate_metrics_used": False,
            }
            for key, expected in expected_selection.items():
                if selection.get(key) != expected:
                    errors.append(
                        f"refresh selection result {key} is {selection.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            candidates = selection.get("candidates")
            parsed_candidates: list[tuple[int, float]] = []
            if not isinstance(candidates, list) or len(candidates) != 4:
                errors.append("refresh selection result must contain exactly four candidates")
            else:
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        errors.append("refresh selection candidate must be an object")
                        continue
                    step = candidate.get("step")
                    loss = candidate.get("eval_loss")
                    checkpoint = candidate.get("checkpoint")
                    if step not in REFRESH_CHECKPOINT_STEPS or not _is_number(loss):
                        errors.append(f"invalid refresh selection candidate: {candidate!r}")
                        continue
                    if not isinstance(checkpoint, str) or Path(checkpoint).name != (
                        f"checkpoint-{step}"
                    ):
                        errors.append(f"refresh candidate step {step} names the wrong checkpoint")
                    parsed_candidates.append((int(step), float(loss)))

                    state_path = output / f"checkpoint-{step}/trainer_state.json"
                    if state_path.is_file():
                        state = _load_json(state_path, errors)
                        history = state.get("log_history", []) if state is not None else []
                        state_losses = [
                            float(item["eval_loss"])
                            for item in history
                            if isinstance(item, dict)
                            and item.get("step") == step
                            and _is_number(item.get("eval_loss"))
                        ]
                        if not state_losses or abs(state_losses[-1] - float(loss)) > 1e-12:
                            errors.append(
                                f"refresh candidate step {step} eval_loss does not match "
                                "trainer_state.json"
                            )
            if sorted(step for step, _ in parsed_candidates) != list(REFRESH_CHECKPOINT_STEPS):
                errors.append(
                    "refresh selection candidates do not cover steps 32, 64, 96, "
                    "and 128 exactly once"
                )
            elif parsed_candidates:
                expected_step, expected_loss = min(
                    parsed_candidates, key=lambda item: (item[1], item[0])
                )
                if selection.get("selected_step") != expected_step:
                    errors.append(
                        f"refresh selection chose step {selection.get('selected_step')!r}, "
                        f"expected {expected_step}"
                    )
                if not _is_number(selection.get("selected_eval_loss")) or abs(
                    float(selection.get("selected_eval_loss", 0.0)) - expected_loss
                ) > 1e-12:
                    errors.append("refresh selected_eval_loss does not match the winning candidate")
                selected_checkpoint = selection.get("selected_checkpoint")
                if not isinstance(selected_checkpoint, str) or Path(
                    selected_checkpoint
                ).name != f"checkpoint-{expected_step}":
                    errors.append("refresh selected_checkpoint does not match selected_step")
                observations["refresh_selection"] = {
                    "selected_step": selection.get("selected_step"),
                    "selected_eval_loss": selection.get("selected_eval_loss"),
                    "candidate_losses": {
                        str(step): loss for step, loss in sorted(parsed_candidates)
                    },
                }

                selected_dir = root / "lora-refresh-selected"
                source_dir = output / f"checkpoint-{expected_step}"
                for name in ("adapter_config.json", "adapter_model.safetensors"):
                    source = source_dir / name
                    copied = selected_dir / name
                    if source.is_file() and copied.is_file() and sha256(source) != sha256(copied):
                        errors.append(
                            f"selected adapter {name} does not match checkpoint-{expected_step}"
                        )

    _audit_refresh_gate(root, errors, observations)

    manifest_path = root / "checksums.sha256"
    if manifest_path.is_file():
        _audit_refresh_manifest(root, manifest_path, errors)

    return {"ok": not errors, "errors": errors, "observations": observations}


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
    parser.add_argument(
        "--profile",
        choices=("v2", "refresh"),
        default="v2",
        help="artifact layout to audit (default: v2)",
    )
    parser.add_argument("--write-manifest", type=Path, help="write a sorted SHA-256 manifest")
    parser.add_argument("--verify-manifest", type=Path, help="verify an existing SHA-256 manifest")
    parser.add_argument("--audit", action="store_true", help="audit the selected Week-2 layout")
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
        audit = (
            audit_refresh_bundle(args.root)
            if args.profile == "refresh"
            else audit_bundle(args.root)
        )
        result["audit"] = audit
        failed = failed or not audit["ok"]
    if not (args.write_manifest or args.verify_manifest or args.audit):
        parser.error("select --write-manifest, --verify-manifest, or --audit")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

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

The ``corrected-full`` profile is the Worker-side, fully reconstructible final
Week-2 correction bundle.  Absolute runtime paths recorded inside JSON files
are treated as provenance only; portable identity is established with hashes
and the fixed relative tree below::

    recipe2/
      lora-refresh/checkpoint-{32,64,96,128}/
      lora-refresh-selected/
      metadata/
    fresh384/
      cases.jsonl
      gold.jsonl
      leakage_audit.json
      selection_policy.json
      recipe2_cross_audit.json
    candidates/
      lambda_{0000,0125,0250,0375,0500,0625,0750,0875,1000}/
        adapter/
        eval/
      recipe2/
        adapter/
        eval/
    selection/
      comparison.json
      selection.json
      adapter_identities.json
      endpoint_configs/{v2,refresh1,recipe2}.adapter_config.json
    adaptive_day14/
      gold.jsonl
      agent_pipeline_records.jsonl
      planner.log
      metrics.json
      scorer.log
      gate_summary.json
    checksums.sha256

The fresh-384 selector is the only model-selection source.  The one Day-14
run is explicitly marked adaptive/non-confirmatory and must target exactly the
already-selected adapter.

The ``corrected-thin`` profile uses the same tree and preserves every raw
evaluation, run manifest, interpolation provenance file, and selection audit,
but keeps adapter weights only for the final selected candidate.  It replaces
the recipe2 checkpoint/model blobs with trainer states plus
``recipe2/metadata/selected_adapter_identity.json``.  This is the durable
backup profile; its verifier recomputes metrics and bootstrap selection from
the raw records rather than trusting the summary JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from evaluation.agent_only_score import score as _score_day14
    from scripts.run_day14_gate import RELEASED_BASELINE, GateValidationError
    from scripts.run_day14_gate import evaluate_gate as _evaluate_day14_gate
    from scripts.select_lora_interpolation import (
        SelectionValidationError,
        paired_stratified_bootstrap,
    )
    from scripts.select_lora_interpolation import (
        _metric_and_strata as _recompute_corrected_metrics,
    )
except ModuleNotFoundError:  # direct ``python scripts/...py`` execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.agent_only_score import score as _score_day14
    from scripts.run_day14_gate import RELEASED_BASELINE, GateValidationError
    from scripts.run_day14_gate import evaluate_gate as _evaluate_day14_gate
    from scripts.select_lora_interpolation import (
        SelectionValidationError,
        paired_stratified_bootstrap,
    )
    from scripts.select_lora_interpolation import (
        _metric_and_strata as _recompute_corrected_metrics,
    )

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

CORRECTED_FULL_PROFILE = "corrected-full"
CORRECTED_THIN_PROFILE = "corrected-thin"
CORRECTED_ADAPTER_CONFIG_SHA256 = {
    "v2": "bcf2fa6ad9862df7fe5281d6f9f0aa21fc2b451751270d858adcf4076f464dd3",
    "refresh1": "9dc73dd51504976bc8b39ebab144fd14f498aa68e0049cf8b6a58b986851ef64",
    "recipe2": "d77a118094c39b4f4059522ca412f5e268f5572c75701ec571732599f6275156",
}
CORRECTED_LAMBDA_GRID = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
CORRECTED_CATEGORY_COUNTS = {
    "no_search_negative": 128,
    "true_search_positive": 64,
    "routing_control": 64,
    "mask_control": 64,
    "rewrite_retention": 64,
}
CORRECTED_NO_SEARCH_COUNTS = {
    "generic_style": 64,
    "generic_background": 32,
    "ordinary_target": 32,
}
CORRECTED_MASK_COUNTS = {"triggered": 32, "not_triggered": 32}
CORRECTED_ELIGIBILITY_THRESHOLDS = {
    "complete_prediction_rows": 384,
    "strict_raw_json_validity_min": 1.0,
    "subtask_accuracy_min": 0.95,
    "no_search_specificity_min": 0.95,
    "true_search_trigger_recall_min": 0.95,
    "search_query_end_to_end_recall_min": 0.95,
    "mask_trigger_f1_min": 0.95,
    "rewrite_constraint_retention_min": 0.85,
}
CORRECTED_THRESHOLD_METRICS = {
    "complete_prediction_rows": "prediction_rows",
    "strict_raw_json_validity_min": "strict_raw_json_validity",
    "subtask_accuracy_min": "subtask_accuracy",
    "no_search_specificity_min": "no_search_specificity",
    "true_search_trigger_recall_min": "true_search_trigger_recall",
    "search_query_end_to_end_recall_min": "search_query_end_to_end_recall",
    "mask_trigger_f1_min": "mask_trigger_f1",
    "rewrite_constraint_retention_min": "rewrite_constraint_retention",
}
CORRECTED_STRICT_LEAKAGE_FIELDS = (
    "prior_identifier_overlap",
    "prior_basename_overlap",
    "prior_video_sha256_overlap",
    "prior_exact_prompt_overlap",
    "prior_concept_overlap",
    "prior_template_id_overlap",
    "prior_template_signature_overlap",
    "prior_constraint_value_overlap",
    "prior_concept_phrase_hit_count",
    "prior_constraint_phrase_hit_count",
)
CORRECTED_INPUT_HASH_KEYS = {
    "source_manifest",
    "v2_train",
    "v2_eval",
    "refresh1_train",
    "refresh1_eval",
    "day14_forbidden_cases",
}
CORRECTED_CROSS_AUDIT_BLOCKERS = {
    "exact_prompt",
    "exact_indexed_concept",
    "concept_phrase",
    "constraint_phrase",
    "exact_template_signature_strict",
    "wildcard_template_signature_strict",
    "source_path",
    "source_basename",
    "source_sample_id",
    "recipe_paths_missing_from_locked_v2",
}
CORRECTED_SUBTASKS = {
    "global_style",
    "remove_object",
    "add_object",
    "replace_object",
    "change_background",
    "change_color",
    "change_weather",
    "add_effect",
    "customization",
    "combined_tasks",
    "camera_edit",
}
CORRECTED_FULL_RECIPE_REQUIRED_FILES = (
    "recipe2/lora-refresh/adapter_config.json",
    "recipe2/lora-refresh/adapter_model.safetensors",
    "recipe2/lora-refresh/train.log",
    "recipe2/lora-refresh/trainer_log.jsonl",
    "recipe2/lora-refresh/trainer_state.json",
    "recipe2/lora-refresh/exit_code",
    "recipe2/lora-refresh-selected/adapter_config.json",
    "recipe2/lora-refresh-selected/adapter_model.safetensors",
    "recipe2/metadata/refresh_train.jsonl",
    "recipe2/metadata/refresh_eval.jsonl",
    "recipe2/metadata/refresh_generation_audit.json",
    "recipe2/metadata/train_refresh.yaml",
    "recipe2/metadata/refresh_selection_policy.json",
    "recipe2/metadata/refresh_selection.json",
)
CORRECTED_COMMON_REQUIRED_FILES = (
    "fresh384/cases.jsonl",
    "fresh384/gold.jsonl",
    "fresh384/leakage_audit.json",
    "fresh384/selection_policy.json",
    "selection/comparison.json",
    "selection/selection.json",
    "selection/adapter_identities.json",
    "selection/endpoint_configs/v2.adapter_config.json",
    "selection/endpoint_configs/refresh1.adapter_config.json",
    "selection/endpoint_configs/recipe2.adapter_config.json",
    "fresh384/recipe2_cross_audit.json",
    "adaptive_day14/gold.jsonl",
    "adaptive_day14/agent_pipeline_records.jsonl",
    "adaptive_day14/planner.log",
    "adaptive_day14/metrics.json",
    "adaptive_day14/scorer.log",
    "adaptive_day14/gate_summary.json",
    "checksums.sha256",
)
CORRECTED_FULL_REQUIRED_FILES = (
    *CORRECTED_FULL_RECIPE_REQUIRED_FILES,
    *CORRECTED_COMMON_REQUIRED_FILES,
)
CORRECTED_THIN_REQUIRED_FILES = (
    "recipe2/lora-refresh/train.log",
    "recipe2/lora-refresh/trainer_log.jsonl",
    "recipe2/lora-refresh/trainer_state.json",
    "recipe2/lora-refresh/exit_code",
    "recipe2/metadata/refresh_train.jsonl",
    "recipe2/metadata/refresh_eval.jsonl",
    "recipe2/metadata/refresh_generation_audit.json",
    "recipe2/metadata/train_refresh.yaml",
    "recipe2/metadata/refresh_selection_policy.json",
    "recipe2/metadata/refresh_selection.json",
    "recipe2/metadata/selected_adapter_identity.json",
    *CORRECTED_COMMON_REQUIRED_FILES,
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


def _require_nonempty_files(root: Path, relatives: Iterable[str], errors: list[str]) -> None:
    for relative in relatives:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing required file: {relative}")
        elif path.stat().st_size == 0:
            errors.append(f"empty required file: {relative}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _lambda_slug(value: float) -> str:
    return f"lambda_{round(value * 1000):04d}"


def _valid_corrected_plan(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "refined_text_instruction",
        "subtask",
        "image_search",
        "mask",
    }:
        return False
    if not isinstance(value["refined_text_instruction"], str) or not value[
        "refined_text_instruction"
    ].strip():
        return False
    if value["subtask"] not in CORRECTED_SUBTASKS:
        return False
    return all(
        item is False or (isinstance(item, str) and bool(item.strip()))
        for item in (value["image_search"], value["mask"])
    )


def _audit_recipe2(
    root: Path,
    errors: list[str],
    observations: dict[str, Any],
    *,
    thin: bool = False,
) -> dict[str, str] | None:
    """Audit the bounded recipe-2 retrain and return selected adapter hashes."""

    output = root / "recipe2/lora-refresh"
    selected = root / "recipe2/lora-refresh-selected"
    metadata = root / "recipe2/metadata"
    checkpoint_names = sorted(
        path.name for path in output.glob("checkpoint-*") if path.is_dir()
    )
    expected_names = [f"checkpoint-{step}" for step in REFRESH_CHECKPOINT_STEPS]
    observations["recipe2_checkpoint_dirs"] = checkpoint_names
    if set(checkpoint_names) != set(expected_names) or len(checkpoint_names) != 4:
        errors.append(
            f"recipe2 checkpoints are {checkpoint_names!r}, expected exactly {expected_names!r}"
        )
    for step in REFRESH_CHECKPOINT_STEPS:
        checkpoint = output / f"checkpoint-{step}"
        checkpoint_files = ("trainer_state.json",) if thin else REFRESH_CHECKPOINT_FILES
        for name in checkpoint_files:
            path = checkpoint / name
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"recipe2 checkpoint-{step} is missing non-empty {name}")
        if thin:
            forbidden = set(REFRESH_CHECKPOINT_FILES) - {"trainer_state.json"}
            present = sorted(name for name in forbidden if (checkpoint / name).exists())
            if present:
                errors.append(
                    f"thin recipe2 checkpoint-{step} retains reconstructible blobs: {present}"
                )

    if thin:
        forbidden_top_level = (
            output / "adapter_config.json",
            output / "adapter_model.safetensors",
            selected / "adapter_config.json",
            selected / "adapter_model.safetensors",
        )
        present = [
            path.relative_to(root).as_posix()
            for path in forbidden_top_level
            if path.exists() or path.is_symlink()
        ]
        if present:
            errors.append(f"thin recipe2 layout retains model blobs/configs: {present}")
        if selected.exists() or selected.is_symlink():
            errors.append("thin recipe2 layout must omit lora-refresh-selected")

    exit_code = output / "exit_code"
    if exit_code.is_file():
        try:
            value = int(exit_code.read_text(encoding="utf-8").strip())
            observations["recipe2_training_exit_code"] = value
            if value != 0:
                errors.append(f"recipe2 training exit code is {value}, expected 0")
        except (OSError, ValueError) as exc:
            errors.append(f"invalid recipe2 training exit_code: {exc}")

    split_rows: dict[str, list[dict[str, Any]]] = {}
    for name, expected in (("refresh_train.jsonl", 1_024), ("refresh_eval.jsonl", 256)):
        path = metadata / name
        if path.is_file():
            rows = _read_jsonl_objects(path, errors)
            split_rows[name] = rows
            observations[f"recipe2_rows:{name}"] = len(rows)
            if len(rows) != expected:
                errors.append(f"recipe2 {name}: found {len(rows)} rows, expected {expected}")
    if set(split_rows) == {"refresh_train.jsonl", "refresh_eval.jsonl"}:
        try:
            train_videos = {str(row["videos"][0]) for row in split_rows["refresh_train.jsonl"]}
            eval_videos = {str(row["videos"][0]) for row in split_rows["refresh_eval.jsonl"]}
            overlap = train_videos & eval_videos
            observations["recipe2_video_overlap"] = len(overlap)
            if overlap:
                errors.append(f"recipe2 train/eval source-video overlap: {len(overlap)}")
        except (KeyError, IndexError, TypeError) as exc:
            errors.append(f"cannot audit recipe2 grouped split: {exc}")

    generation_path = metadata / "refresh_generation_audit.json"
    if generation_path.is_file():
        generation = _load_json(generation_path, errors)
        if generation is not None:
            if generation.get("recipe_version") != 2:
                errors.append("recipe2 generation audit must record recipe_version 2")
            counts = generation.get("counts")
            expected_counts = {
                "train": 1_024,
                "validation": 256,
                "validation_cases": 256,
                "validation_gold": 256,
            }
            if not isinstance(counts, dict):
                errors.append("recipe2 generation audit must contain a counts object")
            else:
                for key, expected in expected_counts.items():
                    if counts.get(key) != expected:
                        errors.append(
                            f"recipe2 generation audit counts.{key} is "
                            f"{counts.get(key)!r}, expected {expected}"
                        )
            isolation = generation.get("isolation")
            if not isinstance(isolation, dict):
                errors.append("recipe2 generation audit must contain an isolation object")
            else:
                for key in ("video_overlap", "normalized_prompt_overlap", "concept_overlap", "template_overlap"):
                    if isolation.get(key) != 0:
                        errors.append(
                            f"recipe2 generation audit isolation.{key} is "
                            f"{isolation.get(key)!r}, expected 0"
                        )
            forbidden = generation.get("forbidden_audit")
            if not isinstance(forbidden, dict):
                errors.append("recipe2 generation audit must contain forbidden_audit")
            else:
                for key in ("exact_prompt_overlap", "phrase_hit_count"):
                    if forbidden.get(key) != 0:
                        errors.append(
                            f"recipe2 generation audit forbidden_audit.{key} is "
                            f"{forbidden.get(key)!r}, expected 0"
                        )
                paired = forbidden.get("synthetic_prompt_target_ngram_overlap")
                if not isinstance(paired, dict):
                    errors.append("recipe2 generation audit lacks paired synthetic n-gram audit")
                else:
                    if paired.get("n") != 4:
                        errors.append("recipe2 paired synthetic n-gram audit must use n=4")
                    if paired.get("hit_count") != 0 or paired.get("hits") != []:
                        errors.append("recipe2 paired synthetic n-gram audit must have zero hits")

    config_path = metadata / "train_refresh.yaml"
    if config_path.is_file():
        _audit_refresh_config(config_path, errors)

    policy_path = metadata / "refresh_selection_policy.json"
    policy = _load_json(policy_path, errors) if policy_path.is_file() else None
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
                    f"recipe2 selection policy {key} is {policy.get(key)!r}, expected {expected!r}"
                )
        for key, name in (("train_sha256", "refresh_train.jsonl"), ("eval_sha256", "refresh_eval.jsonl")):
            path = metadata / name
            if path.is_file() and policy.get(key) != sha256(path):
                errors.append(f"recipe2 selection policy {key} does not match {name}")

    selection_path = metadata / "refresh_selection.json"
    selection = _load_json(selection_path, errors) if selection_path.is_file() else None
    if selection is None:
        return None
    for key, expected in {
        "selection_source": "refresh_eval",
        "selection_metric": "eval_loss",
        "selection_policy": REFRESH_SELECTION_POLICY,
        "external_gate_metrics_used": False,
    }.items():
        if selection.get(key) != expected:
            errors.append(
                f"recipe2 selection result {key} is {selection.get(key)!r}, expected {expected!r}"
            )
    raw_candidates = selection.get("candidates")
    parsed: list[tuple[int, float]] = []
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 4:
        errors.append("recipe2 selection must contain exactly four candidates")
    else:
        for candidate in raw_candidates:
            if not isinstance(candidate, dict):
                errors.append("recipe2 selection candidate must be an object")
                continue
            step, loss = candidate.get("step"), candidate.get("eval_loss")
            if step not in REFRESH_CHECKPOINT_STEPS or not _is_number(loss):
                errors.append(f"invalid recipe2 selection candidate: {candidate!r}")
                continue
            checkpoint_value = candidate.get("checkpoint")
            if not isinstance(checkpoint_value, str) or Path(checkpoint_value).name != f"checkpoint-{step}":
                errors.append(f"recipe2 candidate step {step} names the wrong checkpoint")
            parsed.append((int(step), float(loss)))
            state_path = output / f"checkpoint-{step}/trainer_state.json"
            if state_path.is_file():
                state = _load_json(state_path, errors)
                history = state.get("log_history", []) if state is not None else []
                losses = [
                    float(item["eval_loss"])
                    for item in history
                    if isinstance(item, dict)
                    and item.get("step") == step
                    and _is_number(item.get("eval_loss"))
                ]
                if not losses or not math.isclose(losses[-1], float(loss), rel_tol=0.0, abs_tol=1e-12):
                    errors.append(
                        f"recipe2 candidate step {step} eval_loss does not match trainer_state.json"
                    )
    if sorted(step for step, _ in parsed) != list(REFRESH_CHECKPOINT_STEPS):
        errors.append("recipe2 selection candidates must cover steps 32, 64, 96, and 128 once")
        return None
    expected_step, expected_loss = min(parsed, key=lambda item: (item[1], item[0]))
    if selection.get("selected_step") != expected_step:
        errors.append(
            f"recipe2 selected step is {selection.get('selected_step')!r}, expected {expected_step}"
        )
    if not _is_number(selection.get("selected_eval_loss")) or not math.isclose(
        float(selection.get("selected_eval_loss", 0.0)), expected_loss, rel_tol=0.0, abs_tol=1e-12
    ):
        errors.append("recipe2 selected_eval_loss does not match the winning checkpoint")
    checkpoint_value = selection.get("selected_checkpoint")
    if not isinstance(checkpoint_value, str) or Path(checkpoint_value).name != f"checkpoint-{expected_step}":
        errors.append("recipe2 selected_checkpoint does not match selected_step")
    source = output / f"checkpoint-{expected_step}"
    if not thin:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            source_file, selected_file = source / name, selected / name
            if source_file.is_file() and selected_file.is_file() and sha256(source_file) != sha256(selected_file):
                errors.append(f"recipe2 selected adapter {name} does not match checkpoint-{expected_step}")
    observations["recipe2_selection"] = {
        "selected_step": expected_step,
        "selected_eval_loss": expected_loss,
    }
    if thin:
        identity_path = metadata / "selected_adapter_identity.json"
        identity_value = _load_json(identity_path, errors) if identity_path.is_file() else None
        if identity_value is None:
            return None
        if identity_value.get("schema_version") != 1:
            errors.append("recipe2 selected adapter identity must use schema_version 1")
        if identity_value.get("selected_step") != expected_step:
            errors.append("recipe2 selected adapter identity step mismatch")
        identity = {
            "adapter_model_sha256": identity_value.get("adapter_model_sha256"),
            "adapter_config_sha256": identity_value.get("adapter_config_sha256"),
        }
        if not all(_is_sha256(value) for value in identity.values()):
            errors.append("recipe2 selected adapter identity contains invalid hashes")
            return None
    else:
        model_path = selected / "adapter_model.safetensors"
        config_path = selected / "adapter_config.json"
        if not model_path.is_file() or not config_path.is_file():
            return None
        identity = {
            "adapter_model_sha256": sha256(model_path),
            "adapter_config_sha256": sha256(config_path),
        }
    observations["recipe2_selected_adapter"] = identity
    return identity


def _audit_fresh384(
    root: Path, recipe2_identity: dict[str, str] | None, errors: list[str], observations: dict[str, Any]
) -> dict[str, Any] | None:
    fresh = root / "fresh384"
    cases_path, gold_path = fresh / "cases.jsonl", fresh / "gold.jsonl"
    cases = _read_jsonl_objects(cases_path, errors) if cases_path.is_file() else []
    gold = _read_jsonl_objects(gold_path, errors) if gold_path.is_file() else []
    if len(cases) != 384:
        errors.append(f"fresh384 cases has {len(cases)} rows, expected 384")
    if len(gold) != 384:
        errors.append(f"fresh384 gold has {len(gold)} rows, expected 384")
    case_ids = [row.get("bench_id") for row in cases]
    gold_ids = [row.get("bench_id") for row in gold]
    valid_case_ids = all(isinstance(value, str) and bool(value) for value in case_ids)
    if not valid_case_ids or len(set(case_ids)) != len(case_ids):
        errors.append("fresh384 cases must have unique non-empty bench_id values")
    elif any(re.fullmatch(r"interp_\d{4}", value) is None for value in case_ids):
        errors.append("fresh384 contains a non-interpolation bench_id")
    if case_ids != gold_ids:
        errors.append("fresh384 cases/gold bench_id order does not match")
    for case, gold_row in zip(cases, gold):
        for key in (
            "prompt",
            "video_path",
            "axis",
            "category",
            "subtype",
            "edit_type",
            "source",
            "catalog",
        ):
            if case.get(key) != gold_row.get(key):
                errors.append(
                    f"fresh384 cases/gold {key} mismatch for {case.get('bench_id')!r}"
                )
                break
        if not _valid_corrected_plan(gold_row.get("gold_plan")):
            errors.append(f"fresh384 gold has invalid plan for {gold_row.get('bench_id')!r}")
            continue
        plan = gold_row["gold_plan"]
        search_triggered = isinstance(plan["image_search"], str)
        if search_triggered != (gold_row.get("category") == "true_search_positive"):
            errors.append(
                "all and only fresh384 true_search_positive rows must trigger gold search: "
                f"{gold_row.get('bench_id')!r}"
            )
        if gold_row.get("category") == "rewrite_retention":
            constraints = gold_row.get("constraints")
            if not isinstance(constraints, list) or len(constraints) != 4:
                errors.append(
                    f"fresh384 rewrite row must contain four constraints: {gold_row.get('bench_id')!r}"
                )
    categories = {key: 0 for key in CORRECTED_CATEGORY_COUNTS}
    for row in cases:
        category = row.get("category")
        if category in categories:
            categories[str(category)] += 1
        else:
            errors.append(f"fresh384 case has unexpected category: {category!r}")
    if categories != CORRECTED_CATEGORY_COUNTS:
        errors.append(f"fresh384 category counts are {categories!r}, expected {CORRECTED_CATEGORY_COUNTS!r}")
    gold_categories = {
        key: sum(row.get("category") == key for row in gold)
        for key in CORRECTED_CATEGORY_COUNTS
    }
    if gold_categories != CORRECTED_CATEGORY_COUNTS:
        errors.append("fresh384 gold category counts do not match the locked allocation")
    no_search_counts = {
        key: sum(
            row.get("category") == "no_search_negative" and row.get("subtype") == key
            for row in cases
        )
        for key in CORRECTED_NO_SEARCH_COUNTS
    }
    if no_search_counts != CORRECTED_NO_SEARCH_COUNTS:
        errors.append("fresh384 no-search subtype counts do not match the locked allocation")
    mask_counts = {
        key: sum(
            row.get("category") == "mask_control" and row.get("subtype") == key
            for row in cases
        )
        for key in CORRECTED_MASK_COUNTS
    }
    if mask_counts != CORRECTED_MASK_COUNTS:
        errors.append("fresh384 mask subtype counts do not match the locked allocation")
    gold_mask_counts = {
        "triggered": sum(
            row.get("category") == "mask_control"
            and isinstance(row.get("gold_plan"), dict)
            and isinstance(row["gold_plan"].get("mask"), str)
            for row in gold
        ),
        "not_triggered": sum(
            row.get("category") == "mask_control"
            and isinstance(row.get("gold_plan"), dict)
            and row["gold_plan"].get("mask") is False
            for row in gold
        ),
    }
    if gold_mask_counts != CORRECTED_MASK_COUNTS:
        errors.append("fresh384 gold mask-trigger counts do not match the locked allocation")
    try:
        video_paths = [str(row["video_path"]) for row in cases]
        source_ids = [str(row["source"]["sample_id"]) for row in cases]
        video_hashes = [str(row["source"]["video_sha256"]) for row in cases]
        if not (len(set(video_paths)) == len(set(source_ids)) == len(set(video_hashes)) == 384):
            errors.append("fresh384 must use 384 path-, source-id-, and byte-unique videos")
        if any(not _is_sha256(value) for value in video_hashes):
            errors.append("fresh384 contains an invalid source-video SHA-256")
    except (KeyError, TypeError) as exc:
        errors.append(f"cannot audit fresh384 source identities: {exc}")

    audit_path = fresh / "leakage_audit.json"
    audit = _load_json(audit_path, errors) if audit_path.is_file() else None
    if audit is not None:
        if audit.get("recipe_version") != 2:
            errors.append("fresh384 leakage audit must record recipe_version 2")
        counts = audit.get("counts")
        if not isinstance(counts, dict):
            errors.append("fresh384 leakage audit lacks counts")
        else:
            if counts.get("cases") != 384 or counts.get("gold") != 384:
                errors.append("fresh384 leakage audit must record 384 cases and 384 gold rows")
            if counts.get("by_category") != dict(sorted(CORRECTED_CATEGORY_COUNTS.items())):
                errors.append("fresh384 leakage audit category counts are not locked")
            if counts.get("no_search_by_subtype") != dict(sorted(CORRECTED_NO_SEARCH_COUNTS.items())):
                errors.append("fresh384 leakage audit no-search counts are not locked")
            if counts.get("mask_trigger") != dict(sorted(CORRECTED_MASK_COUNTS.items())):
                errors.append("fresh384 leakage audit mask counts are not locked")
        isolation = audit.get("isolation")
        if not isinstance(isolation, dict):
            errors.append("fresh384 leakage audit lacks isolation")
        else:
            for key in CORRECTED_STRICT_LEAKAGE_FIELDS:
                if isolation.get(key) != 0:
                    errors.append(
                        f"fresh384 strict leakage {key} is {isolation.get(key)!r}, expected 0"
                    )
            if isolation.get("explicit_forbidden_phrase_hits") != []:
                errors.append("fresh384 leakage audit has explicit forbidden phrase hits")
            for prefix in ("allowed_generic_concept_phrase", "allowed_generic_template_signature"):
                hits = isolation.get(f"{prefix}_hits")
                count = isolation.get(f"{prefix}_hit_count")
                if not isinstance(hits, list) or count != len(hits):
                    errors.append(f"fresh384 leakage audit has inconsistent {prefix} records")
                    continue
                if any(
                    not isinstance(item, dict)
                    or item.get("category") not in {"routing_control", "mask_control"}
                    for item in hits
                ):
                    errors.append(f"fresh384 leakage audit {prefix} includes a strict category")
            concept_policy = isolation.get("concept_phrase_policy")
            concept_audit_categories = (
                concept_policy.get("phrase_containment_audit_only_categories")
                if isinstance(concept_policy, dict)
                else None
            )
            if (
                not isinstance(concept_policy, dict)
                or concept_policy.get("exact_concept_overlap_forbidden_for_all_categories")
                is not True
                or not isinstance(concept_audit_categories, list)
                or set(concept_audit_categories) != {"routing_control", "mask_control"}
            ):
                errors.append("fresh384 concept phrase policy is missing or weakened")
            template_policy = isolation.get("template_signature_policy")
            template_audit_categories = (
                template_policy.get("wildcard_signature_overlap_audit_only_categories")
                if isinstance(template_policy, dict)
                else None
            )
            if (
                not isinstance(template_policy, dict)
                or template_policy.get(
                    "exact_prompt_and_template_id_overlap_forbidden_for_all_categories"
                )
                is not True
                or not isinstance(template_audit_categories, list)
                or set(template_audit_categories) != {"routing_control", "mask_control"}
            ):
                errors.append("fresh384 template-signature policy is missing or weakened")
        artifacts = audit.get("artifacts")
        if not isinstance(artifacts, dict):
            errors.append("fresh384 leakage audit lacks artifact hashes")
        else:
            if cases_path.is_file() and artifacts.get("cases_sha256") != sha256(cases_path):
                errors.append("fresh384 leakage audit cases_sha256 mismatch")
            if gold_path.is_file() and artifacts.get("gold_sha256") != sha256(gold_path):
                errors.append("fresh384 leakage audit gold_sha256 mismatch")
            inputs = artifacts.get("input_sha256")
            if not isinstance(inputs, dict) or set(inputs) != CORRECTED_INPUT_HASH_KEYS or any(
                not _is_sha256(value) for value in inputs.values()
            ):
                errors.append("fresh384 leakage audit input hashes are incomplete or invalid")

    policy_path = fresh / "selection_policy.json"
    policy = _load_json(policy_path, errors) if policy_path.is_file() else None
    if policy is None:
        return None
    if policy.get("version") != 2 or policy.get("created_before_candidate_inference") is not True:
        errors.append("fresh384 policy must be pre-registered version 2")
    candidate_rule = policy.get("candidate_rule")
    if not isinstance(candidate_rule, dict):
        errors.append("fresh384 policy lacks candidate_rule")
        candidate_rule = {}
    expected_grid = list(CORRECTED_LAMBDA_GRID)
    for key, expected in {
        "method": "exact_parameter_delta_interpolation",
        "lambda_grid": expected_grid,
        "candidate_count": 9,
        "additional_training_allowed": False,
        "day14_outputs_or_metrics_allowed_for_generation_or_selection": False,
    }.items():
        if candidate_rule.get(key) != expected:
            errors.append(f"fresh384 policy candidate_rule.{key} is not locked")
    for key in ("v2_adapter_model_sha256", "refresh1_adapter_model_sha256"):
        if not _is_sha256(candidate_rule.get(key)):
            errors.append(f"fresh384 policy has invalid {key}")
    validation = policy.get("validation_artifacts")
    if not isinstance(validation, dict):
        errors.append("fresh384 policy lacks validation_artifacts")
        validation = {}
    expected_validation = {
        "expected_cases": 384,
        "expected_unique_videos": 384,
        "expected_category_counts": CORRECTED_CATEGORY_COUNTS,
        "expected_no_search_subtype_counts": CORRECTED_NO_SEARCH_COUNTS,
        "expected_mask_trigger_counts": CORRECTED_MASK_COUNTS,
    }
    for key, expected in expected_validation.items():
        if validation.get(key) != expected:
            errors.append(f"fresh384 policy validation_artifacts.{key} is not locked")
    for key, path in (
        ("cases_sha256", cases_path),
        ("gold_sha256", gold_path),
        ("leakage_audit_sha256", audit_path),
    ):
        if path.is_file() and validation.get(key) != sha256(path):
            errors.append(f"fresh384 policy {key} mismatch")
    if (
        audit is not None
        and isinstance(audit.get("artifacts"), dict)
        and validation.get("input_sha256") != audit["artifacts"].get("input_sha256")
    ):
        errors.append("fresh384 policy/audit input hashes disagree")
    final = policy.get("final_decision")
    if not isinstance(final, dict) or final.get("rule") != "primary_if_eligible_else_grid":
        errors.append("fresh384 policy must use primary_if_eligible_else_grid")
        primary: dict[str, Any] = {}
    else:
        primary_value = final.get("primary_candidate")
        primary = primary_value if isinstance(primary_value, dict) else {}
    if primary.get("candidate_id") != "recipe2" or primary.get("directory_name") != "recipe2":
        errors.append("fresh384 primary candidate must be candidates/recipe2")
    if not _is_sha256(primary.get("adapter_model_sha256")):
        errors.append("fresh384 primary recipe2 hash is invalid")
    elif recipe2_identity is not None and primary["adapter_model_sha256"] != recipe2_identity["adapter_model_sha256"]:
        errors.append("fresh384 primary recipe2 hash does not match selected recipe2 checkpoint")
    selection_rule = policy.get("selection")
    if not isinstance(selection_rule, dict):
        errors.append("fresh384 selection policy lacks selection")
    else:
        if selection_rule.get("rule") != "one_standard_error_then_smallest_lambda":
            errors.append("fresh384 selection policy has the wrong one-SE rule")
        if selection_rule.get("reference_tie_breaker") != "smallest lambda" or selection_rule.get("tie_breaker") != "smallest lambda":
            errors.append("fresh384 selection policy tie-breakers are not locked")
        if selection_rule.get("day14_metrics_used") is not False:
            errors.append("fresh384 selection policy must exclude Day-14 metrics")
    if policy.get("eligibility_thresholds") != CORRECTED_ELIGIBILITY_THRESHOLDS:
        errors.append("fresh384 eligibility thresholds are not locked")
    utility = policy.get("utility")
    if not isinstance(utility, dict) or utility.get("formula") != "0.5*no_search_specificity+0.5*rewrite_constraint_retention" or utility.get("higher_is_better") is not True:
        errors.append("fresh384 utility rule is not locked")
    bootstrap = policy.get("bootstrap")
    if (
        not isinstance(bootstrap, dict)
        or bootstrap.get("method")
        != "paired stratified case bootstrap over the two utility strata"
        or bootstrap.get("seed") != 20260916
        or bootstrap.get("draws") != 10_000
        or bootstrap.get("strata")
        != {"no_search_negative": 128, "rewrite_retention": 64}
    ):
        errors.append("fresh384 bootstrap policy is not locked")
    observations["fresh384"] = {
        "cases": len(cases),
        "categories": categories,
        "strict_leakage_zero": not any(
            isinstance(audit, dict)
            and isinstance(audit.get("isolation"), dict)
            and audit["isolation"].get(key) != 0
            for key in CORRECTED_STRICT_LEAKAGE_FIELDS
        ),
    }
    return {
        "policy": policy,
        "case_ids": {str(value) for value in case_ids if isinstance(value, str)},
        "cases": cases,
        "gold": gold,
        "cases_sha256": sha256(cases_path) if cases_path.is_file() else None,
        "gold_sha256": sha256(gold_path) if gold_path.is_file() else None,
        "audit_sha256": sha256(audit_path) if audit_path.is_file() else None,
        "policy_sha256": sha256(policy_path),
        "v2_sha256": candidate_rule.get("v2_adapter_model_sha256"),
        "refresh1_sha256": candidate_rule.get("refresh1_adapter_model_sha256"),
        "primary": primary,
    }


def _audit_complete_eval_run(
    eval_dir: Path,
    *,
    expected_cases: int,
    expected_ids: set[str],
    case_rows: list[dict[str, Any]],
    cases_sha256: str,
    gold_sha256: str,
    adapter_config_sha256: str,
    adapter_model_sha256: str,
    provenance_sha256: str | None,
    errors: list[str],
    label: str,
) -> dict[str, str] | None:
    required = {
        "predictions": eval_dir / "agent_pipeline_records.jsonl",
        "planner_log": eval_dir / "planner.log",
        "metrics": eval_dir / "metrics.json",
        "scorer_log": eval_dir / "scorer.log",
    }
    manifest_path = eval_dir / "run_manifest.json"
    for name, path in {**required, "run_manifest": manifest_path}.items():
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"{label} missing non-empty {name}: {path.name}")
    if not all(path.is_file() for path in required.values()) or not manifest_path.is_file():
        return None

    rows = _read_jsonl_objects(required["predictions"], errors)
    ids = [row.get("bench_id") for row in rows]
    if len(rows) != expected_cases:
        errors.append(f"{label} has {len(rows)} predictions, expected {expected_cases}")
    valid_ids = all(isinstance(value, str) and bool(value) for value in ids)
    if not valid_ids or len(set(ids)) != len(ids) or set(ids) != expected_ids:
        errors.append(f"{label} prediction IDs are incomplete or duplicated")
    if any("error" in row for row in rows):
        errors.append(f"{label} predictions contain per-case errors")
    metrics = _load_json(required["metrics"], errors)
    if metrics is not None and metrics.get("num_cases") != expected_cases:
        errors.append(f"{label} metrics num_cases is not {expected_cases}")

    manifest = _load_json(manifest_path, errors)
    if manifest is None:
        return None
    if manifest.get("schema_version") != 1 or manifest.get("runner") != "scripts.run_planner_eval":
        errors.append(f"{label} has an unsupported run manifest")
    if manifest.get("status") != "complete" or manifest.get("error") is not None:
        errors.append(f"{label} run manifest is not complete/error-free")
    if manifest.get("expected_cases") != expected_cases:
        errors.append(f"{label} run manifest expected_cases mismatch")
    exit_state = manifest.get("exit_state")
    expected_exit = {
        "stage": "complete",
        "agent_returncode": 0,
        "predictions_validated": True,
        "scorer_returncode": 0,
        "metrics_validated": True,
        "inputs_reverified": True,
        "outputs_reverified": True,
    }
    if not isinstance(exit_state, dict):
        errors.append(f"{label} run manifest lacks exit_state")
    else:
        for key, expected in expected_exit.items():
            if exit_state.get(key) != expected:
                errors.append(f"{label} exit_state.{key} is not {expected!r}")

    inputs = manifest.get("inputs")
    runtime_identity: dict[str, str] | None = None
    if not isinstance(inputs, dict):
        errors.append(f"{label} run manifest lacks inputs")
    else:
        expected_inputs = {
            "cases": cases_sha256,
            "gold": gold_sha256,
            "adapter_config": adapter_config_sha256,
            "adapter_weights": adapter_model_sha256,
        }
        for key, expected in expected_inputs.items():
            entry = inputs.get(key)
            if not isinstance(entry, dict) or entry.get("sha256") != expected:
                errors.append(f"{label} run manifest input {key} hash mismatch")
        expected_source_hashes: dict[str, str] = {}
        expected_reference_keys: set[str] = set()
        for case in case_rows:
            bench_id = case.get("bench_id")
            source = case.get("source")
            if isinstance(bench_id, str) and isinstance(source, dict):
                source_hash = source.get("video_sha256")
                if _is_sha256(source_hash):
                    expected_source_hashes[f"source_video:{bench_id}"] = source_hash
            if isinstance(bench_id, str) and case.get("ref_image_path") not in (None, ""):
                expected_reference_keys.add(f"reference_image:{bench_id}")
        core_keys = {*expected_inputs, "base_config"}
        expected_input_keys = core_keys | set(expected_source_hashes) | expected_reference_keys
        provenance_entry = inputs.get("adapter_provenance")
        if provenance_sha256 is None:
            if provenance_entry is not None:
                errors.append(f"{label} unexpectedly records interpolation provenance")
        elif not isinstance(provenance_entry, dict) or provenance_entry.get("sha256") != provenance_sha256:
            errors.append(f"{label} run manifest provenance hash mismatch")
        else:
            expected_input_keys.add("adapter_provenance")
        if set(inputs) != expected_input_keys:
            errors.append(f"{label} run manifest input key set is incomplete or unexpected")
        base_entry = inputs.get("base_config")
        if (
            not isinstance(base_entry, dict)
            or not _is_sha256(base_entry.get("sha256"))
            or not isinstance(base_entry.get("path"), str)
            or Path(base_entry["path"]).name != "config.json"
        ):
            errors.append(f"{label} run manifest has an invalid base_config identity")
        for key, expected in expected_source_hashes.items():
            entry = inputs.get(key)
            if not isinstance(entry, dict) or entry.get("sha256") != expected:
                errors.append(f"{label} run manifest source-video hash mismatch for {key}")
        for key in expected_reference_keys:
            entry = inputs.get(key)
            if not isinstance(entry, dict) or not _is_sha256(entry.get("sha256")):
                errors.append(f"{label} run manifest reference-image identity is invalid for {key}")
        runtime_keys = {"base_config", *expected_source_hashes, *expected_reference_keys}
        if all(
            isinstance(inputs.get(key), dict) and _is_sha256(inputs[key].get("sha256"))
            for key in runtime_keys
        ):
            runtime_identity = {
                key: str(inputs[key]["sha256"]) for key in sorted(runtime_keys)
            }

        planner_text = required["planner_log"].read_text(encoding="utf-8")
        planner_lines = planner_text.splitlines()
        config_entry = inputs.get("adapter_config")
        for declaration, entry, config_name in (
            ("base", base_entry, "config.json"),
            ("adapter", config_entry, "adapter_config.json"),
        ):
            entry_path = entry.get("path") if isinstance(entry, dict) else None
            expected_line = (
                f"Agent {declaration}: {Path(entry_path).parent}"
                if isinstance(entry_path, str) and Path(entry_path).name == config_name
                else None
            )
            observed_lines = [
                line
                for line in planner_lines
                if line.startswith(f"Agent {declaration}:")
            ]
            if expected_line is None or observed_lines != [expected_line]:
                errors.append(
                    f"{label} planner log does not uniquely bind the {declaration} path"
                )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        errors.append(f"{label} run manifest lacks outputs")
    else:
        for key, path in required.items():
            entry = outputs.get(key)
            if not isinstance(entry, dict) or entry.get("sha256") != sha256(path):
                errors.append(f"{label} run manifest output {key} hash mismatch")
    return runtime_identity


def _audit_recipe2_cross_audit(
    root: Path, errors: list[str], observations: dict[str, Any]
) -> None:
    """Validate the immutable post-seal recipe2/fresh384 isolation supplement."""

    path = root / "fresh384/recipe2_cross_audit.json"
    audit = _load_json(path, errors) if path.is_file() else None
    if audit is None:
        return
    if audit.get("schema_version") != 1:
        errors.append("recipe2 cross-audit must use schema_version 1")
    if audit.get("audit_role") != "supplemental_post_seal":
        errors.append("recipe2 cross-audit must declare supplemental_post_seal")
    if audit.get("passed") is not True:
        errors.append("recipe2 cross-audit must record passed=true")
    method = audit.get("method")
    if not isinstance(method, dict) or method.get("paired_ngram_role") != "diagnostic_only":
        errors.append("recipe2 cross-audit must keep paired n-grams diagnostic-only")
    sealed_paths = {
        "cases_sha256": root / "fresh384/cases.jsonl",
        "gold_sha256": root / "fresh384/gold.jsonl",
        "selection_policy_sha256": root / "fresh384/selection_policy.json",
    }
    recipe2_paths = {
        "train_sha256": root / "recipe2/metadata/refresh_train.jsonl",
        "eval_sha256": root / "recipe2/metadata/refresh_eval.jsonl",
    }
    if not all(item.is_file() for item in (*sealed_paths.values(), *recipe2_paths.values())):
        return
    expected_sealed = {key: sha256(item) for key, item in sealed_paths.items()}
    if audit.get("sealed_artifacts") != expected_sealed:
        errors.append("recipe2 cross-audit sealed artifact hashes mismatch")
    expected_recipe2 = {key: sha256(item) for key, item in recipe2_paths.items()}
    if audit.get("recipe2_artifacts") != expected_recipe2:
        errors.append("recipe2 cross-audit train/eval hashes mismatch")
    blockers = audit.get("blocking_overlaps")
    if not isinstance(blockers, dict) or set(blockers) != CORRECTED_CROSS_AUDIT_BLOCKERS:
        errors.append("recipe2 cross-audit blocker key set is not locked")
    elif any(
        isinstance(value, bool) or not isinstance(value, int) or value != 0
        for value in blockers.values()
    ):
        errors.append("recipe2 cross-audit has a non-zero blocking overlap")
    blocking_examples = audit.get("blocking_examples")
    if not isinstance(blocking_examples, dict) or set(blocking_examples) != CORRECTED_CROSS_AUDIT_BLOCKERS or any(
        value != [] for value in blocking_examples.values()
    ):
        errors.append("recipe2 cross-audit blocking examples are incomplete or non-empty")

    diagnostics = audit.get("diagnostics")
    expected_diagnostics = {
        "generic_template_overlaps",
        "paired_prompt_target_4gram",
        "concept_constraint_value_4gram",
        "source_proof",
        "counts",
    }
    if not isinstance(diagnostics, dict) or set(diagnostics) != expected_diagnostics:
        errors.append("recipe2 cross-audit diagnostic key set is incomplete")
        return
    generic = diagnostics["generic_template_overlaps"]
    if not isinstance(generic, dict):
        errors.append("recipe2 cross-audit generic-template diagnostics must be an object")
    else:
        hits = generic.get("hits")
        if generic.get("blocking") is not False or generic.get("allowed_categories") != [
            "mask_control",
            "routing_control",
        ]:
            errors.append("recipe2 cross-audit generic-template policy is not audit-only")
        for key in ("exact_hit_count", "wildcard_hit_count"):
            value = generic.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"recipe2 cross-audit generic diagnostics has invalid {key}")
        signatures = generic.get("unique_signatures")
        if not isinstance(signatures, list) or any(
            not isinstance(value, str) or not value for value in signatures
        ) or signatures != sorted(set(signatures)):
            errors.append("recipe2 cross-audit unique signatures are invalid")
        if not isinstance(hits, list) or any(
            not isinstance(hit, dict)
            or hit.get("category") not in {"routing_control", "mask_control"}
            for hit in hits
        ):
            errors.append("recipe2 cross-audit generic hits include a strict category")
    for key in ("paired_prompt_target_4gram", "concept_constraint_value_4gram"):
        item = diagnostics[key]
        if not isinstance(item, dict) or item.get("n") != 4:
            errors.append(f"recipe2 cross-audit {key} must be an n=4 object")
        elif not isinstance(item.get("examples"), list) or not isinstance(
            item.get("by_fresh_category", item.get("by_category")), dict
        ):
            errors.append(f"recipe2 cross-audit {key} lacks category counts/examples")
        elif item.get("blocking") is not False:
            errors.append(f"recipe2 cross-audit {key} must remain diagnostic-only")
    source_proof = diagnostics["source_proof"]
    if not isinstance(source_proof, dict):
        errors.append("recipe2 cross-audit source proof must be an object")
    else:
        if source_proof.get("recipe_paths_subset_locked_v2") is not True:
            errors.append("recipe2 cross-audit does not prove recipe paths are a v2 subset")
        if source_proof.get("fresh_prior_video_sha256_overlap") != 0:
            errors.append("recipe2 cross-audit source proof has fresh video overlap")
    expected_counts = {
        "fresh_cases": 384,
        "fresh_gold": 384,
        "recipe2_train_rows": 1_024,
        "recipe2_eval_rows": 256,
        "recipe2_rows": 1_280,
    }
    if diagnostics["counts"] != expected_counts:
        errors.append("recipe2 cross-audit counts do not match the locked datasets")
    observations["recipe2_cross_audit"] = {
        "audit_role": audit.get("audit_role"),
        "blocking_overlaps": blockers,
    }


def _audit_endpoint_configs(
    root: Path, errors: list[str], observations: dict[str, Any]
) -> dict[str, Any] | None:
    """Bind the bundled endpoints and derive the one legal grid config."""

    configs: dict[str, dict[str, Any]] = {}
    for name, expected_sha in CORRECTED_ADAPTER_CONFIG_SHA256.items():
        path = root / f"selection/endpoint_configs/{name}.adapter_config.json"
        value = _load_json(path, errors) if path.is_file() else None
        if value is None:
            continue
        observed_sha = sha256(path)
        if observed_sha != expected_sha:
            errors.append(
                f"endpoint config {name} differs from the locked production bytes"
            )
        configs[name] = value
    if set(configs) != {"v2", "refresh1", "recipe2"}:
        return None

    canonical = []
    for name in ("v2", "refresh1"):
        value = dict(configs[name])
        targets = value.get("target_modules")
        if not isinstance(targets, list) or any(
            not isinstance(item, str) or not item for item in targets
        ):
            errors.append(f"endpoint config {name} has invalid target_modules")
            return None
        value["target_modules"] = sorted(targets)
        canonical.append(value)
    if canonical[0] != canonical[1]:
        errors.append("v2 and refresh1 endpoint configs are not semantically compatible")
        return None
    source = configs["v2"]
    rank, alpha = source.get("r"), source.get("lora_alpha")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank <= 0
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or float(alpha) <= 0
    ):
        errors.append("v2 endpoint config has invalid LoRA rank/alpha")
        return None
    output = dict(source)
    output["r"] = rank * 2
    output["lora_alpha"] = alpha * 2
    output["target_modules"] = sorted(source["target_modules"])
    serialized = (
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    contract = {
        "serialized": serialized,
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "lora": {
            "input_rank": rank,
            "input_lora_alpha": alpha,
            "input_scaling": float(alpha) / rank,
            "output_rank": rank * 2,
            "output_lora_alpha": alpha * 2,
            "output_scaling": float(alpha * 2) / (rank * 2),
            "dtype": "float32",
        },
    }
    observations["grid_adapter_config_sha256"] = contract["sha256"]
    return contract


def _audit_adapter_identities(
    root: Path,
    fresh: dict[str, Any] | None,
    recipe2_identity: dict[str, str] | None,
    errors: list[str],
    observations: dict[str, Any],
) -> dict[str, dict[str, str]] | None:
    path = root / "selection/adapter_identities.json"
    value = _load_json(path, errors) if path.is_file() else None
    if value is None:
        return None
    if value.get("schema_version") != 1 or set(value) != {
        "schema_version",
        "v2",
        "refresh1",
        "recipe2",
    }:
        errors.append("adapter identities must use the locked schema-version-1 key set")
    identities: dict[str, dict[str, str]] = {}
    for name in ("v2", "refresh1", "recipe2"):
        identity = value.get(name)
        if not isinstance(identity, dict) or set(identity) != {
            "adapter_model_sha256",
            "adapter_config_sha256",
        } or not all(_is_sha256(item) for item in identity.values()):
            errors.append(f"adapter identity {name} is incomplete or invalid")
            continue
        identities[name] = dict(identity)
    if set(identities) != {"v2", "refresh1", "recipe2"}:
        return None
    for name in CORRECTED_ADAPTER_CONFIG_SHA256:
        config_path = root / f"selection/endpoint_configs/{name}.adapter_config.json"
        observed = sha256(config_path) if config_path.is_file() else None
        if identities[name]["adapter_config_sha256"] != observed:
            errors.append(
                f"adapter identity {name} config hash differs from the bundled endpoint config"
            )
    if fresh is not None:
        expected_models = {
            "v2": fresh["v2_sha256"],
            "refresh1": fresh["refresh1_sha256"],
            "recipe2": fresh["primary"].get("adapter_model_sha256"),
        }
        for name, expected in expected_models.items():
            if identities.get(name, {}).get("adapter_model_sha256") != expected:
                errors.append(f"adapter identity {name} model hash differs from locked policy")
    if recipe2_identity is not None and identities.get("recipe2") != recipe2_identity:
        errors.append("adapter identity recipe2 differs from selected recipe2 checkpoint")
    observations["adapter_identities"] = identities
    return identities


def _audit_candidates_and_selection(
    root: Path,
    fresh: dict[str, Any] | None,
    recipe2_identity: dict[str, str] | None,
    endpoint_identities: dict[str, dict[str, str]] | None,
    grid_contract: dict[str, Any] | None,
    errors: list[str],
    observations: dict[str, Any],
    *,
    thin: bool = False,
) -> dict[str, Any] | None:
    if fresh is None or endpoint_identities is None:
        return None
    candidates_root = root / "candidates"
    expected_grid_ids = [_lambda_slug(value) for value in CORRECTED_LAMBDA_GRID]
    expected_ids = [*expected_grid_ids, "recipe2"]
    actual_dirs = sorted(path.name for path in candidates_root.iterdir() if path.is_dir()) if candidates_root.is_dir() else []
    if set(actual_dirs) != set(expected_ids) or len(actual_dirs) != len(expected_ids):
        errors.append(f"candidate directories are {actual_dirs!r}, expected exactly {sorted(expected_ids)!r}")

    primary = fresh["primary"]
    comparison_path = root / "selection/comparison.json"
    selection_path = root / "selection/selection.json"
    comparison = _load_json(comparison_path, errors) if comparison_path.is_file() else None
    selection = _load_json(selection_path, errors) if selection_path.is_file() else None
    if comparison is None or selection is None:
        return None
    preselected_id = selection.get("selected_candidate_id")
    candidate_identity: dict[str, dict[str, Any]] = {}
    shared_runtime_identity: dict[str, str] | None = None
    for candidate_id in expected_ids:
        candidate = candidates_root / candidate_id
        adapter = candidate / "adapter"
        eval_dir = candidate / "eval"
        config_path = adapter / "adapter_config.json"
        model_path = adapter / "adapter_model.safetensors"
        if not config_path.is_file() or config_path.stat().st_size == 0:
            errors.append(f"candidate {candidate_id} missing non-empty adapter config")
        run_manifest_path = eval_dir / "run_manifest.json"
        run_manifest = (
            _load_json(run_manifest_path, errors) if run_manifest_path.is_file() else None
        )
        inputs = run_manifest.get("inputs") if isinstance(run_manifest, dict) else None
        config_entry = inputs.get("adapter_config") if isinstance(inputs, dict) else None
        model_entry = inputs.get("adapter_weights") if isinstance(inputs, dict) else None
        base_entry = inputs.get("base_config") if isinstance(inputs, dict) else None
        manifest_config_sha = (
            config_entry.get("sha256") if isinstance(config_entry, dict) else None
        )
        manifest_model_sha = (
            model_entry.get("sha256") if isinstance(model_entry, dict) else None
        )
        config_runtime_path = (
            config_entry.get("path") if isinstance(config_entry, dict) else None
        )
        model_runtime_path = (
            model_entry.get("path") if isinstance(model_entry, dict) else None
        )
        runtime_adapter_dir: str | None = None
        runtime_base_dir: str | None = None
        if (
            isinstance(config_runtime_path, str)
            and isinstance(model_runtime_path, str)
            and Path(config_runtime_path).name == "adapter_config.json"
            and Path(model_runtime_path).name == "adapter_model.safetensors"
            and Path(config_runtime_path).parent == Path(model_runtime_path).parent
        ):
            runtime_adapter_dir = str(Path(config_runtime_path).parent)
        else:
            errors.append(f"candidate {candidate_id} run manifest adapter paths disagree")
        base_runtime_path = base_entry.get("path") if isinstance(base_entry, dict) else None
        if isinstance(base_runtime_path, str) and Path(base_runtime_path).name == "config.json":
            runtime_base_dir = str(Path(base_runtime_path).parent)
        else:
            errors.append(f"candidate {candidate_id} run manifest base path is invalid")
        if not _is_sha256(manifest_config_sha) or not _is_sha256(manifest_model_sha):
            errors.append(f"candidate {candidate_id} run manifest lacks model hashes")
        if not config_path.is_file() or not _is_sha256(manifest_config_sha) or not _is_sha256(manifest_model_sha):
            continue
        config_sha, model_sha = sha256(config_path), str(manifest_model_sha)
        if config_sha != manifest_config_sha:
            errors.append(f"candidate {candidate_id} config differs from run manifest")
        weights_required = not thin or candidate_id == preselected_id
        if weights_required:
            if not model_path.is_file() or model_path.stat().st_size == 0:
                errors.append(f"candidate {candidate_id} missing selected/full adapter weights")
            elif sha256(model_path) != model_sha:
                errors.append(f"candidate {candidate_id} weights differ from run manifest")
        elif model_path.exists() or model_path.is_symlink():
            errors.append(
                f"thin bundle must omit reconstructible non-selected weights for {candidate_id}"
            )
        provenance_path = adapter / "interpolation_provenance.json"
        provenance_sha: str | None = None
        value: float | None = None
        if candidate_id == "recipe2":
            if provenance_path.exists() or provenance_path.is_symlink():
                errors.append("recipe2 primary candidate must not contain interpolation provenance")
            if recipe2_identity is not None:
                if config_sha != recipe2_identity["adapter_config_sha256"]:
                    errors.append("recipe2 candidate config differs from selected recipe2 checkpoint")
                if model_sha != recipe2_identity["adapter_model_sha256"]:
                    errors.append("recipe2 candidate weights differ from selected recipe2 checkpoint")
            if model_sha != primary.get("adapter_model_sha256"):
                errors.append("recipe2 candidate weights differ from locked primary hash")
        else:
            value = CORRECTED_LAMBDA_GRID[expected_grid_ids.index(candidate_id)]
            if (
                grid_contract is not None
                and config_path.read_bytes() != grid_contract["serialized"]
            ):
                errors.append(
                    f"candidate {candidate_id} config is not the exact rank-concat output config"
                )
            if not provenance_path.is_file() or provenance_path.stat().st_size == 0:
                errors.append(f"candidate {candidate_id} lacks interpolation provenance")
            else:
                provenance_sha = sha256(provenance_path)
                provenance = _load_json(provenance_path, errors)
                if provenance is not None:
                    if provenance.get("schema_version") != 1 or provenance.get("method") != "exact_delta_space_rank_concat":
                        errors.append(f"candidate {candidate_id} has unsupported provenance")
                    if (
                        provenance.get("equation")
                        != "delta_out=(1-lambda_b)*delta_a+lambda_b*delta_b"
                    ):
                        errors.append(f"candidate {candidate_id} provenance equation mismatch")
                    coefficient = provenance.get("coefficient")
                    if not isinstance(coefficient, dict):
                        errors.append(f"candidate {candidate_id} provenance lacks coefficient")
                    else:
                        expected_coefficients = {
                            "lambda_b": value,
                            "adapter_a_weight": 1.0 - value,
                            "adapter_b_weight": value,
                        }
                        for key, expected in expected_coefficients.items():
                            observed = coefficient.get(key)
                            if not _is_number(observed) or not math.isclose(
                                float(observed), expected, rel_tol=0.0, abs_tol=1e-12
                            ):
                                errors.append(f"candidate {candidate_id} provenance {key} mismatch")
                    sources = provenance.get("sources")
                    if not isinstance(sources, dict):
                        errors.append(f"candidate {candidate_id} provenance lacks sources")
                    else:
                        source_a = sources.get("adapter_a")
                        source_b = sources.get("adapter_b")
                        if not isinstance(source_a, dict) or source_a.get(
                            "adapter_model_sha256"
                        ) != fresh["v2_sha256"] or source_a.get(
                            "adapter_config_sha256"
                        ) != endpoint_identities["v2"]["adapter_config_sha256"]:
                            errors.append(f"candidate {candidate_id} provenance v2 endpoint mismatch")
                        if not isinstance(source_b, dict) or source_b.get(
                            "adapter_model_sha256"
                        ) != fresh["refresh1_sha256"] or source_b.get(
                            "adapter_config_sha256"
                        ) != endpoint_identities["refresh1"]["adapter_config_sha256"]:
                            errors.append(f"candidate {candidate_id} provenance refresh1 endpoint mismatch")
                    output = provenance.get("output")
                    if not isinstance(output, dict) or output.get("adapter_config_sha256") != config_sha or output.get("adapter_model_sha256") != model_sha:
                        errors.append(f"candidate {candidate_id} provenance output hashes mismatch")
                    elif not isinstance(output.get("path"), str) or not isinstance(
                        output.get("copied_inference_metadata"), list
                    ):
                        errors.append(f"candidate {candidate_id} provenance output metadata is incomplete")
                    lora = provenance.get("lora")
                    if not isinstance(lora, dict) or grid_contract is None:
                        errors.append(f"candidate {candidate_id} provenance lacks LoRA contract")
                    else:
                        for key, expected in grid_contract["lora"].items():
                            observed = lora.get(key)
                            if _is_number(expected):
                                if not _is_number(observed) or not math.isclose(
                                    float(observed),
                                    float(expected),
                                    rel_tol=0.0,
                                    abs_tol=1e-12,
                                ):
                                    errors.append(
                                        f"candidate {candidate_id} provenance lora.{key} mismatch"
                                    )
                            elif observed != expected:
                                errors.append(
                                    f"candidate {candidate_id} provenance lora.{key} mismatch"
                                )
                        module_count = lora.get("module_count")
                        tensor_count = lora.get("tensor_count")
                        if (
                            isinstance(module_count, bool)
                            or not isinstance(module_count, int)
                            or module_count <= 0
                            or isinstance(tensor_count, bool)
                            or not isinstance(tensor_count, int)
                            or tensor_count != module_count * 2
                        ):
                            errors.append(
                                f"candidate {candidate_id} provenance LoRA tensor counts are invalid"
                            )
        candidate_identity[candidate_id] = {
            "kind": "primary" if candidate_id == "recipe2" else "grid",
            "lambda": value,
            "adapter_config_sha256": config_sha,
            "adapter_model_sha256": model_sha,
            "provenance_sha256": provenance_sha,
            "runtime_adapter_dir": runtime_adapter_dir,
            "runtime_base_dir": runtime_base_dir,
        }
        runtime_identity = _audit_complete_eval_run(
            eval_dir,
            expected_cases=384,
            expected_ids=fresh["case_ids"],
            case_rows=fresh["cases"],
            cases_sha256=fresh["cases_sha256"],
            gold_sha256=fresh["gold_sha256"],
            adapter_config_sha256=config_sha,
            adapter_model_sha256=model_sha,
            provenance_sha256=provenance_sha,
            errors=errors,
            label=f"candidate {candidate_id}",
        )
        if runtime_identity is not None:
            if shared_runtime_identity is None:
                shared_runtime_identity = runtime_identity
            elif runtime_identity != shared_runtime_identity:
                errors.append(
                    f"candidate {candidate_id} did not use the shared base/media inputs"
                )
        suspicious = [
            path.relative_to(candidate).as_posix()
            for path in candidate.rglob("*")
            if any(re.search(r"day[_-]?14", part, re.IGNORECASE) for part in path.relative_to(candidate).parts)
        ]
        if suspicious:
            errors.append(f"candidate {candidate_id} contains Day-14-named artifacts")

    model_hashes = [item["adapter_model_sha256"] for item in candidate_identity.values()]
    if len(set(model_hashes)) != len(model_hashes):
        errors.append("registered candidate adapter hashes are not unique")

    if comparison.get("schema_version") != 1 or comparison.get("selector") != "fresh384_interpolation_selector":
        errors.append("candidate comparison has an unsupported schema/selector")
    if comparison.get("day14_metrics_used") is not False:
        errors.append("candidate comparison must exclude Day-14 metrics")
    comparison_policy = comparison.get("policy")
    if not isinstance(comparison_policy, dict) or comparison_policy.get("lambda_grid") != list(CORRECTED_LAMBDA_GRID):
        errors.append("candidate comparison lambda grid mismatch")
    elif comparison_policy.get("primary_if_eligible_else_grid") != primary:
        errors.append("candidate comparison primary policy differs from fresh384 policy")
    if isinstance(comparison_policy, dict):
        if comparison_policy.get("bootstrap_seed") != 20260916 or comparison_policy.get("bootstrap_draws") != 10_000:
            errors.append("candidate comparison bootstrap settings mismatch")
        if comparison_policy.get("eligibility_thresholds") != CORRECTED_ELIGIBILITY_THRESHOLDS:
            errors.append("candidate comparison eligibility thresholds mismatch")
    expected_validation_hashes = {
        "cases_sha256": fresh["cases_sha256"],
        "gold_sha256": fresh["gold_sha256"],
        "leakage_audit_sha256": fresh["audit_sha256"],
        "policy_sha256": fresh["policy_sha256"],
    }
    if comparison.get("validation_artifacts") != expected_validation_hashes:
        errors.append("candidate comparison validation artifact hashes mismatch")
    raw_results = comparison.get("candidates")
    by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_results, list) or len(raw_results) != 10:
        errors.append("candidate comparison must contain nine grid candidates plus recipe2")
    else:
        for result in raw_results:
            if not isinstance(result, dict) or not isinstance(result.get("candidate_id"), str):
                errors.append("candidate comparison contains an invalid candidate")
                continue
            candidate_id = result["candidate_id"]
            if candidate_id in by_id:
                errors.append(f"candidate comparison duplicates {candidate_id}")
                continue
            by_id[candidate_id] = result
        if set(by_id) != set(expected_ids):
            errors.append("candidate comparison IDs do not match registered candidates")
    recomputed_eligibility: dict[str, bool] = {}
    recomputed_strata: dict[str, dict[str, list[float]]] = {}
    recomputed_metrics: dict[str, dict[str, Any]] = {}
    for candidate_id, identity in candidate_identity.items():
        result = by_id.get(candidate_id)
        if result is None:
            continue
        if result.get("kind") != identity["kind"] or result.get("lambda") != identity["lambda"]:
            errors.append(f"comparison kind/lambda mismatch for {candidate_id}")
        recorded = result.get("adapter_identity")
        if not isinstance(recorded, dict):
            errors.append(f"comparison lacks adapter identity for {candidate_id}")
        else:
            for key in ("adapter_config_sha256", "adapter_model_sha256", "provenance_sha256"):
                if recorded.get(key) != identity[key]:
                    errors.append(f"comparison {candidate_id} {key} mismatch")
        prediction = result.get("prediction_artifact")
        records = root / f"candidates/{candidate_id}/eval/agent_pipeline_records.jsonl"
        if not isinstance(prediction, dict) or prediction.get("rows") != 384 or prediction.get("sha256") != (sha256(records) if records.is_file() else None):
            errors.append(f"comparison prediction artifact mismatch for {candidate_id}")
        planner = result.get("planner_log")
        planner_path = root / f"candidates/{candidate_id}/eval/planner.log"
        if not isinstance(planner, dict) or planner.get("sha256") != (sha256(planner_path) if planner_path.is_file() else None):
            errors.append(f"comparison planner log mismatch for {candidate_id}")
        eligibility = result.get("eligibility")
        prediction_rows = _read_jsonl_objects(records, errors) if records.is_file() else []
        prediction_index = {
            str(row.get("bench_id")): row
            for row in prediction_rows
            if isinstance(row.get("bench_id"), str)
        }
        try:
            metrics, strata, _diagnostics = _recompute_corrected_metrics(
                fresh["gold"], prediction_rows, prediction_index
            )
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            ZeroDivisionError,
        ) as exc:
            errors.append(f"cannot recompute candidate {candidate_id} metrics: {exc}")
            metrics, strata = {}, {}
        else:
            recomputed_metrics[candidate_id] = metrics
            recomputed_strata[candidate_id] = strata
            if result.get("metrics") != metrics:
                errors.append(f"comparison metrics differ from raw predictions for {candidate_id}")
        recomputed_checks: dict[str, dict[str, Any]] = {}
        for threshold_name, metric_name in CORRECTED_THRESHOLD_METRICS.items():
            value = metrics.get(metric_name)
            threshold = CORRECTED_ELIGIBILITY_THRESHOLDS[threshold_name]
            if not _is_number(value):
                passed = False
            elif threshold_name == "complete_prediction_rows":
                passed = value == threshold
            else:
                passed = float(value) >= float(threshold)
            recomputed_checks[threshold_name] = {
                "metric": metric_name,
                "value": value,
                "operator": "==" if threshold_name == "complete_prediction_rows" else ">=",
                "threshold": threshold,
                "passed": passed,
            }
        recomputed_eligible = all(item["passed"] for item in recomputed_checks.values())
        specificity = metrics.get("no_search_specificity")
        retention = metrics.get("rewrite_constraint_retention")
        utility_value = metrics.get("utility")
        if not all(_is_number(value) for value in (specificity, retention, utility_value)) or not math.isclose(
            float(utility_value or 0.0),
            0.5 * float(specificity or 0.0) + 0.5 * float(retention or 0.0),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            errors.append(f"comparison utility does not recompute for {candidate_id}")
            recomputed_eligible = False
        recomputed_eligibility[candidate_id] = recomputed_eligible
        if not isinstance(eligibility, dict) or not isinstance(eligibility.get("eligible"), bool):
            errors.append(f"comparison eligibility is invalid for {candidate_id}")
        elif eligibility.get("eligible") != recomputed_eligible or eligibility.get("checks") != recomputed_checks:
            errors.append(f"comparison eligibility does not recompute for {candidate_id}")
        bootstrap = result.get("bootstrap")
        if not isinstance(bootstrap, dict) or not _is_number(bootstrap.get("standard_error")) or float(bootstrap["standard_error"]) < 0:
            errors.append(f"comparison bootstrap is invalid for {candidate_id}")

    recomputed_bootstrap: dict[str, dict[str, float | int | str]] = {}
    if set(recomputed_strata) == set(expected_ids):
        try:
            recomputed_bootstrap = paired_stratified_bootstrap(
                recomputed_strata, seed=20260916, draws=10_000
            )
        except SelectionValidationError as exc:
            errors.append(f"cannot recompute candidate bootstrap: {exc}")
        else:
            for candidate_id, expected in recomputed_bootstrap.items():
                if by_id.get(candidate_id, {}).get("bootstrap") != expected:
                    errors.append(
                        f"comparison bootstrap differs from raw predictions for {candidate_id}"
                    )

    grid_decision = comparison.get("grid_decision")
    eligible_grid_ids = [
        candidate_id
        for candidate_id in expected_grid_ids
        if recomputed_eligibility.get(candidate_id) is True
    ]
    expected_grid_selected: str | None = None
    expected_grid_decision: dict[str, Any]
    if eligible_grid_ids and all(
        candidate_id in recomputed_bootstrap for candidate_id in eligible_grid_ids
    ):
        best_utility = max(
            float(recomputed_metrics[candidate_id]["utility"])
            for candidate_id in eligible_grid_ids
        )
        reference_id = min(
            (
                candidate_id
                for candidate_id in eligible_grid_ids
                if math.isclose(
                    float(recomputed_metrics[candidate_id]["utility"]),
                    best_utility,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ),
            key=lambda candidate_id: float(candidate_identity[candidate_id]["lambda"]),
        )
        reference_se = float(
            recomputed_bootstrap[reference_id]["standard_error"]
        )
        cutoff = best_utility - reference_se
        one_se_ids = [
            candidate_id
            for candidate_id in eligible_grid_ids
            if float(recomputed_metrics[candidate_id]["utility"]) >= cutoff
        ]
        expected_grid_selected = min(
            one_se_ids,
            key=lambda candidate_id: float(candidate_identity[candidate_id]["lambda"]),
        )
        expected_grid_decision = {
            "reference_candidate_id": reference_id,
            "reference_lambda": candidate_identity[reference_id]["lambda"],
            "best_utility": best_utility,
            "reference_bootstrap_se": reference_se,
            "one_se_cutoff": cutoff,
            "one_se_candidate_ids": [
                candidate_id
                for candidate_id in sorted(
                    one_se_ids,
                    key=lambda item: float(candidate_identity[item]["lambda"]),
                )
            ],
            "selected_candidate_id": expected_grid_selected,
            "selected_lambda": candidate_identity[expected_grid_selected]["lambda"],
        }
    else:
        if eligible_grid_ids:
            errors.append("cannot apply grid decision without recomputed bootstrap values")
        expected_grid_decision = {
            "reference_candidate_id": None,
            "reference_lambda": None,
            "best_utility": None,
            "reference_bootstrap_se": None,
            "one_se_cutoff": None,
            "one_se_candidate_ids": [],
            "selected_candidate_id": None,
            "selected_lambda": None,
        }
    if not isinstance(grid_decision, dict):
        errors.append("candidate comparison lacks grid_decision")
        grid_selected = None
    else:
        grid_selected = grid_decision.get("selected_candidate_id")
        if grid_selected is not None and grid_selected not in expected_grid_ids:
            errors.append("grid_decision did not select a registered grid candidate")
        elif grid_selected is not None and recomputed_eligibility.get(str(grid_selected)) is not True:
            errors.append("grid_decision selected an ineligible candidate")
        for key, expected in expected_grid_decision.items():
            observed = grid_decision.get(key)
            if _is_number(expected):
                if not _is_number(observed) or not math.isclose(
                    float(observed), float(expected), rel_tol=0.0, abs_tol=1e-15
                ):
                    errors.append(f"grid_decision {key} does not recompute")
            elif observed != expected:
                errors.append(f"grid_decision {key} does not recompute")
        if grid_selected != expected_grid_selected:
            errors.append("grid_decision violates the locked one-SE/smallest-lambda rule")

    if selection.get("schema_version") != 1 or selection.get("selected") is not True:
        errors.append("final selection must be a successful schema-version-1 decision")
    if selection.get("day14_metrics_used") is not False:
        errors.append("final selection must exclude Day-14 metrics")
    if selection.get("comparison_sha256") != sha256(comparison_path):
        errors.append("final selection comparison_sha256 mismatch")
    if selection.get("validation_artifacts") != expected_validation_hashes:
        errors.append("final selection validation artifact hashes mismatch")
    selected_id = selection.get("selected_candidate_id")
    if selected_id not in expected_ids:
        errors.append("final selection names an unknown candidate")
        return None
    primary_eligible = recomputed_eligibility.get("recipe2")
    expected_selected = "recipe2" if primary_eligible is True else grid_selected
    if selected_id != expected_selected:
        errors.append("final selection violates primary-if-eligible-else-grid")
    selected_identity = candidate_identity.get(selected_id)
    if selected_identity is None:
        return None
    if selection.get("selected_kind") != selected_identity["kind"]:
        errors.append("final selection selected_kind mismatch")
    if selection.get("selected_lambda") != selected_identity["lambda"]:
        errors.append("final selection selected_lambda mismatch")
    if selection.get("selected_adapter_model_sha256") != selected_identity["adapter_model_sha256"]:
        errors.append("final selection adapter hash mismatch")
    if selection.get("selected_adapter_dir") != selected_identity["runtime_adapter_dir"]:
        errors.append("final selection adapter path differs from the selected run manifest")
    if selection.get("primary_candidate_id") != "recipe2" or selection.get("primary_eligible") != primary_eligible:
        errors.append("final selection primary status mismatch")
    if selection.get("grid_decision") != grid_decision:
        errors.append("final selection grid_decision differs from comparison")
    if selection.get("selected_metrics") != by_id.get(selected_id, {}).get("metrics"):
        errors.append("final selection metrics differ from selected comparison row")
    observations["corrected_selection"] = {
        "selected_candidate_id": selected_id,
        "selected_kind": selected_identity["kind"],
        "selected_lambda": selected_identity["lambda"],
        "selected_adapter_model_sha256": selected_identity["adapter_model_sha256"],
        "primary_eligible": primary_eligible,
    }
    return {
        "candidate_id": selected_id,
        "selected_adapter_dir": selection.get("selected_adapter_dir"),
        **selected_identity,
    }


def _audit_adaptive_day14(
    root: Path,
    selected: dict[str, Any] | None,
    fresh: dict[str, Any] | None,
    errors: list[str],
    observations: dict[str, Any],
) -> None:
    adaptive = root / "adaptive_day14"
    expected_prediction_files = {
        root / f"candidates/{_lambda_slug(value)}/eval/agent_pipeline_records.jsonl"
        for value in CORRECTED_LAMBDA_GRID
    } | {
        root / "candidates/recipe2/eval/agent_pipeline_records.jsonl",
        adaptive / "agent_pipeline_records.jsonl",
    }
    actual_prediction_files = set(root.rglob("agent_pipeline_records.jsonl"))
    if actual_prediction_files != expected_prediction_files:
        extras = sorted(
            path.relative_to(root).as_posix()
            for path in actual_prediction_files - expected_prediction_files
        )
        missing = sorted(
            path.relative_to(root).as_posix()
            for path in expected_prediction_files - actual_prediction_files
        )
        errors.append(
            "bundle must contain exactly ten candidate prediction sets and one "
            f"adaptive Day-14 run: missing={missing}, extra={extras}"
        )
    expected_run_manifests = {
        root / f"candidates/{_lambda_slug(value)}/eval/run_manifest.json"
        for value in CORRECTED_LAMBDA_GRID
    } | {
        root / "candidates/recipe2/eval/run_manifest.json",
    }
    actual_run_manifests = set(root.rglob("run_manifest.json"))
    if actual_run_manifests != expected_run_manifests:
        extras = sorted(
            path.relative_to(root).as_posix()
            for path in actual_run_manifests - expected_run_manifests
        )
        missing = sorted(
            path.relative_to(root).as_posix()
            for path in expected_run_manifests - actual_run_manifests
        )
        errors.append(
            f"unexpected evaluation-run manifest set: missing={missing}, extra={extras}"
        )
    gold_path = adaptive / "gold.jsonl"
    predictions_path = adaptive / "agent_pipeline_records.jsonl"
    gold = _read_jsonl_objects(gold_path, errors) if gold_path.is_file() else []
    predictions = (
        _read_jsonl_objects(predictions_path, errors) if predictions_path.is_file() else []
    )
    gold_ids = [row.get("bench_id") for row in gold]
    prediction_ids = [row.get("bench_id") for row in predictions]
    if len(gold) != 100 or len(predictions) != 100:
        errors.append("adaptive Day-14 gold/predictions must each contain exactly 100 rows")
    valid_gold_ids = all(
        isinstance(value, str) and bool(value) for value in gold_ids
    )
    valid_prediction_ids = all(
        isinstance(value, str) and bool(value) for value in prediction_ids
    )
    if (
        not valid_gold_ids
        or not valid_prediction_ids
        or gold_ids != prediction_ids
        or len(set(gold_ids)) != len(gold_ids)
    ):
        errors.append("adaptive Day-14 gold/prediction IDs must be unique and aligned")
    if any("error" in row for row in predictions):
        errors.append("adaptive Day-14 predictions contain per-case errors")
    if fresh is not None and gold_path.is_file():
        locked_day14_hash = fresh["policy"].get("validation_artifacts", {}).get(
            "input_sha256", {}
        ).get("day14_forbidden_cases")
        if sha256(gold_path) != locked_day14_hash:
            errors.append("adaptive Day-14 gold differs from the exclusion-locked suite")
    metrics = _load_json(adaptive / "metrics.json", errors)
    recomputed_metrics: dict[str, Any] | None = None
    if metrics is not None:
        missing = [key for key in REQUIRED_GATE_KEYS if key not in metrics]
        if missing:
            errors.append(f"adaptive Day-14 metrics missing keys: {', '.join(missing)}")
        if metrics.get("num_cases") != 100:
            errors.append("adaptive Day-14 metrics num_cases is not 100")
        try:
            recomputed_metrics = _score_day14(gold, predictions)
        except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"cannot recompute adaptive Day-14 metrics: {exc}")
        else:
            if metrics != recomputed_metrics:
                errors.append("adaptive Day-14 metrics differ from bundled gold/predictions")
    summary = _load_json(adaptive / "gate_summary.json", errors)
    if summary is not None:
        if not isinstance(summary.get("passed"), bool):
            errors.append("adaptive Day-14 gate summary must record a boolean passed value")
        if not isinstance(summary.get("checks"), dict):
            errors.append("adaptive Day-14 gate summary must contain checks")
        if not isinstance(summary.get("diagnostics"), dict):
            errors.append("adaptive Day-14 gate summary must contain diagnostics")
        if recomputed_metrics is not None:
            try:
                expected_summary = _evaluate_day14_gate(
                    recomputed_metrics, RELEASED_BASELINE
                )
            except GateValidationError as exc:
                errors.append(f"cannot recompute adaptive Day-14 gate: {exc}")
            else:
                expected_summary["released_baseline_source"] = (
                    "pre-registered constants: routing=0.78, retention=0.48"
                )
                if summary != expected_summary:
                    errors.append(
                        "adaptive Day-14 gate summary differs from recomputed default gate"
                    )
    if selected is not None:
        planner_path = adaptive / "planner.log"
        try:
            text = planner_path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"cannot read adaptive Day-14 planner log: {exc}")
        else:
            lines = text.splitlines()
            for label, key in (
                ("base", "runtime_base_dir"),
                ("adapter", "runtime_adapter_dir"),
            ):
                expected = selected.get(key)
                declarations = [
                    line for line in lines if line.startswith(f"Agent {label}:")
                ]
                if not isinstance(expected, str) or declarations != [
                    f"Agent {label}: {expected}"
                ]:
                    errors.append(
                        "adaptive Day-14 planner did not uniquely use the exact "
                        f"selected {label} path"
                    )
    observations["adaptive_day14"] = {
        "evaluation_role": "adaptive_regression",
        "confirmatory": False,
        "used_for_model_selection": False,
        "run_count": 1,
        "num_cases": len(predictions),
        "passed": summary.get("passed") if summary else None,
    }


def _audit_corrected_bundle(root: Path, *, thin: bool) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    profile = CORRECTED_THIN_PROFILE if thin else CORRECTED_FULL_PROFILE
    observations: dict[str, Any] = {"profile": profile}
    required = CORRECTED_THIN_REQUIRED_FILES if thin else CORRECTED_FULL_REQUIRED_FILES
    _require_nonempty_files(root, required, errors)
    recipe2 = _audit_recipe2(root, errors, observations, thin=thin)
    fresh = _audit_fresh384(root, recipe2, errors, observations)
    _audit_recipe2_cross_audit(root, errors, observations)
    grid_contract = _audit_endpoint_configs(root, errors, observations)
    identities = _audit_adapter_identities(root, fresh, recipe2, errors, observations)
    selected = _audit_candidates_and_selection(
        root,
        fresh,
        recipe2,
        identities,
        grid_contract,
        errors,
        observations,
        thin=thin,
    )
    if thin and selected is not None:
        selected_weights = (
            root
            / f"candidates/{selected['candidate_id']}/adapter/adapter_model.safetensors"
        )
        actual_weights = set(root.rglob("adapter_model.safetensors"))
        if actual_weights != {selected_weights}:
            extras = sorted(
                path.relative_to(root).as_posix()
                for path in actual_weights - {selected_weights}
            )
            errors.append(
                "thin bundle must contain only the final selected adapter weights: "
                f"extra={extras}"
            )
    _audit_adaptive_day14(root, selected, fresh, errors, observations)
    manifest_path = root / "checksums.sha256"
    if manifest_path.is_file():
        _audit_refresh_manifest(root, manifest_path, errors)
    return {"ok": not errors, "errors": errors, "observations": observations}


def audit_corrected_full_bundle(root: Path) -> dict[str, Any]:
    """Audit all recipe2 checkpoints and all ten candidate adapter blobs."""

    return _audit_corrected_bundle(root, thin=False)


def audit_corrected_thin_bundle(root: Path) -> dict[str, Any]:
    """Audit the durable evidence bundle with only final selected adapter bytes."""

    return _audit_corrected_bundle(root, thin=True)


def audit_corrected_selection_bundle(root: Path) -> dict[str, Any]:
    """Backward-compatible Python alias for the explicit full profile."""

    return audit_corrected_full_bundle(root)


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
        choices=("v2", "refresh", CORRECTED_FULL_PROFILE, CORRECTED_THIN_PROFILE),
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
        if args.profile == "refresh":
            audit = audit_refresh_bundle(args.root)
        elif args.profile == CORRECTED_FULL_PROFILE:
            audit = audit_corrected_full_bundle(args.root)
        elif args.profile == CORRECTED_THIN_PROFILE:
            audit = audit_corrected_thin_bundle(args.root)
        else:
            audit = audit_bundle(args.root)
        result["audit"] = audit
        failed = failed or not audit["ok"]
    if not (args.write_manifest or args.verify_manifest or args.audit):
        parser.error("select --write-manifest, --verify-manifest, or --audit")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Audit and select a LoRA candidate on the locked interpolation benchmark.

The selector deliberately ignores precomputed metric files.  It verifies the
pre-registered validation bundle and every candidate adapter, recomputes the
locked metrics from raw planner records, and applies the policy's eligibility
thresholds and paired stratified bootstrap rule.  This keeps Day-14 results out
of model selection and makes the final decision reproducible from immutable
inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

try:
    from evaluation.agent_only_score import (
        constraint_is_retained,
        extract_plan,
        score,
        triggered,
        valid_plan,
    )
except ModuleNotFoundError:  # direct ``python scripts/...py`` execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.agent_only_score import (
        constraint_is_retained,
        extract_plan,
        score,
        triggered,
        valid_plan,
    )


EXPECTED_GRID = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
INTERPOLATION_METHOD = "exact_delta_space_rank_concat"
EXPECTED_CATEGORY_COUNTS = {
    "no_search_negative": 128,
    "true_search_positive": 64,
    "routing_control": 64,
    "mask_control": 64,
    "rewrite_retention": 64,
}
EXPECTED_CATEGORIES = set(EXPECTED_CATEGORY_COUNTS)
EXPECTED_NO_SEARCH_SUBTYPE_COUNTS = {
    "generic_style": 64,
    "generic_background": 32,
    "ordinary_target": 32,
}
EXPECTED_MASK_TRIGGER_COUNTS = {"triggered": 32, "not_triggered": 32}
EXPECTED_THRESHOLDS = {
    "complete_prediction_rows": 384,
    "strict_raw_json_validity_min": 1.0,
    "subtask_accuracy_min": 0.95,
    "no_search_specificity_min": 0.95,
    "true_search_trigger_recall_min": 0.95,
    "search_query_end_to_end_recall_min": 0.95,
    "mask_trigger_f1_min": 0.95,
    "rewrite_constraint_retention_min": 0.85,
}
EXPECTED_BOOTSTRAP_SEED = 20260916
EXPECTED_BOOTSTRAP_DRAWS = 10_000
THRESHOLD_METRICS = {
    "complete_prediction_rows": "prediction_rows",
    "strict_raw_json_validity_min": "strict_raw_json_validity",
    "subtask_accuracy_min": "subtask_accuracy",
    "no_search_specificity_min": "no_search_specificity",
    "true_search_trigger_recall_min": "true_search_trigger_recall",
    "search_query_end_to_end_recall_min": "search_query_end_to_end_recall",
    "mask_trigger_f1_min": "mask_trigger_f1",
    "rewrite_constraint_retention_min": "rewrite_constraint_retention",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
DAY14_PATH_RE = re.compile(r"(?<![a-z0-9])day[_-]?14(?![a-z0-9])", re.IGNORECASE)
RUN_MANIFEST = "run_manifest.json"
METRICS = "metrics.json"
SCORER_LOG = "scorer.log"


class SelectionValidationError(RuntimeError):
    """Raised when a locked input or candidate artifact is not auditable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise SelectionValidationError(f"{label}: invalid strict JSON: {error}") from error


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        value = _parse_json(path.read_text(encoding="utf-8"), str(path))
    except (OSError, UnicodeDecodeError) as error:
        raise SelectionValidationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise SelectionValidationError(f"{label} must be a JSON object: {path}")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require_file(path, label)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise SelectionValidationError(f"cannot read {label} {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        value = _parse_json(line, f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise SelectionValidationError(
                f"{path}:{line_number}: expected a JSON object"
            )
        rows.append(value)
    return rows


def _require_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SelectionValidationError(
            f"{label} must be a non-empty regular file (not a symlink): {path}"
        )


def _sha(value: Any, label: str) -> str:
    normalized = str(value).strip().lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise SelectionValidationError(f"{label} must be a SHA-256 digest")
    return normalized


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SelectionValidationError(f"{label} must be finite")
    return result


def _positive_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SelectionValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectionValidationError(f"{label} must be an object")
    return value


def _lambda_slug(value: float) -> str:
    scaled = value * 1000
    rounded = round(scaled)
    if not math.isclose(scaled, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise SelectionValidationError(
            f"lambda {value!r} cannot be represented by the locked directory convention"
        )
    return f"lambda_{rounded:04d}"


def _normalise_primary(policy: dict[str, Any]) -> dict[str, Any] | None:
    """Read the optional primary-first rule, accepting the two documented spellings."""

    final = policy.get("final_decision")
    direct = policy.get("primary_if_eligible_else_grid")
    if final is not None and direct is not None:
        raise SelectionValidationError(
            "policy must not define both final_decision and primary_if_eligible_else_grid"
        )
    if direct is not None:
        config = _mapping(direct, "primary_if_eligible_else_grid")
    elif final is None:
        return None
    else:
        final_map = _mapping(final, "final_decision")
        if "primary_if_eligible_else_grid" in final_map:
            if set(final_map) != {"primary_if_eligible_else_grid"}:
                raise SelectionValidationError(
                    "final_decision wrapper may only contain primary_if_eligible_else_grid"
                )
            config = _mapping(
                final_map["primary_if_eligible_else_grid"],
                "final_decision.primary_if_eligible_else_grid",
            )
        else:
            if final_map.get("rule") != "primary_if_eligible_else_grid":
                raise SelectionValidationError(
                    "unsupported final_decision rule; expected primary_if_eligible_else_grid"
                )
            config = _mapping(final_map.get("primary_candidate"), "primary_candidate")

    allowed = {
        "candidate_id",
        "directory_name",
        "adapter_model_sha256",
        "adapter_config_sha256",
        "inference_adapter_path",
    }
    extras = sorted(set(config) - allowed)
    if extras:
        raise SelectionValidationError(f"unknown primary candidate fields: {extras}")
    candidate_id = str(config.get("candidate_id", "")).strip()
    if not candidate_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", candidate_id):
        raise SelectionValidationError("primary candidate_id is missing or unsafe")
    directory_name = str(config.get("directory_name", candidate_id)).strip()
    if (
        not directory_name
        or directory_name in {".", ".."}
        or Path(directory_name).name != directory_name
    ):
        raise SelectionValidationError("primary directory_name must be one safe path component")
    result = {
        "candidate_id": candidate_id,
        "directory_name": directory_name,
        "adapter_model_sha256": _sha(
            config.get("adapter_model_sha256"), "primary adapter_model_sha256"
        ),
    }
    if config.get("adapter_config_sha256") is not None:
        result["adapter_config_sha256"] = _sha(
            config["adapter_config_sha256"], "primary adapter_config_sha256"
        )
    if config.get("inference_adapter_path") is not None:
        path = str(config["inference_adapter_path"]).strip()
        if not path:
            raise SelectionValidationError("primary inference_adapter_path is empty")
        result["inference_adapter_path"] = path
    return result


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    policy_version = policy.get("version")
    if (
        isinstance(policy_version, bool)
        or policy_version not in {1, 2}
        or policy.get("created_before_candidate_inference") is not True
    ):
        raise SelectionValidationError(
            "policy must be a supported version and created before candidate inference"
        )
    candidate_rule = _mapping(policy.get("candidate_rule"), "candidate_rule")
    if candidate_rule.get("method") != "exact_parameter_delta_interpolation":
        raise SelectionValidationError("unsupported candidate interpolation method")
    raw_grid = candidate_rule.get("lambda_grid")
    if not isinstance(raw_grid, list):
        raise SelectionValidationError("candidate_rule.lambda_grid must be a list")
    grid = tuple(
        _finite_number(value, f"lambda_grid[{index}]")
        for index, value in enumerate(raw_grid)
    )
    if grid != EXPECTED_GRID:
        raise SelectionValidationError(
            f"lambda grid differs from the pre-registered fresh-384 grid: {grid}"
        )
    if candidate_rule.get("candidate_count") != len(grid):
        raise SelectionValidationError("candidate_count does not match lambda_grid")
    if candidate_rule.get("additional_training_allowed") is not False:
        raise SelectionValidationError("interpolation policy unexpectedly allows training")
    if candidate_rule.get("day14_outputs_or_metrics_allowed_for_generation_or_selection") is not False:
        raise SelectionValidationError("policy allows Day-14 leakage into selection")

    validation = _mapping(policy.get("validation_artifacts"), "validation_artifacts")
    expected_cases = _positive_int(validation.get("expected_cases"), "expected_cases")
    if validation.get("expected_unique_videos") != expected_cases:
        raise SelectionValidationError("expected_unique_videos must equal expected_cases")
    category_counts = _mapping(
        validation.get("expected_category_counts"), "expected_category_counts"
    )
    if set(category_counts) != EXPECTED_CATEGORIES:
        raise SelectionValidationError("policy category set is not the locked five-category set")
    parsed_counts = {
        key: _positive_int(value, f"expected_category_counts.{key}")
        for key, value in category_counts.items()
    }
    if expected_cases != 384 or parsed_counts != EXPECTED_CATEGORY_COUNTS:
        raise SelectionValidationError(
            "policy is not the locked 384-case category allocation"
        )
    if validation.get("expected_no_search_subtype_counts") != EXPECTED_NO_SEARCH_SUBTYPE_COUNTS:
        raise SelectionValidationError("policy no-search subtype allocation is not locked")
    if validation.get("expected_mask_trigger_counts") != EXPECTED_MASK_TRIGGER_COUNTS:
        raise SelectionValidationError("policy mask-trigger allocation is not locked")
    for name in ("cases_sha256", "gold_sha256", "leakage_audit_sha256"):
        _sha(validation.get(name), f"validation_artifacts.{name}")
    input_hashes = _mapping(validation.get("input_sha256"), "input_sha256")
    if "day14_forbidden_cases" not in input_hashes:
        raise SelectionValidationError("policy does not record the Day-14 exclusion input")
    for name, value in input_hashes.items():
        _sha(value, f"input_sha256.{name}")

    thresholds = _mapping(policy.get("eligibility_thresholds"), "eligibility_thresholds")
    if set(thresholds) != set(THRESHOLD_METRICS):
        raise SelectionValidationError(
            "eligibility threshold names differ from the supported locked policy"
        )
    complete_rows = _positive_int(
        thresholds["complete_prediction_rows"], "complete_prediction_rows"
    )
    if complete_rows != expected_cases:
        raise SelectionValidationError("complete_prediction_rows must equal expected_cases")
    for name, value in thresholds.items():
        if name == "complete_prediction_rows":
            continue
        threshold = _finite_number(value, name)
        if not 0.0 <= threshold <= 1.0:
            raise SelectionValidationError(f"{name} must be in [0, 1]")
    if thresholds != EXPECTED_THRESHOLDS:
        raise SelectionValidationError("eligibility thresholds differ from the locked policy")

    utility = _mapping(policy.get("utility"), "utility")
    if utility.get("formula") != "0.5*no_search_specificity+0.5*rewrite_constraint_retention":
        raise SelectionValidationError("unsupported utility formula")
    if utility.get("higher_is_better") is not True:
        raise SelectionValidationError("utility must declare higher_is_better=true")
    bootstrap = _mapping(policy.get("bootstrap"), "bootstrap")
    if bootstrap.get("method") != "paired stratified case bootstrap over the two utility strata":
        raise SelectionValidationError("unsupported bootstrap method")
    seed = _positive_int(bootstrap.get("seed"), "bootstrap.seed", minimum=0)
    draws = _positive_int(bootstrap.get("draws"), "bootstrap.draws", minimum=2)
    if seed != EXPECTED_BOOTSTRAP_SEED or draws != EXPECTED_BOOTSTRAP_DRAWS:
        raise SelectionValidationError(
            "bootstrap seed/draw count differ from the pre-registered policy"
        )
    strata = _mapping(bootstrap.get("strata"), "bootstrap.strata")
    expected_strata = {
        "no_search_negative": parsed_counts["no_search_negative"],
        "rewrite_retention": parsed_counts["rewrite_retention"],
    }
    if strata != expected_strata:
        raise SelectionValidationError(
            f"bootstrap strata differ from validation counts: {strata} != {expected_strata}"
        )
    selection = _mapping(policy.get("selection"), "selection")
    if selection.get("rule") != "one_standard_error_then_smallest_lambda":
        raise SelectionValidationError("unsupported selection rule")
    if selection.get("tie_breaker") != "smallest lambda" or selection.get(
        "reference_tie_breaker"
    ) != "smallest lambda":
        raise SelectionValidationError("selection tie-breakers are not locked to smallest lambda")
    if selection.get("day14_metrics_used") is not False:
        raise SelectionValidationError("selection policy declares Day-14 metric use")

    return {
        "grid": grid,
        "v2_sha256": _sha(
            candidate_rule.get("v2_adapter_model_sha256"), "v2 endpoint hash"
        ),
        "refresh1_sha256": _sha(
            candidate_rule.get("refresh1_adapter_model_sha256"),
            "refresh1 endpoint hash",
        ),
        "expected_cases": expected_cases,
        "category_counts": parsed_counts,
        "thresholds": thresholds,
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "primary": _normalise_primary(policy),
        "policy_version": policy_version,
    }


def _unique_ids(rows: Sequence[dict[str, Any]], label: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        bench_id = row.get("bench_id")
        if not isinstance(bench_id, str) or not bench_id.strip():
            raise SelectionValidationError(f"{label} contains an empty/non-string bench_id")
        if bench_id in seen:
            raise SelectionValidationError(f"{label} contains duplicate bench_id {bench_id!r}")
        seen.add(bench_id)
        ids.append(bench_id)
    return ids


def validate_validation_bundle(
    cases_path: Path,
    gold_path: Path,
    audit_path: Path,
    policy: dict[str, Any],
    normalized_policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validation = policy["validation_artifacts"]
    observed_hashes = {
        "cases_sha256": sha256_file(cases_path),
        "gold_sha256": sha256_file(gold_path),
        "leakage_audit_sha256": sha256_file(audit_path),
    }
    for name, observed in observed_hashes.items():
        if observed != str(validation[name]).lower():
            raise SelectionValidationError(
                f"locked validation artifact hash mismatch for {name}: {observed}"
            )
    cases = load_jsonl(cases_path, "validation cases")
    gold = load_jsonl(gold_path, "validation gold")
    audit = load_json_object(audit_path, "leakage audit")
    expected_cases = normalized_policy["expected_cases"]
    if len(cases) != expected_cases or len(gold) != expected_cases:
        raise SelectionValidationError(
            f"validation rows must be exactly {expected_cases}: cases={len(cases)}, gold={len(gold)}"
        )
    case_ids = _unique_ids(cases, "cases")
    gold_ids = _unique_ids(gold, "gold")
    if case_ids != gold_ids:
        raise SelectionValidationError("case/gold bench_id order differs")
    if any(re.fullmatch(r"interp_\d{4}", bench_id) is None for bench_id in case_ids):
        raise SelectionValidationError("validation contains a non-interpolation bench_id")

    shared_fields = (
        "bench_id",
        "video_path",
        "prompt",
        "edit_type",
        "axis",
        "category",
        "subtype",
        "source",
        "catalog",
    )
    for case, gold_row in zip(cases, gold):
        if any(case.get(name) != gold_row.get(name) for name in shared_fields):
            raise SelectionValidationError(
                f"case/gold metadata differs for {case['bench_id']}"
            )
        if not valid_plan(gold_row.get("gold_plan")):
            raise SelectionValidationError(f"invalid gold plan for {case['bench_id']}")
    observed_categories = Counter(str(row.get("category")) for row in gold)
    if observed_categories != Counter(normalized_policy["category_counts"]):
        raise SelectionValidationError(
            f"validation category counts differ from policy: {dict(observed_categories)}"
        )
    no_search_subtypes = Counter(
        str(row.get("subtype"))
        for row in gold
        if row.get("category") == "no_search_negative"
    )
    if no_search_subtypes != Counter(EXPECTED_NO_SEARCH_SUBTYPE_COUNTS):
        raise SelectionValidationError(
            f"validation no-search subtype counts are not locked: {dict(no_search_subtypes)}"
        )
    mask_trigger_counts = Counter(
        "triggered" if triggered(row["gold_plan"]["mask"]) else "not_triggered"
        for row in gold
        if row.get("category") == "mask_control"
    )
    if mask_trigger_counts != Counter(EXPECTED_MASK_TRIGGER_COUNTS):
        raise SelectionValidationError(
            f"validation mask-trigger counts are not locked: {dict(mask_trigger_counts)}"
        )
    for row in gold:
        search_expected = triggered(row["gold_plan"]["image_search"])
        if search_expected != (row.get("category") == "true_search_positive"):
            raise SelectionValidationError(
                "all and only true_search_positive rows must trigger gold search: "
                f"{row['bench_id']}"
            )
        if row.get("category") == "rewrite_retention" and len(row.get("constraints", [])) != 4:
            raise SelectionValidationError(
                f"rewrite row must contain four locked constraints: {row['bench_id']}"
            )
    video_paths = [str(row.get("video_path", "")) for row in cases]
    video_hashes = [str(_mapping(row.get("source"), "case.source").get("video_sha256", "")) for row in cases]
    if len(set(video_paths)) != expected_cases or len(set(video_hashes)) != expected_cases:
        raise SelectionValidationError("validation videos are not path- and byte-unique")
    if any(SHA256_RE.fullmatch(value) is None for value in video_hashes):
        raise SelectionValidationError("validation source contains an invalid video SHA-256")

    audit_counts = _mapping(audit.get("counts"), "leakage_audit.counts")
    if audit.get("recipe_version") != normalized_policy["policy_version"]:
        raise SelectionValidationError(
            "leakage-audit recipe version differs from selection-policy version"
        )
    if audit_counts.get("cases") != expected_cases or audit_counts.get("gold") != expected_cases:
        raise SelectionValidationError("leakage audit row counts disagree with policy")
    if audit_counts.get("by_category") != dict(sorted(normalized_policy["category_counts"].items())):
        raise SelectionValidationError("leakage audit category counts disagree with policy")
    if audit_counts.get("no_search_by_subtype") != dict(
        sorted(EXPECTED_NO_SEARCH_SUBTYPE_COUNTS.items())
    ):
        raise SelectionValidationError("leakage audit no-search subtype counts disagree")
    if audit_counts.get("mask_trigger") != dict(
        sorted(EXPECTED_MASK_TRIGGER_COUNTS.items())
    ):
        raise SelectionValidationError("leakage audit mask-trigger counts disagree")
    isolation = _mapping(audit.get("isolation"), "leakage_audit.isolation")
    blocking_isolation_fields = {
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
    }
    for name in blocking_isolation_fields:
        if isolation.get(name) != 0:
            raise SelectionValidationError(
                f"leakage audit is non-zero or missing: {name}={isolation.get(name)!r}"
            )
    if isolation.get("explicit_forbidden_phrase_hits") != []:
        raise SelectionValidationError("leakage audit has forbidden phrase hits")
    if normalized_policy["policy_version"] >= 2:
        expected_generic_categories = {"mask_control", "routing_control"}
        concept_hits = isolation.get("allowed_generic_concept_phrase_hits")
        template_hits = isolation.get("allowed_generic_template_signature_hits")
        if not isinstance(concept_hits, list) or not isinstance(template_hits, list):
            raise SelectionValidationError("v2 leakage audit lacks generic-axis hit records")
        if isolation.get("allowed_generic_concept_phrase_hit_count") != len(concept_hits):
            raise SelectionValidationError("generic concept hit count disagrees with records")
        if isolation.get("allowed_generic_template_signature_hit_count") != len(template_hits):
            raise SelectionValidationError("generic template hit count disagrees with records")
        for label, rows in (("concept", concept_hits), ("template", template_hits)):
            if any(
                not isinstance(row, dict)
                or row.get("category") not in expected_generic_categories
                for row in rows
            ):
                raise SelectionValidationError(
                    f"allowed generic {label} hits contain a non-generic category"
                )
    audit_artifacts = _mapping(audit.get("artifacts"), "leakage_audit.artifacts")
    for name in ("cases_sha256", "gold_sha256"):
        if str(audit_artifacts.get(name, "")).lower() != observed_hashes[name]:
            raise SelectionValidationError(f"leakage audit internal hash mismatch: {name}")
    if audit_artifacts.get("input_sha256") != validation.get("input_sha256"):
        raise SelectionValidationError("leakage audit and policy input hashes differ")
    return cases, gold, audit


def _find_candidate_artifacts(candidate_dir: Path) -> dict[str, Path]:
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise SelectionValidationError(f"candidate directory is missing or a symlink: {candidate_dir}")
    suspicious = [
        str(path.relative_to(candidate_dir))
        for path in candidate_dir.rglob("*")
        if any(DAY14_PATH_RE.search(part) for part in path.relative_to(candidate_dir).parts)
    ]
    if suspicious:
        raise SelectionValidationError(
            f"candidate directory contains Day-14-named artifacts: {suspicious[:5]}"
        )

    adapter_options = [
        path
        for path in (candidate_dir / "adapter", candidate_dir)
        if (path / "adapter_model.safetensors").is_file()
        and (path / "adapter_config.json").is_file()
    ]
    if len(adapter_options) != 1:
        raise SelectionValidationError(
            f"candidate must contain exactly one adapter at ./adapter or ./: {candidate_dir}"
        )
    records_options = [
        path
        for path in (
            candidate_dir / "eval" / "agent_pipeline_records.jsonl",
            candidate_dir / "validation" / "agent_pipeline_records.jsonl",
            candidate_dir / "agent_pipeline_records.jsonl",
        )
        if path.is_file()
    ]
    if len(records_options) != 1:
        raise SelectionValidationError(
            f"candidate must contain exactly one agent_pipeline_records.jsonl: {candidate_dir}"
        )
    records = records_options[0]
    log_options = [
        path
        for path in (records.parent / "planner.log", candidate_dir / "planner.log")
        if path.is_file()
    ]
    # The two expressions are the same path when records live at candidate root.
    log_options = list(dict.fromkeys(log_options))
    if len(log_options) != 1:
        raise SelectionValidationError(
            f"candidate must contain exactly one planner.log next to records or at root: {candidate_dir}"
        )
    evaluation_dir = records.parent
    required_outputs = {
        "run_manifest": evaluation_dir / RUN_MANIFEST,
        "metrics": evaluation_dir / METRICS,
        "scorer_log": evaluation_dir / SCORER_LOG,
    }
    for label, path in required_outputs.items():
        _require_file(path, f"candidate {label}")
    return {
        "adapter": adapter_options[0],
        "records": records,
        "planner_log": log_options[0],
        **required_outputs,
    }


def _validate_adapter_config_semantics(config_value: dict[str, Any], path: Path) -> str:
    if config_value.get("peft_type") != "LORA":
        raise SelectionValidationError(f"adapter config peft_type must be LORA: {path}")
    rank = config_value.get("r")
    alpha = config_value.get("lora_alpha")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise SelectionValidationError(f"adapter config has invalid rank: {path}")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or float(alpha) <= 0
    ):
        raise SelectionValidationError(f"adapter config has invalid lora_alpha: {path}")
    targets = config_value.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(item, str) or not item for item in targets)
        or len(set(targets)) != len(targets)
    ):
        raise SelectionValidationError(
            f"adapter config has invalid target_modules: {path}"
        )
    if config_value.get("bias", "none") != "none":
        raise SelectionValidationError(f"adapter config bias must be none: {path}")
    base_model = config_value.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise SelectionValidationError(
            f"adapter config has no base_model_name_or_path: {path}"
        )
    return base_model.strip()


def _validate_adapter_files(adapter: Path) -> tuple[str, str, str]:
    if adapter.is_symlink() or not adapter.is_dir():
        raise SelectionValidationError(f"adapter directory is missing or a symlink: {adapter}")
    config = adapter / "adapter_config.json"
    model = adapter / "adapter_model.safetensors"
    _require_file(config, "adapter config")
    _require_file(model, "adapter weights")
    config_value = load_json_object(config, "adapter config")
    base_model = _validate_adapter_config_semantics(config_value, config)
    return sha256_file(config), sha256_file(model), base_model


def _validate_interpolation_provenance(
    adapter: Path,
    expected_lambda: float,
    normalized_policy: dict[str, Any],
) -> dict[str, Any]:
    provenance_path = adapter / "interpolation_provenance.json"
    provenance = load_json_object(provenance_path, "interpolation provenance")
    config_sha, model_sha, base_model = _validate_adapter_files(adapter)
    if provenance.get("schema_version") != 1 or provenance.get("method") != INTERPOLATION_METHOD:
        raise SelectionValidationError(f"unsupported interpolation provenance: {provenance_path}")
    if provenance.get("equation") != "delta_out=(1-lambda_b)*delta_a+lambda_b*delta_b":
        raise SelectionValidationError("interpolation provenance equation is not exact delta-space")
    coefficient = _mapping(provenance.get("coefficient"), "provenance.coefficient")
    observed_lambda = _finite_number(coefficient.get("lambda_b"), "provenance lambda_b")
    if not math.isclose(observed_lambda, expected_lambda, rel_tol=0.0, abs_tol=1e-12):
        raise SelectionValidationError(
            f"provenance lambda mismatch: expected {expected_lambda}, got {observed_lambda}"
        )
    expected_weights = (1.0 - expected_lambda, expected_lambda)
    observed_weights = (
        _finite_number(coefficient.get("adapter_a_weight"), "adapter_a_weight"),
        _finite_number(coefficient.get("adapter_b_weight"), "adapter_b_weight"),
    )
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(observed_weights, expected_weights)
    ):
        raise SelectionValidationError("provenance interpolation weights are inconsistent")
    sources = _mapping(provenance.get("sources"), "provenance.sources")
    source_a = _mapping(sources.get("adapter_a"), "provenance.sources.adapter_a")
    source_b = _mapping(sources.get("adapter_b"), "provenance.sources.adapter_b")
    if _sha(source_a.get("adapter_model_sha256"), "adapter_a source hash") != normalized_policy["v2_sha256"]:
        raise SelectionValidationError("candidate adapter_a is not the locked v2 endpoint")
    if _sha(source_b.get("adapter_model_sha256"), "adapter_b source hash") != normalized_policy["refresh1_sha256"]:
        raise SelectionValidationError("candidate adapter_b is not the locked refresh1 endpoint")
    source_config_hashes = {
        "adapter_a": _sha(
            source_a.get("adapter_config_sha256"), "adapter_a source config hash"
        ),
        "adapter_b": _sha(
            source_b.get("adapter_config_sha256"), "adapter_b source config hash"
        ),
    }
    output = _mapping(provenance.get("output"), "provenance.output")
    if _sha(output.get("adapter_config_sha256"), "provenance output config hash") != config_sha:
        raise SelectionValidationError("candidate adapter config hash differs from provenance")
    if _sha(output.get("adapter_model_sha256"), "provenance output model hash") != model_sha:
        raise SelectionValidationError("candidate adapter model hash differs from provenance")
    lora = _mapping(provenance.get("lora"), "provenance.lora")
    input_rank = _positive_int(lora.get("input_rank"), "provenance.lora.input_rank")
    output_rank = _positive_int(lora.get("output_rank"), "provenance.lora.output_rank")
    input_alpha = _finite_number(
        lora.get("input_lora_alpha"), "provenance.lora.input_lora_alpha"
    )
    output_alpha = _finite_number(
        lora.get("output_lora_alpha"), "provenance.lora.output_lora_alpha"
    )
    input_scaling = _finite_number(
        lora.get("input_scaling"), "provenance.lora.input_scaling"
    )
    output_scaling = _finite_number(
        lora.get("output_scaling"), "provenance.lora.output_scaling"
    )
    module_count = _positive_int(
        lora.get("module_count"), "provenance.lora.module_count"
    )
    tensor_count = _positive_int(
        lora.get("tensor_count"), "provenance.lora.tensor_count"
    )
    if (
        input_alpha <= 0
        or output_alpha <= 0
        or output_rank != 2 * input_rank
        or not math.isclose(output_alpha, 2 * input_alpha, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(input_scaling, input_alpha / input_rank, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(output_scaling, output_alpha / output_rank, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(input_scaling, output_scaling, rel_tol=0.0, abs_tol=1e-12)
        or tensor_count != 2 * module_count
        or lora.get("dtype") != "float32"
    ):
        raise SelectionValidationError(
            "interpolation provenance does not preserve exact rank-concat LoRA scaling"
        )
    output_config = load_json_object(
        adapter / "adapter_config.json", "interpolation adapter config"
    )
    if (
        output_config.get("r") != output_rank
        or not math.isclose(
            _finite_number(output_config.get("lora_alpha"), "adapter lora_alpha"),
            output_alpha,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise SelectionValidationError(
            "interpolation adapter config differs from provenance rank/alpha"
        )
    return {
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": sha256_file(provenance_path),
        "adapter_config_sha256": config_sha,
        "adapter_model_sha256": model_sha,
        "base_model_name_or_path": base_model,
        "provenance_output_path": str(output.get("path", "")),
        "source_adapter_config_sha256": source_config_hashes,
        "input_rank": input_rank,
        "input_lora_alpha": input_alpha,
        "output_rank": output_rank,
        "output_lora_alpha": output_alpha,
    }


def _validate_primary_adapter(
    adapter: Path,
    primary: dict[str, Any],
    reference_adapter: Path,
) -> dict[str, Any]:
    config_sha, model_sha, base_model = _validate_adapter_files(adapter)
    if model_sha != primary["adapter_model_sha256"]:
        raise SelectionValidationError("primary adapter model hash differs from locked policy")
    if "adapter_config_sha256" in primary and config_sha != primary["adapter_config_sha256"]:
        raise SelectionValidationError("primary adapter config hash differs from locked policy")
    reference_adapter = reference_adapter.expanduser().resolve()
    if reference_adapter == adapter.resolve():
        raise SelectionValidationError(
            "primary reference adapter must be the separate preselected training artifact"
        )
    reference_config_sha, reference_model_sha, reference_base = (
        _validate_adapter_files(reference_adapter)
    )
    if reference_model_sha != primary["adapter_model_sha256"]:
        raise SelectionValidationError(
            "primary reference adapter model hash differs from locked policy"
        )
    if reference_model_sha != model_sha or reference_config_sha != config_sha:
        raise SelectionValidationError(
            "primary candidate is not byte-identical to the preselected reference adapter"
        )
    if reference_base != base_model:
        raise SelectionValidationError(
            "primary candidate and reference adapter base models differ"
        )
    return {
        "provenance_path": None,
        "provenance_sha256": None,
        "adapter_config_sha256": config_sha,
        "adapter_model_sha256": model_sha,
        "base_model_name_or_path": base_model,
        "provenance_output_path": None,
        "primary_reference": {
            "adapter_dir": str(reference_adapter),
            "adapter_config_sha256": reference_config_sha,
            "adapter_model_sha256": reference_model_sha,
        },
    }


def _verify_planner_log(
    path: Path,
    adapter: Path,
    *,
    provenance_output_path: str | None,
    policy_inference_path: str | None = None,
    expected_base_model: str,
) -> str:
    _require_file(path, "planner log")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SelectionValidationError(f"cannot read planner log {path}: {error}") from error
    logged = re.findall(r"(?m)^Agent adapter:\s*(\S.*?)\s*$", text)
    if len(logged) != 1:
        raise SelectionValidationError(
            f"planner log must contain exactly one Agent adapter line: {path}"
        )
    allowed = {str(adapter.resolve())}
    if provenance_output_path:
        allowed.add(str(Path(provenance_output_path).expanduser().resolve()))
    if policy_inference_path:
        allowed.add(str(Path(policy_inference_path).expanduser().resolve()))
    observed = str(Path(logged[0]).expanduser().resolve())
    if observed not in allowed:
        raise SelectionValidationError(
            f"planner log used an unverified adapter path {observed}; allowed={sorted(allowed)}"
        )
    logged_bases = re.findall(r"(?m)^Agent base:\s*(\S.*?)\s*$", text)
    if len(logged_bases) != 1:
        raise SelectionValidationError(
            f"planner log must contain exactly one Agent base line: {path}"
        )
    observed_base = str(Path(logged_bases[0]).expanduser().resolve())
    expected_base = str(Path(expected_base_model).expanduser().resolve())
    if observed_base != expected_base:
        raise SelectionValidationError(
            f"planner log base differs from adapter config: {observed_base} != {expected_base}"
        )
    return observed_base


def _recorded_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SelectionValidationError(f"{label} path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SelectionValidationError(f"{label} path must be absolute: {value!r}")
    return path.resolve()


def _runtime_case_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SelectionValidationError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def _manifest_input_entry(
    inputs: dict[str, Any],
    name: str,
    *,
    expected_path: Path,
    expected_sha256: str,
) -> None:
    entry = _mapping(inputs.get(name), f"run_manifest.inputs.{name}")
    observed_path = _recorded_path(entry.get("path"), f"run_manifest.inputs.{name}")
    if observed_path != expected_path.expanduser().resolve():
        raise SelectionValidationError(
            f"run manifest {name} path mismatch: {observed_path} != {expected_path.resolve()}"
        )
    observed_sha = _sha(entry.get("sha256"), f"run_manifest.inputs.{name}.sha256")
    if observed_sha != expected_sha256:
        raise SelectionValidationError(
            f"run manifest {name} SHA-256 mismatch: {observed_sha} != {expected_sha256}"
        )


def _manifest_output_entry(
    outputs: dict[str, Any],
    name: str,
    *,
    expected_path: Path,
) -> str:
    entry = _mapping(outputs.get(name), f"run_manifest.outputs.{name}")
    observed_path = _recorded_path(entry.get("path"), f"run_manifest.outputs.{name}")
    expected_path = expected_path.expanduser().resolve()
    if observed_path != expected_path:
        raise SelectionValidationError(
            f"run manifest {name} output path mismatch: {observed_path} != {expected_path}"
        )
    _require_file(expected_path, f"run manifest {name} output")
    observed_bytes = entry.get("bytes")
    if (
        isinstance(observed_bytes, bool)
        or not isinstance(observed_bytes, int)
        or observed_bytes != expected_path.stat().st_size
    ):
        raise SelectionValidationError(
            f"run manifest {name} byte count differs from the current output"
        )
    expected_sha = _sha(entry.get("sha256"), f"run_manifest.outputs.{name}.sha256")
    current_sha = sha256_file(expected_path)
    if expected_sha != current_sha:
        raise SelectionValidationError(
            f"run manifest {name} SHA-256 differs from the current output"
        )
    return current_sha


def _command_flag(command: list[str], flag: str, label: str) -> str:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise SelectionValidationError(
            f"{label} must contain exactly one {flag} value"
        )
    return command[positions[0] + 1]


def _validate_command(command: Any, module: str, label: str) -> list[str]:
    if (
        not isinstance(command, list)
        or len(command) < 3
        or any(not isinstance(token, str) for token in command)
        or command[1:3] != ["-m", module]
    ):
        raise SelectionValidationError(
            f"run manifest {label} command is not `python -m {module}`"
        )
    return command


def _validate_run_manifest(
    *,
    manifest_path: Path,
    cases_path: Path,
    gold_path: Path,
    cases: Sequence[dict[str, Any]],
    adapter: Path,
    records_path: Path,
    planner_log_path: Path,
    metrics_path: Path,
    scorer_log_path: Path,
    identity: dict[str, Any],
    logged_base: str,
    expected_cases: int,
) -> tuple[dict[str, Any], dict[str, tuple[Path, str]]]:
    """Bind scored bytes to one complete, TOCTOU-checked generic runner run."""

    manifest_sha = sha256_file(manifest_path)
    manifest = load_json_object(manifest_path, "candidate run manifest")
    if manifest.get("schema_version") != 1 or manifest.get("runner") != "scripts.run_planner_eval":
        raise SelectionValidationError("candidate run manifest has an unsupported schema/runner")
    if manifest.get("status") != "complete" or manifest.get("error") is not None:
        raise SelectionValidationError("candidate run manifest is not a clean complete run")
    if manifest.get("expected_cases") != expected_cases:
        raise SelectionValidationError("candidate run manifest expected_cases is not locked")

    exit_state = _mapping(manifest.get("exit_state"), "run_manifest.exit_state")
    required_state = {
        "stage": "complete",
        "agent_returncode": 0,
        "predictions_validated": True,
        "scorer_returncode": 0,
        "metrics_validated": True,
        "inputs_reverified": True,
        "outputs_reverified": True,
    }
    for name, expected in required_state.items():
        observed = exit_state.get(name)
        if type(observed) is not type(expected) or observed != expected:
            raise SelectionValidationError(
                f"run manifest exit_state.{name}={observed!r}, expected {expected!r}"
            )

    cases_path = cases_path.expanduser().resolve()
    gold_path = gold_path.expanduser().resolve()
    adapter = adapter.expanduser().resolve()
    records_path = records_path.expanduser().resolve()
    planner_log_path = planner_log_path.expanduser().resolve()
    metrics_path = metrics_path.expanduser().resolve()
    scorer_log_path = scorer_log_path.expanduser().resolve()
    base_path = Path(logged_base).expanduser().resolve()
    base_config_path = base_path / "config.json"
    if not base_config_path.is_file() or base_config_path.stat().st_size <= 0:
        raise SelectionValidationError(
            f"run manifest base config is missing or empty: {base_config_path}"
        )

    cases_sha = sha256_file(cases_path)
    gold_sha = sha256_file(gold_path)
    base_config_sha = sha256_file(base_config_path)
    inputs = _mapping(manifest.get("inputs"), "run_manifest.inputs")
    _manifest_input_entry(
        inputs, "cases", expected_path=cases_path, expected_sha256=cases_sha
    )
    _manifest_input_entry(
        inputs, "gold", expected_path=gold_path, expected_sha256=gold_sha
    )
    _manifest_input_entry(
        inputs,
        "base_config",
        expected_path=base_config_path,
        expected_sha256=base_config_sha,
    )
    _manifest_input_entry(
        inputs,
        "adapter_config",
        expected_path=adapter / "adapter_config.json",
        expected_sha256=identity["adapter_config_sha256"],
    )
    _manifest_input_entry(
        inputs,
        "adapter_weights",
        expected_path=adapter / "adapter_model.safetensors",
        expected_sha256=identity["adapter_model_sha256"],
    )

    expected_input_names = {
        "cases",
        "gold",
        "base_config",
        "adapter_config",
        "adapter_weights",
    }
    provenance_path = identity.get("provenance_path")
    if provenance_path is not None:
        provenance = Path(str(provenance_path)).expanduser().resolve()
        _manifest_input_entry(
            inputs,
            "adapter_provenance",
            expected_path=provenance,
            expected_sha256=identity["provenance_sha256"],
        )
        expected_input_names.add("adapter_provenance")
    elif "adapter_provenance" in inputs:
        raise SelectionValidationError(
            "primary run manifest contains an unregistered adapter provenance input"
        )

    for case in cases:
        bench_id = str(case["bench_id"])
        video_label = f"source_video:{bench_id}"
        source = _mapping(case.get("source"), f"case {bench_id} source")
        expected_video_sha = _sha(
            source.get("video_sha256"), f"case {bench_id} source.video_sha256"
        )
        _manifest_input_entry(
            inputs,
            video_label,
            expected_path=_runtime_case_path(
                case.get("video_path"), f"case {bench_id} video_path"
            ),
            expected_sha256=expected_video_sha,
        )
        expected_input_names.add(video_label)

        raw_reference = case.get("ref_image_path")
        if raw_reference not in (None, ""):
            reference_label = f"reference_image:{bench_id}"
            reference_path = _runtime_case_path(
                raw_reference, f"case {bench_id} ref_image_path"
            )
            _require_file(reference_path, f"reference image for {bench_id}")
            _manifest_input_entry(
                inputs,
                reference_label,
                expected_path=reference_path,
                expected_sha256=sha256_file(reference_path),
            )
            expected_input_names.add(reference_label)
    if set(inputs) != expected_input_names:
        raise SelectionValidationError(
            "run manifest input set differs from the locked evaluation inputs: "
            f"missing={sorted(expected_input_names-set(inputs))}, "
            f"extra={sorted(set(inputs)-expected_input_names)}"
        )

    outputs = _mapping(manifest.get("outputs"), "run_manifest.outputs")
    expected_outputs = {
        "predictions": records_path,
        "planner_log": planner_log_path,
        "metrics": metrics_path,
        "scorer_log": scorer_log_path,
    }
    if set(outputs) != set(expected_outputs):
        raise SelectionValidationError(
            "run manifest output set is incomplete or unexpected: "
            f"observed={sorted(outputs)}"
        )
    output_hashes = {
        name: _manifest_output_entry(outputs, name, expected_path=path)
        for name, path in expected_outputs.items()
    }
    metrics = load_json_object(metrics_path, "candidate runner metrics")
    if type(metrics.get("num_cases")) is not int or metrics.get("num_cases") != expected_cases:
        raise SelectionValidationError(
            "candidate runner metrics num_cases differs from the locked case count"
        )

    commands = _mapping(manifest.get("commands"), "run_manifest.commands")
    if set(commands) != {"agent", "scorer", "cwd"}:
        raise SelectionValidationError("run manifest command set is incomplete or unexpected")
    agent_command = _validate_command(commands.get("agent"), "aurora.agent", "agent")
    scorer_command = _validate_command(
        commands.get("scorer"), "evaluation.agent_only_score", "scorer"
    )
    if agent_command.count("--custom_only") != 1 or agent_command.count("--plan_only") != 1:
        raise SelectionValidationError(
            "run manifest agent command must enable custom-only plan-only inference"
        )
    expected_agent_values = {
        "--custom_cases_jsonl": cases_path,
        "--agent_base": base_path,
        "--agent_adapter": adapter,
        "--out_dir": records_path.parent,
    }
    for flag, expected_path in expected_agent_values.items():
        observed = _recorded_path(
            _command_flag(agent_command, flag, "run manifest agent command"),
            f"run manifest agent command {flag}",
        )
        if observed != expected_path:
            raise SelectionValidationError(
                f"run manifest agent command {flag} path mismatch: {observed} != {expected_path}"
            )
    if _command_flag(agent_command, "--mask_backend", "run manifest agent command") != "none":
        raise SelectionValidationError("run manifest agent command executed mask tools")
    device = _command_flag(agent_command, "--device", "run manifest agent command")
    if not device.strip():
        raise SelectionValidationError("run manifest agent command has an empty device")
    python_path = _recorded_path(agent_command[0], "run manifest Python executable")
    if not python_path.is_file():
        raise SelectionValidationError(
            f"run manifest Python executable is unavailable: {python_path}"
        )
    expected_agent_command = [
        agent_command[0],
        "-m",
        "aurora.agent",
        "--custom_cases_jsonl",
        str(cases_path),
        "--custom_only",
        "--plan_only",
        "--mask_backend",
        "none",
        "--agent_base",
        str(base_path),
        "--agent_adapter",
        str(adapter),
        "--device",
        device,
        "--out_dir",
        str(records_path.parent),
    ]
    if agent_command != expected_agent_command:
        raise SelectionValidationError(
            "run manifest agent command differs from the locked generic runner command"
        )

    expected_scorer_values = {
        "--gold": gold_path,
        "--predictions": records_path,
        "--out": metrics_path,
    }
    for flag, expected_path in expected_scorer_values.items():
        observed = _recorded_path(
            _command_flag(scorer_command, flag, "run manifest scorer command"),
            f"run manifest scorer command {flag}",
        )
        if observed != expected_path:
            raise SelectionValidationError(
                f"run manifest scorer command {flag} path mismatch: {observed} != {expected_path}"
            )
    expected_scorer_command = [
        agent_command[0],
        "-m",
        "evaluation.agent_only_score",
        "--gold",
        str(gold_path),
        "--predictions",
        str(records_path),
        "--out",
        str(metrics_path),
    ]
    if scorer_command != expected_scorer_command:
        raise SelectionValidationError(
            "run manifest scorer command differs from the locked generic runner command"
        )
    recorded_cwd = _recorded_path(commands.get("cwd"), "run manifest cwd")
    if recorded_cwd != Path(__file__).resolve().parents[1]:
        raise SelectionValidationError(
            "run manifest cwd differs from the selector repository root"
        )

    bound_files: dict[str, tuple[Path, str]] = {
        "run_manifest": (manifest_path.resolve(), manifest_sha),
        "cases": (cases_path, cases_sha),
        "gold": (gold_path, gold_sha),
        "base_config": (base_config_path, base_config_sha),
        "adapter_config": (
            adapter / "adapter_config.json",
            identity["adapter_config_sha256"],
        ),
        "adapter_weights": (
            adapter / "adapter_model.safetensors",
            identity["adapter_model_sha256"],
        ),
        **{
            f"output:{name}": (path, output_hashes[name])
            for name, path in expected_outputs.items()
        },
    }
    if provenance_path is not None:
        bound_files["adapter_provenance"] = (
            Path(str(provenance_path)).resolve(),
            identity["provenance_sha256"],
        )
    primary_reference = identity.get("primary_reference")
    if isinstance(primary_reference, dict):
        reference_root = Path(str(primary_reference["adapter_dir"])).resolve()
        bound_files["primary_reference_config"] = (
            reference_root / "adapter_config.json",
            str(primary_reference["adapter_config_sha256"]),
        )
        bound_files["primary_reference_weights"] = (
            reference_root / "adapter_model.safetensors",
            str(primary_reference["adapter_model_sha256"]),
        )
    return manifest, bound_files


def _reverify_bound_files(bound_files: dict[str, tuple[Path, str]]) -> None:
    changed = [
        name
        for name, (path, expected_sha) in bound_files.items()
        if not path.is_file() or sha256_file(path) != expected_sha
    ]
    if changed:
        raise SelectionValidationError(
            "candidate artifacts changed while selection was running: "
            + ", ".join(sorted(changed))
        )


def _prediction_index(
    path: Path, expected_ids: set[str], expected_cases: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = load_jsonl(path, "candidate predictions")
    if len(rows) != expected_cases:
        raise SelectionValidationError(
            f"candidate has {len(rows)} predictions; expected {expected_cases}: {path}"
        )
    ids = _unique_ids(rows, "candidate predictions")
    actual = set(ids)
    if actual != expected_ids:
        raise SelectionValidationError(
            f"candidate prediction IDs differ: missing={sorted(expected_ids-actual)}, "
            f"extra={sorted(actual-expected_ids)}"
        )
    errors = [row["bench_id"] for row in rows if "error" in row]
    if errors:
        raise SelectionValidationError(f"candidate inference recorded errors: {errors[:10]}")
    return rows, {str(row["bench_id"]): row for row in rows}


def _subset_score(
    rows: Sequence[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return score(list(rows), [predictions[str(row["bench_id"])] for row in rows])


def _metric_and_strata(
    gold: Sequence[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    prediction_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[float]], dict[str, Any]]:
    by_category = {
        category: [row for row in gold if row.get("category") == category]
        for category in EXPECTED_CATEGORIES
    }
    all_metrics = score(list(gold), prediction_rows)
    category_metrics = {
        category: _subset_score(rows, prediction_index)
        for category, rows in by_category.items()
    }
    no_search_metrics = category_metrics["no_search_negative"]
    negative_counts = no_search_metrics["image_search_trigger"]
    negative_total = int(negative_counts["tn"]) + int(negative_counts["fp"])
    specificity = float(negative_counts["tn"]) / negative_total if negative_total else 0.0
    positive_metrics = category_metrics["true_search_positive"]
    scoped = {
        "prediction_rows": len(prediction_rows),
        "strict_raw_json_validity": float(all_metrics["strict_raw_json_validity"]),
        "subtask_accuracy": float(category_metrics["routing_control"]["subtask_accuracy"]),
        "no_search_specificity": specificity,
        "true_search_trigger_recall": float(
            positive_metrics["image_search_trigger"]["recall"]
        ),
        "search_query_end_to_end_recall": float(
            positive_metrics["image_search_query"]["end_to_end_recall"]
        ),
        "mask_trigger_f1": float(category_metrics["mask_control"]["mask_trigger"]["f1"]),
        "rewrite_constraint_retention": float(
            category_metrics["rewrite_retention"]["constraint_retention"]
        ),
    }
    scoped["utility"] = (
        0.5 * scoped["no_search_specificity"]
        + 0.5 * scoped["rewrite_constraint_retention"]
    )

    no_search_scores: list[float] = []
    for row in by_category["no_search_negative"]:
        prediction = prediction_index[str(row["bench_id"])]
        plan = extract_plan(prediction)
        # Match agent_only_score: invalid plans count as not triggering search.
        was_triggered = triggered(plan["image_search"]) if valid_plan(plan) else False
        no_search_scores.append(float(not was_triggered))
    rewrite_scores: list[float] = []
    for row in by_category["rewrite_retention"]:
        prediction = prediction_index[str(row["bench_id"])]
        plan = extract_plan(prediction)
        instruction = str(plan.get("refined_text_instruction", "")) if valid_plan(plan) else ""
        constraints = row.get("constraints", [])
        if not constraints:
            raise SelectionValidationError(
                f"rewrite retention row has no constraints: {row['bench_id']}"
            )
        rewrite_scores.append(
            sum(constraint_is_retained(item, instruction) for item in constraints)
            / len(constraints)
        )
    strata = {
        "no_search_negative": no_search_scores,
        "rewrite_retention": rewrite_scores,
    }
    if not math.isclose(
        sum(no_search_scores) / len(no_search_scores), specificity, abs_tol=1e-15
    ) or not math.isclose(
        sum(rewrite_scores) / len(rewrite_scores),
        scoped["rewrite_constraint_retention"],
        abs_tol=1e-15,
    ):
        raise SelectionValidationError("per-case utility scores disagree with locked scorer")
    diagnostics = {
        "all_case": {
            "json_validity": all_metrics["json_validity"],
            "strict_raw_json_validity": all_metrics["strict_raw_json_validity"],
            "subtask_accuracy": all_metrics["subtask_accuracy"],
            "image_search_trigger": all_metrics["image_search_trigger"],
            "image_search_query": all_metrics["image_search_query"],
            "mask_trigger": all_metrics["mask_trigger"],
            "constraint_retention": all_metrics["constraint_retention"],
        },
        "strict_raw_invalid_ids": all_metrics["details"]["strict_raw_invalid_predictions"],
        "utility_strata": {
            name: {
                "cases": len(values),
                "sum": sum(values),
                "mean": sum(values) / len(values),
                "case_score_sha256": hashlib.sha256(
                    json.dumps(values, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
            for name, values in strata.items()
        },
    }
    return scoped, strata, diagnostics


def _eligibility(
    scoped: dict[str, Any], thresholds: dict[str, Any]
) -> tuple[bool, dict[str, dict[str, Any]]]:
    checks: dict[str, dict[str, Any]] = {}
    for threshold_name, metric_name in THRESHOLD_METRICS.items():
        threshold = thresholds[threshold_name]
        value = scoped[metric_name]
        passed = value == threshold if threshold_name == "complete_prediction_rows" else value >= threshold
        checks[threshold_name] = {
            "metric": metric_name,
            "value": value,
            "operator": "==" if threshold_name == "complete_prediction_rows" else ">=",
            "threshold": threshold,
            "passed": passed,
        }
    return all(item["passed"] for item in checks.values()), checks


def paired_stratified_bootstrap(
    candidate_strata: dict[str, dict[str, list[float]]],
    *,
    seed: int,
    draws: int,
) -> dict[str, dict[str, float | int | str]]:
    if not candidate_strata:
        raise SelectionValidationError("bootstrap has no candidates")
    names = list(candidate_strata)
    lengths = {
        stratum: len(candidate_strata[names[0]][stratum])
        for stratum in ("no_search_negative", "rewrite_retention")
    }
    for name, strata in candidate_strata.items():
        if set(strata) != set(lengths):
            raise SelectionValidationError(f"candidate {name} has unexpected bootstrap strata")
        if any(len(strata[stratum]) != length for stratum, length in lengths.items()):
            raise SelectionValidationError(f"candidate {name} has inconsistent stratum lengths")
        if any(not math.isfinite(value) for values in strata.values() for value in values):
            raise SelectionValidationError(f"candidate {name} has non-finite case scores")

    rng = random.Random(seed)
    samples: dict[str, list[float]] = {name: [] for name in names}
    no_search_n = lengths["no_search_negative"]
    rewrite_n = lengths["rewrite_retention"]
    for _ in range(draws):
        negative_indices = [rng.randrange(no_search_n) for _ in range(no_search_n)]
        rewrite_indices = [rng.randrange(rewrite_n) for _ in range(rewrite_n)]
        for name in names:
            strata = candidate_strata[name]
            negative_mean = math.fsum(
                strata["no_search_negative"][index] for index in negative_indices
            ) / no_search_n
            rewrite_mean = math.fsum(
                strata["rewrite_retention"][index] for index in rewrite_indices
            ) / rewrite_n
            samples[name].append(0.5 * negative_mean + 0.5 * rewrite_mean)

    result: dict[str, dict[str, float | int | str]] = {}
    for name, values in samples.items():
        mean = math.fsum(values) / draws
        variance = math.fsum((value - mean) ** 2 for value in values) / (draws - 1)
        result[name] = {
            "draws": draws,
            "seed": seed,
            "standard_error": math.sqrt(variance),
            "bootstrap_mean": mean,
            "standard_deviation_denominator": "draws-1",
            "rng": "python random.Random(seed), paired randrange indices",
        }
    return result


def _candidate_result(
    *,
    candidate_id: str,
    candidate_dir: Path,
    kind: str,
    expected_ids: set[str],
    cases_path: Path,
    gold_path: Path,
    cases: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    normalized_policy: dict[str, Any],
    interpolation_lambda: float | None = None,
    primary: dict[str, Any] | None = None,
    primary_reference_adapter: Path | None = None,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    artifacts = _find_candidate_artifacts(candidate_dir)
    adapter = artifacts["adapter"]
    if kind == "grid":
        assert interpolation_lambda is not None
        identity = _validate_interpolation_provenance(
            adapter, interpolation_lambda, normalized_policy
        )
        policy_inference_path = None
    elif kind == "primary":
        assert primary is not None and primary_reference_adapter is not None
        identity = _validate_primary_adapter(
            adapter, primary, primary_reference_adapter
        )
        policy_inference_path = primary.get("inference_adapter_path")
    else:  # pragma: no cover - internal API guard
        raise AssertionError(kind)
    logged_base = _verify_planner_log(
        artifacts["planner_log"],
        adapter,
        provenance_output_path=identity["provenance_output_path"],
        policy_inference_path=policy_inference_path,
        expected_base_model=identity["base_model_name_or_path"],
    )
    run_manifest, bound_files = _validate_run_manifest(
        manifest_path=artifacts["run_manifest"],
        cases_path=cases_path,
        gold_path=gold_path,
        cases=cases,
        adapter=adapter,
        records_path=artifacts["records"],
        planner_log_path=artifacts["planner_log"],
        metrics_path=artifacts["metrics"],
        scorer_log_path=artifacts["scorer_log"],
        identity=identity,
        logged_base=logged_base,
        expected_cases=normalized_policy["expected_cases"],
    )
    predictions, prediction_index = _prediction_index(
        artifacts["records"], expected_ids, normalized_policy["expected_cases"]
    )
    scoped, strata, diagnostics = _metric_and_strata(
        gold, predictions, prediction_index
    )
    eligible, checks = _eligibility(scoped, normalized_policy["thresholds"])
    _reverify_bound_files(bound_files)
    result = {
        "candidate_id": candidate_id,
        "kind": kind,
        "lambda": interpolation_lambda,
        "candidate_dir": str(candidate_dir.resolve()),
        "adapter_dir": str(adapter.resolve()),
        "adapter_identity": identity,
        "prediction_artifact": {
            "path": str(artifacts["records"].resolve()),
            "sha256": sha256_file(artifacts["records"]),
            "rows": len(predictions),
        },
        "planner_log": {
            "path": str(artifacts["planner_log"].resolve()),
            "sha256": sha256_file(artifacts["planner_log"]),
            "base_model_path": logged_base,
        },
        "run_manifest": {
            "path": str(artifacts["run_manifest"].resolve()),
            "sha256": sha256_file(artifacts["run_manifest"]),
            "status": run_manifest["status"],
        },
        "metrics": scoped,
        "eligibility": {"eligible": eligible, "checks": checks},
        "diagnostics": diagnostics,
    }
    return result, strata


def _grid_decision(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in candidates if row["eligibility"]["eligible"]]
    if not eligible:
        return {
            "reference_candidate_id": None,
            "reference_lambda": None,
            "best_utility": None,
            "reference_bootstrap_se": None,
            "one_se_cutoff": None,
            "one_se_candidate_ids": [],
            "selected_candidate_id": None,
            "selected_lambda": None,
            "reason": "no interpolation candidate met all eligibility thresholds",
        }
    best_utility = max(row["metrics"]["utility"] for row in eligible)
    tied = [row for row in eligible if row["metrics"]["utility"] == best_utility]
    reference = min(tied, key=lambda row: row["lambda"])
    reference_se = reference["bootstrap"]["standard_error"]
    cutoff = best_utility - reference_se
    one_se = [row for row in eligible if row["metrics"]["utility"] >= cutoff]
    selected = min(one_se, key=lambda row: row["lambda"])
    return {
        "reference_candidate_id": reference["candidate_id"],
        "reference_lambda": reference["lambda"],
        "best_utility": best_utility,
        "reference_bootstrap_se": reference_se,
        "one_se_cutoff": cutoff,
        "one_se_candidate_ids": [
            row["candidate_id"] for row in sorted(one_se, key=lambda row: row["lambda"])
        ],
        "selected_candidate_id": selected["candidate_id"],
        "selected_lambda": selected["lambda"],
        "reason": "smallest eligible lambda within one bootstrap SE of the eligible point-estimate winner",
    }


def evaluate_candidates(
    *,
    cases_path: Path,
    gold_path: Path,
    audit_path: Path,
    policy_path: Path,
    candidates_root: Path,
    primary_reference_adapter: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path, label in (
        (cases_path, "validation cases"),
        (gold_path, "validation gold"),
        (audit_path, "leakage audit"),
        (policy_path, "selection policy"),
    ):
        _require_file(path, label)
    sealed_artifact_hashes = {
        "cases_sha256": sha256_file(cases_path),
        "gold_sha256": sha256_file(gold_path),
        "leakage_audit_sha256": sha256_file(audit_path),
        "policy_sha256": sha256_file(policy_path),
    }
    policy = load_json_object(policy_path, "selection policy")
    normalized_policy = validate_policy(policy)
    primary = normalized_policy["primary"]
    if primary is not None and primary_reference_adapter is None:
        raise SelectionValidationError(
            "policy registers a primary candidate; --primary-reference-adapter is required"
        )
    if primary is None and primary_reference_adapter is not None:
        raise SelectionValidationError(
            "--primary-reference-adapter was provided but policy has no primary candidate"
        )
    if primary_reference_adapter is not None:
        primary_reference_adapter = primary_reference_adapter.expanduser().resolve()
    cases, gold, audit = validate_validation_bundle(
        cases_path, gold_path, audit_path, policy, normalized_policy
    )
    del audit  # validation side effects above are the required use.
    candidates_root = candidates_root.expanduser().resolve()
    if candidates_root.is_symlink() or not candidates_root.is_dir():
        raise SelectionValidationError(
            f"candidates root is missing or a symlink: {candidates_root}"
        )
    suspicious_root_paths = [
        str(path.relative_to(candidates_root))
        for path in candidates_root.rglob("*")
        if any(DAY14_PATH_RE.search(part) for part in path.relative_to(candidates_root).parts)
    ]
    if suspicious_root_paths:
        raise SelectionValidationError(
            "candidates root contains Day-14-named artifacts: "
            f"{suspicious_root_paths[:5]}"
        )
    expected_directories = {_lambda_slug(value) for value in normalized_policy["grid"]}
    if normalized_policy["primary"] is not None:
        expected_directories.add(normalized_policy["primary"]["directory_name"])
    unexpected_lambda_dirs = sorted(
        path.name
        for path in candidates_root.iterdir()
        if path.is_dir() and path.name.startswith("lambda_")
        and path.name not in expected_directories
    )
    if unexpected_lambda_dirs:
        raise SelectionValidationError(
            f"unregistered interpolation candidate directories: {unexpected_lambda_dirs}"
        )
    expected_ids = {str(row["bench_id"]) for row in gold}

    results: list[dict[str, Any]] = []
    strata: dict[str, dict[str, list[float]]] = {}
    for value in normalized_policy["grid"]:
        candidate_id = _lambda_slug(value)
        result, candidate_strata = _candidate_result(
            candidate_id=candidate_id,
            candidate_dir=candidates_root / candidate_id,
            kind="grid",
            expected_ids=expected_ids,
            cases_path=cases_path,
            gold_path=gold_path,
            cases=cases,
            gold=gold,
            normalized_policy=normalized_policy,
            interpolation_lambda=value,
        )
        results.append(result)
        strata[candidate_id] = candidate_strata

    primary_result: dict[str, Any] | None = None
    if primary is not None:
        if primary["candidate_id"] in {row["candidate_id"] for row in results}:
            raise SelectionValidationError("primary candidate_id collides with interpolation grid")
        primary_result, primary_strata = _candidate_result(
            candidate_id=primary["candidate_id"],
            candidate_dir=candidates_root / primary["directory_name"],
            kind="primary",
            expected_ids=expected_ids,
            cases_path=cases_path,
            gold_path=gold_path,
            cases=cases,
            gold=gold,
            normalized_policy=normalized_policy,
            primary=primary,
            primary_reference_adapter=primary_reference_adapter,
        )
        results.append(primary_result)
        strata[primary["candidate_id"]] = primary_strata

    model_hashes = [row["adapter_identity"]["adapter_model_sha256"] for row in results]
    if len(set(model_hashes)) != len(model_hashes):
        raise SelectionValidationError("two registered candidates have identical adapter bytes")
    grid_source_configs = {
        (
            row["adapter_identity"]["source_adapter_config_sha256"]["adapter_a"],
            row["adapter_identity"]["source_adapter_config_sha256"]["adapter_b"],
        )
        for row in results
        if row["kind"] == "grid"
    }
    if len(grid_source_configs) != 1:
        raise SelectionValidationError(
            "interpolation candidates record inconsistent endpoint config identities"
        )
    base_models = {row["planner_log"]["base_model_path"] for row in results}
    if len(base_models) != 1:
        raise SelectionValidationError(
            f"registered candidates used different base models: {sorted(base_models)}"
        )

    bootstrap = paired_stratified_bootstrap(
        strata,
        seed=normalized_policy["bootstrap_seed"],
        draws=normalized_policy["bootstrap_draws"],
    )
    for result in results:
        result["bootstrap"] = bootstrap[result["candidate_id"]]
    grid_results = [row for row in results if row["kind"] == "grid"]
    grid_decision = _grid_decision(grid_results)

    if primary_result is not None and primary_result["eligibility"]["eligible"]:
        final_candidate_id = primary_result["candidate_id"]
        final_reason = "pre-registered primary candidate met every eligibility threshold"
    else:
        final_candidate_id = grid_decision["selected_candidate_id"]
        final_reason = (
            "primary candidate was ineligible; fell back to the locked interpolation-grid rule"
            if primary_result is not None
            else "no primary candidate was registered; used the locked interpolation-grid rule"
        )
    final_result = next(
        (row for row in results if row["candidate_id"] == final_candidate_id), None
    )
    primary_reference_identity = (
        primary_result["adapter_identity"].get("primary_reference")
        if primary_result is not None
        else None
    )
    input_hashes = {
        "cases_sha256": sha256_file(cases_path),
        "gold_sha256": sha256_file(gold_path),
        "leakage_audit_sha256": sha256_file(audit_path),
        "policy_sha256": sha256_file(policy_path),
    }
    if input_hashes != sealed_artifact_hashes:
        raise SelectionValidationError(
            "sealed validation artifacts changed while selection was running"
        )
    comparison = {
        "schema_version": 1,
        "selector": "fresh384_interpolation_selector",
        "day14_metrics_used": False,
        "validation_artifacts": input_hashes,
        "policy": {
            "lambda_grid": list(normalized_policy["grid"]),
            "bootstrap_seed": normalized_policy["bootstrap_seed"],
            "bootstrap_draws": normalized_policy["bootstrap_draws"],
            "eligibility_thresholds": normalized_policy["thresholds"],
            "primary_if_eligible_else_grid": primary,
        },
        "primary_reference_adapter": primary_reference_identity,
        "candidates": results,
        "grid_decision": grid_decision,
    }
    selection = {
        "schema_version": 1,
        "selected": final_result is not None,
        "selected_candidate_id": final_candidate_id,
        "selected_kind": final_result["kind"] if final_result else None,
        "selected_lambda": final_result["lambda"] if final_result else None,
        "selected_adapter_dir": final_result["adapter_dir"] if final_result else None,
        "selected_adapter_model_sha256": (
            final_result["adapter_identity"]["adapter_model_sha256"]
            if final_result else None
        ),
        "selected_metrics": final_result["metrics"] if final_result else None,
        "final_decision_reason": final_reason,
        "primary_candidate_id": primary_result["candidate_id"] if primary_result else None,
        "primary_eligible": (
            primary_result["eligibility"]["eligible"] if primary_result else None
        ),
        "primary_reference_adapter": primary_reference_identity,
        "grid_decision": grid_decision,
        "day14_metrics_used": False,
        "validation_artifacts": input_hashes,
    }
    return comparison, selection


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise SelectionValidationError(f"refusing to overwrite output artifact: {path}") from error


def run_and_write(
    *,
    cases_path: Path,
    gold_path: Path,
    audit_path: Path,
    policy_path: Path,
    candidates_root: Path,
    primary_reference_adapter: Path | None,
    comparison_out: Path,
    selection_out: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = {
        path.expanduser().resolve()
        for path in (cases_path, gold_path, audit_path, policy_path)
    }
    comparison_out = comparison_out.expanduser().resolve()
    selection_out = selection_out.expanduser().resolve()
    if comparison_out == selection_out or comparison_out in inputs or selection_out in inputs:
        raise SelectionValidationError("input and output artifact paths must be distinct")
    if comparison_out.exists() or selection_out.exists():
        raise SelectionValidationError("refusing to overwrite an existing selector output")
    comparison, selection = evaluate_candidates(
        cases_path=cases_path.expanduser().resolve(),
        gold_path=gold_path.expanduser().resolve(),
        audit_path=audit_path.expanduser().resolve(),
        policy_path=policy_path.expanduser().resolve(),
        candidates_root=candidates_root,
        primary_reference_adapter=primary_reference_adapter,
    )
    _write_new_json(comparison_out, comparison)
    selection["comparison_sha256"] = sha256_file(comparison_out)
    _write_new_json(selection_out, selection)
    return comparison, selection


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument(
        "--primary-reference-adapter",
        type=Path,
        help=(
            "preselected primary training artifact; required iff the policy registers "
            "a primary candidate"
        ),
    )
    parser.add_argument("--comparison-out", type=Path, required=True)
    parser.add_argument("--selection-out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        _, selection = run_and_write(
            cases_path=args.cases,
            gold_path=args.gold,
            audit_path=args.leakage_audit,
            policy_path=args.policy,
            candidates_root=args.candidates_root,
            primary_reference_adapter=args.primary_reference_adapter,
            comparison_out=args.comparison_out,
            selection_out=args.selection_out,
        )
    except SelectionValidationError as error:
        print(f"selection aborted: {error}", file=sys.stderr)
        return 1
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0 if selection["selected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

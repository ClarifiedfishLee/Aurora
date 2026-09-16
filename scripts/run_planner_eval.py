"""Run one auditable planner evaluation without model-selection semantics.

The Aurora planner is convenient for exploratory work: it may materialize a
released adapter when a requested path is absent, and it records per-case
errors without necessarily exiting non-zero.  Those behaviours are unsafe for
an interpolation grid.  This wrapper therefore requires local model artifacts,
checks the exact model paths reported by the planner, rejects incomplete or
error-bearing predictions, scores them, and records all input identities in a
fresh result directory.

This module intentionally contains no thresholds, candidate selection, or
Day-14 gate logic.  One invocation evaluates exactly one adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_PROVENANCE = "interpolation_provenance.json"
PREDICTIONS = "agent_pipeline_records.jsonl"
PLANNER_LOG = "planner.log"
METRICS = "metrics.json"
SCORER_LOG = "scorer.log"
RUN_MANIFEST = "run_manifest.json"
PLAN_FIELDS = {"refined_text_instruction", "subtask", "image_search", "mask"}
SUBTASKS = {
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
SHA256_RE = re.compile(r"[0-9a-f]{64}")
INTERPOLATION_METHOD = "exact_delta_space_rank_concat"


class PlannerEvalError(RuntimeError):
    """Raised when an evaluation cannot be proven complete and reproducible."""


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.expanduser().resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise PlannerEvalError(f"{label} is missing or empty: {path}")


def require_owned_output(path: Path, label: str) -> None:
    """Require a non-empty, regular output rather than a redirected symlink."""

    if path.is_symlink():
        raise PlannerEvalError(f"{label} must not be a symlink: {path}")
    require_nonempty_file(path, label)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PlannerEvalError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    require_nonempty_file(path, label)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlannerEvalError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlannerEvalError(f"{label} must contain a JSON object: {path}")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    require_nonempty_file(path, label)
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise PlannerEvalError(f"cannot read {label} {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
        except (json.JSONDecodeError, PlannerEvalError) as exc:
            raise PlannerEvalError(
                f"{path}:{line_number}: invalid JSON object: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise PlannerEvalError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def unique_bench_ids(rows: Iterable[dict[str, Any]], label: str) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        bench_id = row.get("bench_id")
        if not isinstance(bench_id, str) or not bench_id.strip():
            raise PlannerEvalError(f"{label} contains an empty or non-string bench_id")
        if bench_id in ids:
            raise PlannerEvalError(f"{label} contains duplicate bench_id {bench_id!r}")
        ids.add(bench_id)
    return ids


def _runtime_media_path(value: str) -> Path:
    """Resolve a media path exactly as the agent's repository cwd will."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    # Keep symlink components in the recorded path so re-hashing detects a
    # symlink that is retargeted while the evaluation is running.
    return Path(os.path.abspath(path))


def _valid_plan(plan: Any) -> bool:
    if not isinstance(plan, dict) or set(plan) != PLAN_FIELDS:
        return False
    if (
        not isinstance(plan["refined_text_instruction"], str)
        or not plan["refined_text_instruction"].strip()
    ):
        return False
    if not isinstance(plan["subtask"], str) or plan["subtask"] not in SUBTASKS:
        return False
    return all(
        value is False or (isinstance(value, str) and bool(value.strip()))
        for value in (plan["image_search"], plan["mask"])
    )


def _validate_case_gold_semantics(
    case_rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]]
) -> dict[str, dict[str, str]]:
    """Validate row pairing and return the media files used by inference."""

    cases_by_id = {str(row["bench_id"]): row for row in case_rows}
    gold_by_id = {str(row["bench_id"]): row for row in gold_rows}
    media_inputs: dict[str, dict[str, str]] = {}
    for bench_id, case in cases_by_id.items():
        if "gold_plan" in case:
            raise PlannerEvalError(
                f"cases JSONL must not contain gold_plan ({bench_id!r}); "
                "pass gold labels only to --gold"
            )
        prompt = case.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise PlannerEvalError(
                f"cases JSONL has an empty or non-string prompt for {bench_id!r}"
            )
        raw_video = case.get("video_path")
        if not isinstance(raw_video, str) or not raw_video.strip():
            raise PlannerEvalError(
                f"cases JSONL has an empty or non-string video_path for {bench_id!r}"
            )
        video = _runtime_media_path(raw_video)
        require_nonempty_file(video, f"source video for {bench_id!r}")
        video_sha = sha256_file(video)
        source = case.get("source")
        if isinstance(source, dict) and source.get("video_sha256") is not None:
            declared_sha = str(source["video_sha256"]).strip().lower()
            if SHA256_RE.fullmatch(declared_sha) is None:
                raise PlannerEvalError(
                    f"cases JSONL has an invalid source.video_sha256 for {bench_id!r}"
                )
            if declared_sha != video_sha:
                raise PlannerEvalError(
                    f"source video hash differs from case metadata for {bench_id!r}: {video}"
                )
        media_inputs[f"source_video:{bench_id}"] = {
            "path": str(video),
            "sha256": video_sha,
        }

        raw_ref = case.get("ref_image_path")
        if raw_ref not in (None, ""):
            if not isinstance(raw_ref, str) or not raw_ref.strip():
                raise PlannerEvalError(
                    f"cases JSONL has an invalid ref_image_path for {bench_id!r}"
                )
            ref_image = _runtime_media_path(raw_ref)
            require_nonempty_file(ref_image, f"reference image for {bench_id!r}")
            media_inputs[f"reference_image:{bench_id}"] = {
                "path": str(ref_image),
                "sha256": sha256_file(ref_image),
            }

        gold = gold_by_id[bench_id]
        if gold.get("prompt") != prompt:
            raise PlannerEvalError(
                f"cases/gold prompt mismatch for bench_id {bench_id!r}"
            )
        gold_video = gold.get("video_path")
        if not isinstance(gold_video, str) or _runtime_media_path(gold_video) != video:
            raise PlannerEvalError(
                f"cases/gold video_path mismatch for bench_id {bench_id!r}"
            )
        gold_ref = gold.get("ref_image_path")
        if raw_ref not in (None, ""):
            if (
                not isinstance(gold_ref, str)
                or _runtime_media_path(gold_ref) != ref_image
            ):
                raise PlannerEvalError(
                    f"cases/gold ref_image_path mismatch for bench_id {bench_id!r}"
                )
        elif gold_ref not in (None, ""):
            raise PlannerEvalError(
                f"cases/gold ref_image_path mismatch for bench_id {bench_id!r}"
            )
        if not _valid_plan(gold.get("gold_plan")):
            raise PlannerEvalError(
                f"gold JSONL has an invalid gold_plan for bench_id {bench_id!r}"
            )
        for field in ("axis", "category", "subtype", "edit_type"):
            if field in case and gold.get(field) != case[field]:
                raise PlannerEvalError(
                    f"cases/gold {field} mismatch for bench_id {bench_id!r}"
                )
        constraints = gold.get("constraints", [])
        if not isinstance(constraints, list) or any(
            not isinstance(item, dict) or "value" not in item
            for item in constraints
        ):
            raise PlannerEvalError(
                f"gold JSONL has invalid constraints for bench_id {bench_id!r}"
            )
    return media_inputs


def _validate_interpolation_provenance(
    provenance: dict[str, Any], adapter_config_sha: str, adapter_weights_sha: str
) -> None:
    """Ensure a present interpolation provenance describes these exact bytes."""

    if provenance.get("schema_version") != 1:
        raise PlannerEvalError("adapter provenance has an unsupported schema_version")
    if provenance.get("method") != INTERPOLATION_METHOD:
        raise PlannerEvalError("adapter provenance has an unsupported method")
    coefficient = provenance.get("coefficient")
    if not isinstance(coefficient, dict):
        raise PlannerEvalError("adapter provenance is missing coefficient metadata")
    lambda_b = coefficient.get("lambda_b")
    if (
        isinstance(lambda_b, bool)
        or not isinstance(lambda_b, (int, float))
        or not math.isfinite(float(lambda_b))
        or not 0.0 <= float(lambda_b) <= 1.0
    ):
        raise PlannerEvalError("adapter provenance has an invalid lambda_b")
    output = provenance.get("output")
    if not isinstance(output, dict):
        raise PlannerEvalError("adapter provenance is missing output metadata")
    if output.get("adapter_config_sha256") != adapter_config_sha:
        raise PlannerEvalError(
            "adapter config hash differs from interpolation provenance"
        )
    if output.get("adapter_model_sha256") != adapter_weights_sha:
        raise PlannerEvalError(
            "adapter weights hash differs from interpolation provenance"
        )


def validate_inputs(
    *,
    base: Path,
    adapter: Path,
    cases: Path,
    gold: Path,
    out_dir: Path,
    expected_cases: int,
) -> tuple[Path, Path, Path, Path, Path, set[str], dict[str, dict[str, str]]]:
    if (
        isinstance(expected_cases, bool)
        or not isinstance(expected_cases, int)
        or expected_cases <= 0
    ):
        raise PlannerEvalError("expected_cases must be a positive integer")

    base = resolve_path(base)
    adapter = resolve_path(adapter)
    cases = resolve_path(cases)
    gold = resolve_path(gold)
    out_dir = resolve_path(out_dir)
    if out_dir.exists() or out_dir.is_symlink():
        raise PlannerEvalError(f"result directory must not already exist: {out_dir}")
    if out_dir == base or out_dir.is_relative_to(base):
        raise PlannerEvalError(
            f"result directory must not be inside the base model: {out_dir}"
        )
    if out_dir == adapter or out_dir.is_relative_to(adapter):
        raise PlannerEvalError(
            f"result directory must not be inside the adapter: {out_dir}"
        )
    if cases == gold:
        raise PlannerEvalError("cases and gold must be distinct files")

    base_config = base / "config.json"
    adapter_config = adapter / ADAPTER_CONFIG
    adapter_weights = adapter / ADAPTER_WEIGHTS
    require_nonempty_file(base_config, "base model config")
    require_nonempty_file(adapter_config, "adapter config")
    require_nonempty_file(adapter_weights, "adapter weights")
    require_nonempty_file(cases, "cases JSONL")
    require_nonempty_file(gold, "gold JSONL")

    try:
        if cases.samefile(gold):
            raise PlannerEvalError("cases and gold must not alias the same file")
    except OSError as exc:
        raise PlannerEvalError(f"cannot compare cases and gold identities: {exc}") from exc

    # Hash before parsing, then verify all hashes again after validation.  This
    # ensures the bytes we validated are the bytes passed to the subprocesses.
    base_config_sha = sha256_file(base_config)
    adapter_config_sha = sha256_file(adapter_config)
    adapter_weights_sha = sha256_file(adapter_weights)
    cases_sha = sha256_file(cases)
    gold_sha = sha256_file(gold)
    load_json_object(base_config, "base model config")
    load_json_object(adapter_config, "adapter config")
    case_rows = load_jsonl(cases, "cases JSONL")
    gold_rows = load_jsonl(gold, "gold JSONL")
    if len(case_rows) != expected_cases:
        raise PlannerEvalError(
            f"cases JSONL has {len(case_rows)} rows; expected exactly {expected_cases}: {cases}"
        )
    if len(gold_rows) != expected_cases:
        raise PlannerEvalError(
            f"gold JSONL has {len(gold_rows)} rows; expected exactly {expected_cases}: {gold}"
        )
    case_ids = unique_bench_ids(case_rows, "cases JSONL")
    gold_ids = unique_bench_ids(gold_rows, "gold JSONL")
    if case_ids != gold_ids:
        missing = sorted(case_ids - gold_ids)
        extra = sorted(gold_ids - case_ids)
        raise PlannerEvalError(
            f"cases/gold bench_id mismatch: absent_from_gold={missing}, absent_from_cases={extra}"
        )

    media_inputs = _validate_case_gold_semantics(case_rows, gold_rows)

    inputs = {
        "cases": {"path": str(cases), "sha256": cases_sha},
        "gold": {"path": str(gold), "sha256": gold_sha},
        "base_config": {
            "path": str(base_config),
            "sha256": base_config_sha,
        },
        "adapter_config": {
            "path": str(adapter_config),
            "sha256": adapter_config_sha,
        },
        "adapter_weights": {
            "path": str(adapter_weights),
            "sha256": adapter_weights_sha,
        },
        **media_inputs,
    }
    provenance = adapter / ADAPTER_PROVENANCE
    if provenance.exists() or provenance.is_symlink():
        require_nonempty_file(provenance, "adapter provenance")
        provenance_sha = sha256_file(provenance)
        provenance_data = load_json_object(provenance, "adapter provenance")
        _validate_interpolation_provenance(
            provenance_data, adapter_config_sha, adapter_weights_sha
        )
        inputs["adapter_provenance"] = {
            "path": str(provenance),
            "sha256": provenance_sha,
        }
    # Close the parse/hash race before launching the expensive model process.
    verify_input_hashes(inputs)
    return base, adapter, cases, gold, out_dir, case_ids, inputs


def validate_predictions(
    path: Path, expected_ids: set[str], expected_cases: int
) -> None:
    require_owned_output(path, "prediction JSONL")
    rows = load_jsonl(path, "prediction JSONL")
    if len(rows) != expected_cases:
        raise PlannerEvalError(
            f"prediction JSONL has {len(rows)} rows; expected exactly {expected_cases}: {path}"
        )
    actual_ids = unique_bench_ids(rows, "prediction JSONL")
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise PlannerEvalError(
            f"prediction bench_id mismatch: missing={missing}, extra={extra}"
        )
    error_ids = [str(row["bench_id"]) for row in rows if "error" in row]
    if error_ids:
        raise PlannerEvalError(f"planner inference recorded errors for: {error_ids}")


def verify_model_log(path: Path, base: Path, adapter: Path) -> None:
    require_owned_output(path, "planner log")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for label, expected in (
        ("base", f"Agent base: {base}"),
        ("adapter", f"Agent adapter: {adapter}"),
    ):
        declarations = [line for line in lines if line.startswith(f"Agent {label}:")]
        if declarations != [expected]:
            raise PlannerEvalError(
                "planner did not uniquely confirm the exact requested model paths "
                f"({label}): "
                f"expected={expected!r}, observed={declarations!r}"
            )


def validate_metrics(path: Path, expected_cases: int) -> dict[str, Any]:
    require_owned_output(path, "metrics JSON")
    metrics = load_json_object(path, "metrics JSON")
    num_cases = metrics.get("num_cases")
    if (
        isinstance(num_cases, bool)
        or not isinstance(num_cases, int)
        or num_cases != expected_cases
    ):
        raise PlannerEvalError(
            f"metrics num_cases is {num_cases!r}; expected integer {expected_cases}"
        )
    return metrics


def build_agent_command(
    python: Path,
    cases: Path,
    base: Path,
    adapter: Path,
    out_dir: Path,
    device: str,
) -> list[str]:
    return [
        str(python),
        "-m",
        "aurora.agent",
        "--custom_cases_jsonl",
        str(cases),
        "--custom_only",
        "--plan_only",
        "--mask_backend",
        "none",
        "--agent_base",
        str(base),
        "--agent_adapter",
        str(adapter),
        "--device",
        device,
        "--out_dir",
        str(out_dir),
    ]


def build_score_command(
    python: Path, gold: Path, predictions: Path, metrics: Path
) -> list[str]:
    return [
        str(python),
        "-m",
        "evaluation.agent_only_score",
        "--gold",
        str(gold),
        "--predictions",
        str(predictions),
        "--out",
        str(metrics),
    ]


def run_logged(command: list[str], log_path: Path, *, echo: bool) -> int:
    """Run one command and create its log without ever overwriting a file."""

    with log_path.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            if echo:
                print(line, end="", flush=True)
        return process.wait()


def _artifact_entry(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def verify_input_hashes(inputs: dict[str, dict[str, str]]) -> None:
    """Prove that no declared input changed while inference was running."""

    changed: list[str] = []
    for label, entry in inputs.items():
        path = Path(entry["path"])
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            changed.append(label)
    if changed:
        raise PlannerEvalError(
            "evaluation inputs changed during the run: " + ", ".join(changed)
        )


def verify_output_hashes(outputs: dict[str, dict[str, str]]) -> None:
    """Prove scorer execution did not mutate already-validated outputs."""

    changed: list[str] = []
    for label, entry in outputs.items():
        path = Path(entry["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != entry["sha256"]
        ):
            changed.append(label)
    if changed:
        raise PlannerEvalError(
            "validated evaluation outputs changed during the run: "
            + ", ".join(changed)
        )


def run_evaluation(
    *,
    base: Path,
    adapter: Path,
    cases: Path,
    gold: Path,
    out_dir: Path,
    expected_cases: int,
    device: str,
    executor: Callable[[list[str], Path, bool], int] | None = None,
) -> dict[str, Any]:
    """Run one evaluation and return the final, append-once run manifest."""

    (
        base,
        adapter,
        cases,
        gold,
        out_dir,
        expected_ids,
        inputs,
    ) = validate_inputs(
        base=base,
        adapter=adapter,
        cases=cases,
        gold=gold,
        out_dir=out_dir,
        expected_cases=expected_cases,
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    predictions = out_dir / PREDICTIONS
    planner_log = out_dir / PLANNER_LOG
    metrics_path = out_dir / METRICS
    scorer_log = out_dir / SCORER_LOG
    manifest_path = out_dir / RUN_MANIFEST
    python = Path(sys.executable).absolute()
    agent_command = build_agent_command(python, cases, base, adapter, out_dir, device)
    score_command = build_score_command(python, gold, predictions, metrics_path)
    started = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    exit_state: dict[str, Any] = {
        "stage": "agent",
        "agent_returncode": None,
        "predictions_validated": False,
        "scorer_returncode": None,
        "metrics_validated": False,
        "inputs_reverified": False,
        "outputs_reverified": False,
    }
    status = "failed"
    error: str | None = None
    execute = executor or (lambda cmd, log, echo: run_logged(cmd, log, echo=echo))

    try:
        agent_returncode = execute(agent_command, planner_log, True)
        exit_state["agent_returncode"] = agent_returncode
        if agent_returncode != 0:
            raise PlannerEvalError(
                f"planner exited with status {agent_returncode}; inspect {planner_log}"
            )
        verify_model_log(planner_log, base, adapter)
        validate_predictions(predictions, expected_ids, expected_cases)
        exit_state["predictions_validated"] = True
        validated_outputs = {
            "predictions": {
                "path": str(predictions),
                "sha256": sha256_file(predictions),
            },
            "planner_log": {
                "path": str(planner_log),
                "sha256": sha256_file(planner_log),
            },
        }
        exit_state["stage"] = "scorer"

        scorer_returncode = execute(score_command, scorer_log, False)
        exit_state["scorer_returncode"] = scorer_returncode
        if scorer_returncode != 0:
            raise PlannerEvalError(
                f"scorer exited with status {scorer_returncode}; inspect {scorer_log}"
            )
        validate_metrics(metrics_path, expected_cases)
        exit_state["metrics_validated"] = True
        verify_input_hashes(inputs)
        exit_state["inputs_reverified"] = True
        verify_output_hashes(validated_outputs)
        exit_state["outputs_reverified"] = True
        exit_state["stage"] = "complete"
        status = "complete"
    except (OSError, PlannerEvalError) as exc:
        error = str(exc)
    except Exception as exc:  # preserve an auditable failure for runtime faults
        error = f"{type(exc).__name__}: {exc}"

    finished = datetime.now(timezone.utc)
    output_paths = {
        "predictions": predictions,
        "planner_log": planner_log,
        "metrics": metrics_path,
        "scorer_log": scorer_log,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "runner": "scripts.run_planner_eval",
        "status": status,
        "error": error,
        "expected_cases": expected_cases,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": time.monotonic() - started_monotonic,
        "inputs": inputs,
        "commands": {
            "agent": agent_command,
            "scorer": score_command,
            "cwd": str(REPO_ROOT),
        },
        "exit_state": exit_state,
        "outputs": {
            name: entry
            for name, path in output_paths.items()
            if (entry := _artifact_entry(path)) is not None
        },
    }
    _write_manifest(manifest_path, manifest)
    if error is not None:
        raise PlannerEvalError(error)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        manifest = run_evaluation(
            base=args.base,
            adapter=args.adapter,
            cases=args.cases,
            gold=args.gold,
            out_dir=args.out_dir,
            expected_cases=args.expected_cases,
            device=args.device,
        )
    except PlannerEvalError as exc:
        print(f"Planner evaluation aborted: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

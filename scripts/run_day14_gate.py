"""Run the pre-registered Day-14 planner gate with fail-fast checks.

This wrapper exists because ``aurora.agent`` intentionally falls back to the
released Aurora adapter when a requested adapter path is missing and records
per-case inference errors without failing the whole process.  Both behaviours
are convenient for exploratory runs but unsafe for a model-selection gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = Path("/mlx_devbox/users/jieyu.li/models/Qwen3-VL-8B-Instruct")
DEFAULT_ADAPTER = Path("/tmp/aurora-sft-5k/lora-final-12597-best-eval")
DEFAULT_GOLD = Path("data/week1/planner_100.jsonl")
DEFAULT_OUT_DIR = Path("/tmp/aurora-sft-5k/day14_gate")
RELEASED_BASELINE = {"subtask_accuracy": 0.78, "constraint_retention": 0.48}

ABSOLUTE_THRESHOLDS = {
    "json_validity": ("min", 0.99),
    "subtask_accuracy": ("min", 0.95),
    "image_search_trigger.f1": ("min", 0.80),
    "mask_trigger.f1": ("min", 0.95),
    "constraint_retention": ("min", 0.80),
    "constraint_case_accuracy": ("min", 0.65),
    "source_entity_false_trigger.rate": ("max", 0.05),
}


class GateValidationError(RuntimeError):
    """Raised when a gate precondition or inference artifact is unsafe."""


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise GateValidationError(f"{label} is missing or empty: {path}")


def validate_model_paths(base: Path, adapter: Path) -> tuple[Path, Path]:
    base = resolve_path(base)
    adapter = resolve_path(adapter)
    require_nonempty_file(base / "config.json", "base model config")
    require_nonempty_file(adapter / "adapter_config.json", "adapter config")
    require_nonempty_file(adapter / "adapter_model.safetensors", "adapter weights")
    return base, adapter


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise GateValidationError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def unique_bench_ids(rows: Iterable[dict[str, Any]], label: str) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        bench_id = str(row.get("bench_id", "")).strip()
        if not bench_id:
            raise GateValidationError(f"{label} contains an empty bench_id")
        if bench_id in ids:
            raise GateValidationError(f"{label} contains duplicate bench_id {bench_id!r}")
        ids.add(bench_id)
    return ids


def validate_gold(path: Path, expected_cases: int) -> set[str]:
    require_nonempty_file(path, "gold JSONL")
    rows = load_jsonl(path)
    if len(rows) != expected_cases:
        raise GateValidationError(
            f"gold JSONL has {len(rows)} rows; expected exactly {expected_cases}: {path}"
        )
    return unique_bench_ids(rows, "gold JSONL")


def validate_predictions(path: Path, expected_ids: set[str], expected_cases: int) -> None:
    require_nonempty_file(path, "prediction JSONL")
    rows = load_jsonl(path)
    if len(rows) != expected_cases:
        raise GateValidationError(
            f"prediction JSONL has {len(rows)} rows; expected exactly {expected_cases}: {path}"
        )
    actual_ids = unique_bench_ids(rows, "prediction JSONL")
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise GateValidationError(f"prediction bench_id mismatch: missing={missing}, extra={extra}")
    error_ids = [str(row["bench_id"]) for row in rows if "error" in row]
    if error_ids:
        raise GateValidationError(f"planner inference recorded errors for: {error_ids}")


def verify_model_log(path: Path, base: Path, adapter: Path) -> None:
    require_nonempty_file(path, "planner log")
    text = path.read_text(encoding="utf-8")
    expected_base = f"Agent base: {base}"
    expected_adapter = f"Agent adapter: {adapter}"
    missing = [entry for entry in (expected_base, expected_adapter) if entry not in text]
    if missing:
        raise GateValidationError(
            "planner did not confirm the requested model paths in its log: " + ", ".join(missing)
        )


def nested_metric(metrics: dict[str, Any], dotted_name: str) -> float:
    value: Any = metrics
    for part in dotted_name.split("."):
        if not isinstance(value, dict) or part not in value:
            raise GateValidationError(f"metric is missing: {dotted_name}")
        value = value[part]
    if not isinstance(value, (int, float)):
        raise GateValidationError(f"metric is not numeric: {dotted_name}={value!r}")
    return float(value)


def evaluate_gate(metrics: dict[str, Any], released: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    for name, (direction, threshold) in ABSOLUTE_THRESHOLDS.items():
        value = nested_metric(metrics, name)
        passed = value >= threshold if direction == "min" else value <= threshold
        checks[name] = {
            "value": value,
            "operator": ">=" if direction == "min" else "<=",
            "threshold": threshold,
            "passed": passed,
        }

    for name in ("subtask_accuracy", "constraint_retention"):
        value = nested_metric(metrics, name)
        baseline = nested_metric(released, name)
        checks[f"{name}_vs_released"] = {
            "value": value,
            "operator": ">",
            "threshold": baseline,
            "passed": value > baseline,
        }

    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "diagnostics": {
            "strict_raw_json_validity": nested_metric(metrics, "strict_raw_json_validity"),
            "image_search_query.conditional_accuracy": nested_metric(
                metrics, "image_search_query.conditional_accuracy"
            ),
            "image_search_query.end_to_end_recall": nested_metric(
                metrics, "image_search_query.end_to_end_recall"
            ),
        },
    }


def build_agent_command(
    python: Path,
    gold: Path,
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
        str(gold),
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


def build_score_command(python: Path, gold: Path, predictions: Path, metrics: Path) -> list[str]:
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


def run_logged(command: list[str], log_path: Path, *, echo: bool) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
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
        return_code = process.wait()
    if return_code != 0:
        raise GateValidationError(
            f"command exited with status {return_code}; inspect {log_path}: {' '.join(command)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--released-metrics",
        type=Path,
        help="Optional full released-LoRA metrics JSON; defaults to pre-registered 0.78/0.48 values.",
    )
    parser.add_argument("--expected-cases", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    try:
        base, adapter = validate_model_paths(args.base, args.adapter)
        gold = resolve_path(args.gold)
        expected_ids = validate_gold(gold, args.expected_cases)
        if args.released_metrics is None:
            released = dict(RELEASED_BASELINE)
            released_source = "pre-registered constants: routing=0.78, retention=0.48"
        else:
            released_metrics_path = resolve_path(args.released_metrics)
            require_nonempty_file(released_metrics_path, "released Aurora metrics")
            released = json.loads(released_metrics_path.read_text(encoding="utf-8"))
            released_source = str(released_metrics_path)

        out_dir = resolve_path(args.out_dir)
        predictions = out_dir / "agent_pipeline_records.jsonl"
        planner_log = out_dir / "planner.log"
        metrics_path = out_dir / "metrics.json"
        scorer_log = out_dir / "scorer.log"
        summary_path = out_dir / "gate_summary.json"
        artifacts = (predictions, planner_log, metrics_path, scorer_log, summary_path)
        existing = [str(path) for path in artifacts if path.exists()]
        if existing:
            raise GateValidationError(
                "refusing to mix a gate run with existing artifacts; use a fresh --out-dir: "
                + ", ".join(existing)
            )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Preserve the venv entrypoint instead of resolving its symlink to a
        # system interpreter, which would lose the Worker's installed packages.
        python = Path(sys.executable).absolute()
        run_logged(
            build_agent_command(python, gold, base, adapter, out_dir, args.device),
            planner_log,
            echo=True,
        )
        verify_model_log(planner_log, base, adapter)
        validate_predictions(predictions, expected_ids, args.expected_cases)

        run_logged(
            build_score_command(python, gold, predictions, metrics_path),
            scorer_log,
            echo=False,
        )
        require_nonempty_file(metrics_path, "scorer metrics")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary = evaluate_gate(metrics, released)
        summary["released_baseline_source"] = released_source
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0 if summary["passed"] else 2
    except (GateValidationError, json.JSONDecodeError) as exc:
        print(f"Day-14 gate aborted: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

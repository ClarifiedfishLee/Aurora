"""Build and audit a bounded LLaMA-Factory LoRA refresh run.

The refresh deliberately loads an existing adapter through
``adapter_name_or_path`` but starts a new Trainer run.  It never consumes an
old Trainer checkpoint or an external Day-14 gate metric when selecting the
refresh checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from scripts.make_llamafactory_config import validate_rows
except ModuleNotFoundError:  # direct ``python scripts/...py`` execution
    from make_llamafactory_config import validate_rows


TRAIN_DATASET_NAME = "aurora_planner_refresh_train"
EVAL_DATASET_NAME = "aurora_planner_refresh_eval"
EXPECTED_TRAIN_ROWS = 1024
EXPECTED_EVAL_ROWS = 256
GRADIENT_ACCUMULATION_STEPS = 8
CHECKPOINT_INTERVAL = 32
EXPECTED_CHECKPOINT_STEPS = (32, 64, 96, 128)
SELECTION_POLICY = "lowest_refresh_eval_loss_then_earliest_step"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_explicit_splits(
    train_path: Path,
    eval_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = _read_jsonl(train_path)
    eval_rows = _read_jsonl(eval_path)
    if len(train_rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"refresh train split must contain exactly {EXPECTED_TRAIN_ROWS} rows, got {len(train_rows)}"
        )
    if len(eval_rows) != EXPECTED_EVAL_ROWS:
        raise ValueError(
            f"refresh eval split must contain exactly {EXPECTED_EVAL_ROWS} rows, got {len(eval_rows)}"
        )
    validate_rows(train_rows)
    validate_rows(eval_rows)
    train_videos = {str(row["videos"][0]) for row in train_rows}
    eval_videos = {str(row["videos"][0]) for row in eval_rows}
    overlap = train_videos & eval_videos
    if overlap:
        preview = ", ".join(sorted(overlap)[:3])
        raise ValueError(f"refresh train/eval video overlap ({len(overlap)}): {preview}")
    return train_rows, eval_rows


def _write_dataset_info(dataset_dir: Path, train_path: Path, eval_path: Path) -> Path:
    if train_path.parent != dataset_dir or eval_path.parent != dataset_dir:
        raise ValueError("refresh train and eval JSONL files must share one dataset directory")
    dataset_info_path = dataset_dir / "dataset_info.json"
    if dataset_info_path.exists():
        dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        if not isinstance(dataset_info, dict):
            raise ValueError(f"{dataset_info_path}: expected a JSON object")
    else:
        dataset_info = {}
    common = {
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "system": "system", "videos": "videos"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    dataset_info[TRAIN_DATASET_NAME] = {"file_name": train_path.name, **common}
    dataset_info[EVAL_DATASET_NAME] = {"file_name": eval_path.name, **common}
    dataset_info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset_info_path


def write_refresh_bundle(
    train_path: Path,
    eval_path: Path,
    model_path: str,
    adapter_path: Path,
    output_dir: Path,
    config_path: Path,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    """Validate fixed splits and emit the refresh YAML plus selection policy."""

    train_path = train_path.resolve()
    eval_path = eval_path.resolve()
    adapter_path = adapter_path.resolve()
    output_dir = output_dir.resolve()
    config_path = config_path.resolve()
    if "best-eval" not in adapter_path.name:
        raise ValueError("refresh adapter_name_or_path must point to the preselected best-eval adapter")
    if adapter_path == output_dir:
        raise ValueError("refresh output_dir must differ from adapter_name_or_path")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("refresh output_dir must be new or empty; refusing possible Trainer resume state")
    train_rows, eval_rows = _validate_explicit_splits(train_path, eval_path)
    update_steps = math.ceil(len(train_rows) / GRADIENT_ACCUMULATION_STEPS)
    if update_steps != EXPECTED_CHECKPOINT_STEPS[-1]:
        raise ValueError(f"expected 128 refresh update steps, got {update_steps}")
    dataset_dir = train_path.parent
    dataset_info_path = _write_dataset_info(dataset_dir, train_path, eval_path)

    # create_new_adapter=false makes the input adapter trainable.  The new
    # output directory plus overwrite_output_dir avoids Trainer auto-resume;
    # resume_from_checkpoint is intentionally absent, so no old optimizer or
    # scheduler state is loaded.
    yaml = f"""### model
model_name_or_path: {json.dumps(model_path)}
adapter_name_or_path: {json.dumps(str(adapter_path))}
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
create_new_adapter: false

### dataset
dataset: {TRAIN_DATASET_NAME}
eval_dataset: {EVAL_DATASET_NAME}
dataset_dir: {json.dumps(str(dataset_dir))}
template: qwen3_vl_nothink
cutoff_len: 4096
max_samples: {EXPECTED_TRAIN_ROWS}
preprocessing_num_workers: 8
dataloader_num_workers: 4
val_size: 0.0

### output
output_dir: {json.dumps(str(output_dir))}
logging_steps: 4
save_strategy: steps
save_steps: {CHECKPOINT_INTERVAL}
save_total_limit: 4
eval_strategy: steps
eval_steps: {CHECKPOINT_INTERVAL}
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: {GRADIENT_ACCUMULATION_STEPS}
learning_rate: 2.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml, encoding="utf-8")

    policy_path = (policy_path or config_path.with_name("refresh_selection_policy.json")).resolve()
    policy = {
        "version": 1,
        "selection_source": "refresh_eval",
        "selection_metric": "eval_loss",
        "lower_is_better": True,
        "selection_policy": SELECTION_POLICY,
        "tie_breaker": "earliest_step",
        "eligible_checkpoint_steps": list(EXPECTED_CHECKPOINT_STEPS),
        "external_gate_metrics_allowed": False,
        "expected_checkpoint_count": 4,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_sha256": _sha256(train_path),
        "eval_sha256": _sha256(eval_path),
    }
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    return {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_videos": len({str(row["videos"][0]) for row in train_rows}),
        "eval_videos": len({str(row["videos"][0]) for row in eval_rows}),
        "video_overlap": 0,
        "update_steps": update_steps,
        "checkpoint_steps": list(EXPECTED_CHECKPOINT_STEPS),
        "dataset_info": str(dataset_info_path),
        "selection_policy": str(policy_path),
    }


def _checkpoint_eval_loss(checkpoint: Path, step: int) -> float:
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    entries = [
        item
        for item in state.get("log_history", [])
        if item.get("step") == step and isinstance(item.get("eval_loss"), (int, float))
    ]
    if not entries:
        raise ValueError(f"{state_path}: no refresh eval_loss at step {step}")
    return float(entries[-1]["eval_loss"])


def select_refresh_checkpoint(
    output_dir: Path,
    policy_path: Path,
    selection_out: Path | None = None,
) -> dict[str, Any]:
    """Select only from pre-registered refresh-eval losses; never gate scores."""

    output_dir = output_dir.resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("unsupported or changed refresh checkpoint selection policy")
    if policy.get("selection_source") != "refresh_eval" or policy.get("external_gate_metrics_allowed") is not False:
        raise ValueError("selection policy must exclude external Day-14 gate metrics")
    steps = tuple(policy.get("eligible_checkpoint_steps", []))
    if steps != EXPECTED_CHECKPOINT_STEPS:
        raise ValueError(f"eligible checkpoint steps must be {EXPECTED_CHECKPOINT_STEPS}")

    candidates = []
    for step in steps:
        checkpoint = output_dir / f"checkpoint-{step}"
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        for artifact in ("adapter_config.json", "adapter_model.safetensors"):
            artifact_path = checkpoint / artifact
            if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
                raise FileNotFoundError(artifact_path)
        candidates.append(
            {
                "step": step,
                "eval_loss": _checkpoint_eval_loss(checkpoint, step),
                "checkpoint": str(checkpoint),
            }
        )
    selected = min(candidates, key=lambda item: (item["eval_loss"], item["step"]))
    result = {
        "selection_source": "refresh_eval",
        "selection_metric": "eval_loss",
        "selection_policy": SELECTION_POLICY,
        "external_gate_metrics_used": False,
        "selected_step": selected["step"],
        "selected_eval_loss": selected["eval_loss"],
        "selected_checkpoint": selected["checkpoint"],
        "candidates": candidates,
    }
    if selection_out is not None:
        selection_out.parent.mkdir(parents=True, exist_ok=True)
        selection_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="write a bounded refresh training bundle")
    build.add_argument("--train", type=Path, required=True)
    build.add_argument("--eval", type=Path, required=True)
    build.add_argument("--model", required=True)
    build.add_argument("--adapter", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--config-out", type=Path, required=True)
    build.add_argument("--policy-out", type=Path)

    select = subparsers.add_parser("select", help="select the pre-registered refresh checkpoint")
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--policy", type=Path, required=True)
    select.add_argument("--selection-out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "build":
        result = write_refresh_bundle(
            args.train,
            args.eval,
            args.model,
            args.adapter,
            args.output_dir,
            args.config_out,
            policy_path=args.policy_out,
        )
    else:
        result = select_refresh_checkpoint(args.output_dir, args.policy, args.selection_out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

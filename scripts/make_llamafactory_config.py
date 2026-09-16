"""Create a self-contained LLaMA-Factory dataset description and SFT YAML."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


TRAIN_DATASET_NAME = "aurora_planner_sft_train"
EVAL_DATASET_NAME = "aurora_planner_sft_eval"


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("SFT dataset is empty")
    for index, row in enumerate(rows, 1):
        messages = row.get("messages")
        videos = row.get("videos")
        if not isinstance(row.get("system"), str) or not row["system"].strip():
            raise ValueError(f"row {index}: missing system prompt")
        if not isinstance(messages, list) or [message.get("role") for message in messages] != ["user", "assistant"]:
            raise ValueError(f"row {index}: expected user/assistant messages")
        if not isinstance(videos, list) or len(videos) != 1 or messages[0]["content"].count("<video>") != 1:
            raise ValueError(f"row {index}: expected one video and one <video> token")
        if not Path(videos[0]).is_file():
            raise FileNotFoundError(videos[0])


def grouped_split(
    rows: list[dict[str, Any]], eval_ratio: float = 0.02
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval ratio must be between zero and one")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["videos"][0])].append(row)
    if len(groups) < 2:
        raise ValueError("grouped train/eval split requires at least two unique videos")
    ordered_videos = sorted(
        groups,
        key=lambda video: hashlib.sha256(video.encode()).hexdigest(),
    )
    eval_group_count = min(len(groups) - 1, max(1, round(len(groups) * eval_ratio)))
    eval_videos = set(ordered_videos[:eval_group_count])
    train_rows = [row for row in rows if str(row["videos"][0]) not in eval_videos]
    eval_rows = [row for row in rows if str(row["videos"][0]) in eval_videos]
    summary = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_videos": len(groups) - len(eval_videos),
        "eval_videos": len(eval_videos),
        "video_overlap": 0,
    }
    return train_rows, eval_rows, summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_bundle(
    dataset_path: Path,
    model_path: str,
    output_dir: Path,
    config_path: Path,
    eval_ratio: float = 0.02,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line]
    validate_rows(rows)
    train_rows, eval_rows, split_summary = grouped_split(rows, eval_ratio=eval_ratio)
    estimated_update_steps = max(1, math.ceil(len(train_rows) / 8))
    checkpoint_steps = max(10, min(250, estimated_update_steps // 6))
    train_path = dataset_path.with_name(f"{dataset_path.stem}_train.jsonl")
    eval_path = dataset_path.with_name(f"{dataset_path.stem}_eval.jsonl")
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    dataset_info = {
        TRAIN_DATASET_NAME: {
            "file_name": train_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "system": "system", "videos": "videos"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
        EVAL_DATASET_NAME: {
            "file_name": eval_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "system": "system", "videos": "videos"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    (dataset_path.parent / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    yaml = f"""### model
model_name_or_path: {model_path}
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target: all

### dataset
dataset: {TRAIN_DATASET_NAME}
eval_dataset: {EVAL_DATASET_NAME}
dataset_dir: {dataset_path.parent}
template: qwen3_vl_nothink
cutoff_len: 4096
max_samples: {len(train_rows)}
preprocessing_num_workers: 8
dataloader_num_workers: 4
val_size: 0.0

### output
output_dir: {output_dir}
logging_steps: 5
save_steps: {checkpoint_steps}
save_total_limit: 2
eval_steps: {checkpoint_steps}
eval_strategy: steps
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml, encoding="utf-8")
    return {
        **split_summary,
        "train_dataset": str(train_path),
        "eval_dataset": str(eval_path),
        "checkpoint_steps": checkpoint_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    args = parser.parse_args()
    summary = write_bundle(
        args.dataset.resolve(),
        args.model,
        args.output_dir.resolve(),
        args.config_out.resolve(),
        eval_ratio=args.eval_ratio,
    )
    print(json.dumps({"dataset": str(args.dataset), "config": str(args.config_out), **summary}, indent=2))


if __name__ == "__main__":
    main()

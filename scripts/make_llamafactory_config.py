"""Create a self-contained LLaMA-Factory dataset description and SFT YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASET_NAME = "aurora_planner_sft"


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


def write_bundle(dataset_path: Path, model_path: str, output_dir: Path, config_path: Path) -> None:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line]
    validate_rows(rows)
    dataset_info = {
        DATASET_NAME: {
            "file_name": dataset_path.name,
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
dataset: {DATASET_NAME}
dataset_dir: {dataset_path.parent}
template: qwen3_vl_nothink
cutoff_len: 4096
max_samples: {len(rows)}
preprocessing_num_workers: 8
dataloader_num_workers: 4
val_size: 0.02

### output
output_dir: {output_dir}
logging_steps: 5
save_steps: 100
eval_steps: 50
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    args = parser.parse_args()
    write_bundle(args.dataset.resolve(), args.model, args.output_dir.resolve(), args.config_out.resolve())
    print(json.dumps({"dataset": str(args.dataset), "config": str(args.config_out)}, indent=2))


if __name__ == "__main__":
    main()

# Week 2 planner-SFT protocol

Week 2 builds planner data without downloading the 2.09 TB Aurora editor
dataset. The working set contains source video plus clean edit instruction;
target edited videos are not retained.

## 1. Stream a bounded source working set

Start with the 500-case smoke set:

```bash
HF_HOME=/tmp/aurora-hf-cache \
HF_DATASETS_CACHE=/tmp/aurora-hf-cache/datasets \
python -m scripts.stream_sft_sources \
  --out-dir /tmp/aurora-sft-bootstrap-500 \
  --buffer-size 64
```

The default mix is 200 Ditto-combined, 100 ROSE insertion, 100 ROSE removal,
and 100 ROSE text-only edits. The command is resumable through `manifest.jsonl`.
Use a small shuffle buffer because buffering 1,000 video examples delays the
first materialized sample and wastes temporary disk.

## 2. Produce released-LoRA teacher plans

```bash
python -m scripts.build_planner_sft teacher-cases \
  --manifest /tmp/aurora-sft-bootstrap-500/manifest.jsonl \
  --out /tmp/aurora-sft-bootstrap-500/teacher_cases.jsonl

python -m aurora.agent \
  --custom_cases_jsonl /tmp/aurora-sft-bootstrap-500/teacher_cases.jsonl \
  --custom_only --plan_only --mask_backend none \
  --out_dir /tmp/aurora-sft-bootstrap-500/teacher
```

Teacher inference uses the released Aurora LoRA for routing/search/mask. The
clean dataset instruction overrides its rewrite in the final target so that
atomic details are not lost. Dataset metadata corrects obvious subset routes
(`combined_tasks`, `add_object`, and `remove_object`).

## 3. Export canonical and LLaMA-Factory records

```bash
python -m scripts.build_planner_sft compose \
  --manifest /tmp/aurora-sft-bootstrap-500/manifest.jsonl \
  --teacher-records /tmp/aurora-sft-bootstrap-500/teacher/agent_pipeline_records.jsonl \
  --canonical-out /tmp/aurora-sft-bootstrap-500/sft_canonical.jsonl \
  --llama-out /tmp/aurora-sft-bootstrap-500/sft_llama.jsonl
```

The initial degradation is deterministic and meaning-preserving. It exists to
validate the end-to-end training path, not to claim the final eight-category
hard-case contribution. Hard-case generation and semantic validation are added
only after the 500-case training smoke passes.

Create an isolated LLaMA-Factory bundle and launch the one-epoch smoke:

```bash
python -m scripts.make_llamafactory_config \
  --dataset /tmp/aurora-sft-bootstrap-500/sft_llama.jsonl \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --output-dir /tmp/aurora-sft-bootstrap-500/lora-smoke \
  --config-out /tmp/aurora-sft-bootstrap-500/train.yaml

llamafactory-cli train /tmp/aurora-sft-bootstrap-500/train.yaml
```

The generated smoke config matches Aurora's LoRA rank 32 / alpha 64, uses the
official `qwen3_vl_nothink` template, validates one `<video>` token per media
path, and limits video pixels before any GPU allocation.

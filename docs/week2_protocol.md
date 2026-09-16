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

## 4. Smoke result and calibration gate

The 500-case smoke completed one epoch on one A100-SXM4-80GB in 203 seconds
(62 optimizer steps). It reached train loss 0.0842 and eval loss 0.0210. On
the 100-case development regression it retained valid JSON on every case,
mask F1 was 100%, and lexical constraint retention rose to 74.5%. However,
search F1 fell to zero because only 10 of the 500 teacher plans requested a
search. Routing also fell to 72%. This is a useful pipeline pass but a data
balance failure, so it must not be scaled unchanged.

Before the 5K run, add distinct calibration examples for under-search and for
the full route taxonomy:

```bash
python -m scripts.augment_planner_sft \
  --base /tmp/aurora-sft-bootstrap-500/sft_llama.jsonl \
  --out /tmp/aurora-sft-bootstrap-500/sft_calibrated.jsonl \
  --metadata-out /tmp/aurora-sft-bootstrap-500/calibration_metadata.jsonl \
  --search-count 500 --routing-count 500
```

The final calibration generator contains 160 concrete entities across brand
products, IP characters, landmarks, and cultural artifacts. It combines them
with 32 request/spatial templates for addition and replacement. Routing
calibration separately covers all 11 planner
subtasks, including `customization`, with three distinct semantic templates
per subtask. Re-run the same 100-case gate before expanding the working set.

The calibrated 1,500-case run passed that gate:

| Model | Runtime / raw JSON | Routing | Search F1 | Valid-query recall | Mask F1 | Constraint retention | False search |
|---|---:|---:|---:|---:|---:|---:|---:|
| Released Aurora LoRA | 100% / 100% | 78% | 88.9% | 80% | 100% | 48.0% | 0% |
| 500-case bootstrap | 100% / 98% | 72% | 0% | 0% | 100% | 74.5% | 0% |
| 1,500-case calibrated SFT | 100% / 100% | 98% | 88.9% | 80% | 100% | 86.5% | 0% |

The two remaining route errors are one `combined_tasks` case classified as
`change_color` and the sole `customization` case classified as
`replace_object`. The two missed searches are `Starbucks holiday cup` and
`Japanese cherry blossom tree`, the same overall recall level as the released
LoRA. Every query that was triggered by the released and 1,500-case models
contained the correct entity alias, so conditional query accuracy was 100%;
valid-query recall above includes missed triggers. These are
development-regression results, not final held-out benchmark claims.

For the 5K stage, stream 2K Ditto-combined plus 1K each of ROSE insertion,
removal, and v2v. The current `datasets` release requires `.decode(False)` so
the downloader reads encoded source MP4 bytes without installing `torchcodec`
or decoding unused target videos. Teacher planning can then use the resumable
batched runner:

```bash
python -m scripts.run_planner_teacher_vllm \
  --cases /tmp/aurora-sft-5k/teacher_cases.jsonl \
  --merged-model /tmp/aurora-agent-merged \
  --out /tmp/aurora-sft-5k/teacher/agent_pipeline_records.jsonl \
  --batch-size 8 --video-frames 6 --frame-max-side 448
```

## 5. Assemble the final v2 dataset

The source export contains 4,999 valid canonical records; one blank source
instruction is filtered before composition. Strict semantic checks accept
4,598 generated hard requests. The remaining 401 valid sources intentionally
have no hard variant because none passed the quality checks, rather than being
filled with weak synthetic data. One extra hard row derived from the blank
source is dropped and reported.

Assemble canonical, accepted-hard, and calibration records with the partial
hard set explicitly allowed:

```bash
python -m scripts.assemble_final_sft \
  --canonical /tmp/aurora-sft-5k/sft_canonical.jsonl \
  --base-llama /tmp/aurora-sft-5k/sft_llama.jsonl \
  --hard-metadata /tmp/aurora-sft-5k/hard_requests_v2.jsonl \
  --out /tmp/aurora-sft-5k/sft_final_12597_v2.jsonl \
  --summary-out /tmp/aurora-sft-5k/sft_final_12597_v2_summary.json \
  --search-count 2000 --routing-count 1000 \
  --allow-partial-hard --drop-extra-hard
```

The expected v2 composition is:

| Partition | Records |
|---|---:|
| Canonical base | 4,999 |
| Accepted hard requests | 4,598 |
| Under-search calibration | 2,000 |
| Routing calibration | 1,000 |
| **Total** | **12,597** |

Do not bypass the assembly checks: they verify canonical/base alignment,
assistant JSON, media identity, hard-request acceptance flags, calibration
coverage, and unknown or missing hard rows.

## 6. Build the grouped split and train

Split by source-video identity, not by row. Canonical, hard, search, and routing
variants derived from the same source must remain on one side of the split:

```bash
python -m scripts.make_llamafactory_config \
  --dataset /tmp/aurora-sft-5k/sft_final_12597_v2.jsonl \
  --model /mlx_devbox/users/jieyu.li/models/Qwen3-VL-8B-Instruct \
  --output-dir /tmp/aurora-sft-5k/lora-final-12597 \
  --config-out /tmp/aurora-sft-5k/train_final_12597.yaml \
  --eval-ratio 0.02
```

The deterministic grouped split contains 12,339 training rows from 4,899
videos and 258 evaluation rows from 100 videos, with zero source-video overlap.
The generated config retains LoRA rank 32 / alpha 64, evaluates and checkpoints
during the one-epoch run, and keeps only the two newest checkpoints.

Before reading the 100-case development result, select the main SFT adapter as
the checkpoint with the lowest loss on this grouped 258-row evaluation split.
Copy only its inference files to `lora-final-12597-best-eval` as checkpoints
are rotated. This is the pre-registered model-selection rule; do not choose a
checkpoint using the 100-case gate. If the last-step root adapter differs from
the best-eval checkpoint, report it only as a separate diagnostic.

Launch training in the prepared Worker environment:

```bash
HF_HOME=/tmp/llamafactory-hf-cache \
/tmp/llamafactory-venv/bin/llamafactory-cli train \
  /tmp/aurora-sft-5k/train_final_12597.yaml
```

The prepared model working copies, training data, environment, and live output
are under the Worker's `/tmp`; only the base-model path above is persistent.
`/tmp` disappears with the Worker. For this run the persistent `/mlx_devbox`
volume was already full, so copying there would not be a valid backup. After
the trainer exits successfully, use resumable `rsync` to stage the complete
output directory and `lora-final-12597-best-eval` snapshot in the Devbox
master's `/tmp`, then immediately pull them into
`runs/week2/final_12597/` on the local machine. Generate a sorted SHA-256
manifest on the Worker and verify it locally before treating the backup as
complete. Retain the Devbox staging copy until local verification succeeds.

## 7. Day-14 agent-only gate

Run the unchanged 100-case development regression with the final adapter, then
score it with the same deterministic evaluator used for every earlier model:

```bash
python -m aurora.agent \
  --custom_cases_jsonl data/week1/planner_100.jsonl \
  --custom_only --plan_only --mask_backend none \
  --agent_base /mlx_devbox/users/jieyu.li/models/Qwen3-VL-8B-Instruct \
  --agent_adapter /tmp/aurora-sft-5k/lora-final-12597-best-eval \
  --out_dir /tmp/aurora-sft-5k/day14_gate

python -m evaluation.agent_only_score \
  --gold data/week1/planner_100.jsonl \
  --predictions /tmp/aurora-sft-5k/day14_gate/agent_pipeline_records.jsonl \
  --out /tmp/aurora-sft-5k/day14_gate/metrics.json
```

Pre-register the pass criteria before reading the final result: JSON validity
at least 99%, routing accuracy at least 95%, search F1 at least 80%, mask F1 at
least 95%, lexical constraint retention at least 80%, constraint-case
retention at least 65%, and source-entity false-trigger rate at most 5%. The
final model must also show a clear improvement over the released Aurora LoRA
in both routing and retention; meeting only the absolute thresholds is not
sufficient. Do not revise these thresholds after seeing the result.

Report both validity fields emitted by the evaluator. `json_validity` measures
the normalized runtime plan that Aurora can execute; `strict_raw_json_validity`
parses the complete `agent_raw` response and enforces the exact four-field
contract without Aurora's cleanup. The latter is the appropriate measure of
whether the model itself learned structured output.

Also report `image_search_query.conditional_accuracy` and
`image_search_query.end_to_end_recall`. These use explicit, pre-registered
entity aliases for the ten positive search cases, so a triggered but unrelated
query cannot receive credit. Treat them as diagnostics rather than a new kill
threshold because the positive sample is small and entity-overlaps training.

Record the result next to the historical 500/1,500 rows, but do not treat this
reused development suite as the final held-out benchmark. Its videos and raw
requests have zero exact overlap with the v2 training data, but it is only ten
videos crossed with ten recurring request patterns. Seven of its ten external
search entities and all ten weather rewrite targets also occur in training.
Preference-data work begins only after this regression gate passes or the
failure has been diagnosed and the SFT data corrected.

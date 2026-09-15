# Week 1 evaluation protocol

Week 1 turns the smoke-tested pipeline into a reproducible planner baseline.
The immediate deliverables are a Base-vs-Aurora-LoRA comparison on 100 cases
and a 30-50 pair pilot measuring agreement between the video judge and a human.

## 1. Planner regression suite

Generate the deterministic early-development suite:

```bash
python -m scripts.build_week1_planner_cases \
  --out data/week1/planner_100.jsonl
```

It contains 100 cases, with 25 primary cases for each of `search`, `mask`,
`routing`, and `rewrite`. It deliberately reuses the ten smoke videos, so it is
only a regression suite. It must not be reported as the final Mini-AgentEdit
benchmark and must never be mixed into its held-out test split.

Run the released Aurora planner without executing tools:

```bash
python -m aurora.agent \
  --custom_cases_jsonl data/week1/planner_100.jsonl \
  --custom_only --plan_only --mask_backend none \
  --out_dir runs/week1/aurora_lora
```

Run the unadapted Qwen3-VL-8B baseline with the same decoding path:

```bash
python -m aurora.agent \
  --custom_cases_jsonl data/week1/planner_100.jsonl \
  --custom_only --plan_only --mask_backend none --no_agent_adapter \
  --out_dir runs/week1/base
```

`--plan_only` is important: it skips Serper and Grounded-SAM execution while
preserving the planner's predicted `image_search` and `mask` fields. Do not use
`--disable_image_search` for agent-only scoring because that flag intentionally
overwrites the search decision.

Score both runs:

```bash
python -m evaluation.agent_only_score \
  --gold data/week1/planner_100.jsonl \
  --predictions runs/week1/aurora_lora/agent_pipeline_records.jsonl \
  --out runs/week1/aurora_lora/metrics.json

python -m evaluation.agent_only_score \
  --gold data/week1/planner_100.jsonl \
  --predictions runs/week1/base/agent_pipeline_records.jsonl \
  --out runs/week1/base/metrics.json
```

Report JSON validity, routing accuracy, search-trigger F1, mask-trigger F1,
constraint retention, and source-entity false-trigger rate. Inspect errors by
axis before changing prompts or annotations.

## 2. Judge agreement pilot

### Judge deployment

Use the official UniEditBench code, Qwen3-VL-4B-Instruct, and the
`sft_image_video_lora_4b` adapter. On the Merlin Worker run:

```bash
bash scripts/setup_unieditbench_judge.sh
```

The script creates its environment and caches under `/tmp`, because the Merlin
home filesystem does not have enough free space for the official CUDA/vLLM
dependency set. Base weights and the adapter remain in persistent storage.

There are two upstream integration traps captured by the script and wrapper:

1. `decord` is required to read video but is absent from the published
   `requirements.txt`.
2. vLLM ignores LoRA weights attached to visual-tower modules when the adapter
   is loaded dynamically. The script first merges the complete adapter into the
   base model, then serves the merged model so the visual weights are retained.

The published inference script also maps `source_prompt`/`target_prompt` into a
template that expects `original_prompt`/`edited_prompt`. Use the compatibility
wrapper, which accepts both field conventions and fails the run on errors:

```bash
python -m evaluation.unieditbench_infer \
  --metadata data/week1/judge_smoke.json \
  --save runs/week1/judge_smoke_result.json \
  --unieditbench_repo /mlx_devbox/users/jieyu.li/external/UniEditBench \
  --port 8005
```

Select 30-50 single-axis A/B pairs before bulk outcome rendering. Use the same
source video, seed, CFG, frame count, and editor checkpoint for both variants.
Randomize which variant is presented as A to avoid a position bias.

The checked-in pilot builder produces 40 pairs: 10 search, 10 mask, 16
rewrite, and 4 routing negative controls. Routing is a negative control because
the current editor bridge records but does not consume `plan.subtask`; those
pairs should be ties until an actual route-dependent execution path exists.

```bash
python -m scripts.build_week1_ab_pilot prepare-tools

# Run Aurora on data/week1/ab_tool_cases.jsonl with grounded_sam, then:
python -m scripts.build_week1_ab_pilot build-records \
  --resolved runs/week1/ab_tools/agent_pipeline_records.jsonl

python -m aurora.editor_bridge_video \
  --records_jsonl data/week1/ab_editor_records.jsonl \
  --ckpt models/aurora_editor.safetensors \
  --out_dir runs/week1/ab_pilot/videos \
  --num_frames 45 --save_frames 64 --num_inference_steps 50 \
  --seed 42 --cfg_scale 2.0 --image_cfg_scale 1.0 \
  --fallback_to_two_pass_cfg --use_mask_overlay

python -m scripts.build_blind_ab_page \
  --pairs data/week1/ab_pairs.jsonl \
  --videos-dir runs/week1/ab_pilot/videos \
  --out-dir runs/week1/ab_pilot/blind
```

The blind page copies the candidates to opaque `A`/`B` filenames and stores
the answer mapping separately in `blind_key.json`. Keep that key away from
annotators. Human votes are saved in browser local storage and can be exported
as JSONL.

Store one annotation per line:

```json
{"bench_id":"judge_pilot_0001","axis":"search","video_a":"path/to/a.mp4","video_b":"path/to/b.mp4","human_label":"A","judge_label":"A","human_notes":"Target identity is correct only in A","judge_scores":{"A":4.0,"B":2.0}}
```

Allowed labels are `A`, `B`, and `tie`. Use any other value (for example,
`invalid`) to exclude a corrupt or unjudgeable pair while keeping an audit
trail. Compute agreement with:

```bash
python -m evaluation.judge_agreement \
  --annotations data/week1/judge_agreement.jsonl \
  --out runs/week1/judge_agreement_metrics.json
```

The report includes exact agreement, directional agreement after removing ties,
Cohen's kappa, tie rates, confusion matrices, and per-axis breakdowns. Do not
start the 300-500-case render batch until the pilot has useful coverage on all
four axes and disagreements have been reviewed. A practical go/no-go target is
at least 70% directional agreement overall, with no axis dominated by ties; the
threshold is a project decision, not a claim about a universal judge standard.

## 3. Frozen Mini-AgentEdit boundary

The final 150-200-case benchmark must use videos that do not occur in smoke
tests, SFT data, prompt development, or preference mining. During Week 1, record
source ID, license, entity family, and split before downloading media. Freeze
the benchmark only after the pilot rubric is stable.

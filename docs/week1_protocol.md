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

Select 30-50 single-axis A/B pairs before bulk outcome rendering. Use the same
source video, seed, CFG, frame count, and editor checkpoint for both variants.
Randomize which variant is presented as A to avoid a position bias.

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

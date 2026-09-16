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

Run the unchanged 100-case development regression with the final adapter
through the fail-fast wrapper:

```bash
HF_HOME=/tmp/aurora-hf-cache \
.venv/bin/python scripts/run_day14_gate.py
```

The wrapper first requires non-empty base-model and best-eval adapter files.
It then checks that the planner log names those exact paths, rather than a
silently downloaded fallback adapter. Before scoring, it requires exactly 100
unique prediction IDs matching the gold set and zero per-case inference
errors. It writes the raw records, planner/scorer logs, `metrics.json`, and a
threshold-by-threshold `gate_summary.json` under
`/tmp/aurora-sft-5k/day14_gate/`. Exit status 0 means pass, 2 means the run was
complete but missed at least one criterion, and 1 means the run was aborted by
a safety check.

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

## 8. Preserve refresh1 as a failed correction

The first bounded refresh is a result, not the final adapter. Its checkpoint
was selected only by refresh-validation loss, after which the unchanged
Day-14 regression was run once. It corrected the targeted false-search
behaviour but failed the pre-registered strict retention thresholds.

The failure was traced to a data-template collision. Ten generated weather
rows used the same raw-request and target skeleton as the ten recurring snow
cases in Day-14. The generated target phrase `keeping every subject visible`
displaced the gate's literal `clearly visible` constraint. Exact prompt,
concept, and template-ID checks had not detected this shared lexical skeleton.
Do not relabel this run as passing, discard its artifacts, or select another
checkpoint using the observed Day-14 result. Preserve its data, four
checkpoints, eval-loss selection, and failing gate output under the immutable
refresh1 run directory.

## 9. Run the recipe2 correction without reusing Day-14 for selection

Recipe2 keeps the fixed 1,024/256 composition but changes the colliding
weather templates and adds a paired prompt/target forbidden 4-gram audit. A
synthetic row is rejected when both its user prompt and refined target share a
4-gram with the same Day-14 case. Replay rows are excluded from this new
synthetic-template audit because they are inherited rather than generated.
The previous source-video, normalized-prompt, concept, template, contract, and
forbidden-phrase checks remain active.

Generate recipe2 into a fresh directory; never overwrite refresh1:

```bash
python -m scripts.build_oversearch_refresh \
  --base-train /tmp/aurora-sft-5k/sft_final_12597_v2_train.jsonl \
  --base-eval /tmp/aurora-sft-5k/sft_final_12597_v2_eval.jsonl \
  --train-out /tmp/aurora-sft-refresh-recipe2/refresh_train.jsonl \
  --validation-out /tmp/aurora-sft-refresh-recipe2/refresh_eval.jsonl \
  --cases-out /tmp/aurora-sft-refresh-recipe2/refresh_cases.jsonl \
  --gold-out /tmp/aurora-sft-refresh-recipe2/refresh_gold.jsonl \
  --summary-out /tmp/aurora-sft-refresh-recipe2/refresh_generation_audit.json \
  --forbidden-cases data/week1/planner_100.jsonl \
  --forbidden-ngram-n 4
```

Require `recipe_version: 2`, a zero synthetic 4-gram hit count, and all other
isolation checks in `refresh_generation_audit.json` before allocating the GPU.
Build and train the continuation from the already selected v2 adapter:

```bash
python -m scripts.make_llamafactory_refresh_config build \
  --train /tmp/aurora-sft-refresh-recipe2/refresh_train.jsonl \
  --eval /tmp/aurora-sft-refresh-recipe2/refresh_eval.jsonl \
  --model /mlx_devbox/users/jieyu.li/models/Qwen3-VL-8B-Instruct \
  --adapter /tmp/aurora-sft-5k/lora-final-12597-best-eval \
  --output-dir /tmp/aurora-sft-refresh-recipe2/lora-refresh \
  --config-out /tmp/aurora-sft-refresh-recipe2/train_refresh.yaml \
  --policy-out /tmp/aurora-sft-refresh-recipe2/refresh_selection_policy.json

HF_HOME=/tmp/llamafactory-hf-cache \
/tmp/llamafactory-venv/bin/llamafactory-cli train \
  /tmp/aurora-sft-refresh-recipe2/train_refresh.yaml

python -m scripts.make_llamafactory_refresh_config select \
  --output-dir /tmp/aurora-sft-refresh-recipe2/lora-refresh \
  --policy /tmp/aurora-sft-refresh-recipe2/refresh_selection_policy.json \
  --selection-out /tmp/aurora-sft-refresh-recipe2/refresh_selection.json
```

The continuation still evaluates and saves at steps 32, 64, 96, and 128. The
selector requires all four checkpoints, chooses the lowest `refresh_eval`
loss, breaks a tie toward the earliest step, and rejects any policy that
allows an external gate metric. Materialize `lora-refresh-selected/` from that
checkpoint and verify that its adapter config and weights hash exactly match
the selected checkpoint. This portable snapshot is the pre-registered
`recipe2` primary candidate; no Day-14 output may influence this choice.

## 10. Seal the fresh-384 validation bundle and decision policy

Model selection uses 384 newly streamed, distinct source videos rather than
the reused Day-14 suite. The locked allocation is 128 no-search negatives (64
generic style, 32 generic background, and 32 ordinary targets), 64 true-search
positives, 64 routing controls, 64 mask controls, and 64 rewrite-retention
cases. Each rewrite case carries four literal constraints.

The builder rejects source overlap by sample ID, basename, and encoded-video
SHA-256; audits earlier v2 and refresh1 corpora; and uses the Day-14 case file
only as an exclusion input. Run it once after the recipe2 adapter hash is
known, before any candidate inference:

```bash
V2_ADAPTER=/tmp/aurora-sft-5k/lora-final-12597-best-eval
REFRESH1_ADAPTER=/tmp/aurora-sft-refresh/lora-refresh-selected
RECIPE2_ADAPTER=/tmp/aurora-sft-refresh-recipe2/lora-refresh-selected
V2_SHA=$(sha256sum "$V2_ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)
REFRESH1_SHA=$(sha256sum "$REFRESH1_ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)
RECIPE2_SHA=$(sha256sum "$RECIPE2_ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)

python -m scripts.build_interpolation_validation \
  --source-manifest /tmp/aurora-interp-sources/manifest.jsonl \
  --source-root /tmp/aurora-interp-sources \
  --v2-train /tmp/aurora-sft-5k/sft_final_12597_v2_train.jsonl \
  --v2-eval /tmp/aurora-sft-5k/sft_final_12597_v2_eval.jsonl \
  --refresh1-train /tmp/aurora-sft-refresh/refresh_train.jsonl \
  --refresh1-eval /tmp/aurora-sft-refresh/refresh_eval.jsonl \
  --day14-cases data/week1/planner_100.jsonl \
  --cases-out /tmp/aurora-interpolation-validation-v4/cases.jsonl \
  --gold-out /tmp/aurora-interpolation-validation-v4/gold.jsonl \
  --audit-out /tmp/aurora-interpolation-validation-v4/audit.json \
  --policy-out /tmp/aurora-interpolation-validation-v4/policy.json \
  --v2-adapter-sha256 "$V2_SHA" \
  --refresh1-adapter-sha256 "$REFRESH1_SHA" \
  --primary-adapter-sha256 "$RECIPE2_SHA"
```

The output policy hashes the cases, gold, leakage audit, all construction
inputs, both interpolation endpoints, and the recipe2 primary. It also fixes
the nine-point grid `0, 0.125, ..., 1`, bootstrap seed 20260916, 10,000 paired
stratified draws, and the following scoped eligibility thresholds:

| Check | Required value |
|---|---:|
| Complete predictions | 384 |
| Strict raw JSON validity, all cases | 100% |
| Routing accuracy, routing controls | at least 95% |
| No-search specificity, negatives | at least 95% |
| Search-trigger recall, positives | at least 95% |
| Search-query end-to-end recall, positives | at least 95% |
| Mask-trigger F1, mask controls | at least 95% |
| Rewrite constraint retention | at least 85% |

Do not regenerate the validation bundle or policy after inspecting any
candidate output. A failed build must be retried in a fresh output directory,
and candidate inference must wait until all hashes and the 384-video audit are
accepted.

Because recipe2 is the registered primary but was produced after the v2 and
refresh1 corpora used by the builder, preserve the sealed v4 policy and add a
separate post-seal cross-audit before reading candidate metrics:

```bash
python -m scripts.audit_primary_validation_isolation \
  --cases /tmp/aurora-interpolation-validation-v4/cases.jsonl \
  --gold /tmp/aurora-interpolation-validation-v4/gold.jsonl \
  --leakage-audit /tmp/aurora-interpolation-validation-v4/audit.json \
  --policy /tmp/aurora-interpolation-validation-v4/policy.json \
  --recipe2-train /tmp/aurora-sft-refresh-recipe2/refresh_train.jsonl \
  --recipe2-eval /tmp/aurora-sft-refresh-recipe2/refresh_eval.jsonl \
  --v2-train /tmp/aurora-sft-5k/sft_final_12597_v2_train.jsonl \
  --v2-eval /tmp/aurora-sft-5k/sft_final_12597_v2_eval.jsonl \
  --out /tmp/aurora-interpolation-validation-v4/recipe2_cross_audit.json
```

This artifact is explicitly `supplemental_post_seal`: it hashes the immutable
v4 cases, gold, and policy plus the recipe2 train/eval files, but it does not
rewrite the policy or change any threshold. All blocking source, exact-prompt,
concept, strict-template, and constraint overlaps must be zero. Generic
routing/mask skeletons and broad paired 4-grams remain diagnostic-only and
are recorded with bounded examples.

## 11. Build, run, and select the registered candidates

The fallback grid interpolates parameter deltas between the original v2
adapter (`lambda=0`) and the preserved refresh1 adapter (`lambda=1`). Use
exact delta-space rank concatenation; averaging LoRA A/B factors is not
equivalent. The interpolation utility creates rank-64 / alpha-128 adapters while preserving
the parents' scaling:

```bash
GRID_ROOT=/tmp/aurora-interpolation-grid-v1
for PAIR in 0000:0 0125:0.125 0250:0.25 0375:0.375 0500:0.5 0625:0.625 0750:0.75 0875:0.875 1000:1; do
  SLUG=${PAIR%%:*}
  LAMBDA=${PAIR#*:}
  python -m scripts.interpolate_lora_adapters \
    --adapter-a /tmp/aurora-sft-5k/lora-final-12597-best-eval \
    --adapter-b /tmp/aurora-sft-refresh/lora-refresh-selected \
    --lambda-b "$LAMBDA" \
    --out-dir "$GRID_ROOT/lambda_$SLUG/adapter"
done

mkdir -p "$GRID_ROOT/recipe2"
cp -a /tmp/aurora-sft-refresh-recipe2/lora-refresh-selected \
  "$GRID_ROOT/recipe2/adapter"
```

Evaluate every registered adapter with the same fail-fast generic runner. It
requires exact local model paths, exactly 384 predictions, no per-case errors,
unchanged inputs, and a fresh result directory:

```bash
for CANDIDATE in lambda_0000 lambda_0125 lambda_0250 lambda_0375 lambda_0500 lambda_0625 lambda_0750 lambda_0875 lambda_1000 recipe2; do
  python -m scripts.run_planner_eval \
    --base /mlx_devbox/users/jieyu.li/models/Qwen3-VL-8B-Instruct \
    --adapter "$GRID_ROOT/$CANDIDATE/adapter" \
    --cases /tmp/aurora-interpolation-validation-v4/cases.jsonl \
    --gold /tmp/aurora-interpolation-validation-v4/gold.jsonl \
    --out-dir "$GRID_ROOT/$CANDIDATE/eval" \
    --expected-cases 384 \
    --device cuda:0
done
```

Then recompute all scoped metrics from raw planner records and apply the locked
decision with the auditable selector:

```bash
python -m scripts.select_lora_interpolation \
  --cases /tmp/aurora-interpolation-validation-v4/cases.jsonl \
  --gold /tmp/aurora-interpolation-validation-v4/gold.jsonl \
  --leakage-audit /tmp/aurora-interpolation-validation-v4/audit.json \
  --policy /tmp/aurora-interpolation-validation-v4/policy.json \
  --candidates-root /tmp/aurora-interpolation-grid-v1 \
  --primary-reference-adapter /tmp/aurora-sft-refresh-recipe2/lora-refresh-selected \
  --comparison-out /tmp/aurora-interpolation-validation-v4/comparison.json \
  --selection-out /tmp/aurora-interpolation-validation-v4/selection.json
```

`--primary-reference-adapter` is the loss-selected recipe2 training artifact,
not a validation-derived choice. The selector requires its config and weights
to be byte-identical to the copied `recipe2/adapter`, records both hashes, and
rejects a missing or drifted reference without changing the sealed policy.

The final rule is primary-first: select recipe2 if it meets every eligibility
threshold. Otherwise restrict the grid to eligible candidates, define utility
as `0.5 * no-search specificity + 0.5 * rewrite retention`, find the
point-estimate winner, and select the smallest lambda within one bootstrap
standard error of it. The selector still reports the complete grid when
recipe2 is eligible. If neither recipe2 nor any grid candidate is eligible,
there is no selected final adapter; do not relax the thresholds after seeing
the outputs.

## 12. Reuse Day-14 only as an adaptive post-selection regression

After `selection.json` is immutable, run the unchanged Day-14 wrapper at most
once on the selected adapter:

```bash
SELECTED_ADAPTER=$(python -c 'import json; print(json.load(open("/tmp/aurora-interpolation-validation-v4/selection.json"))["selected_adapter_dir"])')

HF_HOME=/tmp/aurora-hf-cache \
.venv/bin/python scripts/run_day14_gate.py \
  --adapter "$SELECTED_ADAPTER" \
  --out-dir /tmp/aurora-day14-selected-adaptive
```

This run is an adaptive regression because the suite has already influenced
diagnosis and exclusion rules. It is not a held-out or confirmatory result and
must not change the selected adapter. Any confirmatory claim requires a
separate sealed evaluation set.

Keep the following trees separate and immutable when staging through the
Devbox and pulling them locally:

```text
/tmp/aurora-sft-refresh/                    # failed refresh1 finding
/tmp/aurora-sft-refresh-recipe2/            # corrected data, training, selection
/tmp/aurora-interpolation-validation-v4/    # sealed cases, audit, policy, decision
/tmp/aurora-interpolation-grid-v1/          # ten adapters and fresh-384 runs
/tmp/aurora-day14-selected-adaptive/        # one post-selection regression
```

Retain logs, raw predictions, all selection JSON, interpolation provenance,
adapter files, and SHA-256 manifests. Verify the same relative-path hashes
after each transfer hop. The existing `runs/week2/final_12597/` bundle remains
the historical v2 baseline and must not be overwritten by these correction
artifacts.

## 13. Assemble and verify the corrected portable bundles

Do this on the live Worker before releasing it. The assembler claims a new
destination atomically, never changes a source artifact, excludes evaluation
keyframes, emits the adapter identity/config proofs, copies the exclusion-
locked Day-14 suite as `adaptive_day14/gold.jsonl`, and writes
`checksums.sha256` last. A failed assembly removes only the new destination it
claimed.

The runtime-to-bundle mapping is fixed:

| Runtime source | Portable destination |
|---|---|
| recipe2 `lora-refresh/` and `lora-refresh-selected/` | `recipe2/` |
| recipe2 train/eval/audit/YAML/policy/selection | `recipe2/metadata/` |
| fresh `cases.jsonl`, `gold.jsonl` | `fresh384/` |
| fresh `audit.json`, `policy.json` | `fresh384/leakage_audit.json`, `fresh384/selection_policy.json` |
| fresh `recipe2_cross_audit.json` | `fresh384/recipe2_cross_audit.json` |
| fresh `comparison.json`, `selection.json` | `selection/` |
| v2 / refresh1 / recipe2 adapter configs and live hashes | `selection/endpoint_configs/`, `selection/adapter_identities.json` |
| ten grid-root candidate directories | `candidates/` |
| adaptive Day-14 five outputs | `adaptive_day14/` |
| locked `data/week1/planner_100.jsonl` | `adaptive_day14/gold.jsonl` |

Build both profiles from the same immutable sources. `corrected-full` keeps
all recipe2 checkpoint state and all ten candidate weights. On the Worker it
hardlinks large files when the filesystem permits, so do not modify a source
file after assembly. `corrected-thin` is the durable transfer profile: it
keeps all raw predictions, logs, manifests, configs, provenance, trainer
states, and the final selected candidate weight, while omitting reconstructible
non-selected weights.

```bash
cd /mlx_devbox/users/jieyu.li/Aurora

RECIPE2_ROOT=/tmp/aurora-sft-refresh-recipe2
FRESH_ROOT=/tmp/aurora-interpolation-validation-v4
CANDIDATES_ROOT=/tmp/aurora-interpolation-grid-v1
ADAPTIVE_ROOT=/tmp/aurora-day14-selected-adaptive
DAY14_GOLD=/mlx_devbox/users/jieyu.li/Aurora/data/week1/planner_100.jsonl
V2_ADAPTER=/tmp/aurora-sft-5k/lora-final-12597-best-eval
REFRESH1_ADAPTER=/tmp/aurora-sft-refresh/lora-refresh-selected

python -m scripts.assemble_week2_corrected_bundle \
  --profile corrected-full \
  --recipe2-root "$RECIPE2_ROOT" \
  --fresh-root "$FRESH_ROOT" \
  --candidates-root "$CANDIDATES_ROOT" \
  --adaptive-day14-root "$ADAPTIVE_ROOT" \
  --day14-gold "$DAY14_GOLD" \
  --v2-adapter "$V2_ADAPTER" \
  --refresh1-adapter "$REFRESH1_ADAPTER" \
  --out-root /tmp/aurora-week2-corrected-full

python -m scripts.assemble_week2_corrected_bundle \
  --profile corrected-thin \
  --recipe2-root "$RECIPE2_ROOT" \
  --fresh-root "$FRESH_ROOT" \
  --candidates-root "$CANDIDATES_ROOT" \
  --adaptive-day14-root "$ADAPTIVE_ROOT" \
  --day14-gold "$DAY14_GOLD" \
  --v2-adapter "$V2_ADAPTER" \
  --refresh1-adapter "$REFRESH1_ADAPTER" \
  --out-root /tmp/aurora-week2-corrected-thin
```

The assembler runs the semantic audit once before keeping either destination.
Run the public verifier explicitly as the handoff record and again after every
transfer hop:

```bash
python -m scripts.verify_week2_bundle \
  --root /tmp/aurora-week2-corrected-full \
  --profile corrected-full \
  --verify-manifest /tmp/aurora-week2-corrected-full/checksums.sha256 \
  --audit

python -m scripts.verify_week2_bundle \
  --root /tmp/aurora-week2-corrected-thin \
  --profile corrected-thin \
  --verify-manifest /tmp/aurora-week2-corrected-thin/checksums.sha256 \
  --audit
```

Do not regenerate a manifest after transfer. A mismatch is evidence of an
incomplete or changed transfer; retry into a fresh destination instead. Keep
the full Worker bundle until the independently transferred thin bundle passes
both checksum verification and the `corrected-thin` semantic audit locally.

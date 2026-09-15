# Aurora-OutcomeDPO local preparation

This repository tracks the upstream Aurora code plus the local assets and
measurement tools needed for the Day 1–3 feasibility check.

## 1. Prepare smoke-test videos

The source images come from Wikimedia Commons. The generated videos and source
images are intentionally ignored by Git; attribution and license metadata are
written to `data/smoke/assets_manifest.json`.

```bash
uv run --no-project --with imageio-ffmpeg python scripts/prepare_smoke_assets.py
python3 scripts/validate_smoke_cases.py
```

## 2. Configure Serper without committing the key

```bash
cp .env.example .env.local
```

Fill `SERPER_KEY_ID` in `.env.local`. Initial offline smoke tests can instead
pass `--disable_image_search`.

## 3. Remote planner smoke test

Run from the repository root so the relative video paths resolve correctly.

```bash
python scripts/profile_command.py --name planner_offline -- \
  python -m aurora.agent \
  --custom_cases_jsonl data/smoke/cases.jsonl \
  --custom_only \
  --max_cases 1 \
  --mask_backend none \
  --disable_image_search \
  --out_dir runs/smoke/agent_offline
```

## 4. Remote editor smoke test

After the planner succeeds, render its record with fixed guidance settings.

```bash
python scripts/profile_command.py --name editor_case_001 -- \
  python -m aurora.editor_bridge_video \
  --records_jsonl runs/smoke/agent_offline/agent_pipeline_records.jsonl \
  --ckpt models/aurora_editor.safetensors \
  --out_dir runs/smoke/editor_case_001 \
  --num_frames 81 \
  --save_frames 64 \
  --cfg_scale 2.0 \
  --image_cfg_scale 1.0 \
  --fallback_to_two_pass_cfg
```

Every profiled command appends one record to `runs/smoke/metrics.jsonl` and
writes the full output to a separate log file. Record the random seed in the
run name or command as soon as the selected Aurora entry point exposes it.

## 5. Git remotes

`upstream` points to `https://github.com/yeates/Aurora.git`. Add a private fork
or project repository as `origin` before pushing:

```bash
git remote add origin <your-private-repository-url>
git push -u origin main
```

Do not push `.env.local`, generated videos, model weights, or run outputs.

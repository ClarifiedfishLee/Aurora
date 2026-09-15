# Mini-AgentEdit v0.1

Mini-AgentEdit is the held-out evaluation set for planner and end-to-end Aurora
experiments. The source videos must not appear in SFT data, preference mining,
or prompt-development examples.

## Pilot and frozen splits

1. Build a 30-50 case `pilot` split to validate annotation and judge rubrics.
2. Revise the schema or rubric only during the pilot.
3. Freeze 150-200 cases across `dev` and `test` before preference mining.
4. Group by source video and entity family. A `holdout_group` may occur in only
   one split.
5. Keep roughly balanced coverage of `search`, `mask`, `routing`, and `rewrite`.
   Use `mixed` only when a case cannot be assigned to one primary axis.

The machine-readable schema is `evaluation/mini_agentedit.schema.json`.

## Required JSONL fields

- `bench_id`: stable `mae_*` identifier.
- `video_path`: repository-relative or mounted-media path.
- `prompt`: raw user request presented to the planner.
- `edit_type`: dataset-side expected edit category.
- `axis`: primary decision under test.
- `split`: `pilot`, `dev`, or `test`.
- `source`: dataset, original source ID, and license metadata.
- `gold_plan`: the four-field Aurora planning contract.
- `constraints`: atomic color, count, spatial, identity, or preservation facts
  that must survive instruction refinement.

Example:

```json
{"bench_id":"mae_pilot_0001","video_path":"/mounted/holdout/clip.mp4","prompt":"make the cup red, keep the two plates","edit_type":"change_color","axis":"rewrite","split":"pilot","holdout_group":"tabletop_cups","source":{"dataset":"Pexels","source_id":"replace-with-source-id","license":"Pexels License"},"gold_plan":{"refined_text_instruction":"Change the cup to red while preserving both plates and the rest of the scene.","subtask":"change_color","image_search":false,"mask":"cup"},"constraints":[{"type":"color","value":"red"},{"type":"count","value":"two","aliases":["2","both"]},{"type":"preservation","value":"plates"}],"source_entities":["cup","plates"]}
```

Do not commit media unless its license explicitly permits redistribution. The
manifest may be committed while `video_path` points to a private mounted copy.
Aurora training shards are for SFT/data-pipeline bootstrap only and must not be
used as Mini-AgentEdit source videos.

## Agent-only scoring

Aurora planner records can be scored directly:

```bash
python -m evaluation.agent_only_score \
  --gold data/mini_agentedit/pilot.jsonl \
  --predictions runs/mini_agentedit/agent_pipeline_records.jsonl \
  --out runs/mini_agentedit/agent_metrics.json
```

The first version reports JSON validity, subtask accuracy, search-trigger F1,
mask-trigger F1, constraint retention, and error IDs. Search-query quality,
mask granularity, and reference selection require separate axis-specific
rubrics and are intentionally not collapsed into the trigger metrics.

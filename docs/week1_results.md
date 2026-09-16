# Week 1 planner baseline results

Status: initial 100-case development regression completed on one A100-SXM4-80GB.
These are engineering baseline results, not final Mini-AgentEdit benchmark
numbers, because the suite deliberately reuses the ten smoke-test videos.

| Model | Runtime JSON | Strict raw JSON | Routing accuracy | Search F1 | Mask F1 | Source-entity false search | Lexical constraint retention | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-VL-8B | 100% | 100% | 76% | 39.2% | 90.9% | 34.4% | 64% | 271.5 s |
| Aurora released LoRA | 100% | 100% | 78% | 88.9% | 100% | 0% | 48% | 266.3 s |

Runtime validity is measured after Aurora normalizes a plan; strict raw
validity parses the complete model response and enforces the exact four-field
contract. A later audit also checked the value of every triggered positive
search query against pre-registered entity aliases. Both models were 100%
correct conditional on triggering; end-to-end valid-query recall was 100% for
Base and 80% for the released LoRA because the latter missed two searches.

The strongest released-LoRA gain is tool discipline. The base model searched
on 31 of 90 cases whose gold plan was self-contained, while the LoRA made zero
such false search triggers. The LoRA missed two of ten required searches. It
also followed the removal-only mask contract exactly; the base model emitted
two extra masks.

Routing improves only two points overall. Ten weather cases are counted as
errors for both models because the released `type1_system.txt` prompt omits
`change_weather` from its allowed subtask list, even though the parser and the
documented planning contract accept it. Excluding those ten known prompt-contract
cases, routing is 84.4% for Base and 86.7% for the released LoRA. The released
baseline result remains unmodified for reproducibility; a contract-fixed prompt
should be reported only as a separate diagnostic.

The deterministic constraint score is lexical, not semantic: it checks whether
gold values or aliases occur as normalized substrings. Manual inspection shows
several LoRA rewrites preserve a constraint through paraphrase—for example,
"sparse ... to avoid obscuring" instead of repeating "clearly visible"—and are
therefore scored as misses. The 64% vs 48% result must not be presented as proof
that Base is semantically better. Week 1 judge work must replace this proxy with
an atomic factual rubric and report lexical and judge-based retention separately.

Artifacts:

- Gold suite: `data/week1/planner_100.jsonl`
- Base records and metrics: `runs/week1/base/`
- Released-LoRA records and metrics: `runs/week1/aurora_lora/`
- Machine-readable comparison: `runs/week1/comparison.json`

Next gate: construct 30-50 single-axis A/B outcome pairs, deploy the official
UniEditBench 4B video evaluator, obtain human labels without seeing judge
outputs, and measure overall plus per-axis agreement before bulk rendering.

## Judge deployment smoke

The official Qwen3-VL-4B image+video evaluator was downloaded and served on the
A100. A complete request using the Stanley smoke edit returned valid five-axis
JSON, but assigned 5/5 to every dimension. This single result is a warning, not
an agreement measurement: it may indicate an overly generous generic rubric on
the fine-grained identity axis. Human-blind labels on the 30-50-pair pilot are
required before accepting its scores as preference labels.

Deployment required three compatibility fixes documented in
`docs/week1_protocol.md`: adding the missing `decord` dependency, mapping the
metadata field names expected by the prompt template, and merging the full LoRA
before vLLM serving so visual-tower adapter weights are not ignored.

## Judge agreement gate

The 40-pair blind pilot yielded 29 usable human preferences and 11 invalid
renders (27.5%). Independent UniEditBench five-dimension scoring reached 76.9%
directional agreement overall, but the aggregate hides severe axis imbalance:
search reached 100%, mask 66.7%, and rewrite 50.0%. Structural fidelity also
penalizes some intended replacements and removals, so it is not a universal
outcome reward.

An axis-aware pairwise protocol was then tested with interleaved media labels
(`SOURCE`, `CANDIDATE A`, `CANDIDATE B`). Qwen3-VL-8B improved rewrite to 83.3%
directional agreement, but reached only 66.7% on search and 50.0% on mask. The
overall directional agreement was 68.8%. Consecutive unlabeled video inputs
were rejected because both 4B and 8B models confused candidate order.

The Week-1 decision is therefore a partial gate:

- use independent UniEditBench scoring for search pairs;
- use labeled Qwen3-VL-8B pairwise scoring for rewrite pairs;
- require deterministic mask checks plus human audit for mask pairs;
- keep routing as a negative control because the current editor bridge records
  but does not consume `plan.subtask`;
- do not begin the 300-500-pair outcome render batch until invalid renders are
  reduced and the mask rubric is validated.

Week 2 planner SFT can proceed independently of this render gate.

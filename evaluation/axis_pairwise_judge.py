"""Run and score an axis-aware pairwise video judge."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.judge_agreement import LABELS, load_jsonl, score


PAIRWISE_LABELS = (*LABELS, "invalid")
AXIS_RUBRICS = {
    "search": (
        "Prioritize whether the requested external identity, branded object, character, or landmark is "
        "visibly correct. Then consider preservation and temporal quality. Do not infer whether a web search ran."
    ),
    "mask": (
        "Prioritize complete removal of exactly the requested source entity, clean localized inpainting, and "
        "preservation of everything outside that entity."
    ),
    "rewrite": (
        "Check every atomic instruction constraint, especially identity, color, count, size, spatial relation, "
        "and explicit preservation. Prefer the candidate satisfying more constraints without collateral changes."
    ),
    "routing": (
        "Compare only the visible outcomes. This is a negative-control axis, so choose tie when the candidates "
        "are effectively indistinguishable."
    ),
}


def parse_response(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    payload = json.loads(match.group(0) if match else raw)
    label = payload.get("label")
    if label not in PAIRWISE_LABELS:
        raise ValueError(f"invalid pairwise label: {label!r}")
    return payload


def build_prompt(pair: dict[str, Any], case: dict[str, Any]) -> str:
    axis = str(pair["axis"])
    constraints = case.get("constraints", [])
    constraint_text = json.dumps(constraints, ensure_ascii=False)
    return f"""You are a strict video-editing outcome evaluator.

You will receive three videos in this exact order: SOURCE, CANDIDATE A, CANDIDATE B.
The requested edit is: {pair['instruction']}
The single decision axis under study is: {axis}
Atomic constraints: {constraint_text}

Axis-specific rubric: {AXIS_RUBRICS[axis]}

Judge only visible evidence. Do not guess hidden plans, tools, or which candidate is expected to be better.
Use exactly these labels:
- A: candidate A is materially better.
- B: candidate B is materially better.
- tie: neither is materially better, including when both fail similarly or neither performs the requested edit.

This is a relative comparison. If either candidate has partial visible success, choose A or B when it is materially
closer to the request even if it remains imperfect.
An edited object that visibly has the requested identity and a close color/material counts as partial core success.

Decision procedure:
1. Score core_edit as 0=no visible attempt, 1=coarse category only, 2=requested identity or main attribute,
   3=identity plus most modifiers/constraints, 4=all core constraints visibly satisfied.
2. Prefer the higher core_edit score. Override it only for clearly catastrophic preservation or temporal damage.
3. Do not choose tie when a named target identity is visibly present in only one candidate.
4. Use tie only when the visible evidence is genuinely indistinguishable or both candidates fail in the same way.

Return exactly one JSON object:
{{"label":"A|B|tie","core_edit":{{"A":0,"B":0}},"preservation":{{"A":0,"B":0}},
"temporal_quality":{{"A":0,"B":0}},"constraint_checks":[],"reason":"concise visible evidence"}}
Scores are integers from 0 to 4. The final label must reflect the axis-specific rubric, not a blind average."""


def build_items(key_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = {str(row["bench_id"]): row for row in case_rows}
    items = []
    for pair in key_rows:
        case_id = str(pair["source_case_id"])
        if case_id not in cases:
            raise ValueError(f"missing planner case: {case_id}")
        items.append(
            {
                "pair_id": pair["pair_id"],
                "axis": pair["axis"],
                "source_case_id": case_id,
                "source_video": pair["source_video"],
                "video_a": pair["video_a"],
                "video_b": pair["video_b"],
                "prompt": build_prompt(pair, cases[case_id]),
            }
        )
    return items


def infer(items: list[dict[str, Any]], save_path: Path, port: int) -> list[dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{port}/v1")
    models = client.models.list().data
    if not models:
        raise RuntimeError("judge server returned no models")
    model = models[0].id
    completed: list[dict[str, Any]] = []
    if save_path.exists():
        completed = json.loads(save_path.read_text(encoding="utf-8"))
    done = {str(row["pair_id"]) for row in completed}
    for index, item in enumerate(items, 1):
        if item["pair_id"] in done:
            continue
        paths = [str(Path(item[name]).resolve()) for name in ("source_video", "video_a", "video_b")]
        for media_path in paths:
            if not Path(media_path).is_file():
                raise FileNotFoundError(media_path)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "SOURCE VIDEO:"},
                        {"type": "video", "video": paths[0]},
                        {"type": "text", "text": "CANDIDATE A VIDEO:"},
                        {"type": "video", "video": paths[1]},
                        {"type": "text", "text": "CANDIDATE B VIDEO:"},
                        {"type": "video", "video": paths[2]},
                        {"type": "text", "text": "EVALUATION RUBRIC AND REQUIRED OUTPUT:"},
                        {"type": "text", "text": item["prompt"]},
                    ],
                }
            ],
            max_tokens=1536,
            temperature=0,
        )
        raw = response.choices[0].message.content
        parsed = parse_response(raw)
        completed.append({**item, "judge_model": model, "response": raw, "parsed": parsed})
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(completed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{index}/{len(items)}] {item['pair_id']}: {parsed['label']}", flush=True)
    return completed


def score_results(results: list[dict[str, Any]], human_rows: list[dict[str, Any]]) -> dict[str, Any]:
    human = {str(row["bench_id"]): row for row in human_rows}
    rows = []
    four_way = []
    for result in results:
        pair_id = str(result["pair_id"])
        if pair_id not in human:
            raise ValueError(f"missing human annotation: {pair_id}")
        human_label = human[pair_id]["human_label"]
        judge_label = result.get("parsed", {}).get("label") or parse_response(result["response"])["label"]
        four_way.append((human_label, judge_label))
        rows.append(
            {
                "bench_id": pair_id,
                "axis": result["axis"],
                "human_label": human_label,
                "judge_label": judge_label,
            }
        )
    invalid_tp = sum(human_label == judge_label == "invalid" for human_label, judge_label in four_way)
    invalid_pred = sum(judge_label == "invalid" for _, judge_label in four_way)
    invalid_gold = sum(human_label == "invalid" for human_label, _ in four_way)
    precision = invalid_tp / invalid_pred if invalid_pred else None
    recall = invalid_tp / invalid_gold if invalid_gold else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall else None
    return {
        "num_pairs": len(rows),
        "label_counts": {
            "human": dict(Counter(left for left, _ in four_way)),
            "judge": dict(Counter(right for _, right in four_way)),
        },
        "four_way_exact_agreement": sum(left == right for left, right in four_way) / len(four_way) if four_way else None,
        "invalid_detection": {"precision": precision, "recall": recall, "f1": f1},
        "valid_preference_agreement": score(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--key", type=Path, required=True)
    run_parser.add_argument("--cases", type=Path, required=True)
    run_parser.add_argument("--save", type=Path, required=True)
    run_parser.add_argument("--port", type=int, default=8005)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--pair-id", action="append", dest="pair_ids")
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--results", type=Path, required=True)
    score_parser.add_argument("--human", type=Path, required=True)
    score_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "run":
        items = build_items(json.loads(args.key.read_text(encoding="utf-8")), load_jsonl(args.cases))
        if args.pair_ids:
            requested = set(args.pair_ids)
            items = [item for item in items if item["pair_id"] in requested]
            found = {item["pair_id"] for item in items}
            if found != requested:
                raise ValueError(f"unknown pair ids: {sorted(requested - found)}")
        infer(items[: args.limit], args.save, args.port)
        return
    summary = score_results(json.loads(args.results.read_text(encoding="utf-8")), load_jsonl(args.human))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

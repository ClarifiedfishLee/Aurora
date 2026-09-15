"""Deterministic agent-only metrics for Mini-AgentEdit JSONL predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


PLAN_FIELDS = {"refined_text_instruction", "subtask", "image_search", "mask"}
SUBTASKS = {
    "global_style",
    "remove_object",
    "add_object",
    "replace_object",
    "change_background",
    "change_color",
    "change_weather",
    "add_effect",
    "customization",
    "combined_tasks",
    "camera_edit",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def extract_plan(row: dict[str, Any]) -> dict[str, Any] | None:
    plan = row.get("plan")
    if isinstance(plan, dict):
        return plan
    raw = row.get("agent_raw")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def valid_plan(plan: dict[str, Any] | None) -> bool:
    if not isinstance(plan, dict) or set(plan) != PLAN_FIELDS:
        return False
    if not isinstance(plan["refined_text_instruction"], str) or not plan["refined_text_instruction"].strip():
        return False
    if plan["subtask"] not in SUBTASKS:
        return False
    return all(value is False or (isinstance(value, str) and bool(value.strip())) for value in (plan["image_search"], plan["mask"]))


def triggered(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def binary_counts(gold: Iterable[bool], predicted: Iterable[bool]) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for expected, actual in zip(gold, predicted):
        if expected and actual:
            tp += 1
        elif not expected and actual:
            fp += 1
        elif expected and not actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE))


def constraint_is_retained(constraint: dict[str, Any], instruction: str) -> bool:
    normalized_instruction = normalize(instruction)
    variants = [constraint["value"], *constraint.get("aliases", [])]
    return any(normalize(str(value)) in normalized_instruction for value in variants)


def score(gold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = {str(row.get("bench_id", "")): row for row in prediction_rows}
    validity: list[bool] = []
    routing: list[bool] = []
    expected_search: list[bool] = []
    predicted_search: list[bool] = []
    expected_mask: list[bool] = []
    predicted_mask: list[bool] = []
    retained_constraints = total_constraints = exact_constraint_cases = constraint_cases = 0
    missing_predictions: list[str] = []
    invalid_predictions: list[str] = []
    routing_errors: list[dict[str, str]] = []

    for gold in gold_rows:
        bench_id = str(gold["bench_id"])
        prediction = predictions.get(bench_id)
        if prediction is None:
            missing_predictions.append(bench_id)
            plan = None
        else:
            plan = extract_plan(prediction)
        is_valid = valid_plan(plan)
        validity.append(is_valid)
        if not is_valid:
            invalid_predictions.append(bench_id)
        gold_plan = gold["gold_plan"]
        actual_subtask = plan.get("subtask") if is_valid else None
        route_match = actual_subtask == gold_plan["subtask"]
        routing.append(route_match)
        if not route_match:
            routing_errors.append({"bench_id": bench_id, "expected": gold_plan["subtask"], "actual": str(actual_subtask)})
        expected_search.append(triggered(gold_plan["image_search"]))
        predicted_search.append(triggered(plan["image_search"]) if is_valid else False)
        expected_mask.append(triggered(gold_plan["mask"]))
        predicted_mask.append(triggered(plan["mask"]) if is_valid else False)

        constraints = gold.get("constraints", [])
        instruction = str(plan.get("refined_text_instruction", "")) if is_valid else ""
        retained = sum(constraint_is_retained(item, instruction) for item in constraints)
        retained_constraints += retained
        total_constraints += len(constraints)
        if constraints:
            constraint_cases += 1
            if retained == len(constraints):
                exact_constraint_cases += 1

    total = len(gold_rows)
    return {
        "num_cases": total,
        "json_validity": sum(validity) / total if total else 0.0,
        "subtask_accuracy": sum(routing) / total if total else 0.0,
        "image_search_trigger": binary_counts(expected_search, predicted_search),
        "mask_trigger": binary_counts(expected_mask, predicted_mask),
        "constraint_retention": retained_constraints / total_constraints if total_constraints else 1.0,
        "constraint_case_accuracy": exact_constraint_cases / constraint_cases if constraint_cases else 1.0,
        "details": {
            "missing_predictions": missing_predictions,
            "invalid_predictions": invalid_predictions,
            "routing_errors": routing_errors,
            "retained_constraints": retained_constraints,
            "total_constraints": total_constraints,
            "constraint_cases": constraint_cases,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    metrics = score(load_jsonl(args.gold), load_jsonl(args.predictions))
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

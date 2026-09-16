"""Deterministic agent-only metrics for Mini-AgentEdit JSONL predictions."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


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


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def extract_strict_raw_plan(row: dict[str, Any]) -> dict[str, Any] | None:
    """Parse the complete raw model response without runtime normalization."""
    raw = row.get("agent_raw")
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def valid_plan(plan: dict[str, Any] | None) -> bool:
    if not isinstance(plan, dict) or set(plan) != PLAN_FIELDS:
        return False
    if not isinstance(plan["refined_text_instruction"], str) or not plan["refined_text_instruction"].strip():
        return False
    if not isinstance(plan["subtask"], str) or plan["subtask"] not in SUBTASKS:
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


def search_query_matches_alias(query: str, aliases: Iterable[Any]) -> bool:
    """Return whether a complete, explicitly allowed alias occurs in a query."""
    normalized_query = f" {normalize(query)} "
    for alias in aliases:
        normalized_alias = normalize(str(alias))
        if normalized_alias and f" {normalized_alias} " in normalized_query:
            return True
    return False


def _index_by_bench_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        bench_id = str(row.get("bench_id", ""))
        if bench_id in indexed:
            raise ValueError(f"duplicate {label} bench_id: {bench_id!r}")
        indexed[bench_id] = row
    return indexed


def _score_rows(
    gold_rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    *,
    extra_prediction_ids: list[str] | None = None,
) -> dict[str, Any]:
    validity: list[bool] = []
    strict_raw_validity: list[bool] = []
    routing: list[bool] = []
    expected_search: list[bool] = []
    predicted_search: list[bool] = []
    expected_mask: list[bool] = []
    predicted_mask: list[bool] = []
    retained_constraints = total_constraints = exact_constraint_cases = constraint_cases = 0
    missing_predictions: list[str] = []
    invalid_predictions: list[str] = []
    strict_raw_invalid_predictions: list[str] = []
    routing_errors: list[dict[str, str]] = []
    source_entity_search_cases = source_entity_false_triggers = 0
    source_entity_false_trigger_ids: list[str] = []
    search_query_gold_positive_cases = 0
    search_query_triggered_cases = 0
    search_query_correct_queries = 0
    search_query_wrong_ids: list[str] = []
    search_query_missed_ids: list[str] = []

    for gold in gold_rows:
        bench_id = str(gold["bench_id"])
        prediction = predictions.get(bench_id)
        if prediction is None:
            missing_predictions.append(bench_id)
            plan = None
        else:
            plan = extract_plan(prediction)
        is_valid = valid_plan(plan)
        is_strict_raw_valid = prediction is not None and valid_plan(extract_strict_raw_plan(prediction))
        validity.append(is_valid)
        strict_raw_validity.append(is_strict_raw_valid)
        if not is_valid:
            invalid_predictions.append(bench_id)
        if not is_strict_raw_valid:
            strict_raw_invalid_predictions.append(bench_id)
        gold_plan = gold["gold_plan"]
        actual_subtask = plan.get("subtask") if is_valid else None
        route_match = actual_subtask == gold_plan["subtask"]
        routing.append(route_match)
        if not route_match:
            routing_errors.append({"bench_id": bench_id, "expected": gold_plan["subtask"], "actual": str(actual_subtask)})
        search_expected = triggered(gold_plan["image_search"])
        expected_search.append(search_expected)
        search_was_triggered = triggered(plan["image_search"]) if is_valid else False
        predicted_search.append(search_was_triggered)
        if search_expected:
            search_query_gold_positive_cases += 1
            if not search_was_triggered:
                search_query_missed_ids.append(bench_id)
            else:
                search_query_triggered_cases += 1
                query = str(plan["image_search"])
                if search_query_matches_alias(query, gold.get("search_query_aliases", [])):
                    search_query_correct_queries += 1
                else:
                    search_query_wrong_ids.append(bench_id)
        if not search_expected and gold.get("source_entities"):
            source_entity_search_cases += 1
            source_entity_false_triggers += int(search_was_triggered)
            if search_was_triggered:
                source_entity_false_trigger_ids.append(bench_id)
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
        "json_validity_method": "normalized runtime plan validity (plan preferred; agent_raw fallback)",
        "strict_raw_json_validity": sum(strict_raw_validity) / total if total else 0.0,
        "subtask_accuracy": sum(routing) / total if total else 0.0,
        "image_search_trigger": binary_counts(expected_search, predicted_search),
        "image_search_query": {
            "gold_positive_cases": search_query_gold_positive_cases,
            "triggered_cases": search_query_triggered_cases,
            "correct_queries": search_query_correct_queries,
            "conditional_accuracy": (
                search_query_correct_queries / search_query_triggered_cases
                if search_query_triggered_cases
                else 0.0
            ),
            "end_to_end_recall": (
                search_query_correct_queries / search_query_gold_positive_cases
                if search_query_gold_positive_cases
                else 0.0
            ),
            "wrong_query_ids": search_query_wrong_ids,
            "missed_trigger_ids": search_query_missed_ids,
        },
        "mask_trigger": binary_counts(expected_mask, predicted_mask),
        "constraint_retention": retained_constraints / total_constraints if total_constraints else 1.0,
        "constraint_retention_method": "normalized substring over gold values and aliases",
        "constraint_case_accuracy": exact_constraint_cases / constraint_cases if constraint_cases else 1.0,
        "source_entity_false_trigger": {
            "num_cases": source_entity_search_cases,
            "false_triggers": source_entity_false_triggers,
            "rate": source_entity_false_triggers / source_entity_search_cases if source_entity_search_cases else 0.0,
            "ids": source_entity_false_trigger_ids,
        },
        "details": {
            "missing_predictions": missing_predictions,
            "invalid_predictions": invalid_predictions,
            "strict_raw_invalid_predictions": strict_raw_invalid_predictions,
            **({"extra_prediction_ids": extra_prediction_ids} if extra_prediction_ids is not None else {}),
            "routing_errors": routing_errors,
            "retained_constraints": retained_constraints,
            "total_constraints": total_constraints,
            "constraint_cases": constraint_cases,
        },
    }


def score(gold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_by_id = _index_by_bench_id(gold_rows, "gold")
    predictions = _index_by_bench_id(prediction_rows, "prediction")
    extra_prediction_ids = sorted(set(predictions) - set(gold_by_id))
    result = _score_rows(gold_rows, predictions, extra_prediction_ids=extra_prediction_ids)
    axes = sorted({str(row["axis"]) for row in gold_rows if row.get("axis")})
    result["by_axis"] = {
        axis: _score_rows(
            [row for row in gold_rows if str(row.get("axis")) == axis], predictions
        )
        for axis in axes
    }
    return result


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

import json
import unittest

from evaluation.agent_only_score import score


class AgentOnlyScoreTest(unittest.TestCase):
    def test_scores_validity_triggers_routing_and_constraints(self) -> None:
        gold = [
            {
                "bench_id": "mae_pilot_0001",
                "axis": "rewrite",
                "gold_plan": {
                    "refined_text_instruction": "Change the cup to red and keep two plates.",
                    "subtask": "change_color",
                    "image_search": False,
                    "mask": "cup",
                },
                "constraints": [
                    {"type": "color", "value": "red"},
                    {"type": "count", "value": "two", "aliases": ["2", "both"]},
                ],
                "source_entities": ["cup", "plates"],
            },
            {
                "bench_id": "mae_pilot_0002",
                "axis": "search",
                "gold_plan": {
                    "refined_text_instruction": "Replace the bottle with a Stanley tumbler.",
                    "subtask": "replace_object",
                    "image_search": "Stanley tumbler cup",
                    "mask": "bottle",
                },
                "search_query_aliases": ["Stanley tumbler"],
                "constraints": [{"type": "identity", "value": "Stanley tumbler"}],
                "source_entities": ["bottle"],
            },
        ]
        predictions = [
            {
                "bench_id": "mae_pilot_0001",
                "plan": {
                    "refined_text_instruction": "Make the cup red; preserve both plates.",
                    "subtask": "change_color",
                    "image_search": False,
                    "mask": "cup",
                },
                "agent_raw": json.dumps(
                    {
                        "refined_text_instruction": "Make the cup red; preserve both plates.",
                        "subtask": "change_color",
                        "image_search": False,
                        "mask": "cup",
                    }
                ),
            },
            {
                "bench_id": "mae_pilot_0002",
                "plan": {
                    "refined_text_instruction": "Replace the bottle with a generic cup.",
                    "subtask": "add_object",
                    "image_search": False,
                    "mask": False,
                },
                "agent_raw": json.dumps(
                    {
                        "refined_text_instruction": "Replace the bottle with a generic cup.",
                        "subtask": "add_object",
                        "image_search": False,
                        "mask": False,
                    }
                ),
            },
        ]

        metrics = score(gold, predictions)

        self.assertEqual(metrics["json_validity"], 1.0)
        self.assertEqual(metrics["strict_raw_json_validity"], 1.0)
        self.assertEqual(metrics["subtask_accuracy"], 0.5)
        self.assertEqual(metrics["image_search_trigger"]["f1"], 0.0)
        self.assertEqual(metrics["mask_trigger"]["recall"], 0.5)
        self.assertEqual(metrics["constraint_retention"], 2 / 3)
        self.assertEqual(metrics["source_entity_false_trigger"]["num_cases"], 1)
        self.assertEqual(metrics["source_entity_false_trigger"]["rate"], 0.0)
        self.assertEqual(metrics["source_entity_false_trigger"]["ids"], [])
        self.assertEqual(metrics["details"]["routing_errors"][0]["bench_id"], "mae_pilot_0002")
        self.assertEqual(metrics["by_axis"]["rewrite"]["subtask_accuracy"], 1.0)
        self.assertEqual(metrics["by_axis"]["search"]["subtask_accuracy"], 0.0)

    def test_scores_explicit_search_query_aliases_directionally(self) -> None:
        def gold_row(bench_id: str, query: str, aliases: list[str]) -> dict:
            return {
                "bench_id": bench_id,
                "axis": "search",
                "gold_plan": {
                    "refined_text_instruction": f"Use {query}.",
                    "subtask": "replace_object",
                    "image_search": query,
                    "mask": False,
                },
                "search_query_aliases": aliases,
            }

        def prediction_row(bench_id: str, query: str | bool) -> dict:
            plan = {
                "refined_text_instruction": "Replace the object.",
                "subtask": "replace_object",
                "image_search": query,
                "mask": False,
            }
            return {"bench_id": bench_id, "plan": plan, "agent_raw": json.dumps(plan)}

        gold = [
            gold_row(
                "qualified",
                "rose quartz Stanley Quencher tumbler",
                ["rose quartz Stanley Quencher"],
            ),
            gold_row(
                "explicit_short_alias",
                "Trek Madone racing bicycle",
                ["Trek Madone racing bicycle", "Trek Madone"],
            ),
            gold_row("generic_noun", "Starbucks holiday cup", ["Starbucks holiday cup"]),
            gold_row(
                "reverse_substring",
                "rose quartz Stanley Quencher tumbler",
                ["rose quartz Stanley Quencher"],
            ),
            gold_row("missed", "Eiffel Tower", ["Eiffel Tower"]),
        ]
        predictions = [
            prediction_row("qualified", "official rose quartz Stanley Quencher product photo"),
            prediction_row("explicit_short_alias", "Trek Madone side view"),
            prediction_row("generic_noun", "cup"),
            prediction_row("reverse_substring", "Stanley"),
            prediction_row("missed", False),
        ]

        metrics = score(gold, predictions)

        self.assertEqual(metrics["image_search_trigger"]["tp"], 4)
        self.assertEqual(metrics["image_search_trigger"]["fn"], 1)
        self.assertEqual(
            metrics["image_search_query"],
            {
                "gold_positive_cases": 5,
                "triggered_cases": 4,
                "correct_queries": 2,
                "conditional_accuracy": 0.5,
                "end_to_end_recall": 0.4,
                "wrong_query_ids": ["generic_noun", "reverse_substring"],
                "missed_trigger_ids": ["missed"],
            },
        )

    def test_strict_raw_validity_does_not_use_normalized_plan(self) -> None:
        gold = [
            {
                "bench_id": "strict_1",
                "axis": "routing",
                "gold_plan": {
                    "refined_text_instruction": "Remove the cup.",
                    "subtask": "remove_object",
                    "image_search": False,
                    "mask": "cup",
                },
            },
            {
                "bench_id": "strict_2",
                "axis": "routing",
                "gold_plan": {
                    "refined_text_instruction": "Remove the cup.",
                    "subtask": "remove_object",
                    "image_search": False,
                    "mask": "cup",
                },
            },
            {
                "bench_id": "strict_3",
                "axis": "routing",
                "gold_plan": {
                    "refined_text_instruction": "Remove the cup.",
                    "subtask": "remove_object",
                    "image_search": False,
                    "mask": "cup",
                },
            },
        ]
        normalized_plan = {
            "refined_text_instruction": "Remove the cup.",
            "subtask": "remove_object",
            "image_search": False,
            "mask": "cup",
        }
        predictions = [
            {
                "bench_id": "strict_1",
                "plan": normalized_plan,
                "agent_raw": json.dumps({**normalized_plan, "explanation": "done"}),
            },
            {
                "bench_id": "strict_2",
                "plan": normalized_plan,
                "agent_raw": f"Result: {json.dumps(normalized_plan)}",
            },
            {
                "bench_id": "strict_3",
                "plan": normalized_plan,
                "agent_raw": json.dumps({**normalized_plan, "subtask": ["remove_object"]}),
            },
        ]

        metrics = score(gold, predictions)

        self.assertEqual(metrics["json_validity"], 1.0)
        self.assertIn("normalized runtime", metrics["json_validity_method"])
        self.assertEqual(metrics["strict_raw_json_validity"], 0.0)
        self.assertEqual(
            metrics["details"]["strict_raw_invalid_predictions"],
            ["strict_1", "strict_2", "strict_3"],
        )

    def test_rejects_duplicate_bench_ids(self) -> None:
        gold_row = {
            "bench_id": "duplicate",
            "gold_plan": {
                "refined_text_instruction": "Remove the cup.",
                "subtask": "remove_object",
                "image_search": False,
                "mask": "cup",
            },
        }
        with self.assertRaisesRegex(ValueError, "duplicate gold bench_id"):
            score([gold_row, dict(gold_row)], [])

        with self.assertRaisesRegex(ValueError, "duplicate prediction bench_id"):
            score(
                [gold_row],
                [{"bench_id": "duplicate"}, {"bench_id": "duplicate"}],
            )

    def test_reports_extra_prediction_ids(self) -> None:
        plan = {
            "refined_text_instruction": "Remove the cup.",
            "subtask": "remove_object",
            "image_search": False,
            "mask": "cup",
        }
        gold = [{"bench_id": "expected", "gold_plan": plan}]
        predictions = [
            {"bench_id": "expected", "plan": plan, "agent_raw": json.dumps(plan)},
            {"bench_id": "unexpected", "plan": plan, "agent_raw": json.dumps(plan)},
        ]

        metrics = score(gold, predictions)

        self.assertEqual(metrics["details"]["extra_prediction_ids"], ["unexpected"])

    def test_reports_source_entity_false_trigger_ids(self) -> None:
        gold = [
            {
                "bench_id": "style_false_search",
                "gold_plan": {
                    "refined_text_instruction": "Restyle the whole video.",
                    "subtask": "global_style",
                    "image_search": False,
                    "mask": False,
                },
                "source_entities": ["dog"],
            }
        ]
        plan = {
            "refined_text_instruction": "Restyle the whole video.",
            "subtask": "global_style",
            "image_search": "named painter",
            "mask": False,
        }
        predictions = [
            {"bench_id": "style_false_search", "plan": plan, "agent_raw": json.dumps(plan)}
        ]

        metrics = score(gold, predictions)

        self.assertEqual(metrics["source_entity_false_trigger"]["false_triggers"], 1)
        self.assertEqual(
            metrics["source_entity_false_trigger"]["ids"], ["style_false_search"]
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from evaluation.agent_only_score import score


class AgentOnlyScoreTest(unittest.TestCase):
    def test_scores_validity_triggers_routing_and_constraints(self) -> None:
        gold = [
            {
                "bench_id": "mae_pilot_0001",
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
            },
            {
                "bench_id": "mae_pilot_0002",
                "gold_plan": {
                    "refined_text_instruction": "Replace the bottle with a Stanley tumbler.",
                    "subtask": "replace_object",
                    "image_search": "Stanley tumbler cup",
                    "mask": "bottle",
                },
                "constraints": [{"type": "identity", "value": "Stanley tumbler"}],
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
            },
            {
                "bench_id": "mae_pilot_0002",
                "plan": {
                    "refined_text_instruction": "Replace the bottle with a generic cup.",
                    "subtask": "add_object",
                    "image_search": False,
                    "mask": False,
                },
            },
        ]

        metrics = score(gold, predictions)

        self.assertEqual(metrics["json_validity"], 1.0)
        self.assertEqual(metrics["subtask_accuracy"], 0.5)
        self.assertEqual(metrics["image_search_trigger"]["f1"], 0.0)
        self.assertEqual(metrics["mask_trigger"]["recall"], 0.5)
        self.assertEqual(metrics["constraint_retention"], 2 / 3)
        self.assertEqual(metrics["details"]["routing_errors"][0]["bench_id"], "mae_pilot_0002")


if __name__ == "__main__":
    unittest.main()

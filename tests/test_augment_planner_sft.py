import json
import unittest

from scripts.augment_planner_sft import ROUTING_TEMPLATES, build_augmented


class AugmentPlannerSftTest(unittest.TestCase):
    def setUp(self):
        self.base = [
            {
                "system": "system",
                "messages": [
                    {"role": "user", "content": "<video>\ndo something"},
                    {"role": "assistant", "content": "{}"},
                ],
                "videos": ["/tmp/example.mp4"],
            }
        ]

    def test_search_examples_are_balanced_and_trigger_search(self):
        rows, metadata = build_augmented(self.base, search_count=4, routing_count=0)
        plans = [json.loads(row["messages"][1]["content"]) for row in rows[1:]]
        self.assertEqual([plan["subtask"] for plan in plans], ["add_object", "replace_object"] * 2)
        self.assertTrue(all(isinstance(plan["image_search"], str) for plan in plans))
        self.assertTrue(all(item["category"] == "under_search" for item in metadata))

    def test_routing_examples_cover_every_template(self):
        rows, _ = build_augmented(self.base, search_count=0, routing_count=len(ROUTING_TEMPLATES))
        plans = [json.loads(row["messages"][1]["content"]) for row in rows[1:]]
        self.assertEqual({plan["subtask"] for plan in plans}, {item[2] for item in ROUTING_TEMPLATES})
        removal = next(plan for plan in plans if plan["subtask"] == "remove_object")
        self.assertIsInstance(removal["mask"], str)
        self.assertTrue(all(plan["image_search"] is False for plan in plans))


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.assemble_final_sft import assemble
from scripts.generate_hard_requests_vllm import (
    choose_category,
    parse_request,
    select_pending,
    validate_request,
)


class HardSftAssemblyTest(unittest.TestCase):
    def test_category_routes_removal_and_composition(self):
        removal = {
            "raw_user_request": "remove the red cup",
            "target_plan": {"subtask": "remove_object", "mask": "red cup"},
        }
        combined = {
            "raw_user_request": "add a cup and make the wall blue",
            "target_plan": {"subtask": "combined_tasks"},
        }
        self.assertEqual(choose_category(removal, 2), "mask_granularity")
        self.assertEqual(choose_category(combined, 1), "compositional_edit")

    def test_category_respects_search_requirement(self):
        row = {
            "raw_user_request": "replace it with a Stanley Quencher",
            "target_plan": {
                "subtask": "replace_object",
                "image_search": "Stanley Quencher tumbler",
            },
        }
        self.assertEqual(choose_category(row, 0), "under_search")

    def test_parse_request_extracts_json(self):
        self.assertEqual(parse_request('prefix {"raw_user_request":"make this blue"}'), "make this blue")
        self.assertIsNone(parse_request("not json"))

    def test_pilot_selection_ignores_completed_rows_outside_limit(self):
        rows = [
            {
                "sample_id": sample_id,
                "raw_user_request": "add the blue cup",
                "target_plan": {"subtask": "add_object"},
            }
            for sample_id in ("s1", "s2", "s3")
        ]
        selected, done, pending = select_pending(rows, {"s1", "s3"}, max_samples=None)
        self.assertEqual([row["sample_id"] for _, row in selected], ["s1", "s2", "s3"])
        self.assertEqual(done, {"s1", "s3"})
        self.assertEqual([(index, row["sample_id"]) for index, row in pending], [(1, "s2")])

    def test_validation_rejects_dropped_constraints_and_search_entity(self):
        row = {
            "raw_user_request": "put two red cups on the right",
            "target_plan": {
                "subtask": "add_object",
                "image_search": "Starbucks holiday cup",
                "mask": False,
            },
        }
        errors = validate_request(row, "under_search", "add cups")
        self.assertIn("missing_search_entity", errors)
        self.assertTrue(any(error.startswith("missing_constraints:") for error in errors))

    def test_assemble_keeps_base_and_adds_hard_record(self):
        plan = {
            "refined_text_instruction": "Make the object blue.",
            "subtask": "change_color",
            "image_search": False,
            "mask": False,
        }
        canonical = [
            {
                "sample_id": "s1",
                "video_path": "a.mp4",
                "raw_user_request": "make it blue",
                "target_plan": plan,
            }
        ]
        base = [
            {
                "system": "system",
                "messages": [
                    {"role": "user", "content": "<video>\nmake it blue"},
                    {
                        "role": "assistant",
                        "content": '{"refined_text_instruction":"Make the object blue.","subtask":"change_color","image_search":false,"mask":false}',
                    },
                ],
                "videos": ["/tmp/a.mp4"],
            }
        ]
        hard = [
            {
                "sample_id": "s1",
                "category": "pronoun_grounding",
                "raw_user_request": "make that one blue",
                "generated": True,
                "accepted": True,
            }
        ]
        rows, summary = assemble(canonical, base, hard, search_count=2, routing_count=1)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[1]["messages"][0]["content"], "<video>\nmake that one blue")
        self.assertEqual(summary["total"], 5)


if __name__ == "__main__":
    unittest.main()

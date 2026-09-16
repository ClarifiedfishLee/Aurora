import unittest

from scripts.assemble_final_sft import assemble
from scripts.generate_hard_requests_vllm import choose_category, parse_request


class HardSftAssemblyTest(unittest.TestCase):
    def test_category_routes_removal_and_composition(self):
        removal = {"target_plan": {"subtask": "remove_object"}}
        combined = {"target_plan": {"subtask": "combined_tasks"}}
        self.assertEqual(choose_category(removal, 2), "mask_granularity")
        self.assertEqual(choose_category(combined, 1), "compositional_edit")

    def test_parse_request_extracts_json(self):
        self.assertEqual(parse_request('prefix {"raw_user_request":"make this blue"}'), "make this blue")
        self.assertIsNone(parse_request("not json"))

    def test_assemble_keeps_base_and_adds_hard_record(self):
        canonical = [{"sample_id": "s1", "target_plan": {"subtask": "change_color"}}]
        base = [
            {
                "system": "system",
                "messages": [
                    {"role": "user", "content": "<video>\nmake it blue"},
                    {"role": "assistant", "content": "{}"},
                ],
                "videos": ["/tmp/a.mp4"],
            }
        ]
        hard = [{"sample_id": "s1", "category": "pronoun_grounding", "raw_user_request": "make that one blue"}]
        rows, summary = assemble(canonical, base, hard, search_count=2, routing_count=1)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[1]["messages"][0]["content"], "<video>\nmake that one blue")
        self.assertEqual(summary["total"], 5)


if __name__ == "__main__":
    unittest.main()

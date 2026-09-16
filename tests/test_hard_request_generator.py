import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_hard_requests_vllm import (
    CATEGORY_GUIDANCE,
    choose_category,
    completed_ids,
    load_resumable_jsonl,
    stratified_rows,
)


def _general_row(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "raw_user_request": "put two small red cups on the right",
        "target_plan": {
            "subtask": "add_object",
            "image_search": False,
            "mask": False,
        },
    }


class HardRequestGeneratorTest(unittest.TestCase):
    def test_stratified_pilot_covers_every_available_category(self):
        rows = []
        general_categories = {
            "entity_ambiguity",
            "pronoun_grounding",
            "implicit_local_edit",
            "over_search_negative",
            "multiple_constraints",
            "rewrite_preservation",
        }
        found: dict[str, list[dict]] = {category: [] for category in general_categories}
        candidate = 0
        while any(len(matches) < 2 for matches in found.values()):
            row = _general_row(f"general-{candidate}")
            category = choose_category(row, candidate)
            if category in found and len(found[category]) < 2:
                found[category].append(row)
            candidate += 1
        rows.extend(row for category in CATEGORY_GUIDANCE for row in found.get(category, []))
        for suffix in range(2):
            rows.extend(
                [
                    {
                        "sample_id": f"search-{suffix}",
                        "raw_user_request": "make it a Stanley Quencher",
                        "target_plan": {
                            "subtask": "replace_object",
                            "image_search": "Stanley Quencher",
                            "mask": False,
                        },
                    },
                    {
                        "sample_id": f"mask-{suffix}",
                        "raw_user_request": "remove the red cup",
                        "target_plan": {
                            "subtask": "remove_object",
                            "image_search": False,
                            "mask": "red cup",
                        },
                    },
                    {
                        "sample_id": f"combined-{suffix}",
                        "raw_user_request": "remove the cup and make the wall blue",
                        "target_plan": {
                            "subtask": "combined_tasks",
                            "image_search": False,
                            "mask": False,
                        },
                    },
                ]
            )
        selected = stratified_rows(rows, max_samples=9)
        categories = {choose_category(row, index) for index, row in selected}
        self.assertEqual(categories, set(CATEGORY_GUIDANCE))

    def test_completed_ids_only_accepts_quality_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hard.jsonl"
            path.write_text(
                json.dumps({"sample_id": "ok", "accepted": True})
                + "\n"
                + json.dumps({"sample_id": "bad", "accepted": False})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(completed_ids(path), {"ok"})

    def test_resume_repairs_only_a_truncated_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hard.jsonl"
            valid = json.dumps({"sample_id": "ok", "accepted": True}) + "\n"
            path.write_bytes(valid.encode() + b'{"sample_id":"partial"')
            self.assertEqual(load_resumable_jsonl(path), [{"sample_id": "ok", "accepted": True}])
            self.assertEqual(path.read_text(encoding="utf-8"), valid)

    def test_resume_rejects_interior_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hard.jsonl"
            path.write_text("not-json\n{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "record 1"):
                load_resumable_jsonl(path)


if __name__ == "__main__":
    unittest.main()

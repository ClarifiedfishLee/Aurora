import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_blind_ab_page import build as build_blind_page
from scripts.build_week1_ab_pilot import MASK_IDS, build_records, load_jsonl


class Week1ABPilotTest(unittest.TestCase):
    def test_builds_40_single_axis_pairs(self):
        planner = load_jsonl(Path("data/week1/planner_100.jsonl"))
        resolved = [
            {
                "bench_id": case_id,
                "mask": {"mask_path": f"/tmp/{case_id}.png"},
                "final_payload": {"object_mask": f"/tmp/{case_id}.png"},
            }
            for case_id in MASK_IDS
        ]
        records, pairs = build_records(planner, resolved)
        self.assertEqual(len(records), 80)
        self.assertEqual(len(pairs), 40)
        by_id = {record["bench_id"]: record for record in records}
        expected = {
            "search": "image_search",
            "mask": "mask",
            "rewrite": "refined_text_instruction",
            "routing": "subtask",
        }
        for pair in pairs:
            good = by_id[pair["good_record_id"]]["plan"]
            bad = by_id[pair["bad_record_id"]]["plan"]
            changed = [field for field in good if good[field] != bad[field]]
            self.assertEqual(changed, [expected[pair["axis"]]])

    def test_blind_page_keeps_answer_key_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            videos = root / "videos"
            source.touch()
            for variant in ("pair_good", "pair_bad"):
                path = videos / variant / "generate.mp4"
                path.parent.mkdir(parents=True)
                path.touch()
            pairs = [
                {
                    "pair_id": "pair",
                    "axis": "rewrite",
                    "instruction": "edit this",
                    "source_video": str(source),
                    "good_record_id": "pair_good",
                    "bad_record_id": "pair_bad",
                    "negative_control": False,
                }
            ]
            page_path, key_path = build_blind_page(pairs, root / "blind", videos, 7)
            page = page_path.read_text(encoding="utf-8")
            key = json.loads(key_path.read_text(encoding="utf-8"))
            self.assertNotIn("good_side", page)
            self.assertNotIn("pair_good", page)
            self.assertIn("join('\\n')", page)
            self.assertIn(key[0]["good_side"], {"A", "B"})


if __name__ == "__main__":
    unittest.main()

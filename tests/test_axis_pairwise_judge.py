import json
import unittest

from evaluation.axis_pairwise_judge import build_items, parse_response, score_results


class AxisPairwiseJudgeTest(unittest.TestCase):
    def test_builds_axis_specific_prompt(self):
        pair = {
            "pair_id": "p1",
            "axis": "rewrite",
            "source_case_id": "c1",
            "instruction": "add three red balls",
            "source_video": "source.mp4",
            "video_a": "a.mp4",
            "video_b": "b.mp4",
        }
        case = {
            "bench_id": "c1",
            "constraints": [{"type": "count", "value": "three"}],
        }
        item = build_items([pair], [case])[0]
        self.assertIn("count", item["prompt"])
        self.assertIn("CANDIDATE A, CANDIDATE B", item["prompt"])
        self.assertNotIn("A|B|tie|invalid", item["prompt"])

    def test_parses_fenced_json(self):
        raw = "```json\n" + json.dumps({"label": "invalid", "reason": "both fail"}) + "\n```"
        self.assertEqual(parse_response(raw)["label"], "invalid")

    def test_scores_invalid_and_valid_preferences(self):
        results = [
            {"pair_id": "p1", "axis": "search", "parsed": {"label": "A"}},
            {"pair_id": "p2", "axis": "mask", "parsed": {"label": "invalid"}},
        ]
        human = [
            {"bench_id": "p1", "human_label": "A"},
            {"bench_id": "p2", "human_label": "invalid"},
        ]
        summary = score_results(results, human)
        self.assertEqual(summary["four_way_exact_agreement"], 1.0)
        self.assertEqual(summary["invalid_detection"]["f1"], 1.0)
        self.assertEqual(summary["valid_preference_agreement"]["num_usable"], 1)


if __name__ == "__main__":
    unittest.main()

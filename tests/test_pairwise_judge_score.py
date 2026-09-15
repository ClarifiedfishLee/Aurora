import json
import unittest

from evaluation.pairwise_judge_score import aggregate, parse_response


def response(alignment: int, background: int = 4) -> str:
    return "```json\n" + json.dumps(
        {
            "structural_fidelity": 4,
            "text_video_alignment": alignment,
            "background_consistency": background,
            "naturalness": 4,
            "temporal_spatial_consistency": 4,
            "explanation": {},
        }
    ) + "\n```"


class PairwiseJudgeScoreTest(unittest.TestCase):
    def test_parses_fenced_response(self):
        self.assertEqual(parse_response(response(3))["text_video_alignment"], 3)

    def test_aggregates_and_excludes_invalid_human_label(self):
        human = [
            {"bench_id": "p1", "human_label": "A", "human_notes": ""},
            {"bench_id": "p2", "human_label": "invalid", "human_notes": "broken"},
        ]
        key = [
            {"pair_id": "p1", "axis": "search", "good_side": "A"},
            {"pair_id": "p2", "axis": "mask", "good_side": "B"},
        ]
        results = [
            {"id": "p1_A", "response": response(5)},
            {"id": "p1_B", "response": response(2)},
            {"id": "p2_A", "response": response(1)},
            {"id": "p2_B", "response": response(5)},
        ]
        summary, details, annotations = aggregate(human, key, results)
        self.assertEqual(summary["num_parsed"], 4)
        self.assertEqual(summary["metrics"]["alignment"]["num_usable"], 1)
        self.assertEqual(summary["metrics"]["alignment"]["overall"]["directional_agreement"], 1.0)
        self.assertEqual(annotations["alignment"][0]["judge_label"], "A")
        self.assertEqual(details[1]["human_label"], "invalid")
        self.assertEqual(details[1]["human_notes"], "broken")


if __name__ == "__main__":
    unittest.main()

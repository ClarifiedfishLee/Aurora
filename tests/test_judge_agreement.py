import unittest

from evaluation.judge_agreement import score


class JudgeAgreementTest(unittest.TestCase):
    def test_reports_overall_and_axis_metrics(self) -> None:
        rows = [
            {"bench_id": "a", "axis": "search", "human_label": "A", "judge_label": "A"},
            {"bench_id": "b", "axis": "search", "human_label": "B", "judge_label": "A"},
            {"bench_id": "c", "axis": "mask", "human_label": "tie", "judge_label": "tie"},
            {"bench_id": "d", "axis": "mask", "human_label": "invalid", "judge_label": "B"},
        ]

        metrics = score(rows)

        self.assertEqual(metrics["num_usable"], 3)
        self.assertEqual(metrics["coverage"], 0.75)
        self.assertEqual(metrics["overall"]["exact_agreement"], 2 / 3)
        self.assertEqual(metrics["overall"]["directional_agreement"], 0.5)
        self.assertEqual(metrics["by_axis"]["mask"]["human_tie_rate"], 1.0)
        self.assertEqual(metrics["excluded"][0]["bench_id"], "d")

    def test_rejects_duplicate_ids(self) -> None:
        rows = [
            {"bench_id": "a", "human_label": "A", "judge_label": "A"},
            {"bench_id": "a", "human_label": "B", "judge_label": "B"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate bench_id"):
            score(rows)


if __name__ == "__main__":
    unittest.main()

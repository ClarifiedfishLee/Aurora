import unittest
from collections import Counter

from scripts.build_week1_planner_cases import build


class Week1PlannerCasesTest(unittest.TestCase):
    def test_builds_balanced_unique_suite(self) -> None:
        rows = build()

        self.assertEqual(len(rows), 100)
        self.assertEqual(len({row["bench_id"] for row in rows}), 100)
        self.assertEqual(Counter(row["axis"] for row in rows), {"search": 25, "mask": 25, "routing": 25, "rewrite": 25})
        self.assertTrue(all(set(row["gold_plan"]) == {"refined_text_instruction", "subtask", "image_search", "mask"} for row in rows))
        self.assertEqual(sum(row["gold_plan"]["mask"] is not False for row in rows), 10)
        self.assertTrue(all(row["gold_plan"]["mask"] is False or row["gold_plan"]["subtask"] == "remove_object" for row in rows))


if __name__ == "__main__":
    unittest.main()

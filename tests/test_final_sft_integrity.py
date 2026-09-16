import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.assemble_final_sft import assemble, write_summary
from scripts.augment_planner_sft import build_augmented, uniform_source_indices


def planner_plan(index: int) -> dict:
    return {
        "refined_text_instruction": f"Change object {index} to blue.",
        "subtask": "change_color",
        "image_search": False,
        "mask": False,
    }


def dataset(size: int = 8) -> tuple[list[dict], list[dict], list[dict]]:
    canonical = []
    base = []
    hard = []
    for index in range(size):
        sample_id = f"s{index}"
        request = f"make object {index} blue"
        plan = planner_plan(index)
        canonical.append(
            {
                "sample_id": sample_id,
                "video_path": f"videos/subset/{sample_id}.mp4",
                "raw_user_request": request,
                "target_plan": plan,
            }
        )
        base.append(
            {
                "system": "system",
                "messages": [
                    {"role": "user", "content": f"<video>\n{request}"},
                    {
                        "role": "assistant",
                        "content": json.dumps(plan, separators=(",", ":")),
                    },
                ],
                "videos": [f"/tmp/work/videos/subset/{sample_id}.mp4"],
            }
        )
        hard.append(
            {
                "sample_id": sample_id,
                "category": "pronoun_grounding",
                "raw_user_request": f"make that object {index} blue",
                "generated": True,
                "accepted": True,
            }
        )
    return canonical, base, hard


class FinalSftIntegrityTest(unittest.TestCase):
    def test_assemble_validates_pairs_and_reports_calibration_coverage(self):
        canonical, base, hard = dataset()
        rows, summary = assemble(canonical, base, hard, search_count=4, routing_count=2)

        self.assertEqual(len(rows), 22)
        self.assertEqual(summary["hard"], 8)
        self.assertTrue(summary["hard_complete"])
        search_audit = summary["calibration_audit"]["by_category"]["under_search"]
        routing_audit = summary["calibration_audit"]["by_category"]["routing_calibration"]
        self.assertEqual(search_audit["unique_sources"], 4)
        self.assertEqual(search_audit["min_source_index"], 1)
        self.assertEqual(search_audit["max_source_index"], 7)
        self.assertEqual(routing_audit["min_source_index"], 2)
        self.assertEqual(routing_audit["max_source_index"], 6)

    def test_rejects_canonical_base_request_plan_and_video_mismatches(self):
        canonical, base, hard = dataset(1)
        mutations = {
            "request": lambda rows: rows[0]["messages"][0].update(content="<video>\nwrong"),
            "plan": lambda rows: rows[0]["messages"][1].update(content="{}"),
            "video": lambda rows: rows[0].update(videos=["/tmp/work/videos/other.mp4"]),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(base)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    assemble(canonical, changed, hard, search_count=0, routing_count=0)

    def test_rejects_duplicate_missing_unknown_and_rejected_hard_rows(self):
        canonical, base, hard = dataset(2)
        cases = [
            (
                canonical + [copy.deepcopy(canonical[0])],
                base + [copy.deepcopy(base[0])],
                hard,
                "duplicate canonical",
            ),
            (canonical, base, hard + [copy.deepcopy(hard[0])], "duplicate hard"),
            (canonical, base, hard[:1], "missing sample_ids"),
            (
                canonical,
                base,
                [*hard, {**hard[0], "sample_id": "unknown"}],
                "unknown sample_ids",
            ),
            (
                canonical,
                base,
                [{**hard[0], "generated": False}, hard[1]],
                "rejected by flags",
            ),
            (
                canonical,
                base,
                [{**hard[0], "accepted": False}, hard[1]],
                "rejected by flags",
            ),
        ]
        for canonical_rows, base_rows, hard_rows, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    assemble(canonical_rows, base_rows, hard_rows, search_count=0, routing_count=0)

    def test_partial_hard_data_requires_explicit_opt_out(self):
        canonical, base, hard = dataset(2)
        rows, summary = assemble(
            canonical,
            base,
            hard[:1],
            search_count=0,
            routing_count=0,
            require_complete_hard=False,
        )
        self.assertEqual(len(rows), 3)
        self.assertFalse(summary["hard_complete"])
        self.assertEqual(summary["hard_missing"], 1)

    def test_extra_hard_data_requires_explicit_drop_and_is_reported(self):
        canonical, base, hard = dataset(1)
        extra = {**hard[0], "sample_id": "filtered-blank-source"}
        rows, summary = assemble(
            canonical,
            base,
            [*hard, extra],
            search_count=0,
            routing_count=0,
            drop_unknown_hard=True,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["hard_extra_dropped"], 1)
        self.assertEqual(summary["hard_extra_dropped_examples"], ["filtered-blank-source"])

    def test_uniform_source_indices_span_the_population(self):
        self.assertEqual(uniform_source_indices(8, 4), [1, 3, 5, 7])
        self.assertEqual(uniform_source_indices(8, 2), [2, 6])
        self.assertEqual(uniform_source_indices(2, 4), [0, 0, 1, 1])
        search_indices = uniform_source_indices(5000, 2000)
        routing_indices = uniform_source_indices(5000, 1000)
        self.assertEqual((min(search_indices), max(search_indices)), (1, 4998))
        self.assertEqual((len(search_indices), len(set(search_indices))), (2000, 2000))
        self.assertEqual((min(routing_indices), max(routing_indices)), (2, 4997))
        self.assertEqual((len(routing_indices), len(set(routing_indices))), (1000, 1000))
        with self.assertRaises(ValueError):
            uniform_source_indices(8, -1)

        _, base, _ = dataset()
        rows, metadata = build_augmented(base, search_count=4, routing_count=2)
        self.assertEqual([row["source_index"] for row in metadata], [1, 3, 5, 7, 2, 6])
        self.assertEqual(rows[8]["videos"], base[1]["videos"])

    def test_write_summary_creates_parent_and_omits_raw_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "summary.json"
            write_summary(path, {"total": 3, "calibration_metadata": [{"large": "record"}]})
            written = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(written, {"total": 3})


if __name__ == "__main__":
    unittest.main()

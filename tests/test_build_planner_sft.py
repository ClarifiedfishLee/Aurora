import unittest
from pathlib import Path

from scripts.build_planner_sft import build_teacher_cases, compose, degrade_instruction


class BuildPlannerSftTest(unittest.TestCase):
    def test_degradation_is_deterministic(self):
        clean = "Remove the tire from the road."
        self.assertEqual(degrade_instruction(clean, "sample-1"), degrade_instruction(clean, "sample-1"))

    def test_compose_uses_clean_instruction_and_subset_route(self):
        manifest = [
            {
                "sample_id": "s1",
                "video_path": "videos/s1.mp4",
                "clean_instruction": "Remove the tire while preserving the road.",
                "subset": "rose-removal",
                "edit_type": "removal_mask_ref",
            }
        ]
        teacher = [
            {
                "bench_id": "s1",
                "video_path": "/tmp/s1.mp4",
                "plan": {
                    "refined_text_instruction": "Remove tire.",
                    "subtask": "global_style",
                    "image_search": False,
                    "mask": "tire",
                },
            }
        ]
        canonical, llama = compose(manifest, teacher, "system")
        self.assertEqual(canonical[0]["target_plan"]["subtask"], "remove_object")
        self.assertEqual(
            canonical[0]["target_plan"]["refined_text_instruction"], manifest[0]["clean_instruction"]
        )
        self.assertEqual(llama[0]["videos"], [str(Path("/tmp/s1.mp4").resolve())])


if __name__ == "__main__":
    unittest.main()

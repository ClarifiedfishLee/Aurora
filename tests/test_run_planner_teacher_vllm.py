import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_planner_teacher_vllm import completed_ids, make_record


class PlannerTeacherVllmTest(unittest.TestCase):
    def test_completed_ids_supports_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text('{"bench_id":"a"}\n{"bench_id":"b"}\n', encoding="utf-8")
            self.assertEqual(completed_ids(path), {"a", "b"})

    def test_record_is_compose_compatible(self):
        case = {"bench_id": "a", "video_path": "/tmp/a.mp4", "prompt": "edit", "edit_type": "custom"}
        plan = {
            "refined_text_instruction": "Edit it.",
            "subtask": "add_effect",
            "image_search": False,
            "mask": False,
        }
        record = make_record(case, plan, json.dumps(plan))
        self.assertEqual(record["plan"], plan)
        self.assertEqual(record["final_payload"]["subtask"], "add_effect")


if __name__ == "__main__":
    unittest.main()

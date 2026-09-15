import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_llamafactory_config import validate_rows, write_bundle


class MakeLlamafactoryConfigTest(unittest.TestCase):
    def test_writes_dataset_info_and_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.mp4"
            video.write_bytes(b"video")
            dataset = root / "sft.jsonl"
            row = {
                "system": "system",
                "messages": [
                    {"role": "user", "content": "<video>\nrequest"},
                    {"role": "assistant", "content": "{}"},
                ],
                "videos": [str(video)],
            }
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            config = root / "train.yaml"
            write_bundle(dataset, "/models/qwen", root / "output", config)
            info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["aurora_planner_sft"]["columns"]["videos"], "videos")
            self.assertIn("lora_rank: 32", config.read_text(encoding="utf-8"))

    def test_rejects_video_count_mismatch(self):
        with self.assertRaises(ValueError):
            validate_rows([{"system": "x", "messages": [], "videos": []}])


if __name__ == "__main__":
    unittest.main()

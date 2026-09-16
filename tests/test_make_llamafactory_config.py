import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_llamafactory_config import grouped_split, validate_rows, write_bundle


class MakeLlamafactoryConfigTest(unittest.TestCase):
    def test_writes_dataset_info_and_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "sft.jsonl"
            rows = []
            for index in range(2):
                video = root / f"sample-{index}.mp4"
                video.write_bytes(b"video")
                rows.append(
                    {
                        "system": "system",
                        "messages": [
                            {"role": "user", "content": "<video>\nrequest"},
                            {"role": "assistant", "content": "{}"},
                        ],
                        "videos": [str(video)],
                    }
                )
            dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            config = root / "train.yaml"
            summary = write_bundle(dataset, "/models/qwen", root / "output", config)
            info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["aurora_planner_sft_train"]["columns"]["videos"], "videos")
            yaml = config.read_text(encoding="utf-8")
            self.assertIn("lora_rank: 32", yaml)
            self.assertIn("eval_dataset: aurora_planner_sft_eval", yaml)
            self.assertIn("val_size: 0.0", yaml)
            self.assertIn("save_total_limit: 2", yaml)
            self.assertEqual(summary["checkpoint_steps"], 10)
            self.assertEqual(summary["video_overlap"], 0)

    def test_grouped_split_keeps_video_variants_together(self):
        rows = [
            {"videos": [f"/tmp/video-{video}.mp4"], "variant": variant}
            for video in range(10)
            for variant in ("base", "hard", "calibration")
        ]
        train, evaluation, summary = grouped_split(rows, eval_ratio=0.2)
        train_videos = {row["videos"][0] for row in train}
        eval_videos = {row["videos"][0] for row in evaluation}
        self.assertFalse(train_videos & eval_videos)
        self.assertEqual((summary["train_videos"], summary["eval_videos"]), (8, 2))
        self.assertEqual((len(train), len(evaluation)), (24, 6))

    def test_rejects_video_count_mismatch(self):
        with self.assertRaises(ValueError):
            validate_rows([{"system": "x", "messages": [], "videos": []}])


if __name__ == "__main__":
    unittest.main()

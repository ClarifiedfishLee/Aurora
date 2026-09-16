import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_llamafactory_refresh_config import (
    EXPECTED_CHECKPOINT_STEPS,
    select_refresh_checkpoint,
    write_refresh_bundle,
)


def _row(video: Path, index: int) -> dict:
    return {
        "system": "system",
        "messages": [
            {"role": "user", "content": f"<video>\nrequest {index}"},
            {"role": "assistant", "content": "{}"},
        ],
        "videos": [str(video)],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class MakeLlamafactoryRefreshConfigTest(unittest.TestCase):
    def test_writes_fixed_refresh_config_and_preregistered_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_video = root / "train.mp4"
            eval_video = root / "eval.mp4"
            train_video.write_bytes(b"video")
            eval_video.write_bytes(b"video")
            train = root / "refresh_train.jsonl"
            evaluation = root / "refresh_eval.jsonl"
            _write_jsonl(train, [_row(train_video, index) for index in range(1024)])
            _write_jsonl(evaluation, [_row(eval_video, index) for index in range(256)])
            config = root / "refresh.yaml"
            policy = root / "selection_policy.json"

            summary = write_refresh_bundle(
                train,
                evaluation,
                "/models/Qwen3-VL-8B-Instruct",
                root / "lora-final-12597-best-eval",
                root / "lora-refresh",
                config,
                policy,
            )

            yaml = config.read_text(encoding="utf-8")
            self.assertIn('model_name_or_path: "/models/Qwen3-VL-8B-Instruct"', yaml)
            self.assertIn("adapter_name_or_path:", yaml)
            self.assertIn("create_new_adapter: false", yaml)
            self.assertNotIn("resume_from_checkpoint", yaml)
            self.assertIn("max_samples: 1024", yaml)
            self.assertIn("learning_rate: 2.0e-5", yaml)
            self.assertIn("warmup_ratio: 0.05", yaml)
            self.assertIn("gradient_accumulation_steps: 8", yaml)
            self.assertIn("save_steps: 32", yaml)
            self.assertIn("eval_steps: 32", yaml)
            self.assertIn("save_total_limit: 4", yaml)
            self.assertIn("overwrite_output_dir: true", yaml)
            self.assertEqual(summary["checkpoint_steps"], list(EXPECTED_CHECKPOINT_STEPS))
            self.assertEqual(summary["video_overlap"], 0)

            registry = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["aurora_planner_refresh_train"]["file_name"], train.name)
            self.assertEqual(registry["aurora_planner_refresh_eval"]["file_name"], evaluation.name)
            selection_policy = json.loads(policy.read_text(encoding="utf-8"))
            self.assertEqual(selection_policy["eligible_checkpoint_steps"], [32, 64, 96, 128])
            self.assertFalse(selection_policy["external_gate_metrics_allowed"])

    def test_rejects_non_exact_train_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_video = root / "train.mp4"
            eval_video = root / "eval.mp4"
            train_video.write_bytes(b"video")
            eval_video.write_bytes(b"video")
            train = root / "train.jsonl"
            evaluation = root / "eval.jsonl"
            _write_jsonl(train, [_row(train_video, index) for index in range(1023)])
            _write_jsonl(evaluation, [_row(eval_video, index) for index in range(256)])
            with self.assertRaisesRegex(ValueError, "exactly 1024"):
                write_refresh_bundle(
                    train, evaluation, "model", root / "adapter-best-eval", root / "out", root / "x.yaml"
                )

    def test_rejects_train_eval_video_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "shared.mp4"
            video.write_bytes(b"video")
            train = root / "train.jsonl"
            evaluation = root / "eval.jsonl"
            _write_jsonl(train, [_row(video, index) for index in range(1024)])
            _write_jsonl(evaluation, [_row(video, index) for index in range(256)])
            with self.assertRaisesRegex(ValueError, "video overlap"):
                write_refresh_bundle(
                    train, evaluation, "model", root / "adapter-best-eval", root / "out", root / "x.yaml"
                )

    def test_requires_best_eval_adapter_and_new_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_video = root / "train.mp4"
            eval_video = root / "eval.mp4"
            train_video.write_bytes(b"video")
            eval_video.write_bytes(b"video")
            train = root / "train.jsonl"
            evaluation = root / "eval.jsonl"
            _write_jsonl(train, [_row(train_video, index) for index in range(1024)])
            _write_jsonl(evaluation, [_row(eval_video, index) for index in range(256)])
            with self.assertRaisesRegex(ValueError, "best-eval"):
                write_refresh_bundle(
                    train, evaluation, "model", root / "last-step", root / "out", root / "x.yaml"
                )
            output = root / "nonempty-output"
            output.mkdir()
            (output / "checkpoint-8").mkdir()
            with self.assertRaisesRegex(ValueError, "new or empty"):
                write_refresh_bundle(
                    train,
                    evaluation,
                    "model",
                    root / "lora-best-eval",
                    output,
                    root / "x.yaml",
                )

    def test_selects_lowest_refresh_eval_loss_with_earliest_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            losses = {32: 0.5, 64: 0.4, 96: 0.4, 128: 0.45}
            for step, loss in losses.items():
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
                (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"log_history": [{"step": step, "eval_loss": loss}]}),
                    encoding="utf-8",
                )
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "selection_source": "refresh_eval",
                        "selection_policy": "lowest_refresh_eval_loss_then_earliest_step",
                        "eligible_checkpoint_steps": [32, 64, 96, 128],
                        "external_gate_metrics_allowed": False,
                    }
                ),
                encoding="utf-8",
            )
            result = select_refresh_checkpoint(root, policy)
            self.assertEqual(result["selected_step"], 64)
            self.assertEqual(result["selected_eval_loss"], 0.4)
            self.assertFalse(result["external_gate_metrics_used"])

    def test_selector_requires_all_four_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "selection_source": "refresh_eval",
                        "selection_policy": "lowest_refresh_eval_loss_then_earliest_step",
                        "eligible_checkpoint_steps": [32, 64, 96, 128],
                        "external_gate_metrics_allowed": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                select_refresh_checkpoint(root, policy)


if __name__ == "__main__":
    unittest.main()

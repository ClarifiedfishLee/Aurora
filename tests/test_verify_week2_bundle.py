import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.verify_week2_bundle import EXPECTED_COUNTS, audit_bundle, verify_manifest, write_manifest


class VerifyWeek2BundleTest(unittest.TestCase):
    def test_manifest_round_trip_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            artifact = root / "nested/adapter.bin"
            artifact.write_bytes(b"adapter")
            manifest = root / "checksums.sha256"

            self.assertEqual(write_manifest(root, manifest), 1)
            self.assertEqual(verify_manifest(root, manifest), [])
            artifact.write_bytes(b"changed")
            self.assertEqual(verify_manifest(root, manifest), ["checksum mismatch: nested/adapter.bin"])

    def test_manifest_is_sorted_and_excludes_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z.txt").write_text("z", encoding="utf-8")
            (root / "a.txt").write_text("a", encoding="utf-8")
            manifest = root / "checksums.sha256"
            write_manifest(root, manifest)

            lines = manifest.read_text(encoding="utf-8").splitlines()
            self.assertEqual([line.split("  ", 1)[1] for line in lines], ["a.txt", "z.txt"])
            self.assertNotIn("checksums.sha256", manifest.read_text(encoding="utf-8"))

    def test_verifier_rejects_unsafe_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = hashlib.sha256(b"x").hexdigest()
            manifest = root / "bad.sha256"
            manifest.write_text(f"{digest}  ../outside\n", encoding="utf-8")
            self.assertEqual(verify_manifest(root, manifest), ["unsafe manifest path: '../outside'"])

    def test_audit_accepts_complete_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "lora-final-12597"
            best = root / "lora-final-12597-best-eval"
            metadata = root / "metadata"
            gate = root / "day14_gate"
            checkpoint = final / "checkpoint-10"
            for path in (final, best, metadata, gate, checkpoint):
                path.mkdir(parents=True, exist_ok=True)
            for relative in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "train.log",
                "trainer_log.jsonl",
                "trainer_state.json",
            ):
                (final / relative).write_text("x", encoding="utf-8")
            (final / "exit_code").write_text("0\n", encoding="utf-8")
            for relative in (
                "adapter_config.json",
                "adapter_model.safetensors",
            ):
                (best / relative).write_text("x", encoding="utf-8")
            (best / "best_eval.json").write_text(
                json.dumps({"best_step": 10, "eval_loss": 0.1}), encoding="utf-8"
            )
            for relative in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "trainer_state.json",
            ):
                (checkpoint / relative).write_text("x", encoding="utf-8")
            (metadata / "train_final_12597.yaml").write_text("model: x\n", encoding="utf-8")
            (metadata / "dataset_info.json").write_text("{}\n", encoding="utf-8")
            (metadata / "sft_final_12597_v2_summary.json").write_text("{}\n", encoding="utf-8")
            train_rows = [{"videos": ["train.mp4"]}]
            eval_rows = [{"videos": ["eval.mp4"]}]
            all_rows = train_rows + eval_rows
            for name, rows in (
                ("sft_final_12597_v2.jsonl", all_rows),
                ("sft_final_12597_v2_train.jsonl", train_rows),
                ("sft_final_12597_v2_eval.jsonl", eval_rows),
            ):
                (metadata / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            records = [{"bench_id": "case"}]
            (gate / "agent_pipeline_records.jsonl").write_text(
                json.dumps(records[0]) + "\n", encoding="utf-8"
            )
            (gate / "planner.log").write_text("planner\n", encoding="utf-8")
            (gate / "scorer.log").write_text("scorer\n", encoding="utf-8")
            gate_metrics = {
                "num_cases": 100,
                "json_validity": 1.0,
                "strict_raw_json_validity": 1.0,
                "subtask_accuracy": 1.0,
                "image_search_trigger": {},
                "image_search_query": {},
                "mask_trigger": {},
                "constraint_retention": 1.0,
                "constraint_case_accuracy": 1.0,
                "source_entity_false_trigger": {},
                "details": {},
                "by_axis": {},
            }
            (gate / "metrics.json").write_text(json.dumps(gate_metrics), encoding="utf-8")
            (gate / "gate_summary.json").write_text(
                json.dumps({"passed": True, "checks": {}, "diagnostics": {}}),
                encoding="utf-8",
            )
            expected = {
                "metadata/sft_final_12597_v2.jsonl": 2,
                "metadata/sft_final_12597_v2_train.jsonl": 1,
                "metadata/sft_final_12597_v2_eval.jsonl": 1,
                "day14_gate/agent_pipeline_records.jsonl": 1,
            }
            with patch.dict(EXPECTED_COUNTS, expected, clear=True):
                result = audit_bundle(root)
            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(result["observations"]["video_overlap"], 0)

            (gate / "gate_summary.json").write_text(
                json.dumps({"passed": False, "checks": {}, "diagnostics": {}}),
                encoding="utf-8",
            )
            with patch.dict(EXPECTED_COUNTS, expected, clear=True):
                failed = audit_bundle(root)
            self.assertFalse(failed["ok"])
            self.assertIn(
                "Day-14 gate_summary.json does not record a passing gate",
                failed["errors"],
            )


if __name__ == "__main__":
    unittest.main()

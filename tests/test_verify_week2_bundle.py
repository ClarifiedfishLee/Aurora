import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.verify_week2_bundle import (
    EXPECTED_COUNTS,
    audit_bundle,
    audit_refresh_bundle,
    verify_manifest,
    write_manifest,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_refresh_fixture(root: Path) -> None:
    output = root / "lora-refresh"
    selected = root / "lora-refresh-selected"
    metadata = root / "metadata"
    gate = root / "day14_gate"
    for path in (output, selected, metadata, gate):
        path.mkdir(parents=True, exist_ok=True)

    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "train.log",
        "trainer_log.jsonl",
        "trainer_state.json",
    ):
        (output / name).write_text(
            "{}\n" if name.endswith(".json") else "x\n", encoding="utf-8"
        )
    (output / "exit_code").write_text("0\n", encoding="utf-8")

    losses = {32: 0.4, 64: 0.3, 96: 0.2, 128: 0.25}
    candidates = []
    for step, loss in losses.items():
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir()
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ):
            (checkpoint / name).write_bytes(f"{name}-{step}".encode())
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"log_history": [{"step": step, "eval_loss": loss}]}),
            encoding="utf-8",
        )
        candidates.append(
            {
                "step": step,
                "eval_loss": loss,
                "checkpoint": f"/worker/run/checkpoint-{step}",
            }
        )

    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (selected / name).write_bytes((output / "checkpoint-96" / name).read_bytes())

    train_rows = [{"videos": ["/videos/train.mp4"]} for _ in range(1024)]
    eval_rows = [{"videos": ["/videos/eval.mp4"]} for _ in range(256)]
    cases = [{"bench_id": f"refresh-{index:04d}"} for index in range(256)]
    gold = [{"bench_id": f"refresh-{index:04d}"} for index in range(256)]
    for name, rows in (
        ("refresh_train.jsonl", train_rows),
        ("refresh_eval.jsonl", eval_rows),
        ("refresh_cases.jsonl", cases),
        ("refresh_gold.jsonl", gold),
    ):
        _write_jsonl(metadata / name, rows)

    generation_audit = {
        "counts": {
            "train": 1024,
            "validation": 256,
            "validation_cases": 256,
            "validation_gold": 256,
        },
        "isolation": {
            "video_overlap": 0,
            "normalized_prompt_overlap": 0,
            "concept_overlap": 0,
            "template_overlap": 0,
        },
        "forbidden_audit": {"exact_prompt_overlap": 0, "phrase_hit_count": 0},
    }
    (metadata / "refresh_generation_audit.json").write_text(
        json.dumps(generation_audit), encoding="utf-8"
    )
    (metadata / "train_refresh.yaml").write_text(
        """dataset: aurora_planner_refresh_train
eval_dataset: aurora_planner_refresh_eval
adapter_name_or_path: /worker/lora-final-12597-best-eval
create_new_adapter: false
max_samples: 1024
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
num_train_epochs: 1.0
warmup_ratio: 0.05
save_steps: 32
eval_steps: 32
save_total_limit: 4
overwrite_output_dir: true
save_only_model: false
""",
        encoding="utf-8",
    )
    selection_policy = {
        "selection_source": "refresh_eval",
        "selection_metric": "eval_loss",
        "lower_is_better": True,
        "selection_policy": "lowest_refresh_eval_loss_then_earliest_step",
        "tie_breaker": "earliest_step",
        "eligible_checkpoint_steps": [32, 64, 96, 128],
        "external_gate_metrics_allowed": False,
        "expected_checkpoint_count": 4,
        "train_rows": 1024,
        "eval_rows": 256,
        "train_sha256": hashlib.sha256(
            (metadata / "refresh_train.jsonl").read_bytes()
        ).hexdigest(),
        "eval_sha256": hashlib.sha256(
            (metadata / "refresh_eval.jsonl").read_bytes()
        ).hexdigest(),
    }
    (metadata / "refresh_selection_policy.json").write_text(
        json.dumps(selection_policy), encoding="utf-8"
    )
    selection_result = {
        "selection_source": "refresh_eval",
        "selection_metric": "eval_loss",
        "selection_policy": "lowest_refresh_eval_loss_then_earliest_step",
        "external_gate_metrics_used": False,
        "selected_step": 96,
        "selected_eval_loss": 0.2,
        "selected_checkpoint": "/worker/run/checkpoint-96",
        "candidates": candidates,
    }
    (metadata / "refresh_selection.json").write_text(
        json.dumps(selection_result), encoding="utf-8"
    )

    _write_jsonl(gate / "agent_pipeline_records.jsonl", [{"bench_id": "case"}] * 100)
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
        json.dumps({"passed": True, "checks": {}, "diagnostics": {}}), encoding="utf-8"
    )
    write_manifest(root, root / "checksums.sha256")


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

    def test_refresh_audit_accepts_complete_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_refresh_fixture(root)

            result = audit_refresh_bundle(root)

            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(result["observations"]["training_exit_code"], 0)
            self.assertEqual(result["observations"]["video_overlap"], 0)
            self.assertEqual(result["observations"]["refresh_selection"]["selected_step"], 96)
            self.assertTrue(result["observations"]["day14_gate_passed"])

    def test_refresh_audit_rejects_wrong_selected_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_refresh_fixture(root)
            (root / "lora-refresh-selected/adapter_model.safetensors").write_bytes(b"wrong")
            write_manifest(root, root / "checksums.sha256")

            result = audit_refresh_bundle(root)

            self.assertFalse(result["ok"])
            self.assertIn(
                "selected adapter adapter_model.safetensors does not match checkpoint-96",
                result["errors"],
            )

    def test_refresh_audit_requires_complete_manifest_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_refresh_fixture(root)
            manifest = root / "checksums.sha256"
            lines = manifest.read_text(encoding="utf-8").splitlines()
            manifest.write_text(
                "\n".join(
                    line for line in lines if not line.endswith("  metadata/refresh_gold.jsonl")
                )
                + "\n",
                encoding="utf-8",
            )

            result = audit_refresh_bundle(root)

            self.assertFalse(result["ok"])
            self.assertTrue(
                any("checksum manifest omits 1 bundle file" in error for error in result["errors"]),
                result["errors"],
            )


if __name__ == "__main__":
    unittest.main()

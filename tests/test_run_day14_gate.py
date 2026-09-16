import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_day14_gate import (
    GateValidationError,
    build_agent_command,
    evaluate_gate,
    validate_model_paths,
    validate_predictions,
    verify_model_log,
)


class Day14GateRunnerTest(unittest.TestCase):
    def test_model_paths_require_nonempty_base_and_adapter_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            adapter = root / "adapter"
            base.mkdir()
            adapter.mkdir()
            (base / "config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")

            self.assertEqual(validate_model_paths(base, adapter), (base.resolve(), adapter.resolve()))
            (adapter / "adapter_model.safetensors").write_bytes(b"")
            with self.assertRaisesRegex(GateValidationError, "adapter weights"):
                validate_model_paths(base, adapter)

    def test_prediction_validation_requires_exact_unique_error_free_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(
                json.dumps({"bench_id": "a"}) + "\n" + json.dumps({"bench_id": "b"}) + "\n",
                encoding="utf-8",
            )
            validate_predictions(path, {"a", "b"}, 2)

            path.write_text(
                json.dumps({"bench_id": "a"}) + "\n" + json.dumps({"bench_id": "a"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GateValidationError, "duplicate bench_id"):
                validate_predictions(path, {"a", "b"}, 2)

            path.write_text(
                json.dumps({"bench_id": "a"})
                + "\n"
                + json.dumps({"bench_id": "b", "error": "OOM"})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GateValidationError, "recorded errors"):
                validate_predictions(path, {"a", "b"}, 2)

    def test_model_log_must_confirm_exact_requested_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "planner.log"
            base = root / "base"
            adapter = root / "best-eval"
            log.write_text(
                f"Agent base: {base}\nAgent adapter: /models/released-adapter\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GateValidationError, "requested model paths"):
                verify_model_log(log, base, adapter)

            log.write_text(f"Agent base: {base}\nAgent adapter: {adapter}\n", encoding="utf-8")
            verify_model_log(log, base, adapter)

    def test_agent_command_preserves_planner_decisions_without_tools(self) -> None:
        command = build_agent_command(
            Path("/venv/python"),
            Path("/repo/gold.jsonl"),
            Path("/models/base"),
            Path("/models/best-eval"),
            Path("/tmp/gate"),
            "cuda:0",
        )
        self.assertIn("--custom_only", command)
        self.assertIn("--plan_only", command)
        self.assertEqual(command[command.index("--mask_backend") + 1], "none")
        self.assertEqual(command[command.index("--agent_adapter") + 1], "/models/best-eval")

    def test_gate_uses_preregistered_thresholds_and_released_comparison(self) -> None:
        metrics = {
            "json_validity": 0.99,
            "strict_raw_json_validity": 0.98,
            "subtask_accuracy": 0.95,
            "image_search_trigger": {"f1": 0.80},
            "image_search_query": {"conditional_accuracy": 0.9, "end_to_end_recall": 0.8},
            "mask_trigger": {"f1": 0.95},
            "constraint_retention": 0.80,
            "constraint_case_accuracy": 0.65,
            "source_entity_false_trigger": {"rate": 0.05},
        }
        released = {"subtask_accuracy": 0.78, "constraint_retention": 0.48}
        summary = evaluate_gate(metrics, released)

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["diagnostics"]["strict_raw_json_validity"], 0.98)
        metrics["source_entity_false_trigger"]["rate"] = 0.051
        self.assertFalse(evaluate_gate(metrics, released)["passed"])


if __name__ == "__main__":
    unittest.main()

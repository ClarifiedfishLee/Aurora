import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_planner_eval import (
    PlannerEvalError,
    build_agent_command,
    run_evaluation,
    validate_inputs,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class FakeExecutor:
    def __init__(
        self,
        ids: list[str],
        *,
        agent_returncode: int = 0,
        scorer_returncode: int = 0,
        log_adapter_override: Path | None = None,
        duplicate_adapter_declaration: bool = False,
        error_id: str | None = None,
        mutate_predictions_during_score: bool = False,
        mutate_cases_during_score: Path | None = None,
        raise_during_score: bool = False,
        metrics_num_cases: object | None = None,
    ) -> None:
        self.ids = ids
        self.agent_returncode = agent_returncode
        self.scorer_returncode = scorer_returncode
        self.log_adapter_override = log_adapter_override
        self.duplicate_adapter_declaration = duplicate_adapter_declaration
        self.error_id = error_id
        self.mutate_predictions_during_score = mutate_predictions_during_score
        self.mutate_cases_during_score = mutate_cases_during_score
        self.raise_during_score = raise_during_score
        self.metrics_num_cases = metrics_num_cases
        self.calls: list[list[str]] = []

    @staticmethod
    def _value(command: list[str], flag: str) -> str:
        return command[command.index(flag) + 1]

    def __call__(self, command: list[str], log_path: Path, echo: bool) -> int:
        self.calls.append(command)
        module = command[command.index("-m") + 1]
        if module == "aurora.agent":
            base = Path(self._value(command, "--agent_base"))
            adapter = self.log_adapter_override or Path(
                self._value(command, "--agent_adapter")
            )
            log_text = (
                f"Loaded {len(self.ids)} cases\n"
                f"Agent base: {base}\n"
                f"Agent adapter: {adapter}\n"
            )
            if self.duplicate_adapter_declaration:
                requested = Path(self._value(command, "--agent_adapter"))
                log_text += f"Agent adapter: {requested}\n"
            log_path.write_text(log_text, encoding="utf-8")
            if self.agent_returncode == 0:
                out_dir = Path(self._value(command, "--out_dir"))
                rows = []
                for bench_id in self.ids:
                    row = {"bench_id": bench_id}
                    if bench_id == self.error_id:
                        row["error"] = "synthetic failure"
                    rows.append(row)
                _write_jsonl(out_dir / "agent_pipeline_records.jsonl", rows)
            return self.agent_returncode

        self.assert_scorer_command(module)
        if self.raise_during_score:
            raise RuntimeError("synthetic scorer crash")
        log_path.write_text("scored\n", encoding="utf-8")
        if self.mutate_predictions_during_score:
            predictions = Path(self._value(command, "--predictions"))
            predictions.write_text(
                predictions.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
        if self.mutate_cases_during_score is not None:
            self.mutate_cases_during_score.write_text(
                self.mutate_cases_during_score.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
        if self.scorer_returncode == 0:
            metrics_path = Path(self._value(command, "--out"))
            num_cases = (
                len(self.ids)
                if self.metrics_num_cases is None
                else self.metrics_num_cases
            )
            metrics_path.write_text(
                json.dumps({"num_cases": num_cases, "json_validity": 1.0}) + "\n",
                encoding="utf-8",
            )
        return self.scorer_returncode

    @staticmethod
    def assert_scorer_command(module: str) -> None:
        if module != "evaluation.agent_only_score":
            raise AssertionError(f"unexpected module: {module}")


class PlannerEvalRunnerTest(unittest.TestCase):
    def _fixture(self, root: Path, count: int = 2) -> dict[str, Path | list[str]]:
        root.mkdir(parents=True, exist_ok=True)
        base = root / "base"
        adapter = root / "adapter"
        base.mkdir()
        adapter.mkdir()
        (base / "config.json").write_text('{"model_type":"qwen3_vl"}\n')
        (adapter / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}\n', encoding="utf-8"
        )
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
        adapter_config_sha = _sha256(adapter / "adapter_config.json")
        adapter_model_sha = _sha256(adapter / "adapter_model.safetensors")
        (adapter / "interpolation_provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "method": "exact_delta_space_rank_concat",
                    "coefficient": {"lambda_b": 0.5},
                    "output": {
                        "adapter_config_sha256": adapter_config_sha,
                        "adapter_model_sha256": adapter_model_sha,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        ids = [f"case-{index}" for index in range(count)]
        videos = []
        for item in ids:
            video = root / f"{item}.mp4"
            video.write_bytes(f"video:{item}".encode())
            videos.append(video)
        cases = root / "cases.jsonl"
        gold = root / "gold.jsonl"
        case_rows = [
            {"bench_id": item, "prompt": "edit", "video_path": str(video)}
            for item, video in zip(ids, videos)
        ]
        _write_jsonl(cases, case_rows)
        _write_jsonl(
            gold,
            [
                {
                    "bench_id": item,
                    "prompt": "edit",
                    "video_path": str(video),
                    "gold_plan": {
                        "refined_text_instruction": "edit",
                        "subtask": "global_style",
                        "image_search": False,
                        "mask": False,
                    },
                }
                for item, video in zip(ids, videos)
            ],
        )
        return {
            "base": base,
            "adapter": adapter,
            "cases": cases,
            "gold": gold,
            "ids": ids,
        }

    def test_success_writes_hashed_manifest_and_runs_generic_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            out_dir = root / "result"
            executor = FakeExecutor(fixture["ids"])

            manifest = run_evaluation(
                base=fixture["base"],
                adapter=fixture["adapter"],
                cases=fixture["cases"],
                gold=fixture["gold"],
                out_dir=out_dir,
                expected_cases=2,
                device="cuda:7",
                executor=executor,
            )

            stored = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertEqual(stored, manifest)
            self.assertEqual(stored["status"], "complete")
            self.assertIsNone(stored["error"])
            self.assertEqual(stored["exit_state"]["agent_returncode"], 0)
            self.assertEqual(stored["exit_state"]["scorer_returncode"], 0)
            self.assertTrue(stored["exit_state"]["predictions_validated"])
            self.assertTrue(stored["exit_state"]["metrics_validated"])
            self.assertTrue(stored["exit_state"]["inputs_reverified"])
            self.assertTrue(stored["exit_state"]["outputs_reverified"])
            self.assertEqual(
                stored["inputs"]["cases"]["sha256"], _sha256(fixture["cases"])
            )
            self.assertEqual(
                stored["inputs"]["adapter_weights"]["sha256"],
                _sha256(fixture["adapter"] / "adapter_model.safetensors"),
            )
            self.assertEqual(
                stored["inputs"]["adapter_provenance"]["sha256"],
                _sha256(fixture["adapter"] / "interpolation_provenance.json"),
            )
            self.assertEqual(len(executor.calls), 2)
            agent = executor.calls[0]
            self.assertIn("--plan_only", agent)
            self.assertIn("--custom_only", agent)
            self.assertEqual(agent[agent.index("--mask_backend") + 1], "none")
            self.assertEqual(agent[agent.index("--device") + 1], "cuda:7")
            self.assertEqual(
                executor.calls[1][executor.calls[1].index("-m") + 1],
                "evaluation.agent_only_score",
            )

            with self.assertRaisesRegex(PlannerEvalError, "must not already exist"):
                run_evaluation(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=out_dir,
                    expected_cases=2,
                    device="cuda:0",
                    executor=executor,
                )

    def test_preflight_rejects_mismatched_duplicate_and_missing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            out_dir = root / "result"
            _write_jsonl(
                fixture["gold"],
                [
                    {"bench_id": "case-0"},
                    {"bench_id": "different"},
                ],
            )
            with self.assertRaisesRegex(PlannerEvalError, "bench_id mismatch"):
                validate_inputs(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=out_dir,
                    expected_cases=2,
                )
            self.assertFalse(out_dir.exists())

            _write_jsonl(
                fixture["gold"],
                [{"bench_id": "case-0"}, {"bench_id": "case-0"}],
            )
            with self.assertRaisesRegex(PlannerEvalError, "duplicate bench_id"):
                validate_inputs(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=out_dir,
                    expected_cases=2,
                )

            (fixture["adapter"] / "adapter_model.safetensors").write_bytes(b"")
            with self.assertRaisesRegex(PlannerEvalError, "adapter weights"):
                validate_inputs(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=out_dir,
                    expected_cases=2,
                )

    def test_per_case_error_fails_before_scorer_and_preserves_failure_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            out_dir = root / "result"
            executor = FakeExecutor(fixture["ids"], error_id="case-1")

            with self.assertRaisesRegex(PlannerEvalError, "recorded errors"):
                run_evaluation(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=out_dir,
                    expected_cases=2,
                    device="cuda:0",
                    executor=executor,
                )

            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["exit_state"]["agent_returncode"], 0)
            self.assertFalse(manifest["exit_state"]["predictions_validated"])
            self.assertIsNone(manifest["exit_state"]["scorer_returncode"])
            self.assertEqual(len(executor.calls), 1)

    def test_preflight_rejects_semantic_mispairing_and_bad_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            out_dir = root / "result"
            gold_rows = [
                json.loads(line)
                for line in fixture["gold"].read_text(encoding="utf-8").splitlines()
            ]
            gold_rows[0]["prompt"] = "a different edit"
            _write_jsonl(fixture["gold"], gold_rows)
            with self.assertRaisesRegex(PlannerEvalError, "prompt mismatch"):
                validate_inputs(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=out_dir,
                    expected_cases=2,
                )
            self.assertFalse(out_dir.exists())

            fixture = self._fixture(root / "second")
            provenance_path = fixture["adapter"] / "interpolation_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["output"]["adapter_model_sha256"] = "0" * 64
            provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                PlannerEvalError, "weights hash differs from interpolation provenance"
            ):
                validate_inputs(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=root / "second-result",
                    expected_cases=2,
                )

    def test_duplicate_model_declaration_cannot_mask_adapter_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            out_dir = root / "result"
            executor = FakeExecutor(
                fixture["ids"],
                log_adapter_override=Path("/models/released-adapter"),
                duplicate_adapter_declaration=True,
            )
            with self.assertRaisesRegex(
                PlannerEvalError, "uniquely confirm the exact requested model paths"
            ):
                run_evaluation(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=out_dir,
                    expected_cases=2,
                    device="cuda:0",
                    executor=executor,
                )
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(len(executor.calls), 1)

    def test_scorer_side_prediction_mutation_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            out_dir = root / "result"
            executor = FakeExecutor(
                fixture["ids"], mutate_predictions_during_score=True
            )
            with self.assertRaisesRegex(
                PlannerEvalError, "validated evaluation outputs changed"
            ):
                run_evaluation(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=out_dir,
                    expected_cases=2,
                    device="cuda:0",
                    executor=executor,
                )
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue(manifest["exit_state"]["metrics_validated"])
            self.assertFalse(manifest["exit_state"]["outputs_reverified"])

    def test_input_mutation_and_unexpected_runtime_error_leave_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            out_dir = root / "mutated"
            executor = FakeExecutor(
                fixture["ids"], mutate_cases_during_score=fixture["cases"]
            )
            with self.assertRaisesRegex(PlannerEvalError, "inputs changed"):
                run_evaluation(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=out_dir,
                    expected_cases=2,
                    device="cuda:0",
                    executor=executor,
                )
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse(manifest["exit_state"]["inputs_reverified"])

            fixture = self._fixture(root / "second")
            out_dir = root / "crashed"
            executor = FakeExecutor(fixture["ids"], raise_during_score=True)
            with self.assertRaisesRegex(PlannerEvalError, "synthetic scorer crash"):
                run_evaluation(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=out_dir,
                    expected_cases=2,
                    device="cuda:0",
                    executor=executor,
                )
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("RuntimeError", manifest["error"])

    def test_boolean_metrics_count_is_not_accepted_as_integer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root, count=1)
            out_dir = root / "result"
            executor = FakeExecutor(fixture["ids"], metrics_num_cases=True)
            with self.assertRaisesRegex(PlannerEvalError, "expected integer 1"):
                run_evaluation(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=out_dir,
                    expected_cases=1,
                    device="cuda:0",
                    executor=executor,
                )
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse(manifest["exit_state"]["metrics_validated"])

    def test_log_path_mismatch_detects_adapter_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            out_dir = root / "result"
            executor = FakeExecutor(
                fixture["ids"], log_adapter_override=Path("/models/released-adapter")
            )
            with self.assertRaisesRegex(
                PlannerEvalError, "exact requested model paths"
            ):
                run_evaluation(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=out_dir,
                    expected_cases=2,
                    device="cuda:0",
                    executor=executor,
                )
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(len(executor.calls), 1)

    def test_nonzero_planner_exit_is_recorded_without_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            out_dir = root / "result"
            executor = FakeExecutor(fixture["ids"], agent_returncode=17)
            with self.assertRaisesRegex(PlannerEvalError, "status 17"):
                run_evaluation(
                    base=fixture["base"],
                    adapter=fixture["adapter"],
                    cases=fixture["cases"],
                    gold=fixture["gold"],
                    out_dir=out_dir,
                    expected_cases=2,
                    device="cuda:0",
                    executor=executor,
                )
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["exit_state"]["agent_returncode"], 17)
            self.assertEqual(manifest["exit_state"]["stage"], "agent")
            self.assertEqual(len(executor.calls), 1)

    def test_command_contains_no_gate_or_tool_execution_flags(self) -> None:
        command = build_agent_command(
            Path("/venv/python"),
            Path("/data/cases.jsonl"),
            Path("/models/base"),
            Path("/models/adapter"),
            Path("/runs/candidate"),
            "cuda:0",
        )
        self.assertIn("--plan_only", command)
        self.assertEqual(command[command.index("--mask_backend") + 1], "none")
        self.assertNotIn("day14", " ".join(command).lower())


if __name__ == "__main__":
    unittest.main()

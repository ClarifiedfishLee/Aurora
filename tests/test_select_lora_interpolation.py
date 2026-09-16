import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.select_lora_interpolation import (
    INTERPOLATION_METHOD,
    SelectionValidationError,
    evaluate_candidates,
    main,
    paired_stratified_bootstrap,
    sha256_file,
)

GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
CATEGORY_COUNTS = {
    "no_search_negative": 128,
    "true_search_positive": 64,
    "routing_control": 64,
    "mask_control": 64,
    "rewrite_retention": 64,
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def lambda_slug(value: float) -> str:
    return f"lambda_{round(value * 1000):04d}"


class SelectorFixture:
    def __init__(self, root: Path, *, primary: bool = False) -> None:
        self.root = root
        self.cases_path = root / "validation/cases.jsonl"
        self.gold_path = root / "validation/gold.jsonl"
        self.audit_path = root / "validation/audit.json"
        self.policy_path = root / "validation/policy.json"
        self.candidates_root = root / "candidates"
        self.base_model = root / "models/Qwen3-VL-8B-Instruct"
        write_json(self.base_model / "config.json", {"model_type": "qwen3_vl"})
        self.endpoint_a = "1" * 64
        self.endpoint_b = "2" * 64
        self.primary_reference_adapter: Path | None = None
        self.cases, self.gold = self._make_validation_rows()
        write_jsonl(self.cases_path, self.cases)
        write_jsonl(self.gold_path, self.gold)
        self.input_hashes = {
            "source_manifest": "3" * 64,
            "v2_train": "4" * 64,
            "v2_eval": "5" * 64,
            "refresh1_train": "6" * 64,
            "refresh1_eval": "7" * 64,
            "day14_forbidden_cases": "8" * 64,
        }
        audit = {
            "recipe_version": 2,
            "counts": {
                "cases": 384,
                "gold": 384,
                "by_category": dict(sorted(CATEGORY_COUNTS.items())),
                "no_search_by_subtype": {
                    "generic_background": 32,
                    "generic_style": 64,
                    "ordinary_target": 32,
                },
                "mask_trigger": {"not_triggered": 32, "triggered": 32},
            },
            "isolation": {
                "prior_identifier_overlap": 0,
                "prior_basename_overlap": 0,
                "prior_video_sha256_overlap": 0,
                "prior_exact_prompt_overlap": 0,
                "prior_concept_overlap": 0,
                "prior_template_id_overlap": 0,
                "prior_template_signature_overlap": 0,
                "prior_constraint_value_overlap": 0,
                "prior_concept_phrase_hit_count": 0,
                "prior_constraint_phrase_hit_count": 0,
                "allowed_generic_concept_phrase_hit_count": 0,
                "allowed_generic_template_signature_hit_count": 0,
                "allowed_generic_concept_phrase_hits": [],
                "allowed_generic_template_signature_hits": [],
                "explicit_forbidden_phrase_hits": [],
            },
            "artifacts": {
                "input_sha256": self.input_hashes,
                "cases_sha256": sha256_file(self.cases_path),
                "gold_sha256": sha256_file(self.gold_path),
            },
        }
        write_json(self.audit_path, audit)
        self.policy = self._make_policy()
        if primary:
            primary_model = self._write_primary_candidate()
            self.policy["final_decision"] = {
                "rule": "primary_if_eligible_else_grid",
                "primary_candidate": {
                    "candidate_id": "recipe2",
                    "directory_name": "recipe2",
                    "adapter_model_sha256": primary_model,
                },
            }
        write_json(self.policy_path, self.policy)
        for value in GRID:
            if value == 0.5:
                false_search, bad_rewrite = 0, 8
            elif value == 0.0:
                false_search, bad_rewrite = 4, 8
            else:
                false_search, bad_rewrite = 6, 9
            self._write_grid_candidate(
                value,
                self._predictions(
                    false_search=false_search, bad_rewrite=bad_rewrite
                ),
            )

    def _make_validation_rows(self) -> tuple[list[dict], list[dict]]:
        cases: list[dict] = []
        gold: list[dict] = []
        categories = [
            category
            for category, count in CATEGORY_COUNTS.items()
            for _ in range(count)
        ]
        category_seen: dict[str, int] = {category: 0 for category in CATEGORY_COUNTS}
        for index, category in enumerate(categories, 1):
            category_seen[category] += 1
            local = category_seen[category]
            bench_id = f"interp_{index:04d}"
            constraints: list[dict] = []
            if category == "true_search_positive":
                subtask = "replace_object"
                image_search: str | bool = f"named product {local}"
                mask: str | bool = "source object"
                refined = f"Replace the source object with named product {local}."
                axis = "search"
                subtype = "named_product"
            elif category == "routing_control":
                subtask = "remove_object"
                image_search = False
                mask = "source object"
                refined = f"Remove source object {local}."
                axis = "routing"
                subtype = "remove_object"
            elif category == "mask_control":
                subtask = "change_color"
                image_search = False
                mask = "source object" if local <= 32 else False
                refined = f"Change source object {local} to blue."
                axis = "mask"
                subtype = "triggered" if local <= 32 else "not_triggered"
            elif category == "rewrite_retention":
                subtask = "replace_object"
                image_search = False
                mask = "source object"
                values = [
                    f"cobalt-{local}",
                    f"item-{local}",
                    f"left-{local}",
                    f"keep-{local}",
                ]
                constraints = [
                    {"type": kind, "value": value}
                    for kind, value in zip(
                        ("color", "identity", "spatial", "preservation"), values
                    )
                ]
                refined = "Apply " + " ".join(values) + "."
                axis = "rewrite"
                subtype = "four_constraints"
            else:
                subtask = "global_style"
                image_search = False
                mask = False
                refined = f"Apply generic style {local}."
                axis = "search"
                subtype = (
                    "generic_style"
                    if local <= 64
                    else "generic_background"
                    if local <= 96
                    else "ordinary_target"
                )
            plan = {
                "refined_text_instruction": refined,
                "subtask": subtask,
                "image_search": image_search,
                "mask": mask,
            }
            source = {
                "sample_id": f"source-{index}",
                "video_sha256": hashlib.sha256(f"video-{index}".encode()).hexdigest(),
            }
            case = {
                "bench_id": bench_id,
                "video_path": f"/fresh/video-{index}.mp4",
                "prompt": refined,
                "edit_type": subtask,
                "axis": axis,
                "category": category,
                "subtype": subtype,
                "source": source,
                "catalog": {
                    "concept_id": f"concept-{index}",
                    "template_id": f"template-{index}",
                    "template_signature": f"signature-{index}",
                },
            }
            gold_row = {
                **case,
                "gold_plan": plan,
                "constraints": constraints,
                "source_entities": [] if image_search else ["source object"],
            }
            if image_search:
                gold_row["search_query_aliases"] = [image_search]
            cases.append(case)
            gold.append(gold_row)
        return cases, gold

    def _make_policy(self) -> dict:
        return {
            "version": 2,
            "created_before_candidate_inference": True,
            "candidate_rule": {
                "method": "exact_parameter_delta_interpolation",
                "lambda_grid": GRID,
                "candidate_count": 9,
                "v2_adapter_model_sha256": self.endpoint_a,
                "refresh1_adapter_model_sha256": self.endpoint_b,
                "additional_training_allowed": False,
                "day14_outputs_or_metrics_allowed_for_generation_or_selection": False,
            },
            "validation_artifacts": {
                "cases_sha256": sha256_file(self.cases_path),
                "gold_sha256": sha256_file(self.gold_path),
                "leakage_audit_sha256": sha256_file(self.audit_path),
                "input_sha256": self.input_hashes,
                "expected_cases": 384,
                "expected_unique_videos": 384,
                "expected_category_counts": CATEGORY_COUNTS,
                "expected_no_search_subtype_counts": {
                    "generic_style": 64,
                    "generic_background": 32,
                    "ordinary_target": 32,
                },
                "expected_mask_trigger_counts": {"triggered": 32, "not_triggered": 32},
            },
            "eligibility_thresholds": {
                "complete_prediction_rows": 384,
                "strict_raw_json_validity_min": 1.0,
                "subtask_accuracy_min": 0.95,
                "no_search_specificity_min": 0.95,
                "true_search_trigger_recall_min": 0.95,
                "search_query_end_to_end_recall_min": 0.95,
                "mask_trigger_f1_min": 0.95,
                "rewrite_constraint_retention_min": 0.85,
            },
            "eligibility_threshold_scopes": {},
            "utility": {
                "formula": "0.5*no_search_specificity+0.5*rewrite_constraint_retention",
                "higher_is_better": True,
            },
            "bootstrap": {
                "method": "paired stratified case bootstrap over the two utility strata",
                "seed": 20260916,
                "draws": 200,
                "strata": {"no_search_negative": 128, "rewrite_retention": 64},
            },
            "selection": {
                "rule": "one_standard_error_then_smallest_lambda",
                "reference_tie_breaker": "smallest lambda",
                "tie_breaker": "smallest lambda",
                "day14_metrics_used": False,
            },
        }

    def _predictions(
        self, *, false_search: int = 0, bad_rewrite: int = 0
    ) -> list[dict]:
        result: list[dict] = []
        negative_seen = rewrite_seen = 0
        for row in self.gold:
            plan = copy.deepcopy(row["gold_plan"])
            if row["category"] == "no_search_negative":
                negative_seen += 1
                if negative_seen <= false_search:
                    plan["image_search"] = "unnecessary query"
            if row["category"] == "rewrite_retention":
                rewrite_seen += 1
                if rewrite_seen <= bad_rewrite:
                    plan["refined_text_instruction"] = "Edit the source object."
            result.append(
                {
                    "bench_id": row["bench_id"],
                    "plan": plan,
                    "agent_raw": json.dumps(plan, separators=(",", ":")),
                }
            )
        return result

    def _write_adapter(self, adapter: Path, payload: bytes) -> tuple[str, str]:
        adapter.mkdir(parents=True)
        write_json(
            adapter / "adapter_config.json",
            {
                "fixture": True,
                "peft_type": "LORA",
                "r": 32,
                "lora_alpha": 64,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
                "base_model_name_or_path": str(self.base_model.resolve()),
            },
        )
        (adapter / "adapter_model.safetensors").write_bytes(payload)
        return (
            sha256_file(adapter / "adapter_config.json"),
            sha256_file(adapter / "adapter_model.safetensors"),
        )

    def _write_grid_candidate(self, value: float, predictions: list[dict]) -> None:
        candidate = self.candidates_root / lambda_slug(value)
        adapter = candidate / "adapter"
        config_sha, model_sha = self._write_adapter(
            adapter, f"weights-{value}".encode()
        )
        provenance = {
            "schema_version": 1,
            "method": INTERPOLATION_METHOD,
            "equation": "delta_out=(1-lambda_b)*delta_a+lambda_b*delta_b",
            "coefficient": {
                "lambda_b": value,
                "adapter_a_weight": 1.0 - value,
                "adapter_b_weight": value,
            },
            "sources": {
                "adapter_a": {
                    "adapter_config_sha256": "a" * 64,
                    "adapter_model_sha256": self.endpoint_a,
                },
                "adapter_b": {
                    "adapter_config_sha256": "b" * 64,
                    "adapter_model_sha256": self.endpoint_b,
                },
            },
            "lora": {
                "input_rank": 16,
                "input_lora_alpha": 32,
                "input_scaling": 2.0,
                "output_rank": 32,
                "output_lora_alpha": 64,
                "output_scaling": 2.0,
                "module_count": 1,
                "tensor_count": 2,
                "dtype": "float32",
            },
            "output": {
                "path": str(adapter.resolve()),
                "adapter_config_sha256": config_sha,
                "adapter_model_sha256": model_sha,
            },
        }
        write_json(adapter / "interpolation_provenance.json", provenance)
        evaluation = candidate / "eval"
        write_jsonl(evaluation / "agent_pipeline_records.jsonl", predictions)
        (evaluation / "planner.log").write_text(
            f"Agent base: {self.base_model.resolve()}\n"
            f"Agent adapter: {adapter.resolve()}\n",
            encoding="utf-8",
        )
        self._write_complete_run_manifest(evaluation, adapter)

    def _write_primary_candidate(self, *, ineligible: bool = False) -> str:
        candidate = self.candidates_root / "recipe2"
        adapter = candidate / "adapter"
        _, model_sha = self._write_adapter(adapter, b"recipe2-weights")
        self.primary_reference_adapter = (
            self.root / "training/lora-refresh-selected"
        )
        self._write_adapter(
            self.primary_reference_adapter, b"recipe2-weights"
        )
        predictions = self._predictions(bad_rewrite=20 if ineligible else 0)
        evaluation = candidate / "eval"
        write_jsonl(evaluation / "agent_pipeline_records.jsonl", predictions)
        (evaluation / "planner.log").write_text(
            f"Agent base: {self.base_model.resolve()}\n"
            f"Agent adapter: {adapter.resolve()}\n",
            encoding="utf-8",
        )
        self._write_complete_run_manifest(evaluation, adapter)
        return model_sha

    def _write_complete_run_manifest(self, evaluation: Path, adapter: Path) -> None:
        metrics = evaluation / "metrics.json"
        scorer_log = evaluation / "scorer.log"
        records = evaluation / "agent_pipeline_records.jsonl"
        planner_log = evaluation / "planner.log"
        write_json(metrics, {"num_cases": 384})
        scorer_log.write_text("scored 384 cases\n", encoding="utf-8")
        inputs = {
            "cases": {
                "path": str(self.cases_path.resolve()),
                "sha256": sha256_file(self.cases_path),
            },
            "gold": {
                "path": str(self.gold_path.resolve()),
                "sha256": sha256_file(self.gold_path),
            },
            "base_config": {
                "path": str((self.base_model / "config.json").resolve()),
                "sha256": sha256_file(self.base_model / "config.json"),
            },
            "adapter_config": {
                "path": str((adapter / "adapter_config.json").resolve()),
                "sha256": sha256_file(adapter / "adapter_config.json"),
            },
            "adapter_weights": {
                "path": str((adapter / "adapter_model.safetensors").resolve()),
                "sha256": sha256_file(adapter / "adapter_model.safetensors"),
            },
        }
        provenance = adapter / "interpolation_provenance.json"
        if provenance.is_file():
            inputs["adapter_provenance"] = {
                "path": str(provenance.resolve()),
                "sha256": sha256_file(provenance),
            }
        for case in self.cases:
            inputs[f"source_video:{case['bench_id']}"] = {
                "path": str(Path(case["video_path"]).resolve()),
                "sha256": case["source"]["video_sha256"],
            }
        outputs = {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in {
                "predictions": records,
                "planner_log": planner_log,
                "metrics": metrics,
                "scorer_log": scorer_log,
            }.items()
        }
        manifest = {
            "schema_version": 1,
            "runner": "scripts.run_planner_eval",
            "status": "complete",
            "error": None,
            "expected_cases": 384,
            "inputs": inputs,
            "commands": {
                "agent": [
                    str(Path(sys.executable).resolve()),
                    "-m",
                    "aurora.agent",
                    "--custom_cases_jsonl",
                    str(self.cases_path.resolve()),
                    "--custom_only",
                    "--plan_only",
                    "--mask_backend",
                    "none",
                    "--agent_base",
                    str(self.base_model.resolve()),
                    "--agent_adapter",
                    str(adapter.resolve()),
                    "--device",
                    "cuda:0",
                    "--out_dir",
                    str(evaluation.resolve()),
                ],
                "scorer": [
                    str(Path(sys.executable).resolve()),
                    "-m",
                    "evaluation.agent_only_score",
                    "--gold",
                    str(self.gold_path.resolve()),
                    "--predictions",
                    str(records.resolve()),
                    "--out",
                    str(metrics.resolve()),
                ],
                "cwd": str(Path(__file__).resolve().parents[1]),
            },
            "exit_state": {
                "stage": "complete",
                "agent_returncode": 0,
                "predictions_validated": True,
                "scorer_returncode": 0,
                "metrics_validated": True,
                "inputs_reverified": True,
                "outputs_reverified": True,
            },
            "outputs": outputs,
        }
        write_json(evaluation / "run_manifest.json", manifest)


class SelectLoraInterpolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.draw_patch = patch(
            "scripts.select_lora_interpolation.EXPECTED_BOOTSTRAP_DRAWS", 200
        )
        self.draw_patch.start()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()
        self.draw_patch.stop()

    def test_grid_selection_uses_reference_se_then_smallest_lambda(self) -> None:
        fixture = SelectorFixture(self.root)
        comparison, selection = evaluate_candidates(
            cases_path=fixture.cases_path,
            gold_path=fixture.gold_path,
            audit_path=fixture.audit_path,
            policy_path=fixture.policy_path,
            candidates_root=fixture.candidates_root,
            primary_reference_adapter=fixture.primary_reference_adapter,
        )

        decision = comparison["grid_decision"]
        self.assertEqual(decision["reference_candidate_id"], "lambda_0500")
        self.assertGreater(decision["reference_bootstrap_se"], 0)
        self.assertIn("lambda_0000", decision["one_se_candidate_ids"])
        self.assertEqual(decision["selected_candidate_id"], "lambda_0000")
        self.assertEqual(selection["selected_candidate_id"], "lambda_0000")
        self.assertEqual(selection["selected_kind"], "grid")
        self.assertFalse(comparison["day14_metrics_used"])
        by_id = {row["candidate_id"]: row for row in comparison["candidates"]}
        self.assertAlmostEqual(
            by_id["lambda_0500"]["metrics"]["rewrite_constraint_retention"],
            0.875,
        )
        self.assertTrue(by_id["lambda_0000"]["eligibility"]["eligible"])

    def test_primary_candidate_wins_when_eligible_and_grid_is_still_reported(self) -> None:
        fixture = SelectorFixture(self.root, primary=True)
        comparison, selection = evaluate_candidates(
            cases_path=fixture.cases_path,
            gold_path=fixture.gold_path,
            audit_path=fixture.audit_path,
            policy_path=fixture.policy_path,
            candidates_root=fixture.candidates_root,
            primary_reference_adapter=fixture.primary_reference_adapter,
        )

        self.assertEqual(selection["selected_candidate_id"], "recipe2")
        self.assertEqual(selection["selected_kind"], "primary")
        self.assertTrue(selection["primary_eligible"])
        self.assertEqual(
            comparison["grid_decision"]["selected_candidate_id"], "lambda_0000"
        )
        self.assertEqual(
            selection["primary_reference_adapter"]["adapter_dir"],
            str(fixture.primary_reference_adapter.resolve()),
        )
        primary = next(row for row in comparison["candidates"] if row["kind"] == "primary")
        self.assertIsNone(primary["adapter_identity"]["provenance_path"])

    def test_ineligible_primary_falls_back_to_grid(self) -> None:
        fixture = SelectorFixture(self.root)
        primary_model = fixture._write_primary_candidate(ineligible=True)
        fixture.policy["primary_if_eligible_else_grid"] = {
            "candidate_id": "recipe2",
            "adapter_model_sha256": primary_model,
        }
        write_json(fixture.policy_path, fixture.policy)

        comparison, selection = evaluate_candidates(
            cases_path=fixture.cases_path,
            gold_path=fixture.gold_path,
            audit_path=fixture.audit_path,
            policy_path=fixture.policy_path,
            candidates_root=fixture.candidates_root,
            primary_reference_adapter=fixture.primary_reference_adapter,
        )

        self.assertFalse(selection["primary_eligible"])
        self.assertEqual(selection["selected_candidate_id"], "lambda_0000")
        primary = next(row for row in comparison["candidates"] if row["kind"] == "primary")
        self.assertFalse(
            primary["eligibility"]["checks"]["rewrite_constraint_retention_min"]["passed"]
        )

    def test_primary_requires_byte_identical_preselected_reference(self) -> None:
        fixture = SelectorFixture(self.root / "missing", primary=True)
        with self.assertRaisesRegex(
            SelectionValidationError, "primary-reference-adapter is required"
        ):
            evaluate_candidates(
                cases_path=fixture.cases_path,
                gold_path=fixture.gold_path,
                audit_path=fixture.audit_path,
                policy_path=fixture.policy_path,
                candidates_root=fixture.candidates_root,
            )

        fixture = SelectorFixture(self.root / "config", primary=True)
        config_path = fixture.primary_reference_adapter / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["reference_only_mutation"] = True
        write_json(config_path, config)
        with self.assertRaisesRegex(SelectionValidationError, "not byte-identical"):
            evaluate_candidates(
                cases_path=fixture.cases_path,
                gold_path=fixture.gold_path,
                audit_path=fixture.audit_path,
                policy_path=fixture.policy_path,
                candidates_root=fixture.candidates_root,
                primary_reference_adapter=fixture.primary_reference_adapter,
            )

    def test_rejects_tampered_provenance_endpoint(self) -> None:
        fixture = SelectorFixture(self.root)
        path = (
            fixture.candidates_root
            / "lambda_0250/adapter/interpolation_provenance.json"
        )
        provenance = json.loads(path.read_text())
        provenance["sources"]["adapter_a"]["adapter_model_sha256"] = "9" * 64
        write_json(path, provenance)

        with self.assertRaisesRegex(SelectionValidationError, "locked v2 endpoint"):
            evaluate_candidates(
                cases_path=fixture.cases_path,
                gold_path=fixture.gold_path,
                audit_path=fixture.audit_path,
                policy_path=fixture.policy_path,
                candidates_root=fixture.candidates_root,
            )

    def test_rejects_non_exact_rank_concat_provenance(self) -> None:
        fixture = SelectorFixture(self.root)
        path = (
            fixture.candidates_root
            / "lambda_0250/adapter/interpolation_provenance.json"
        )
        provenance = json.loads(path.read_text(encoding="utf-8"))
        provenance["lora"]["output_scaling"] = 1.0
        write_json(path, provenance)
        with self.assertRaisesRegex(
            SelectionValidationError, "does not preserve exact rank-concat"
        ):
            evaluate_candidates(
                cases_path=fixture.cases_path,
                gold_path=fixture.gold_path,
                audit_path=fixture.audit_path,
                policy_path=fixture.policy_path,
                candidates_root=fixture.candidates_root,
            )

    def test_rejects_duplicate_prediction_ids_and_day14_named_artifacts(self) -> None:
        fixture = SelectorFixture(self.root)
        records = fixture.candidates_root / "lambda_0000/eval/agent_pipeline_records.jsonl"
        rows = [json.loads(line) for line in records.read_text().splitlines()]
        rows[-1]["bench_id"] = rows[0]["bench_id"]
        write_jsonl(records, rows)
        manifest_path = records.parent / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["outputs"]["predictions"]["bytes"] = records.stat().st_size
        manifest["outputs"]["predictions"]["sha256"] = sha256_file(records)
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(SelectionValidationError, "duplicate bench_id"):
            evaluate_candidates(
                cases_path=fixture.cases_path,
                gold_path=fixture.gold_path,
                audit_path=fixture.audit_path,
                policy_path=fixture.policy_path,
                candidates_root=fixture.candidates_root,
            )

    def test_rejects_incomplete_or_unbound_run_manifest(self) -> None:
        fixture = SelectorFixture(self.root / "status")
        manifest_path = (
            fixture.candidates_root / "lambda_0000/eval/run_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "failed"
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(SelectionValidationError, "not a clean complete run"):
            evaluate_candidates(
                cases_path=fixture.cases_path,
                gold_path=fixture.gold_path,
                audit_path=fixture.audit_path,
                policy_path=fixture.policy_path,
                candidates_root=fixture.candidates_root,
            )

        fixture = SelectorFixture(self.root / "source")
        manifest_path = (
            fixture.candidates_root / "lambda_0000/eval/run_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["inputs"]["source_video:interp_0001"]["sha256"] = "0" * 64
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(SelectionValidationError, "source_video:interp_0001 SHA-256 mismatch"):
            evaluate_candidates(
                cases_path=fixture.cases_path,
                gold_path=fixture.gold_path,
                audit_path=fixture.audit_path,
                policy_path=fixture.policy_path,
                candidates_root=fixture.candidates_root,
            )

        fixture = SelectorFixture(self.root / "output")
        records = (
            fixture.candidates_root
            / "lambda_0000/eval/agent_pipeline_records.jsonl"
        )
        records.write_text(records.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(SelectionValidationError, "predictions byte count"):
            evaluate_candidates(
                cases_path=fixture.cases_path,
                gold_path=fixture.gold_path,
                audit_path=fixture.audit_path,
                policy_path=fixture.policy_path,
                candidates_root=fixture.candidates_root,
            )

        fixture = SelectorFixture(self.root / "command")
        manifest_path = (
            fixture.candidates_root / "lambda_0000/eval/run_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["commands"]["agent"].extend(["--max_cases", "384"])
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(SelectionValidationError, "generic runner command"):
            evaluate_candidates(
                cases_path=fixture.cases_path,
                gold_path=fixture.gold_path,
                audit_path=fixture.audit_path,
                policy_path=fixture.policy_path,
                candidates_root=fixture.candidates_root,
            )

        fixture = SelectorFixture(self.root / "second")
        (fixture.candidates_root / "lambda_0000/day14_metrics.json").write_text("{}")
        with self.assertRaisesRegex(SelectionValidationError, "Day-14-named"):
            evaluate_candidates(
                cases_path=fixture.cases_path,
                gold_path=fixture.gold_path,
                audit_path=fixture.audit_path,
                policy_path=fixture.policy_path,
                candidates_root=fixture.candidates_root,
            )

    def test_bootstrap_is_paired_and_deterministic(self) -> None:
        strata = {
            "a": {
                "no_search_negative": [0.0, 1.0, 1.0],
                "rewrite_retention": [0.25, 1.0],
            },
            "b": {
                "no_search_negative": [1.0, 0.0, 0.0],
                "rewrite_retention": [0.75, 0.0],
            },
        }
        first = paired_stratified_bootstrap(strata, seed=17, draws=100)
        second = paired_stratified_bootstrap(strata, seed=17, draws=100)
        self.assertEqual(first, second)
        self.assertGreater(first["a"]["standard_error"], 0)

    def test_cli_writes_new_artifacts_and_refuses_overwrite(self) -> None:
        fixture = SelectorFixture(self.root)
        comparison = self.root / "out/comparison.json"
        selection = self.root / "out/selection.json"
        args = [
            "--cases", str(fixture.cases_path),
            "--gold", str(fixture.gold_path),
            "--leakage-audit", str(fixture.audit_path),
            "--policy", str(fixture.policy_path),
            "--candidates-root", str(fixture.candidates_root),
            "--comparison-out", str(comparison),
            "--selection-out", str(selection),
        ]
        self.assertEqual(main(args), 0)
        self.assertTrue(comparison.is_file())
        selected = json.loads(selection.read_text())
        self.assertEqual(selected["comparison_sha256"], sha256_file(comparison))
        self.assertEqual(main(args), 1)


if __name__ == "__main__":
    unittest.main()

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import scripts.build_interpolation_validation as interpolation_builder
from evaluation.agent_only_score import constraint_is_retained, normalize
from scripts.build_interpolation_validation import (
    PriorIndex,
    ROUTING_CONTROL_COUNTS,
    _assert_output_contract,
    _eligible_spec,
    _mask_specs,
    _retention_specs,
    _routing_specs,
    _search_specs,
    _select_specs,
    build_prior_index,
    load_jsonl,
    select_fresh_sources,
    sha256_file,
    write_bundle,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _prior_row(video: Path, prompt: str, *, sample_id: str | None = None) -> dict:
    row = {
        "system": "Return a planner JSON object.",
        "messages": [
            {"role": "user", "content": f"<video>\n{prompt}"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "refined_text_instruction": prompt,
                        "subtask": "change_color",
                        "image_search": False,
                        "mask": False,
                    }
                ),
            },
        ],
        "videos": [str(video)],
    }
    if sample_id is not None:
        row["sample_id"] = sample_id
    return row


class BuildInterpolationValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_root = self.root / "fresh"
        self.source_root.mkdir()
        self.prior_root = self.root / "prior"
        self.prior_root.mkdir()

        prior_files = []
        for label in ("v2-train", "v2-eval", "refresh-train", "refresh-eval", "day14"):
            path = self.prior_root / f"{label}.mp4"
            path.write_bytes(f"prior video {label}".encode())
            prior_files.append(path)

        self.v2_train = self.root / "v2_train.jsonl"
        self.v2_eval = self.root / "v2_eval.jsonl"
        self.refresh_train = self.root / "refresh_train.jsonl"
        self.refresh_eval = self.root / "refresh_eval.jsonl"
        self.day14 = self.root / "day14.jsonl"
        v2_train_row = _prior_row(
            prior_files[0],
            "recast this with a verdigris and cream block-print aesthetic",
            sample_id="historical-sample",
        )
        _write_jsonl(self.v2_train, [v2_train_row])
        _write_jsonl(
            self.v2_eval,
            [_prior_row(
                prior_files[1],
                "use Alessi Juicy Salif citrus squeezer with a folded-paper relief treatment",
            )],
        )
        refresh_train_row = _prior_row(
            prior_files[2], "recast the whole clip with a historical woven aesthetic"
        )
        _write_jsonl(self.refresh_train, [refresh_train_row])
        _write_jsonl(
            self.refresh_eval,
            [
                _prior_row(
                    prior_files[3],
                    "an inventory note mentions the frayed cord by the stool without requesting removal",
                ),
                _prior_row(prior_files[3], "erase the unrelated loose cable"),
            ],
        )
        _write_jsonl(
            self.day14,
            [
                {
                    "bench_id": "day14-1",
                    "video_path": str(prior_files[4]),
                    "prompt": "put the landmark on a sunset waterfront",
                    "gold_plan": {
                        "refined_text_instruction": "Keep everything clearly visible and preserve the surrounding scene.",
                        "subtask": "add_object",
                        "image_search": "Forbidden Test Landmark",
                        "mask": False,
                    },
                    "constraints": [
                        {"type": "preservation", "value": "clearly visible"},
                        {"type": "preservation", "value": "surrounding scene"},
                        {"type": "identity", "value": "sunset waterfront"},
                    ],
                    "source": {"source_id": "nested-prior-source"},
                }
            ],
        )

        manifest = []
        for index in range(390):
            path = self.source_root / f"fresh-{index:04d}.mp4"
            path.write_bytes(f"fresh video {index}".encode())
            manifest.append(
                {
                    "sample_id": f"fresh-{index:04d}",
                    "video_path": path.name,
                    "subset": "fresh-holdout",
                    "source_dataset": "unit-fixture",
                    "license": "test-only",
                }
            )

        # Three independent prior-video collisions: id, basename, and bytes.
        id_collision = self.source_root / "id-collision.mp4"
        id_collision.write_bytes(b"unique id collision bytes")
        manifest.append({"sample_id": "historical-sample", "video_path": id_collision.name})

        nested_id_collision = self.source_root / "nested-id-collision.mp4"
        nested_id_collision.write_bytes(b"unique nested id collision bytes")
        manifest.append({
            "sample_id": "nested-prior-source",
            "video_path": nested_id_collision.name,
        })

        basename_dir = self.source_root / "nested"
        basename_dir.mkdir()
        basename_collision = basename_dir / "v2-eval.mp4"
        basename_collision.write_bytes(b"unique basename collision bytes")
        manifest.append({"sample_id": "new-basename-id", "video_path": "nested/v2-eval.mp4"})

        sha_collision = self.source_root / "sha-collision.mp4"
        sha_collision.write_bytes(prior_files[2].read_bytes())
        manifest.append({"sample_id": "new-sha-id", "video_path": sha_collision.name})

        self.manifest = self.root / "manifest.jsonl"
        _write_jsonl(self.manifest, manifest)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _build(self) -> dict:
        return write_bundle(
            source_manifest=self.manifest,
            source_root=self.source_root,
            v2_train=self.v2_train,
            v2_eval=self.v2_eval,
            refresh1_train=self.refresh_train,
            refresh1_eval=self.refresh_eval,
            day14_cases=self.day14,
            cases_out=self.root / "out/cases.jsonl",
            gold_out=self.root / "out/gold.jsonl",
            audit_out=self.root / "out/audit.json",
            policy_out=self.root / "out/policy.json",
            v2_adapter_sha256="1" * 64,
            refresh1_adapter_sha256="2" * 64,
            primary_adapter_sha256="3" * 64,
        )

    def test_builds_exact_fresh_balanced_suite_and_leakage_audit(self) -> None:
        result = self._build()
        cases = result["cases"]
        gold = result["gold"]
        audit = result["audit"]

        self.assertEqual(audit["recipe_version"], 2)
        self.assertEqual((len(cases), len(gold)), (384, 384))
        self.assertEqual(len({row["video_path"] for row in cases}), 384)
        self.assertEqual(len({row["source"]["video_sha256"] for row in cases}), 384)
        self.assertEqual(
            Counter(row["category"] for row in cases),
            {
                "no_search_negative": 128,
                "true_search_positive": 64,
                "routing_control": 64,
                "mask_control": 64,
                "rewrite_retention": 64,
            },
        )
        self.assertEqual(
            Counter(row["subtype"] for row in cases if row["category"] == "no_search_negative"),
            {"generic_style": 64, "generic_background": 32, "ordinary_target": 32},
        )
        mask_counts = Counter(
            "triggered" if isinstance(row["gold_plan"]["mask"], str) else "not_triggered"
            for row in gold
            if row["category"] == "mask_control"
        )
        self.assertEqual(mask_counts, {"triggered": 32, "not_triggered": 32})
        self.assertEqual(sum(isinstance(row["gold_plan"]["image_search"], str) for row in gold), 64)
        search_subtypes = Counter(
            row["subtype"] for row in cases if row["category"] == "true_search_positive"
        )
        self.assertEqual(set(search_subtypes.values()), {8})
        self.assertEqual(len(search_subtypes), 8)
        routing_subtypes = Counter(
            row["subtype"] for row in cases if row["category"] == "routing_control"
        )
        self.assertEqual(routing_subtypes, Counter(ROUTING_CONTROL_COUNTS))

        self.assertEqual(
            audit["source_filter"]["rejections_by_reason"],
            {
                "prior_basename": 1,
                "prior_sample_id": 2,
                "prior_video_sha256": 1,
            },
        )
        self.assertGreaterEqual(
            audit["catalog_rejections_by_reason"].get("prior_concept_phrase", 0), 1
        )
        self.assertGreaterEqual(
            audit["catalog_rejections_by_reason"].get("prior_template_signature", 0), 1
        )
        for key, value in audit["isolation"].items():
            if key.endswith("overlap"):
                self.assertEqual(value, 0, key)
        self.assertEqual(audit["isolation"]["prior_concept_phrase_hit_count"], 0)
        self.assertEqual(audit["isolation"]["prior_constraint_phrase_hit_count"], 0)
        self.assertEqual(audit["isolation"]["prior_concept_phrase_hits"], [])
        self.assertEqual(audit["isolation"]["prior_constraint_phrase_hits"], [])
        self.assertGreaterEqual(
            audit["isolation"]["allowed_generic_concept_phrase_hit_count"], 1
        )
        self.assertIn(
            {
                "category": "routing_control",
                "subtype": "remove_object",
                "concept_id": "frayed cord by the stool",
            },
            audit["isolation"]["allowed_generic_concept_phrase_hits"],
        )
        self.assertGreaterEqual(
            audit["isolation"]["allowed_generic_template_signature_hit_count"], 1
        )
        self.assertIn(
            {
                "category": "routing_control",
                "subtype": "remove_object",
                "template_signature": "erase the __slot__",
            },
            audit["isolation"]["allowed_generic_template_signature_hits"],
        )
        self.assertEqual(
            audit["isolation"]["concept_phrase_policy"],
            {
                "exact_concept_overlap_forbidden_for_all_categories": True,
                "phrase_containment_forbidden_categories": [
                    "no_search_negative",
                    "rewrite_retention",
                    "true_search_positive",
                ],
                "phrase_containment_audit_only_categories": [
                    "mask_control",
                    "routing_control",
                ],
            },
        )
        self.assertEqual(
            audit["isolation"]["template_signature_policy"],
            {
                "exact_prompt_and_template_id_overlap_forbidden_for_all_categories": True,
                "wildcard_signature_overlap_forbidden_categories": [
                    "no_search_negative",
                    "rewrite_retention",
                    "true_search_positive",
                ],
                "wildcard_signature_overlap_audit_only_categories": [
                    "mask_control",
                    "routing_control",
                ],
            },
        )
        self.assertEqual(audit["isolation"]["explicit_forbidden_phrase_hits"], [])
        self.assertTrue(audit["lexical_retention"]["all_values_in_request"])
        self.assertTrue(audit["lexical_retention"]["all_values_in_refined_instruction"])

    def test_retention_values_are_literal_in_request_and_gold_rewrite(self) -> None:
        result = self._build()
        retention = [row for row in result["gold"] if row["category"] == "rewrite_retention"]
        self.assertEqual(len(retention), 64)
        self.assertEqual({len(row["constraints"]) for row in retention}, {4})
        self.assertEqual(
            Counter(item["type"] for row in retention for item in row["constraints"]),
            {"color": 64, "identity": 64, "spatial": 64, "preservation": 64},
        )
        for row in retention:
            refined = row["gold_plan"]["refined_text_instruction"]
            for constraint in row["constraints"]:
                self.assertTrue(constraint_is_retained(constraint, row["prompt"]))
                self.assertTrue(constraint_is_retained(constraint, refined))
        rendered = json.dumps(result["gold"]).casefold()
        for phrase in ("clearly visible", "surrounding scene", "sunset waterfront"):
            self.assertNotIn(phrase, rendered)

    def test_concept_leakage_policy_is_category_aware(self) -> None:
        def prior(*texts: str, concepts: tuple[str, ...] = ()) -> PriorIndex:
            return PriorIndex(
                identifiers=frozenset(),
                basenames=frozenset(),
                video_sha256=frozenset(),
                prompts=frozenset(),
                concepts=frozenset(concepts),
                template_ids=frozenset(),
                template_signatures=frozenset(),
                constraint_values=frozenset(),
                normalized_texts=tuple(normalize(text) for text in texts),
                video_paths=0,
            )

        routing = next(spec for spec in _routing_specs() if spec.subtype == "remove_object")
        mask = _mask_specs()["triggered"][0]
        for spec in (routing, mask):
            with self.subTest(category=spec.category, kind="phrase"):
                eligible, reason = _eligible_spec(
                    spec,
                    prior(f"inventory note containing {spec.concept_id} for context"),
                )
                self.assertTrue(eligible)
                self.assertIsNone(reason)
            with self.subTest(category=spec.category, kind="exact-index"):
                eligible, reason = _eligible_spec(
                    spec, prior(concepts=(spec.concept_id,))
                )
                self.assertFalse(eligible)
                self.assertEqual(reason, "prior_concept_exact")

        search = _search_specs()[0]
        eligible, reason = _eligible_spec(
            search, prior(f"inventory note containing {search.concept_id} for context")
        )
        self.assertFalse(eligible)
        self.assertEqual(reason, "prior_concept_phrase")

        retention = _retention_specs()[0]
        retained_value = retention.constraints[0]["value"]
        eligible, reason = _eligible_spec(
            retention, prior(f"inventory note containing {retained_value} for context")
        )
        self.assertFalse(eligible)
        self.assertEqual(reason, "prior_constraint_value")

    def test_template_leakage_policy_is_category_aware(self) -> None:
        def prior_with_prompt(prompt: str) -> PriorIndex:
            return PriorIndex(
                identifiers=frozenset(),
                basenames=frozenset(),
                video_sha256=frozenset(),
                prompts=frozenset({normalize(prompt)}),
                concepts=frozenset(),
                template_ids=frozenset(),
                template_signatures=frozenset(),
                constraint_values=frozenset(),
                normalized_texts=(normalize(prompt),),
                video_paths=0,
            )

        routing = next(spec for spec in _routing_specs() if spec.subtype == "remove_object")
        eligible, reason = _eligible_spec(
            routing, prior_with_prompt("erase the unrelated loose cable")
        )
        self.assertTrue(eligible)
        self.assertIsNone(reason)

        search = _search_specs()[0]
        eligible, reason = _eligible_spec(
            search,
            prior_with_prompt("introduce an unrelated lamp beside the foremost subject"),
        )
        self.assertFalse(eligible)
        self.assertEqual(reason, "prior_template_signature")

    def test_catalog_exhaustion_reports_reasons_and_examples(self) -> None:
        search = _search_specs()[0]
        prior = PriorIndex(
            identifiers=frozenset(),
            basenames=frozenset(),
            video_sha256=frozenset(),
            prompts=frozenset(),
            concepts=frozenset(),
            template_ids=frozenset(),
            template_signatures=frozenset(),
            constraint_values=frozenset(),
            normalized_texts=(normalize(f"old request featuring {search.concept_id}"),),
            video_paths=0,
        )
        with self.assertRaisesRegex(
            ValueError,
            r"diagnostic: only 0 of 1.*prior_concept_phrase.*Alessi Juicy Salif",
        ):
            _select_specs([search], 1, prior, "diagnostic")

    def test_policy_locks_grid_hashes_thresholds_and_one_se_rule(self) -> None:
        result = self._build()
        policy = result["policy"]
        self.assertEqual(policy["version"], 2)
        self.assertTrue(policy["created_before_candidate_inference"])
        self.assertEqual(
            policy["candidate_rule"]["lambda_grid"],
            [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        )
        self.assertEqual(policy["candidate_rule"]["candidate_count"], 9)
        self.assertEqual(policy["candidate_rule"]["v2_adapter_model_sha256"], "1" * 64)
        self.assertEqual(policy["candidate_rule"]["refresh1_adapter_model_sha256"], "2" * 64)
        self.assertFalse(
            policy["candidate_rule"]["day14_outputs_or_metrics_allowed_for_generation_or_selection"]
        )
        self.assertIn("not confirmatory", policy["candidate_rule"]["day14_use"])
        self.assertIn("sealed evaluation set", policy["candidate_rule"]["confirmation_requirement"])
        self.assertEqual(
            policy["utility"]["formula"],
            "0.5*no_search_specificity+0.5*rewrite_constraint_retention",
        )
        self.assertEqual(
            policy["utility"]["no_search_specificity_scope"],
            "128 no_search_negative cases only",
        )
        self.assertEqual(
            policy["utility"]["rewrite_constraint_retention_scope"],
            "64 rewrite_retention cases only",
        )
        self.assertEqual(policy["bootstrap"]["seed"], 20260916)
        self.assertEqual(policy["bootstrap"]["draws"], 10000)
        self.assertEqual(
            policy["bootstrap"]["method"],
            "paired stratified case bootstrap over the two utility strata",
        )
        self.assertEqual(
            policy["bootstrap"]["strata"],
            {"no_search_negative": 128, "rewrite_retention": 64},
        )
        self.assertEqual(policy["selection"]["rule"], "one_standard_error_then_smallest_lambda")
        self.assertEqual(policy["selection"]["tie_breaker"], "smallest lambda")
        self.assertEqual(policy["selection"]["reference_tie_breaker"], "smallest lambda")
        self.assertIn("SE(best)", policy["selection"]["one_se_set"])
        self.assertFalse(policy["selection"]["day14_metrics_used"])
        self.assertEqual(
            policy["final_decision"],
            {
                "rule": "primary_if_eligible_else_grid",
                "primary_candidate": {
                    "candidate_id": "recipe2",
                    "adapter_model_sha256": "3" * 64,
                    "directory_name": "recipe2",
                },
            },
        )
        self.assertEqual(
            policy["eligibility_thresholds"],
            {
                "complete_prediction_rows": 384,
                "strict_raw_json_validity_min": 1.0,
                "subtask_accuracy_min": 0.95,
                "no_search_specificity_min": 0.95,
                "true_search_trigger_recall_min": 0.95,
                "search_query_end_to_end_recall_min": 0.95,
                "mask_trigger_f1_min": 0.95,
                "rewrite_constraint_retention_min": 0.85,
            },
        )
        self.assertEqual(
            policy["validation_artifacts"]["expected_category_counts"],
            {
                "no_search_negative": 128,
                "true_search_positive": 64,
                "routing_control": 64,
                "mask_control": 64,
                "rewrite_retention": 64,
            },
        )
        self.assertEqual(
            policy["eligibility_threshold_scopes"]["mask_trigger_f1"],
            "64 mask_control cases only",
        )
        self.assertEqual(set(policy["validation_artifacts"]["input_sha256"]), {
            "source_manifest", "v2_train", "v2_eval", "refresh1_train",
            "refresh1_eval", "day14_forbidden_cases",
        })
        expected_inputs = {
            "source_manifest": self.manifest,
            "v2_train": self.v2_train,
            "v2_eval": self.v2_eval,
            "refresh1_train": self.refresh_train,
            "refresh1_eval": self.refresh_eval,
            "day14_forbidden_cases": self.day14,
        }
        self.assertEqual(
            policy["validation_artifacts"]["input_sha256"],
            {name: sha256_file(path) for name, path in sorted(expected_inputs.items())},
        )
        artifact_hashes = policy["validation_artifacts"]
        self.assertEqual(artifact_hashes["cases_sha256"], sha256_file(self.root / "out/cases.jsonl"))
        self.assertEqual(artifact_hashes["gold_sha256"], sha256_file(self.root / "out/gold.jsonl"))
        self.assertEqual(artifact_hashes["leakage_audit_sha256"], sha256_file(self.root / "out/audit.json"))

    def test_output_contract_rejects_search_trigger_category_swap(self) -> None:
        result = self._build()
        cases = copy.deepcopy(result["cases"])
        gold = copy.deepcopy(result["gold"])
        positive = next(row for row in gold if row["category"] == "true_search_positive")
        negative = next(row for row in gold if row["category"] == "no_search_negative")
        query = positive["gold_plan"]["image_search"]
        positive["gold_plan"]["image_search"] = False
        positive.pop("search_query_aliases")
        negative["gold_plan"]["image_search"] = query
        negative["search_query_aliases"] = [query]
        with self.assertRaisesRegex(AssertionError, "all and only"):
            _assert_output_contract(cases, gold)

    def test_source_manifest_paths_must_remain_under_source_root(self) -> None:
        prior = build_prior_index(
            [self.v2_train, self.v2_eval, self.refresh_train, self.refresh_eval, self.day14]
        )
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"outside fresh root")
        for raw_path in ("../outside.mp4", str(outside.resolve())):
            with self.subTest(raw_path=raw_path):
                with self.assertRaisesRegex(ValueError, "escapes source_root"):
                    select_fresh_sources(
                        [{"sample_id": f"escape-{raw_path}", "video_path": raw_path}],
                        self.source_root,
                        prior,
                    )

    def test_input_and_output_artifact_paths_must_be_distinct(self) -> None:
        original_manifest = self.manifest.read_bytes()
        common = dict(
            source_manifest=self.manifest,
            source_root=self.source_root,
            v2_train=self.v2_train,
            v2_eval=self.v2_eval,
            refresh1_train=self.refresh_train,
            refresh1_eval=self.refresh_eval,
            day14_cases=self.day14,
            audit_out=self.root / "alias/audit.json",
            policy_out=self.root / "alias/policy.json",
            v2_adapter_sha256="1" * 64,
            refresh1_adapter_sha256="2" * 64,
        )
        with self.assertRaisesRegex(ValueError, "artifact paths must be distinct"):
            write_bundle(
                **common,
                cases_out=self.manifest,
                gold_out=self.root / "alias/gold.jsonl",
            )
        self.assertEqual(self.manifest.read_bytes(), original_manifest)
        shared = self.root / "alias/shared.jsonl"
        with self.assertRaisesRegex(ValueError, "artifact paths must be distinct"):
            write_bundle(**common, cases_out=shared, gold_out=shared)
        shared_metadata = self.root / "alias/shared-metadata.json"
        metadata_alias = {
            **common,
            "audit_out": shared_metadata,
            "policy_out": shared_metadata,
        }
        with self.assertRaisesRegex(ValueError, "artifact paths must be distinct"):
            write_bundle(
                **metadata_alias,
                cases_out=self.root / "alias/cases.jsonl",
                gold_out=self.root / "alias/gold-2.jsonl",
            )

    def test_existing_output_preflight_rejects_entire_bundle(self) -> None:
        for occupied_name in ("cases.jsonl", "gold.jsonl", "audit.json", "policy.json"):
            with self.subTest(occupied_name=occupied_name):
                output_root = self.root / f"existing-{occupied_name.replace('.', '-')}"
                outputs = {
                    "cases_out": output_root / "cases.jsonl",
                    "gold_out": output_root / "gold.jsonl",
                    "audit_out": output_root / "audit.json",
                    "policy_out": output_root / "policy.json",
                }
                occupied = output_root / occupied_name
                occupied.parent.mkdir(parents=True)
                occupied.write_text("sealed sentinel\n", encoding="utf-8")
                with patch(
                    "scripts.build_interpolation_validation.build_prior_index"
                ) as build_prior:
                    with self.assertRaisesRegex(
                        ValueError, "refusing to overwrite existing output"
                    ):
                        write_bundle(
                            source_manifest=self.manifest,
                            source_root=self.source_root,
                            v2_train=self.v2_train,
                            v2_eval=self.v2_eval,
                            refresh1_train=self.refresh_train,
                            refresh1_eval=self.refresh_eval,
                            day14_cases=self.day14,
                            **outputs,
                            v2_adapter_sha256="1" * 64,
                            refresh1_adapter_sha256="2" * 64,
                        )
                build_prior.assert_not_called()
                self.assertEqual(
                    occupied.read_text(encoding="utf-8"), "sealed sentinel\n"
                )
                self.assertEqual(
                    sorted(path.name for path in output_root.iterdir()),
                    [occupied_name],
                )

    def test_exclusive_write_preserves_racing_artifact_and_cleans_outputs(
        self,
    ) -> None:
        output_root = self.root / "exclusive-race"
        cases_out = output_root / "cases.jsonl"
        gold_out = output_root / "gold.jsonl"
        audit_out = output_root / "audit.json"
        policy_out = output_root / "policy.json"
        original_make_policy = interpolation_builder.make_selection_policy

        def create_racing_policy(**kwargs: object) -> dict:
            policy_out.write_text("concurrent sentinel\n", encoding="utf-8")
            return original_make_policy(**kwargs)

        with patch.object(
            interpolation_builder,
            "make_selection_policy",
            side_effect=create_racing_policy,
        ):
            with self.assertRaisesRegex(
                ValueError, "refusing to overwrite existing output"
            ):
                write_bundle(
                    source_manifest=self.manifest,
                    source_root=self.source_root,
                    v2_train=self.v2_train,
                    v2_eval=self.v2_eval,
                    refresh1_train=self.refresh_train,
                    refresh1_eval=self.refresh_eval,
                    day14_cases=self.day14,
                    cases_out=cases_out,
                    gold_out=gold_out,
                    audit_out=audit_out,
                    policy_out=policy_out,
                    v2_adapter_sha256="1" * 64,
                    refresh1_adapter_sha256="2" * 64,
                )

        self.assertEqual(
            policy_out.read_text(encoding="utf-8"), "concurrent sentinel\n"
        )
        self.assertFalse(cases_out.exists())
        self.assertFalse(gold_out.exists())
        self.assertFalse(audit_out.exists())

    def test_broken_output_symlink_is_treated_as_existing(self) -> None:
        output_root = self.root / "broken-symlink"
        output_root.mkdir()
        cases_out = output_root / "cases.jsonl"
        cases_out.symlink_to(output_root / "missing-target.jsonl")
        with self.assertRaisesRegex(
            ValueError, "refusing to overwrite existing output"
        ):
            write_bundle(
                source_manifest=self.manifest,
                source_root=self.source_root,
                v2_train=self.v2_train,
                v2_eval=self.v2_eval,
                refresh1_train=self.refresh_train,
                refresh1_eval=self.refresh_eval,
                day14_cases=self.day14,
                cases_out=cases_out,
                gold_out=output_root / "gold.jsonl",
                audit_out=output_root / "audit.json",
                policy_out=output_root / "policy.json",
                v2_adapter_sha256="1" * 64,
                refresh1_adapter_sha256="2" * 64,
            )
        self.assertTrue(cases_out.is_symlink())
        self.assertFalse((output_root / "missing-target.jsonl").exists())

    def test_fails_closed_when_fewer_than_384_fresh_videos_remain(self) -> None:
        prior = build_prior_index(
            [self.v2_train, self.v2_eval, self.refresh_train, self.refresh_eval, self.day14]
        )
        rows = load_jsonl(self.manifest)[:383]
        with self.assertRaisesRegex(ValueError, "need 384"):
            select_fresh_sources(rows, self.source_root, prior)

    def test_prior_repo_root_relative_video_path_resolves_safely(self) -> None:
        repo_root = self.root / "repo"
        dataset = repo_root / "data/week1/cases.jsonl"
        video = repo_root / "data/smoke/root-relative.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"repo-root relative video")
        _write_jsonl(
            dataset,
            [{
                "bench_id": "root-relative",
                "video_path": "data/smoke/root-relative.mp4",
                "prompt": "fixture request",
                "gold_plan": {
                    "refined_text_instruction": "Fixture request.",
                    "subtask": "change_color",
                    "image_search": False,
                    "mask": False,
                },
            }],
        )
        with patch("scripts.build_interpolation_validation.REPO_ROOT", repo_root):
            prior = build_prior_index([dataset])
        self.assertEqual(prior.video_paths, 1)
        self.assertEqual(prior.video_sha256, {sha256_file(video)})

    def test_prior_relative_video_path_rejects_ambiguous_resolution(self) -> None:
        repo_root = self.root / "ambiguous-repo"
        dataset = repo_root / "data/week1/cases.jsonl"
        repo_video = repo_root / "clips/shared.mp4"
        local_video = dataset.parent / "clips/shared.mp4"
        repo_video.parent.mkdir(parents=True)
        local_video.parent.mkdir(parents=True)
        repo_video.write_bytes(b"repo candidate")
        local_video.write_bytes(b"dataset candidate")
        _write_jsonl(
            dataset,
            [{
                "bench_id": "ambiguous",
                "video_path": "clips/shared.mp4",
                "prompt": "fixture request",
                "gold_plan": {
                    "refined_text_instruction": "Fixture request.",
                    "subtask": "change_color",
                    "image_search": False,
                    "mask": False,
                },
            }],
        )
        with patch("scripts.build_interpolation_validation.REPO_ROOT", repo_root):
            with self.assertRaisesRegex(ValueError, "ambiguous relative video path"):
                build_prior_index([dataset])

    def test_invalid_adapter_digest_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "64-character"):
            write_bundle(
                source_manifest=self.manifest,
                source_root=self.source_root,
                v2_train=self.v2_train,
                v2_eval=self.v2_eval,
                refresh1_train=self.refresh_train,
                refresh1_eval=self.refresh_eval,
                day14_cases=self.day14,
                cases_out=self.root / "bad/cases.jsonl",
                gold_out=self.root / "bad/gold.jsonl",
                audit_out=self.root / "bad/audit.json",
                policy_out=self.root / "bad/policy.json",
                v2_adapter_sha256="not-a-digest",
                refresh1_adapter_sha256="2" * 64,
            )

    def test_invalid_primary_adapter_digest_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "primary adapter hash"):
            write_bundle(
                source_manifest=self.manifest,
                source_root=self.source_root,
                v2_train=self.v2_train,
                v2_eval=self.v2_eval,
                refresh1_train=self.refresh_train,
                refresh1_eval=self.refresh_eval,
                day14_cases=self.day14,
                cases_out=self.root / "bad-primary/cases.jsonl",
                gold_out=self.root / "bad-primary/gold.jsonl",
                audit_out=self.root / "bad-primary/audit.json",
                policy_out=self.root / "bad-primary/policy.json",
                v2_adapter_sha256="1" * 64,
                refresh1_adapter_sha256="2" * 64,
                primary_adapter_sha256="not-a-digest",
            )

    def test_identical_adapter_endpoints_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct endpoints"):
            write_bundle(
                source_manifest=self.manifest,
                source_root=self.source_root,
                v2_train=self.v2_train,
                v2_eval=self.v2_eval,
                refresh1_train=self.refresh_train,
                refresh1_eval=self.refresh_eval,
                day14_cases=self.day14,
                cases_out=self.root / "same/cases.jsonl",
                gold_out=self.root / "same/gold.jsonl",
                audit_out=self.root / "same/audit.json",
                policy_out=self.root / "same/policy.json",
                v2_adapter_sha256="3" * 64,
                refresh1_adapter_sha256="3" * 64,
            )


if __name__ == "__main__":
    unittest.main()

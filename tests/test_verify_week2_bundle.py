import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.agent_only_score import score as score_agent_plans
from scripts.run_day14_gate import RELEASED_BASELINE, evaluate_gate
from scripts.verify_week2_bundle import (
    CORRECTED_ADAPTER_CONFIG_SHA256,
    CORRECTED_CATEGORY_COUNTS,
    CORRECTED_LAMBDA_GRID,
    EXPECTED_COUNTS,
    audit_bundle,
    audit_corrected_selection_bundle,
    audit_corrected_thin_bundle,
    audit_refresh_bundle,
    verify_manifest,
    write_manifest,
)

_ADAPTER_CONFIG_TEMPLATE = {
    "alora_invocation_tokens": None,
    "alpha_pattern": {},
    "arrow_config": None,
    "auto_mapping": None,
    "base_model_name_or_path": "/mlx_devbox/users/jieyu.li/models/Qwen3-VL-8B-Instruct",
    "bias": "none",
    "corda_config": None,
    "ensure_weight_tying": False,
    "eva_config": None,
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "layer_replication": None,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": 64,
    "lora_bias": False,
    "lora_dropout": 0.05,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "peft_type": "LORA",
    "peft_version": "0.18.1",
    "qalora_group_size": 16,
    "r": 32,
    "rank_pattern": {},
    "revision": None,
    "target_modules": [],
    "target_parameters": None,
    "task_type": "CAUSAL_LM",
    "trainable_token_indices": None,
    "use_dora": False,
    "use_qalora": False,
    "use_rslora": False,
}
_ADAPTER_TARGET_MODULES = {
    "v2": ["q_proj", "gate_proj", "k_proj", "o_proj", "v_proj", "down_proj", "up_proj"],
    "refresh1": ["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "recipe2": ["gate_proj", "down_proj", "k_proj", "v_proj", "q_proj", "up_proj", "o_proj"],
}


def _adapter_config_bytes(name: str) -> bytes:
    value = dict(_ADAPTER_CONFIG_TEMPLATE)
    value["target_modules"] = _ADAPTER_TARGET_MODULES[name]
    payload = json.dumps(value, indent=2).encode()
    assert hashlib.sha256(payload).hexdigest() == CORRECTED_ADAPTER_CONFIG_SHA256[name]
    return payload


def _grid_adapter_config_bytes() -> bytes:
    value = json.loads(_adapter_config_bytes("v2"))
    value["r"] *= 2
    value["lora_alpha"] *= 2
    value["target_modules"] = sorted(value["target_modules"])
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


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


def _write_complete_eval_manifest(
    eval_dir: Path,
    *,
    cases: Path,
    gold: Path,
    adapter: Path,
    expected_cases: int,
) -> None:
    base_config = Path("/models/qwen/config.json")
    runtime_adapter = adapter.resolve()
    (eval_dir / "planner.log").write_text(
        f"Agent base: {base_config.parent}\nAgent adapter: {runtime_adapter}\n",
        encoding="utf-8",
    )
    outputs = {
        "predictions": eval_dir / "agent_pipeline_records.jsonl",
        "planner_log": eval_dir / "planner.log",
        "metrics": eval_dir / "metrics.json",
        "scorer_log": eval_dir / "scorer.log",
    }
    inputs = {
        "cases": {"path": str(cases), "sha256": hashlib.sha256(cases.read_bytes()).hexdigest()},
        "gold": {"path": str(gold), "sha256": hashlib.sha256(gold.read_bytes()).hexdigest()},
        "base_config": {"path": str(base_config), "sha256": "e" * 64},
        "adapter_config": {
            "path": str(runtime_adapter / "adapter_config.json"),
            "sha256": hashlib.sha256((adapter / "adapter_config.json").read_bytes()).hexdigest(),
        },
        "adapter_weights": {
            "path": str(runtime_adapter / "adapter_model.safetensors"),
            "sha256": hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
        },
    }
    for row in (
        json.loads(line)
        for line in cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ):
        bench_id = row["bench_id"]
        inputs[f"source_video:{bench_id}"] = {
            "path": row["video_path"],
            "sha256": row["source"]["video_sha256"],
        }
        if row.get("ref_image_path") not in (None, ""):
            inputs[f"reference_image:{bench_id}"] = {
                "path": row["ref_image_path"],
                "sha256": hashlib.sha256(
                    f"reference-{bench_id}".encode()
                ).hexdigest(),
            }
    provenance = adapter / "interpolation_provenance.json"
    if provenance.is_file():
        inputs["adapter_provenance"] = {
            "path": str(provenance),
            "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema_version": 1,
        "runner": "scripts.run_planner_eval",
        "status": "complete",
        "error": None,
        "expected_cases": expected_cases,
        "inputs": inputs,
        "exit_state": {
            "stage": "complete",
            "agent_returncode": 0,
            "predictions_validated": True,
            "scorer_returncode": 0,
            "metrics_validated": True,
            "inputs_reverified": True,
            "outputs_reverified": True,
        },
        "outputs": {
            key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in outputs.items()
        },
    }
    (eval_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_corrected_selection_fixture(root: Path) -> None:
    recipe_output = root / "recipe2/lora-refresh"
    recipe_selected = root / "recipe2/lora-refresh-selected"
    recipe_metadata = root / "recipe2/metadata"
    fresh = root / "fresh384"
    selection_dir = root / "selection"
    adaptive = root / "adaptive_day14"
    for path in (recipe_output, recipe_selected, recipe_metadata, fresh, selection_dir, adaptive):
        path.mkdir(parents=True, exist_ok=True)

    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "train.log",
        "trainer_log.jsonl",
        "trainer_state.json",
    ):
        (recipe_output / name).write_text(
            "{}\n" if name.endswith(".json") else "x\n", encoding="utf-8"
        )
    (recipe_output / "exit_code").write_text("0\n", encoding="utf-8")
    losses = {32: 0.4, 64: 0.3, 96: 0.2, 128: 0.25}
    candidates = []
    for step, loss in losses.items():
        checkpoint = recipe_output / f"checkpoint-{step}"
        checkpoint.mkdir()
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ):
            payload = (
                _adapter_config_bytes("recipe2")
                if name == "adapter_config.json"
                else f"{name}-{step}".encode()
            )
            (checkpoint / name).write_bytes(payload)
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"log_history": [{"step": step, "eval_loss": loss}]}),
            encoding="utf-8",
        )
        candidates.append(
            {"step": step, "eval_loss": loss, "checkpoint": f"/worker/checkpoint-{step}"}
        )
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (recipe_selected / name).write_bytes(
            (recipe_output / "checkpoint-96" / name).read_bytes()
        )
    train_rows = [{"videos": [f"/videos/train-{index}.mp4"]} for index in range(1024)]
    eval_rows = [{"videos": [f"/videos/eval-{index}.mp4"]} for index in range(256)]
    _write_jsonl(recipe_metadata / "refresh_train.jsonl", train_rows)
    _write_jsonl(recipe_metadata / "refresh_eval.jsonl", eval_rows)
    generation = {
        "recipe_version": 2,
        "counts": {"train": 1024, "validation": 256, "validation_cases": 256, "validation_gold": 256},
        "isolation": {"video_overlap": 0, "normalized_prompt_overlap": 0, "concept_overlap": 0, "template_overlap": 0},
        "forbidden_audit": {
            "exact_prompt_overlap": 0,
            "phrase_hit_count": 0,
            "synthetic_prompt_target_ngram_overlap": {"n": 4, "hit_count": 0, "hits": []},
        },
    }
    (recipe_metadata / "refresh_generation_audit.json").write_text(json.dumps(generation), encoding="utf-8")
    (recipe_metadata / "train_refresh.yaml").write_text(
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
    recipe_policy = {
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
        "train_sha256": hashlib.sha256((recipe_metadata / "refresh_train.jsonl").read_bytes()).hexdigest(),
        "eval_sha256": hashlib.sha256((recipe_metadata / "refresh_eval.jsonl").read_bytes()).hexdigest(),
    }
    (recipe_metadata / "refresh_selection_policy.json").write_text(json.dumps(recipe_policy), encoding="utf-8")
    (recipe_metadata / "refresh_selection.json").write_text(
        json.dumps(
            {
                "selection_source": "refresh_eval",
                "selection_metric": "eval_loss",
                "selection_policy": "lowest_refresh_eval_loss_then_earliest_step",
                "external_gate_metrics_used": False,
                "selected_step": 96,
                "selected_eval_loss": 0.2,
                "selected_checkpoint": "/worker/checkpoint-96",
                "candidates": candidates,
            }
        ),
        encoding="utf-8",
    )

    categories = [
        category for category, count in CORRECTED_CATEGORY_COUNTS.items() for _ in range(count)
    ]
    seen = {key: 0 for key in CORRECTED_CATEGORY_COUNTS}
    fresh_cases = []
    fresh_gold = []
    for index, category in enumerate(categories, 1):
        seen[category] += 1
        local = seen[category]
        if category == "no_search_negative":
            subtype = "generic_style" if local <= 64 else "generic_background" if local <= 96 else "ordinary_target"
        elif category == "mask_control":
            subtype = "triggered" if local <= 32 else "not_triggered"
        else:
            subtype = "control"
        bench_id = f"interp_{index:04d}"
        prompt = f"edit fixture item {index}"
        axis = (
            "search"
            if category in {"no_search_negative", "true_search_positive"}
            else "routing"
            if category == "routing_control"
            else "mask"
            if category == "mask_control"
            else "rewrite"
        )
        subtask = (
            "remove_object"
            if category == "mask_control" and subtype == "triggered"
            else "combined_tasks"
            if category == "rewrite_retention"
            else "global_style"
        )
        image_search = f"named item {index}" if category == "true_search_positive" else False
        mask = f"fixture object {index}" if subtask == "remove_object" else False
        constraint_values = [
            f"cobalt-{index}",
            f"item-{index}",
            f"left-{index}",
            f"keep-{index}",
        ]
        constraints = (
            [
                {"type": kind, "value": value}
                for kind, value in zip(
                    ("color", "identity", "spatial", "preservation"),
                    constraint_values,
                )
            ]
            if category == "rewrite_retention"
            else []
        )
        refined = (
            "Apply " + " ".join(constraint_values) + "."
            if constraints
            else f"Edit fixture item {index}."
        )
        case = {
            "bench_id": bench_id,
            "video_path": f"/fresh/video-{index}.mp4",
            "prompt": prompt,
            "axis": axis,
            "edit_type": subtask,
            "category": category,
            "subtype": subtype,
            "source": {
                "sample_id": f"source-{index}",
                "video_sha256": hashlib.sha256(f"video-{index}".encode()).hexdigest(),
            },
            "catalog": {
                "concept_id": f"concept-{index}",
                "template_id": f"template-{index}",
                "template_signature": f"signature-{index}",
            },
        }
        fresh_cases.append(case)
        fresh_gold.append(
            {
                **case,
                "gold_plan": {
                    "refined_text_instruction": refined,
                    "subtask": subtask,
                    "image_search": image_search,
                    "mask": mask,
                },
                "constraints": constraints,
                "source_entities": [] if image_search else ["source object"],
                **({"search_query_aliases": [image_search]} if image_search else {}),
            }
        )
    _write_jsonl(fresh / "cases.jsonl", fresh_cases)
    _write_jsonl(fresh / "gold.jsonl", fresh_gold)
    adaptive_gold = []
    for category in CORRECTED_CATEGORY_COUNTS:
        selected_rows = [row for row in fresh_gold if row["category"] == category][:20]
        for row in selected_rows:
            copied = json.loads(json.dumps(row))
            copied["bench_id"] = f"day14-{len(adaptive_gold):03d}"
            adaptive_gold.append(copied)
    adaptive_predictions = [
        {
            "bench_id": row["bench_id"],
            "plan": row["gold_plan"],
            "agent_raw": json.dumps(row["gold_plan"], separators=(",", ":")),
        }
        for row in adaptive_gold
    ]
    _write_jsonl(adaptive / "gold.jsonl", adaptive_gold)
    input_hashes = {
        "source_manifest": "1" * 64,
        "v2_train": "2" * 64,
        "v2_eval": "3" * 64,
        "refresh1_train": "4" * 64,
        "refresh1_eval": "5" * 64,
        "day14_forbidden_cases": hashlib.sha256(
            (adaptive / "gold.jsonl").read_bytes()
        ).hexdigest(),
    }
    strict_isolation = {
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
        "concept_phrase_policy": {
            "exact_concept_overlap_forbidden_for_all_categories": True,
            "phrase_containment_audit_only_categories": ["mask_control", "routing_control"],
        },
        "template_signature_policy": {
            "exact_prompt_and_template_id_overlap_forbidden_for_all_categories": True,
            "wildcard_signature_overlap_audit_only_categories": ["mask_control", "routing_control"],
        },
        "explicit_forbidden_phrase_hits": [],
    }
    fresh_audit = {
        "recipe_version": 2,
        "counts": {
            "cases": 384,
            "gold": 384,
            "by_category": dict(sorted(CORRECTED_CATEGORY_COUNTS.items())),
            "no_search_by_subtype": {"generic_background": 32, "generic_style": 64, "ordinary_target": 32},
            "mask_trigger": {"not_triggered": 32, "triggered": 32},
        },
        "isolation": strict_isolation,
        "artifacts": {
            "input_sha256": input_hashes,
            "cases_sha256": hashlib.sha256((fresh / "cases.jsonl").read_bytes()).hexdigest(),
            "gold_sha256": hashlib.sha256((fresh / "gold.jsonl").read_bytes()).hexdigest(),
        },
    }
    (fresh / "leakage_audit.json").write_text(json.dumps(fresh_audit), encoding="utf-8")
    primary_hash = hashlib.sha256((recipe_selected / "adapter_model.safetensors").read_bytes()).hexdigest()
    primary = {"candidate_id": "recipe2", "directory_name": "recipe2", "adapter_model_sha256": primary_hash}
    selection_policy = {
        "version": 2,
        "created_before_candidate_inference": True,
        "candidate_rule": {
            "method": "exact_parameter_delta_interpolation",
            "lambda_grid": list(CORRECTED_LAMBDA_GRID),
            "candidate_count": 9,
            "v2_adapter_model_sha256": "a" * 64,
            "refresh1_adapter_model_sha256": "b" * 64,
            "additional_training_allowed": False,
            "day14_outputs_or_metrics_allowed_for_generation_or_selection": False,
        },
        "validation_artifacts": {
            "cases_sha256": hashlib.sha256((fresh / "cases.jsonl").read_bytes()).hexdigest(),
            "gold_sha256": hashlib.sha256((fresh / "gold.jsonl").read_bytes()).hexdigest(),
            "leakage_audit_sha256": hashlib.sha256((fresh / "leakage_audit.json").read_bytes()).hexdigest(),
            "input_sha256": input_hashes,
            "expected_cases": 384,
            "expected_unique_videos": 384,
            "expected_category_counts": CORRECTED_CATEGORY_COUNTS,
            "expected_no_search_subtype_counts": {"generic_style": 64, "generic_background": 32, "ordinary_target": 32},
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
        "utility": {
            "formula": "0.5*no_search_specificity+0.5*rewrite_constraint_retention",
            "higher_is_better": True,
        },
        "bootstrap": {
            "method": "paired stratified case bootstrap over the two utility strata",
            "seed": 20260916,
            "draws": 10000,
            "strata": {"no_search_negative": 128, "rewrite_retention": 64},
        },
        "selection": {
            "rule": "one_standard_error_then_smallest_lambda",
            "reference_tie_breaker": "smallest lambda",
            "tie_breaker": "smallest lambda",
            "day14_metrics_used": False,
        },
        "final_decision": {"rule": "primary_if_eligible_else_grid", "primary_candidate": primary},
    }
    (fresh / "selection_policy.json").write_text(json.dumps(selection_policy), encoding="utf-8")
    cross_audit = {
        "schema_version": 1,
        "audit_role": "supplemental_post_seal",
        "passed": True,
        "method": {"paired_ngram_role": "diagnostic_only"},
        "sealed_artifacts": {
            "cases_sha256": hashlib.sha256((fresh / "cases.jsonl").read_bytes()).hexdigest(),
            "gold_sha256": hashlib.sha256((fresh / "gold.jsonl").read_bytes()).hexdigest(),
            "selection_policy_sha256": hashlib.sha256(
                (fresh / "selection_policy.json").read_bytes()
            ).hexdigest(),
        },
        "recipe2_artifacts": {
            "train_sha256": hashlib.sha256(
                (recipe_metadata / "refresh_train.jsonl").read_bytes()
            ).hexdigest(),
            "eval_sha256": hashlib.sha256(
                (recipe_metadata / "refresh_eval.jsonl").read_bytes()
            ).hexdigest(),
        },
        "blocking_overlaps": {
            "exact_prompt": 0,
            "exact_indexed_concept": 0,
            "concept_phrase": 0,
            "constraint_phrase": 0,
            "exact_template_signature_strict": 0,
            "wildcard_template_signature_strict": 0,
            "source_path": 0,
            "source_basename": 0,
            "source_sample_id": 0,
            "recipe_paths_missing_from_locked_v2": 0,
        },
        "blocking_examples": {
            "exact_prompt": [],
            "exact_indexed_concept": [],
            "concept_phrase": [],
            "constraint_phrase": [],
            "exact_template_signature_strict": [],
            "wildcard_template_signature_strict": [],
            "source_path": [],
            "source_basename": [],
            "source_sample_id": [],
            "recipe_paths_missing_from_locked_v2": [],
        },
        "diagnostics": {
            "generic_template_overlaps": {
                "allowed_categories": ["mask_control", "routing_control"],
                "blocking": False,
                "exact_hit_count": 0,
                "wildcard_hit_count": 0,
                "unique_signatures": [],
                "hits": [],
            },
            "paired_prompt_target_4gram": {
                "blocking": False,
                "n": 4,
                "pair_count": 0,
                "fresh_case_count": 384,
                "recipe_row_count": 1280,
                "by_fresh_category": {},
                "examples": [],
            },
            "concept_constraint_value_4gram": {
                "blocking": False,
                "n": 4,
                "hit_count": 0,
                "fresh_case_count": 384,
                "by_category": {},
                "examples": [],
            },
            "source_proof": {
                "recipe_unique_paths": 1280,
                "locked_v2_unique_paths": 12597,
                "recipe_paths_subset_locked_v2": True,
                "fresh_prior_video_sha256_overlap": 0,
            },
            "counts": {
                "fresh_cases": 384,
                "fresh_gold": 384,
                "recipe2_train_rows": 1024,
                "recipe2_eval_rows": 256,
                "recipe2_rows": 1280,
            },
        },
    }
    (fresh / "recipe2_cross_audit.json").write_text(
        json.dumps(cross_audit), encoding="utf-8"
    )
    endpoint_configs = selection_dir / "endpoint_configs"
    endpoint_configs.mkdir()
    for name in ("v2", "refresh1", "recipe2"):
        (endpoint_configs / f"{name}.adapter_config.json").write_bytes(
            _adapter_config_bytes(name)
        )
    endpoint_identities = {
        "schema_version": 1,
        "v2": {
            "adapter_model_sha256": "a" * 64,
            "adapter_config_sha256": CORRECTED_ADAPTER_CONFIG_SHA256["v2"],
        },
        "refresh1": {
            "adapter_model_sha256": "b" * 64,
            "adapter_config_sha256": CORRECTED_ADAPTER_CONFIG_SHA256["refresh1"],
        },
        "recipe2": {
            "adapter_model_sha256": primary_hash,
            "adapter_config_sha256": CORRECTED_ADAPTER_CONFIG_SHA256["recipe2"],
        },
    }
    (selection_dir / "adapter_identities.json").write_text(
        json.dumps(endpoint_identities), encoding="utf-8"
    )

    validation_hashes = {
        "cases_sha256": hashlib.sha256((fresh / "cases.jsonl").read_bytes()).hexdigest(),
        "gold_sha256": hashlib.sha256((fresh / "gold.jsonl").read_bytes()).hexdigest(),
        "leakage_audit_sha256": hashlib.sha256((fresh / "leakage_audit.json").read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256((fresh / "selection_policy.json").read_bytes()).hexdigest(),
    }
    passing_metrics = {
        "prediction_rows": 384,
        "strict_raw_json_validity": 1.0,
        "subtask_accuracy": 1.0,
        "no_search_specificity": 1.0,
        "true_search_trigger_recall": 1.0,
        "search_query_end_to_end_recall": 1.0,
        "mask_trigger_f1": 1.0,
        "rewrite_constraint_retention": 1.0,
        "utility": 1.0,
    }
    threshold_metrics = {
        "complete_prediction_rows": ("prediction_rows", "==", 384),
        "strict_raw_json_validity_min": ("strict_raw_json_validity", ">=", 1.0),
        "subtask_accuracy_min": ("subtask_accuracy", ">=", 0.95),
        "no_search_specificity_min": ("no_search_specificity", ">=", 0.95),
        "true_search_trigger_recall_min": ("true_search_trigger_recall", ">=", 0.95),
        "search_query_end_to_end_recall_min": ("search_query_end_to_end_recall", ">=", 0.95),
        "mask_trigger_f1_min": ("mask_trigger_f1", ">=", 0.95),
        "rewrite_constraint_retention_min": ("rewrite_constraint_retention", ">=", 0.85),
    }
    passing_checks = {
        name: {
            "metric": metric,
            "value": passing_metrics[metric],
            "operator": operator,
            "threshold": threshold,
            "passed": True,
        }
        for name, (metric, operator, threshold) in threshold_metrics.items()
    }
    perfect_predictions = [
        {
            "bench_id": row["bench_id"],
            "plan": row["gold_plan"],
            "agent_raw": json.dumps(row["gold_plan"], separators=(",", ":")),
        }
        for row in fresh_gold
    ]
    perfect_bootstrap = {
        "draws": 10000,
        "seed": 20260916,
        "standard_error": 0.0,
        "bootstrap_mean": 1.0,
        "standard_deviation_denominator": "draws-1",
        "rng": "python random.Random(seed), paired randrange indices",
    }
    comparison_candidates = []
    for value in CORRECTED_LAMBDA_GRID:
        candidate_id = f"lambda_{round(value * 1000):04d}"
        candidate = root / "candidates" / candidate_id
        adapter = candidate / "adapter"
        eval_dir = candidate / "eval"
        adapter.mkdir(parents=True)
        eval_dir.mkdir()
        (adapter / "adapter_config.json").write_bytes(_grid_adapter_config_bytes())
        (adapter / "adapter_model.safetensors").write_bytes(f"grid-{value}".encode())
        config_sha = hashlib.sha256((adapter / "adapter_config.json").read_bytes()).hexdigest()
        model_sha = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
        provenance = {
            "schema_version": 1,
            "method": "exact_delta_space_rank_concat",
            "equation": "delta_out=(1-lambda_b)*delta_a+lambda_b*delta_b",
            "coefficient": {"lambda_b": value, "adapter_a_weight": 1.0 - value, "adapter_b_weight": value},
            "sources": {
                "adapter_a": {"path": "/worker/v2", **endpoint_identities["v2"]},
                "adapter_b": {"path": "/worker/refresh1", **endpoint_identities["refresh1"]},
            },
            "lora": {
                "input_rank": 32,
                "input_lora_alpha": 64,
                "input_scaling": 2.0,
                "output_rank": 64,
                "output_lora_alpha": 128,
                "output_scaling": 2.0,
                "module_count": 252,
                "tensor_count": 504,
                "dtype": "float32",
            },
            "output": {
                "path": str(adapter),
                "adapter_config_sha256": config_sha,
                "adapter_model_sha256": model_sha,
                "copied_inference_metadata": [],
            },
        }
        (adapter / "interpolation_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        _write_jsonl(eval_dir / "agent_pipeline_records.jsonl", perfect_predictions)
        (eval_dir / "planner.log").write_text("planner\n", encoding="utf-8")
        (eval_dir / "metrics.json").write_text(json.dumps({"num_cases": 384}), encoding="utf-8")
        (eval_dir / "scorer.log").write_text("scorer\n", encoding="utf-8")
        _write_complete_eval_manifest(eval_dir, cases=fresh / "cases.jsonl", gold=fresh / "gold.jsonl", adapter=adapter, expected_cases=384)
        comparison_candidates.append(
            {
                "candidate_id": candidate_id,
                "kind": "grid",
                "lambda": value,
                "adapter_identity": {
                    "adapter_config_sha256": config_sha,
                    "adapter_model_sha256": model_sha,
                    "provenance_sha256": hashlib.sha256((adapter / "interpolation_provenance.json").read_bytes()).hexdigest(),
                },
                "prediction_artifact": {"rows": 384, "sha256": hashlib.sha256((eval_dir / "agent_pipeline_records.jsonl").read_bytes()).hexdigest()},
                "planner_log": {"sha256": hashlib.sha256((eval_dir / "planner.log").read_bytes()).hexdigest()},
                "metrics": dict(passing_metrics),
                "eligibility": {"eligible": True, "checks": passing_checks},
                "bootstrap": perfect_bootstrap,
            }
        )

    primary_candidate = root / "candidates/recipe2"
    primary_adapter = primary_candidate / "adapter"
    primary_eval = primary_candidate / "eval"
    primary_adapter.mkdir(parents=True)
    primary_eval.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (primary_adapter / name).write_bytes((recipe_selected / name).read_bytes())
    _write_jsonl(primary_eval / "agent_pipeline_records.jsonl", perfect_predictions)
    (primary_eval / "planner.log").write_text("planner\n", encoding="utf-8")
    (primary_eval / "metrics.json").write_text(json.dumps({"num_cases": 384}), encoding="utf-8")
    (primary_eval / "scorer.log").write_text("scorer\n", encoding="utf-8")
    _write_complete_eval_manifest(primary_eval, cases=fresh / "cases.jsonl", gold=fresh / "gold.jsonl", adapter=primary_adapter, expected_cases=384)
    primary_metrics = dict(passing_metrics)
    comparison_candidates.append(
        {
            "candidate_id": "recipe2",
            "kind": "primary",
            "lambda": None,
            "adapter_identity": {
                "adapter_config_sha256": hashlib.sha256((primary_adapter / "adapter_config.json").read_bytes()).hexdigest(),
                "adapter_model_sha256": primary_hash,
                "provenance_sha256": None,
            },
            "prediction_artifact": {"rows": 384, "sha256": hashlib.sha256((primary_eval / "agent_pipeline_records.jsonl").read_bytes()).hexdigest()},
            "planner_log": {"sha256": hashlib.sha256((primary_eval / "planner.log").read_bytes()).hexdigest()},
            "metrics": primary_metrics,
            "eligibility": {"eligible": True, "checks": passing_checks},
            "bootstrap": perfect_bootstrap,
        }
    )
    grid_decision = {
        "reference_candidate_id": "lambda_0000",
        "reference_lambda": 0.0,
        "best_utility": 1.0,
        "reference_bootstrap_se": 0.0,
        "one_se_cutoff": 1.0,
        "one_se_candidate_ids": [f"lambda_{round(value * 1000):04d}" for value in CORRECTED_LAMBDA_GRID],
        "selected_candidate_id": "lambda_0000",
        "selected_lambda": 0.0,
    }
    comparison = {
        "schema_version": 1,
        "selector": "fresh384_interpolation_selector",
        "day14_metrics_used": False,
        "validation_artifacts": validation_hashes,
        "policy": {
            "lambda_grid": list(CORRECTED_LAMBDA_GRID),
            "bootstrap_seed": 20260916,
            "bootstrap_draws": 10000,
            "eligibility_thresholds": selection_policy["eligibility_thresholds"],
            "primary_if_eligible_else_grid": primary,
        },
        "candidates": comparison_candidates,
        "grid_decision": grid_decision,
    }
    (selection_dir / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
    final_selection = {
        "schema_version": 1,
        "selected": True,
        "selected_candidate_id": "recipe2",
        "selected_kind": "primary",
        "selected_lambda": None,
        "selected_adapter_dir": str(primary_adapter.resolve()),
        "selected_adapter_model_sha256": primary_hash,
        "selected_metrics": primary_metrics,
        "primary_candidate_id": "recipe2",
        "primary_eligible": True,
        "grid_decision": grid_decision,
        "day14_metrics_used": False,
        "validation_artifacts": validation_hashes,
        "comparison_sha256": hashlib.sha256((selection_dir / "comparison.json").read_bytes()).hexdigest(),
    }
    (selection_dir / "selection.json").write_text(json.dumps(final_selection), encoding="utf-8")

    _write_jsonl(adaptive / "agent_pipeline_records.jsonl", adaptive_predictions)
    (adaptive / "planner.log").write_text(
        "Agent base: /models/qwen\n"
        f"Agent adapter: {primary_adapter.resolve()}\n",
        encoding="utf-8",
    )
    gate_metrics = score_agent_plans(adaptive_gold, adaptive_predictions)
    (adaptive / "metrics.json").write_text(json.dumps(gate_metrics), encoding="utf-8")
    (adaptive / "scorer.log").write_text("scorer\n", encoding="utf-8")
    gate_summary = evaluate_gate(gate_metrics, RELEASED_BASELINE)
    gate_summary["released_baseline_source"] = (
        "pre-registered constants: routing=0.78, retention=0.48"
    )
    (adaptive / "gate_summary.json").write_text(
        json.dumps(gate_summary),
        encoding="utf-8",
    )
    write_manifest(root, root / "checksums.sha256")


def _write_corrected_thin_fixture(root: Path) -> None:
    _write_corrected_selection_fixture(root)
    recipe_selected = root / "recipe2/lora-refresh-selected"
    identity = {
        "schema_version": 1,
        "selected_step": 96,
        "adapter_config_sha256": hashlib.sha256(
            (recipe_selected / "adapter_config.json").read_bytes()
        ).hexdigest(),
        "adapter_model_sha256": hashlib.sha256(
            (recipe_selected / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
    }
    (root / "recipe2/metadata/selected_adapter_identity.json").write_text(
        json.dumps(identity), encoding="utf-8"
    )
    for path in (
        root / "recipe2/lora-refresh/adapter_config.json",
        root / "recipe2/lora-refresh/adapter_model.safetensors",
        recipe_selected / "adapter_config.json",
        recipe_selected / "adapter_model.safetensors",
    ):
        path.unlink()
    recipe_selected.rmdir()
    for step in (32, 64, 96, 128):
        checkpoint = root / f"recipe2/lora-refresh/checkpoint-{step}"
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ):
            (checkpoint / name).unlink()
    for value in CORRECTED_LAMBDA_GRID:
        (root / f"candidates/lambda_{round(value * 1000):04d}/adapter/adapter_model.safetensors").unlink()
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

    def test_corrected_selection_audit_accepts_portable_complete_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_selection_fixture(root)

            result = audit_corrected_selection_bundle(root)

            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(result["observations"]["profile"], "corrected-full")
            self.assertEqual(
                result["observations"]["corrected_selection"]["selected_candidate_id"],
                "recipe2",
            )
            self.assertFalse(result["observations"]["adaptive_day14"]["confirmatory"])
            self.assertEqual(result["observations"]["adaptive_day14"]["run_count"], 1)

    def test_corrected_thin_audit_accepts_only_selected_adapter_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_thin_fixture(root)

            result = audit_corrected_thin_bundle(root)

            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(result["observations"]["profile"], "corrected-thin")
            self.assertTrue(
                (root / "candidates/recipe2/adapter/adapter_model.safetensors").is_file()
            )
            self.assertFalse(
                (root / "candidates/lambda_0000/adapter/adapter_model.safetensors").exists()
            )

    def test_corrected_thin_rejects_reconstructible_recipe2_blobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_thin_fixture(root)
            extra = root / "recipe2/lora-refresh/checkpoint-32/adapter_model.safetensors"
            extra.write_bytes(b"reconstructible")
            write_manifest(root, root / "checksums.sha256")

            result = audit_corrected_thin_bundle(root)

            self.assertFalse(result["ok"])
            self.assertTrue(
                any("retains reconstructible blobs" in error for error in result["errors"]),
                result["errors"],
            )
            self.assertTrue(
                any("only the final selected adapter weights" in error for error in result["errors"]),
                result["errors"],
            )

    def test_corrected_selection_rejects_leakage_and_selection_hash_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_selection_fixture(root)
            audit_path = root / "fresh384/leakage_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["isolation"]["prior_exact_prompt_overlap"] = 1
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            selection_path = root / "selection/selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["selected_adapter_model_sha256"] = "0" * 64
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            write_manifest(root, root / "checksums.sha256")

            result = audit_corrected_selection_bundle(root)

            self.assertFalse(result["ok"])
            self.assertTrue(
                any("prior_exact_prompt_overlap" in error for error in result["errors"]),
                result["errors"],
            )
            self.assertIn("final selection adapter hash mismatch", result["errors"])

    def test_corrected_selection_rejects_wrong_adapter_or_extra_day14_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_selection_fixture(root)
            (root / "adaptive_day14/planner.log").write_text(
                "Agent adapter: /wrong/adapter\n", encoding="utf-8"
            )
            extra = root / "day14_second/agent_pipeline_records.jsonl"
            extra.parent.mkdir()
            extra.write_text('{"bench_id":"extra"}\n', encoding="utf-8")
            write_manifest(root, root / "checksums.sha256")

            result = audit_corrected_selection_bundle(root)

            self.assertFalse(result["ok"])
            self.assertTrue(
                any("exact selected adapter path" in error for error in result["errors"]),
                result["errors"],
            )
            self.assertTrue(
                any("one adaptive Day-14 run" in error for error in result["errors"]),
                result["errors"],
            )

    def test_corrected_selection_rejects_disguised_extra_prediction_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_selection_fixture(root)
            extra = root / "rerun/agent_pipeline_records.jsonl"
            extra.parent.mkdir()
            extra.write_bytes(
                (root / "adaptive_day14/agent_pipeline_records.jsonl").read_bytes()
            )
            write_manifest(root, root / "checksums.sha256")

            result = audit_corrected_selection_bundle(root)

            self.assertFalse(result["ok"])
            self.assertTrue(
                any("one adaptive Day-14 run" in error for error in result["errors"]),
                result["errors"],
            )

    def test_corrected_selection_binds_day14_to_selected_run_adapter_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_selection_fixture(root)
            selection_path = root / "selection/selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["selected_adapter_dir"] = "/wrong/adapter"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            (root / "adaptive_day14/planner.log").write_text(
                "Agent adapter: /wrong/adapter\n", encoding="utf-8"
            )
            write_manifest(root, root / "checksums.sha256")

            result = audit_corrected_selection_bundle(root)

            self.assertFalse(result["ok"])
            self.assertIn(
                "final selection adapter path differs from the selected run manifest",
                result["errors"],
            )

    def test_corrected_selection_recomputes_adaptive_day14_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_selection_fixture(root)
            metrics_path = root / "adaptive_day14/metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["subtask_accuracy"] = 0.99
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            (root / "adaptive_day14/planner.log").write_text(
                "Agent base: /wrong/base\n"
                f"Agent adapter: {(root / 'candidates/recipe2/adapter').resolve()}\n",
                encoding="utf-8",
            )
            write_manifest(root, root / "checksums.sha256")

            result = audit_corrected_selection_bundle(root)

            self.assertFalse(result["ok"])
            self.assertIn(
                "adaptive Day-14 metrics differ from bundled gold/predictions",
                result["errors"],
            )
            self.assertTrue(
                any("exact selected base path" in error for error in result["errors"]),
                result["errors"],
            )

    def test_corrected_selection_rejects_endpoint_config_or_provenance_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_selection_fixture(root)
            endpoint = root / "selection/endpoint_configs/v2.adapter_config.json"
            endpoint.write_text("{}", encoding="utf-8")
            provenance_path = (
                root / "candidates/lambda_0000/adapter/interpolation_provenance.json"
            )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["sources"]["adapter_a"]["adapter_config_sha256"] = "0" * 64
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            write_manifest(root, root / "checksums.sha256")

            result = audit_corrected_selection_bundle(root)

            self.assertFalse(result["ok"])
            self.assertIn(
                "endpoint config v2 differs from the locked production bytes",
                result["errors"],
            )
            self.assertIn(
                "candidate lambda_0000 provenance v2 endpoint mismatch",
                result["errors"],
            )

    def test_corrected_selection_rejects_forged_grid_config_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_selection_fixture(root)
            adapter = root / "candidates/lambda_0000/adapter"
            config_path = adapter / "adapter_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": "/wrong",
                        "r": 1,
                        "lora_alpha": 1,
                    }
                ),
                encoding="utf-8",
            )
            config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
            provenance_path = adapter / "interpolation_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["output"]["adapter_config_sha256"] = config_sha
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            manifest_path = root / "candidates/lambda_0000/eval/run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["inputs"]["adapter_config"]["sha256"] = config_sha
            manifest["inputs"]["adapter_provenance"]["sha256"] = hashlib.sha256(
                provenance_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            comparison_path = root / "selection/comparison.json"
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            row = next(
                item
                for item in comparison["candidates"]
                if item["candidate_id"] == "lambda_0000"
            )
            row["adapter_identity"]["adapter_config_sha256"] = config_sha
            row["adapter_identity"]["provenance_sha256"] = hashlib.sha256(
                provenance_path.read_bytes()
            ).hexdigest()
            comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
            selection_path = root / "selection/selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["comparison_sha256"] = hashlib.sha256(
                comparison_path.read_bytes()
            ).hexdigest()
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            write_manifest(root, root / "checksums.sha256")

            result = audit_corrected_selection_bundle(root)

            self.assertFalse(result["ok"])
            self.assertIn(
                "candidate lambda_0000 config is not the exact rank-concat output config",
                result["errors"],
            )

    def test_corrected_selection_rejects_rewrite_without_four_constraints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_selection_fixture(root)
            gold_path = root / "fresh384/gold.jsonl"
            rows = [json.loads(line) for line in gold_path.read_text().splitlines()]
            rewrite = next(row for row in rows if row["category"] == "rewrite_retention")
            rewrite["constraints"] = rewrite["constraints"][:1]
            _write_jsonl(gold_path, rows)
            write_manifest(root, root / "checksums.sha256")

            result = audit_corrected_selection_bundle(root)

            self.assertFalse(result["ok"])
            self.assertTrue(
                any("must contain four constraints" in error for error in result["errors"]),
                result["errors"],
            )

    def test_corrected_selection_rejects_unbound_candidate_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_selection_fixture(root)
            manifest_path = root / "candidates/lambda_0000/eval/run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["inputs"]["source_video:interp_0001"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            write_manifest(root, root / "checksums.sha256")

            result = audit_corrected_selection_bundle(root)

            self.assertFalse(result["ok"])
            self.assertTrue(
                any("source-video hash mismatch" in error for error in result["errors"]),
                result["errors"],
            )

    def test_corrected_selection_reports_malformed_ids_and_plans_without_crashing(self):
        mutations = ("bench_id", "gold_plan")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_corrected_selection_fixture(root)
                cases_path = root / "fresh384/cases.jsonl"
                gold_path = root / "fresh384/gold.jsonl"
                cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
                gold = [json.loads(line) for line in gold_path.read_text().splitlines()]
                if mutation == "bench_id":
                    cases[0]["bench_id"] = ["not", "hashable"]
                    gold[0]["bench_id"] = ["not", "hashable"]
                    _write_jsonl(cases_path, cases)
                else:
                    gold[0]["gold_plan"] = []
                _write_jsonl(gold_path, gold)
                write_manifest(root, root / "checksums.sha256")

                result = audit_corrected_selection_bundle(root)

                self.assertFalse(result["ok"])
                self.assertIsInstance(result["errors"], list)
                self.assertTrue(result["errors"])

    def test_corrected_selection_recomputes_locked_grid_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_selection_fixture(root)
            comparison_path = root / "selection/comparison.json"
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            comparison["grid_decision"]["selected_candidate_id"] = "lambda_1000"
            comparison["grid_decision"]["selected_lambda"] = 1.0
            comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
            selection_path = root / "selection/selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["grid_decision"] = comparison["grid_decision"]
            selection["comparison_sha256"] = hashlib.sha256(
                comparison_path.read_bytes()
            ).hexdigest()
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            write_manifest(root, root / "checksums.sha256")

            result = audit_corrected_selection_bundle(root)

            self.assertFalse(result["ok"])
            self.assertTrue(
                any("one-SE/smallest-lambda" in error for error in result["errors"]),
                result["errors"],
            )

    def test_corrected_selection_reports_missing_candidate_lambda_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_selection_fixture(root)
            comparison_path = root / "selection/comparison.json"
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            del comparison["candidates"][0]["lambda"]
            comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
            selection_path = root / "selection/selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["comparison_sha256"] = hashlib.sha256(
                comparison_path.read_bytes()
            ).hexdigest()
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            write_manifest(root, root / "checksums.sha256")

            result = audit_corrected_selection_bundle(root)

            self.assertFalse(result["ok"])
            self.assertIn(
                "comparison kind/lambda mismatch for lambda_0000", result["errors"]
            )

    def test_corrected_selection_rejects_nonzero_recipe2_cross_audit_blocker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_corrected_selection_fixture(root)
            cross_path = root / "fresh384/recipe2_cross_audit.json"
            cross = json.loads(cross_path.read_text(encoding="utf-8"))
            cross["blocking_overlaps"]["exact_prompt"] = 1
            cross_path.write_text(json.dumps(cross), encoding="utf-8")
            write_manifest(root, root / "checksums.sha256")

            result = audit_corrected_selection_bundle(root)

            self.assertFalse(result["ok"])
            self.assertIn(
                "recipe2 cross-audit has a non-zero blocking overlap", result["errors"]
            )


if __name__ == "__main__":
    unittest.main()

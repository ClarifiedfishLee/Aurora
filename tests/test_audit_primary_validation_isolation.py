from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_primary_validation_isolation import (
    BLOCKING_OVERLAP_KEYS,
    IsolationAuditError,
    _parse_recipe_rows,
    analyze_isolation,
    write_fresh_json,
)


def _plan(refined: str, subtask: str, *, search=False, mask=False):
    return {
        "refined_text_instruction": refined,
        "subtask": subtask,
        "image_search": search,
        "mask": mask,
    }


def _share(prompt: str, plan: dict, video: str) -> dict:
    return {
        "system": "planner",
        "messages": [
            {"role": "user", "content": f"<video>{prompt}"},
            {"role": "assistant", "content": json.dumps(plan)},
        ],
        "videos": [video],
    }


def _fresh(
    bench_id: str,
    *,
    category: str,
    subtype: str,
    prompt: str,
    concept: str,
    signature: str,
    video: str,
    plan: dict,
    constraints: list[dict] | None = None,
) -> tuple[dict, dict]:
    case = {
        "bench_id": bench_id,
        "video_path": video,
        "prompt": prompt,
        "category": category,
        "subtype": subtype,
        "source": {"sample_id": Path(video).stem, "video_sha256": "a" * 64},
        "catalog": {"concept_id": concept, "template_signature": signature},
    }
    return case, {
        **case,
        "gold_plan": plan,
        "constraints": constraints or [],
    }


class IsolationAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fresh_audit = {"isolation": {"prior_video_sha256_overlap": 0}}

    def test_generic_template_overlap_is_audit_only(self) -> None:
        case, gold = _fresh(
            "interp_0001",
            category="routing_control",
            subtype="remove_object",
            prompt="erase the frayed cord by the stool",
            concept="frayed cord by the stool",
            signature="erase the __slot__",
            video="/fresh/fresh_1.mp4",
            plan=_plan(
                "Remove the frayed cord by the stool.",
                "remove_object",
                mask="frayed cord by the stool",
            ),
        )
        recipe = [
            {
                "split": "train",
                "index": 1,
                "prompt": "erase the small item beside the chair",
                "plan": _plan(
                    "Remove the small item beside the chair.",
                    "remove_object",
                    mask="small item beside the chair",
                ),
                "video": "/v2/old_1.mp4",
            }
        ]
        blocking, diagnostics, examples = analyze_isolation(
            cases=[case],
            gold=[gold],
            recipe=recipe,
            locked_v2_video_paths={"/v2/old_1.mp4"},
            fresh_leakage_audit=self.fresh_audit,
        )
        self.assertEqual(tuple(blocking), BLOCKING_OVERLAP_KEYS)
        self.assertTrue(all(value == 0 for value in blocking.values()))
        generic = diagnostics["generic_template_overlaps"]
        self.assertEqual(generic["exact_hit_count"], 1)
        self.assertEqual(generic["wildcard_hit_count"], 1)
        self.assertEqual(generic["unique_signatures"], ["erase the __slot__"])
        self.assertTrue(all(not rows for rows in examples.values()))

    def test_non_generic_prompt_concept_constraint_and_template_overlap_block(self) -> None:
        case, gold = _fresh(
            "interp_0002",
            category="rewrite_retention",
            subtype="lexical_copy",
            prompt="make the central item ochre",
            concept="ochre item",
            signature="make the central item __slot__",
            video="/fresh/shared.mp4",
            plan=_plan("Make the central item ochre.", "change_color"),
            constraints=[{"type": "color", "value": "ochre"}],
        )
        recipe = [
            {
                "split": "train",
                "index": 1,
                "prompt": "make the central item ochre",
                "plan": _plan(
                    "Make the ochre item and preserve the ochre finish.",
                    "change_color",
                    search="ochre item",
                ),
                "video": "/fresh/shared.mp4",
            }
        ]
        blocking, _, examples = analyze_isolation(
            cases=[case],
            gold=[gold],
            recipe=recipe,
            locked_v2_video_paths={"/v2/unrelated.mp4"},
            fresh_leakage_audit=self.fresh_audit,
        )
        self.assertEqual(blocking["exact_prompt"], 1)
        self.assertEqual(blocking["exact_indexed_concept"], 1)
        self.assertEqual(blocking["concept_phrase"], 1)
        self.assertEqual(blocking["constraint_phrase"], 1)
        self.assertGreaterEqual(blocking["wildcard_template_signature_strict"], 1)
        self.assertEqual(blocking["source_path"], 1)
        self.assertEqual(blocking["source_basename"], 1)
        self.assertEqual(blocking["source_sample_id"], 1)
        self.assertEqual(blocking["recipe_paths_missing_from_locked_v2"], 1)
        self.assertTrue(examples["exact_prompt"])

    def test_paired_fourgram_overlap_is_recorded_not_blocking(self) -> None:
        case, gold = _fresh(
            "interp_0003",
            category="no_search_negative",
            subtype="ordinary_target",
            prompt="swap the nearest loose prop for a plain clay cup",
            concept="plain clay cup",
            signature="swap the nearest loose prop for __slot__",
            video="/fresh/fresh_3.mp4",
            plan=_plan("Replace the loose prop with a plain clay cup.", "replace_object"),
        )
        recipe = [
            {
                "split": "eval",
                "index": 7,
                "prompt": "exchange the closest prop for a plain wood bowl",
                "plan": _plan(
                    "Exchange the side prop with a plain wood bowl.",
                    "replace_object",
                ),
                "video": "/v2/old_7.mp4",
            }
        ]
        blocking, diagnostics, _ = analyze_isolation(
            cases=[case],
            gold=[gold],
            recipe=recipe,
            locked_v2_video_paths={"/v2/old_7.mp4"},
            fresh_leakage_audit=self.fresh_audit,
        )
        self.assertEqual(sum(blocking.values()), 0)
        paired = diagnostics["paired_prompt_target_4gram"]
        self.assertFalse(paired["blocking"])
        self.assertEqual(paired["pair_count"], 1)
        self.assertEqual(paired["fresh_case_count"], 1)

    def test_nonzero_prior_video_sha_audit_fails_closed(self) -> None:
        case, gold = _fresh(
            "interp_0004",
            category="routing_control",
            subtype="camera_edit",
            prompt="use a slow left arc",
            concept="slow left arc",
            signature="use __slot__",
            video="/fresh/fresh_4.mp4",
            plan=_plan("Use a slow left arc.", "camera_edit"),
        )
        with self.assertRaisesRegex(IsolationAuditError, "video-SHA"):
            analyze_isolation(
                cases=[case],
                gold=[gold],
                recipe=[],
                locked_v2_video_paths=set(),
                fresh_leakage_audit={"isolation": {"prior_video_sha256_overlap": 1}},
            )


class IsolationHelpersTest(unittest.TestCase):
    def test_recipe_row_counts_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(IsolationAuditError, "recipe2 train"):
            _parse_recipe_rows([], [])

    def test_output_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            write_fresh_json(path, {"schema_version": 1})
            self.assertEqual(json.loads(path.read_text()), {"schema_version": 1})
            with self.assertRaisesRegex(IsolationAuditError, "refusing to overwrite"):
                write_fresh_json(path, {"schema_version": 1})


if __name__ == "__main__":
    unittest.main()

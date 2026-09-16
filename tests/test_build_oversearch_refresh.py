import json
import unittest
from collections import Counter
from unittest.mock import patch

import scripts.build_oversearch_refresh as refresh_builder
from evaluation.agent_only_score import constraint_is_retained, valid_plan
from scripts.augment_planner_sft import EXTERNAL_ENTITY_GROUPS
from scripts.build_oversearch_refresh import (
    DEFAULT_FORBIDDEN_NGRAM_N,
    DEFAULT_FORBIDDEN_CONCEPTS,
    RECIPE_VERSION,
    SUBTASK_ORDER,
    TRAIN_CATEGORY_COUNTS,
    VALIDATION_CATEGORY_COUNTS,
    build_refresh,
    paired_prompt_target_ngram_overlap,
)


def make_row(index: int, split: str, prompt: str, plan: dict) -> dict:
    return {
        "system": "Return one planner JSON object.",
        "messages": [
            {"role": "user", "content": f"<video>\n{prompt}"},
            {"role": "assistant", "content": json.dumps(plan, separators=(",", ":"))},
        ],
        "videos": [f"/fixtures/{split}/video-{index:05d}.mp4"],
    }


def plan(refined: str, subtask: str, search=False, mask=False) -> dict:
    return {
        "refined_text_instruction": refined,
        "subtask": subtask,
        "image_search": search,
        "mask": mask,
    }


def fixture_rows() -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    index = 0
    # More than 32 eligible concepts in every category/subtask matrix cell.
    for group, entities in EXTERNAL_ENTITY_GROUPS.items():
        for subtask in ("add_object", "replace_object"):
            for entity_index, entity in enumerate(entities):
                prompt = f"fixture {subtask} {entity} variant {entity_index}"
                train.append(
                    make_row(
                        index,
                        "train",
                        prompt,
                        plan(
                            f"Use {entity} for fixture {subtask} variant {entity_index}.",
                            subtask,
                            search=entity,
                        ),
                    )
                )
                index += 1

    # Thirty no-search examples per route support the 23/24-row route quotas.
    for subtask in SUBTASK_ORDER:
        for variant in range(30):
            mask = f"fixture removable object {variant}" if subtask == "remove_object" else False
            train.append(
                make_row(
                    index,
                    "train",
                    f"fixture route request {subtask} number {variant}",
                    plan(
                        f"Fixture route instruction {subtask} number {variant}.",
                        subtask,
                        mask=mask,
                    ),
                )
            )
            index += 1

    # A large canonical/hard-like pool remains after search and route replay.
    for variant in range(450):
        subtask = SUBTASK_ORDER[variant % len(SUBTASK_ORDER)]
        mask = f"canonical removable prop {variant}" if subtask == "remove_object" else False
        train.append(
            make_row(
                index,
                "train",
                f"canonical colloquial request unique {variant}",
                plan(f"Canonical preservation instruction unique {variant}.", subtask, mask=mask),
            )
        )
        index += 1

    train.append(
        make_row(
            index,
            "train",
            "a held-out gate sentence that must never appear",
            plan("Keep this ordinary fixture out of refresh replay.", "change_color"),
        )
    )

    evaluation = []
    for eval_index in range(180):
        subtask = SUBTASK_ORDER[eval_index % len(SUBTASK_ORDER)]
        mask = f"evaluation removable prop {eval_index}" if subtask == "remove_object" else False
        evaluation.append(
            make_row(
                eval_index,
                "eval",
                f"held out evaluation request unique {eval_index}",
                plan(
                    f"Held out evaluation instruction unique {eval_index}.",
                    subtask,
                    mask=mask,
                ),
            )
        )
    return train, evaluation


def normalized_prompt(row: dict) -> str:
    return " ".join(row["messages"][0]["content"].casefold().split())


class BuildOversearchRefreshTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_train, cls.base_eval = fixture_rows()
        cls.forbidden_cases = [
            {
                "bench_id": "fixture_forbidden_entity",
                "prompt": "a held-out gate sentence that must never appear",
                "gold_plan": plan(
                    "Use the held-out proper entity.",
                    "replace_object",
                    search="Held-Out Proper Entity",
                ),
            },
            {
                "bench_id": "fixture_forbidden_weather",
                "prompt": "make it lightly snowing but keep everything visible",
                "gold_plan": plan(
                    "Change the weather to light snowfall while keeping every subject clearly visible.",
                    "change_weather",
                ),
            },
        ]
        (
            cls.train,
            cls.validation,
            cls.cases,
            cls.gold,
            cls.summary,
        ) = build_refresh(
            cls.base_train,
            cls.base_eval,
            forbidden_case_rows=cls.forbidden_cases,
        )

    def test_exact_recipe_counts_and_sharegpt_contract(self) -> None:
        self.assertEqual(self.summary["recipe_version"], RECIPE_VERSION)
        self.assertEqual((len(self.train), len(self.validation)), (1024, 256))
        self.assertEqual((len(self.cases), len(self.gold)), (256, 256))
        self.assertEqual(
            self.summary["counts"]["train_by_category"], dict(sorted(TRAIN_CATEGORY_COUNTS.items()))
        )
        self.assertEqual(
            self.summary["counts"]["validation_by_category"],
            dict(sorted(VALIDATION_CATEGORY_COUNTS.items())),
        )
        for row in [*self.train, *self.validation]:
            self.assertEqual([message["role"] for message in row["messages"]], ["user", "assistant"])
            self.assertEqual(row["messages"][0]["content"].count("<video>"), 1)
            self.assertEqual(len(row["videos"]), 1)
            parsed_plan = json.loads(row["messages"][1]["content"])
            self.assertTrue(valid_plan(parsed_plan))
            if parsed_plan["subtask"] == "remove_object":
                self.assertIsInstance(parsed_plan["mask"], str)
            else:
                self.assertIs(parsed_plan["mask"], False)

    def test_search_replay_is_balanced_by_entity_group_and_add_replace(self) -> None:
        matrix = self.summary["train_search_replay_matrix"]
        self.assertEqual(len(matrix), 8)
        self.assertEqual(set(matrix.values()), {32})
        self.assertEqual(
            set(self.summary["train_search_unique_concepts_by_matrix"].values()), {32}
        )
        for group in EXTERNAL_ENTITY_GROUPS:
            self.assertEqual(matrix[f"{group}:add_object"], 32)
            self.assertEqual(matrix[f"{group}:replace_object"], 32)
        self.assertEqual(self.summary["search_distribution"]["validation"]["triggered"], 64)

    def test_video_prompt_concept_and_template_splits_are_isolated(self) -> None:
        isolation = self.summary["isolation"]
        self.assertEqual(isolation["video_overlap"], 0)
        self.assertEqual(isolation["normalized_prompt_overlap"], 0)
        self.assertEqual(isolation["concept_overlap"], 0)
        self.assertEqual(isolation["template_overlap"], 0)
        self.assertFalse(
            {row["videos"][0] for row in self.train}
            & {row["videos"][0] for row in self.validation}
        )
        self.assertFalse(
            {normalized_prompt(row) for row in self.train}
            & {normalized_prompt(row) for row in self.validation}
        )

    def test_forbidden_gate_phrases_and_exact_prompts_are_absent(self) -> None:
        rendered = json.dumps(
            {"train": self.train, "validation": self.validation, "cases": self.cases, "gold": self.gold},
            ensure_ascii=False,
        ).casefold()
        for phrase in (*DEFAULT_FORBIDDEN_CONCEPTS, "Held-Out Proper Entity"):
            self.assertNotIn(phrase.casefold(), rendered)
        self.assertNotIn("a held-out gate sentence that must never appear", rendered)
        self.assertEqual(
            self.summary["forbidden_audit"]["exact_prompt_overlap"], 0
        )
        self.assertEqual(self.summary["forbidden_audit"]["phrase_hit_count"], 0)
        ngram_audit = self.summary["forbidden_audit"][
            "synthetic_prompt_target_ngram_overlap"
        ]
        self.assertEqual(ngram_audit["n"], DEFAULT_FORBIDDEN_NGRAM_N)
        self.assertEqual(ngram_audit["generated_rows_checked"], 448)
        self.assertEqual(ngram_audit["forbidden_cases_checked"], 2)
        self.assertEqual(ngram_audit["hit_count"], 0)
        self.assertEqual(ngram_audit["hits"], [])

    def test_old_weather_prompt_and_target_collision_is_caught(self) -> None:
        overlap = paired_prompt_target_ngram_overlap(
            "make the weather soft drifting snow but keep everything visible",
            (
                "Change the weather to soft drifting snow while keeping every "
                "subject visible and preserving the original action."
            ),
            "make it lightly snowing but keep everything visible",
            (
                "Change the weather to light snowfall while keeping every "
                "subject clearly visible."
            ),
        )
        self.assertIn("but keep everything visible", overlap["prompt_ngrams"])
        self.assertIn("while keeping every subject", overlap["target_ngrams"])

    def test_recipe_v2_weather_templates_are_disjoint(self) -> None:
        forbidden_prompt = "make it lightly snowing but keep everything visible"
        forbidden_target = (
            "Change the weather to light snowfall while keeping every subject clearly visible."
        )
        for validation in (False, True):
            generated_prompt, generated_target = refresh_builder._ordinary_text(
                "change_weather", "soft drifting snow", validation=validation
            )
            self.assertEqual(
                paired_prompt_target_ngram_overlap(
                    generated_prompt,
                    generated_target,
                    forbidden_prompt,
                    forbidden_target,
                ),
                {"prompt_ngrams": [], "target_ngrams": []},
            )

    def test_generation_fails_on_paired_forbidden_ngram_hit(self) -> None:
        ordinary_text = refresh_builder._ordinary_text

        def colliding_ordinary_text(subtask: str, target: str, *, validation: bool):
            if subtask == "change_weather":
                return (
                    f"make the weather {target} but keep everything visible",
                    (
                        f"Change the weather to {target} while keeping every subject "
                        "visible and preserving the original action."
                    ),
                )
            return ordinary_text(subtask, target, validation=validation)

        with patch.object(
            refresh_builder,
            "_ordinary_text",
            side_effect=colliding_ordinary_text,
        ):
            with self.assertRaisesRegex(
                AssertionError,
                "forbidden synthetic prompt/target n-gram overlap",
            ):
                build_refresh(
                    self.base_train,
                    self.base_eval,
                    forbidden_case_rows=self.forbidden_cases,
                )

    def test_validation_gold_matches_cases_and_has_search_aliases(self) -> None:
        self.assertEqual([row["bench_id"] for row in self.cases], [row["bench_id"] for row in self.gold])
        self.assertEqual(len({row["bench_id"] for row in self.gold}), 256)
        search_gold = [row for row in self.gold if isinstance(row["gold_plan"]["image_search"], str)]
        self.assertEqual(len(search_gold), 64)
        self.assertTrue(
            all(row["search_query_aliases"] == [row["gold_plan"]["image_search"]] for row in search_gold)
        )
        self.assertEqual(Counter(row["axis"] for row in self.gold)["search"], 128)
        for row in self.gold:
            refined = row["gold_plan"]["refined_text_instruction"]
            self.assertTrue(
                all(constraint_is_retained(constraint, refined) for constraint in row["constraints"])
            )

    def test_rejects_source_video_overlap(self) -> None:
        overlapping_eval = list(self.base_eval)
        overlapping_eval[0] = {**overlapping_eval[0], "videos": self.base_train[0]["videos"]}
        with self.assertRaisesRegex(ValueError, "base train/eval video overlap"):
            build_refresh(
                self.base_train,
                overlapping_eval,
                forbidden_case_rows=self.forbidden_cases,
            )

    def test_rejects_malformed_sharegpt_row(self) -> None:
        malformed = list(self.base_train)
        malformed[0] = {**malformed[0], "messages": [{"role": "user", "content": "missing video"}]}
        with self.assertRaisesRegex(ValueError, "expected exactly two messages"):
            build_refresh(malformed, self.base_eval, forbidden_case_rows=self.forbidden_cases)

    def test_contract_invalid_history_is_audited_and_never_replayed(self) -> None:
        polluted_train = list(self.base_train)
        polluted_eval = list(self.base_eval)
        train_prompt = "polluted historical combined label"
        eval_prompt = "polluted historical removal label"
        polluted_train[0] = make_row(
            99998,
            "train",
            train_prompt,
            plan(
                "Change the main object and its setting.",
                "combined_tasks",
                mask="main object",
            ),
        )
        polluted_eval[0] = make_row(
            99999,
            "eval",
            eval_prompt,
            plan("Remove the foreground prop.", "remove_object", mask=False),
        )

        train, validation, _, _, summary = build_refresh(
            polluted_train,
            polluted_eval,
            forbidden_case_rows=self.forbidden_cases,
        )

        exclusions = summary["input_contract_exclusions"]
        self.assertEqual(exclusions["train"]["rows"], 1)
        self.assertEqual(
            exclusions["train"]["by_violation_type"],
            {"mask_not_allowed_for_subtask": 1},
        )
        self.assertEqual(exclusions["train"]["by_subtask"], {"combined_tasks": 1})
        self.assertEqual(exclusions["eval"]["rows"], 1)
        self.assertEqual(
            exclusions["eval"]["by_violation_type"],
            {"remove_object_missing_mask": 1},
        )
        self.assertEqual(exclusions["eval"]["by_subtask"], {"remove_object": 1})

        rendered = json.dumps([*train, *validation], ensure_ascii=False)
        self.assertNotIn(train_prompt, rendered)
        self.assertNotIn(eval_prompt, rendered)
        for row in [*train, *validation]:
            output_plan = json.loads(row["messages"][1]["content"])
            if output_plan["subtask"] == "remove_object":
                self.assertIsInstance(output_plan["mask"], str)
            else:
                self.assertIs(output_plan["mask"], False)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from collections import Counter, defaultdict

from scripts.augment_planner_sft import (
    ALLOWED_SUBTASKS,
    EXTERNAL_ENTITIES,
    EXTERNAL_ENTITY_GROUPS,
    ROUTING_TEMPLATE_GROUPS,
    ROUTING_TEMPLATES,
    SEARCH_TEMPLATES_BY_CATEGORY,
    build_augmented,
)


class AugmentPlannerSftTest(unittest.TestCase):
    def setUp(self):
        self.base = [
            {
                "system": "system",
                "messages": [
                    {"role": "user", "content": f"<video>\ndo something {index}"},
                    {"role": "assistant", "content": "{}"},
                ],
                "videos": [f"/tmp/example-{index}.mp4"],
            }
            for index in range(16)
        ]

    def test_external_entity_catalog_is_large_unique_and_covers_required_groups(self):
        self.assertGreaterEqual(len(EXTERNAL_ENTITIES), 120)
        self.assertEqual(len(EXTERNAL_ENTITIES), len(set(EXTERNAL_ENTITIES)))
        self.assertEqual(
            set(EXTERNAL_ENTITY_GROUPS),
            {"brand_product", "ip_character", "landmark", "cultural_artifact"},
        )
        self.assertTrue(all(len(entities) >= 30 for entities in EXTERNAL_ENTITY_GROUPS.values()))
        self.assertIn("Starbucks holiday cup", EXTERNAL_ENTITIES)
        self.assertIn("Japanese cherry blossom tree", EXTERNAL_ENTITIES)

    def test_search_2k_uses_all_entities_and_diverse_legal_templates(self):
        rows, metadata = build_augmented(self.base, search_count=2000, routing_count=0)
        generated = rows[len(self.base) :]
        plans = [json.loads(row["messages"][1]["content"]) for row in generated]

        self.assertEqual(len(plans), 2000)
        self.assertEqual({plan["subtask"] for plan in plans}, {"add_object", "replace_object"})
        self.assertEqual({plan["image_search"] for plan in plans}, set(EXTERNAL_ENTITIES))
        self.assertTrue(all(plan["mask"] is False for plan in plans))
        self.assertTrue(all(item["category"] == "under_search" for item in metadata))
        self.assertEqual(
            {item["entity_category"] for item in metadata}, set(EXTERNAL_ENTITY_GROUPS)
        )
        self.assertGreaterEqual(len({item["template_id"] for item in metadata}), 8)
        self.assertGreaterEqual(
            sum(len(templates) for templates in SEARCH_TEMPLATES_BY_CATEGORY.values()), 8
        )
        for item in metadata:
            entity = item["target_plan"]["image_search"]
            self.assertIn(entity, item["request"])
            self.assertIn(entity, item["target_plan"]["refined_text_instruction"])

    def test_search_generation_is_deterministic_and_sources_remain_evenly_sampled(self):
        first_rows, first_metadata = build_augmented(
            self.base, search_count=32, routing_count=0
        )
        second_rows, second_metadata = build_augmented(
            self.base, search_count=32, routing_count=0
        )
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first_metadata, second_metadata)
        source_counts = Counter(item["source_index"] for item in first_metadata)
        self.assertEqual(set(source_counts), set(range(len(self.base))))
        self.assertEqual(set(source_counts.values()), {2})
        for row, item in zip(first_rows[len(self.base) :], first_metadata):
            self.assertEqual(row["videos"], self.base[item["source_index"]]["videos"])

    def test_routing_has_three_distinct_legal_templates_per_subtask(self):
        self.assertEqual(set(ROUTING_TEMPLATE_GROUPS), ALLOWED_SUBTASKS)
        self.assertEqual(len(ROUTING_TEMPLATES), len(ALLOWED_SUBTASKS) * 3)
        for subtask, templates in ROUTING_TEMPLATE_GROUPS.items():
            with self.subTest(subtask=subtask):
                self.assertGreaterEqual(len(templates), 3)
                self.assertEqual({item[2] for item in templates}, {subtask})
                self.assertEqual(len({item[0] for item in templates}), len(templates))
                self.assertEqual(len({item[1] for item in templates}), len(templates))

        rows, metadata = build_augmented(
            self.base, search_count=0, routing_count=len(ROUTING_TEMPLATES)
        )
        plans = [json.loads(row["messages"][1]["content"]) for row in rows[len(self.base) :]]
        plans_by_subtask = defaultdict(list)
        for plan in plans:
            plans_by_subtask[plan["subtask"]].append(plan)
        self.assertEqual(set(plans_by_subtask), ALLOWED_SUBTASKS)
        for subtask, subtask_plans in plans_by_subtask.items():
            with self.subTest(subtask=subtask):
                self.assertEqual(len(subtask_plans), 3)
                self.assertEqual(
                    len({plan["refined_text_instruction"] for plan in subtask_plans}), 3
                )
                self.assertTrue(all(plan["image_search"] is False for plan in subtask_plans))
                if subtask == "remove_object":
                    self.assertTrue(
                        all(isinstance(plan["mask"], str) for plan in subtask_plans)
                    )
                else:
                    self.assertTrue(all(plan["mask"] is False for plan in subtask_plans))
        self.assertEqual(
            len({item["template_id"] for item in metadata}), len(ROUTING_TEMPLATES)
        )

    def test_first_routing_variant_interleaves_full_taxonomy(self):
        rows, _ = build_augmented(
            self.base, search_count=0, routing_count=len(ALLOWED_SUBTASKS)
        )
        plans = [json.loads(row["messages"][1]["content"]) for row in rows[len(self.base) :]]
        self.assertEqual({plan["subtask"] for plan in plans}, ALLOWED_SUBTASKS)


if __name__ == "__main__":
    unittest.main()

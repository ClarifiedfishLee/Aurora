import unittest

from evaluation.unieditbench_infer import prompt_fields


class UniEditBenchInferTest(unittest.TestCase):
    def test_maps_readme_metadata_to_template_fields(self) -> None:
        fields = prompt_fields({"source_prompt": "before", "target_prompt": "after"})
        self.assertEqual(fields, {"original_prompt": "before", "edited_prompt": "after"})

    def test_prefers_explicit_template_fields(self) -> None:
        fields = prompt_fields(
            {
                "original_prompt": "explicit before",
                "edited_prompt": "explicit after",
                "source_prompt": "legacy before",
                "target_prompt": "legacy after",
            }
        )
        self.assertEqual(fields["original_prompt"], "explicit before")
        self.assertEqual(fields["edited_prompt"], "explicit after")

    def test_rejects_missing_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "edited_prompt"):
            prompt_fields({"source_prompt": "before"})


if __name__ == "__main__":
    unittest.main()

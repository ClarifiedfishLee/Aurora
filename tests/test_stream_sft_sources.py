import unittest

from scripts.stream_sft_sources import normalize_sample, parse_specs, safe_id, source_video_bytes


class StreamSftSourcesTest(unittest.TestCase):
    def test_parse_specs_combines_duplicates(self):
        self.assertEqual(parse_specs(["rose-removal=2", "rose-removal=3"]), {"rose-removal": 5})

    def test_rejects_invalid_spec(self):
        with self.assertRaises(ValueError):
            parse_specs(["rose-removal=0"])

    def test_normalizes_manifest_row(self):
        sample = {
            "__key__": "abc/123",
            "json": {
                "prompt": "Remove the tire.",
                "subset": "rose-removal",
                "source_dataset": "rose_mask",
                "edit_type": "removal_mask_ref",
                "src_video": "rose/source.mp4",
            },
        }
        row = normalize_sample("rose-removal", sample, "videos/x.mp4")
        self.assertEqual(row["sample_id"], "rose-removal_abc_123")
        self.assertEqual(row["clean_instruction"], "Remove the tire.")
        self.assertEqual(row["video_path"], "videos/x.mp4")

    def test_safe_id_rejects_empty_key(self):
        with self.assertRaises(ValueError):
            safe_id("rose-removal", "///")

    def test_accepts_legacy_raw_bytes(self):
        self.assertEqual(source_video_bytes({"source.mp4": b"video"}), b"video")

    def test_accepts_datasets_decode_false_record(self):
        sample = {"source.mp4": {"bytes": b"encoded", "path": "clip.mp4"}}
        self.assertEqual(source_video_bytes(sample), b"encoded")

    def test_rejects_missing_media(self):
        self.assertIsNone(source_video_bytes({"source.mp4": {"bytes": None}}))


if __name__ == "__main__":
    unittest.main()

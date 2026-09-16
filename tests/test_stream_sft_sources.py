import unittest

from scripts.stream_sft_sources import source_video_bytes


class StreamSftSourcesTest(unittest.TestCase):
    def test_accepts_legacy_raw_bytes(self):
        self.assertEqual(source_video_bytes({"source.mp4": b"video"}), b"video")

    def test_accepts_datasets_decode_false_record(self):
        sample = {"source.mp4": {"bytes": b"encoded", "path": "clip.mp4"}}
        self.assertEqual(source_video_bytes(sample), b"encoded")

    def test_rejects_missing_media(self):
        self.assertIsNone(source_video_bytes({"source.mp4": {"bytes": None}}))


if __name__ == "__main__":
    unittest.main()

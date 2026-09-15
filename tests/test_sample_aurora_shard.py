import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.sample_aurora_shard import materialize, read_samples


class SampleAuroraShardTest(unittest.TestCase):
    def test_materializes_only_source_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            shard = root / "sample.tar"
            metadata = json.dumps(
                {
                    "prompt": "Make the coat blue.",
                    "subset": "example",
                    "source_dataset": "unit-test",
                    "edit_type": "change_color",
                    "provenance": {"pipeline": "test"},
                }
            ).encode()
            with tarfile.open(shard, "w") as archive:
                for name, payload in (
                    ("item_001.json", metadata),
                    ("item_001.source.mp4", b"source-video"),
                    ("item_001.mp4", b"target-video"),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

            self.assertEqual(len(read_samples(shard)), 1)
            rows = materialize(shard, root / "output", limit=1, seed=42)

            self.assertEqual(rows[0]["prompt"], "Make the coat blue.")
            self.assertEqual((root / "output/source_videos/item_001.mp4").read_bytes(), b"source-video")
            self.assertFalse((root / "output/target_videos/item_001.mp4").exists())


if __name__ == "__main__":
    unittest.main()

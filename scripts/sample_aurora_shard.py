#!/usr/bin/env python3
"""Create a small source-video manifest from an Aurora WebDataset tar shard."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
from pathlib import Path
from typing import Any


def read_samples(shard: Path) -> list[dict[str, Any]]:
    with tarfile.open(shard) as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        samples: list[dict[str, Any]] = []
        for metadata_name in sorted(name for name in members if name.endswith(".json")):
            sample_id = metadata_name.removesuffix(".json")
            source_name = f"{sample_id}.source.mp4"
            if source_name not in members:
                continue
            metadata_file = archive.extractfile(members[metadata_name])
            if metadata_file is None:
                continue
            metadata = json.load(metadata_file)
            samples.append(
                {
                    "sample_id": sample_id,
                    "source_member": source_name,
                    "prompt": metadata["prompt"],
                    "subset": metadata.get("subset"),
                    "source_dataset": metadata.get("source_dataset"),
                    "edit_type": metadata.get("edit_type"),
                    "provenance": metadata.get("provenance", {}),
                }
            )
    return samples


def materialize(shard: Path, output_dir: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    samples = read_samples(shard)
    if limit < 1:
        raise ValueError("limit must be positive")
    selected = random.Random(seed).sample(samples, min(limit, len(samples)))
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "source_videos"
    video_dir.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    with tarfile.open(shard) as archive:
        for sample in selected:
            destination = video_dir / f"{sample['sample_id']}.mp4"
            source_file = archive.extractfile(sample["source_member"])
            if source_file is None:
                raise FileNotFoundError(sample["source_member"])
            with destination.open("wb") as output_file:
                shutil.copyfileobj(source_file, output_file)
            row = {key: value for key, value in sample.items() if key != "source_member"}
            row["video_path"] = destination.relative_to(output_dir).as_posix()
            rows.append(row)

    manifest = output_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = materialize(args.shard, args.out_dir, args.limit, args.seed)
    print(f"Materialized {len(rows)} source-only samples in {args.out_dir}")


if __name__ == "__main__":
    main()

"""Stream a bounded planner-SFT source-video working set from Aurora data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SPECS = ("ditto-combined=200", "rose-insertion=100", "rose-removal=100", "rose-v2v=100")
LICENSES = {
    "ditto-combined": "Ditto-1M upstream terms; research use only",
    "rose-insertion": "ROSE upstream terms; research use only",
    "rose-removal": "ROSE upstream terms; research use only",
    "rose-v2v": "ROSE upstream terms; research use only",
    "effecterase-removal": "EffectErase upstream terms; research use only",
}


def parse_specs(values: list[str]) -> dict[str, int]:
    specs: dict[str, int] = {}
    for value in values:
        name, separator, raw_count = value.partition("=")
        if not separator or not name or not raw_count.isdigit() or int(raw_count) < 1:
            raise ValueError(f"invalid spec {value!r}; expected config=positive_count")
        specs[name] = specs.get(name, 0) + int(raw_count)
    return specs


def safe_id(config: str, key: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", key).strip("_")
    if not normalized:
        raise ValueError(f"empty sample key after normalization: {key!r}")
    return f"{config}_{normalized}"


def normalize_sample(config: str, sample: dict[str, Any], video_path: str) -> dict[str, Any]:
    metadata = sample["json"]
    return {
        "sample_id": safe_id(config, str(sample["__key__"])),
        "video_path": video_path,
        "clean_instruction": metadata["prompt"],
        "subset": metadata.get("subset", config),
        "source_dataset": metadata.get("source_dataset"),
        "edit_type": metadata.get("edit_type"),
        "source_video_id": metadata.get("src_video"),
        "license": LICENSES.get(config, "consult upstream dataset terms"),
        "provenance": metadata.get("provenance", {}),
    }


def stream_sources(specs: dict[str, int], out_dir: Path, seed: int, buffer_size: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    existing = []
    if manifest_path.exists():
        existing = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    existing_ids = {row["sample_id"] for row in existing}
    rows = list(existing)
    counts = Counter(row["subset"] for row in existing)

    for config, target_count in specs.items():
        if counts[config] >= target_count:
            continue
        dataset = load_dataset(
            "yeates/aurora-training-data", config, split="train", streaming=True
        ).shuffle(seed=seed, buffer_size=buffer_size)
        video_dir = out_dir / "videos" / config
        video_dir.mkdir(parents=True, exist_ok=True)
        for sample in dataset:
            sample_id = safe_id(config, str(sample["__key__"]))
            if sample_id in existing_ids:
                continue
            source_bytes = sample.get("source.mp4")
            if not isinstance(source_bytes, bytes) or not source_bytes:
                continue
            destination = video_dir / f"{sample_id}.mp4"
            destination.write_bytes(source_bytes)
            relative_path = destination.relative_to(out_dir).as_posix()
            row = normalize_sample(config, sample, relative_path)
            rows.append(row)
            existing_ids.add(sample_id)
            counts[config] += 1
            manifest_path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8"
            )
            if counts[config] % 25 == 0 or counts[config] == target_count:
                print(f"{config}: {counts[config]}/{target_count}", flush=True)
            if counts[config] >= target_count:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spec", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=1000)
    args = parser.parse_args()
    specs = parse_specs(args.spec or list(DEFAULT_SPECS))
    rows = stream_sources(specs, args.out_dir, args.seed, args.buffer_size)
    print(json.dumps({"num_samples": len(rows), "by_subset": dict(Counter(row["subset"] for row in rows))}, indent=2))


if __name__ == "__main__":
    main()

"""Run the official UniEditBench judge with metadata-field compatibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def prompt_fields(item: dict[str, Any]) -> dict[str, str]:
    original = item.get("original_prompt") or item.get("source_prompt")
    edited = item.get("edited_prompt") or item.get("target_prompt")
    if not isinstance(original, str) or not original.strip():
        raise ValueError("metadata item requires original_prompt or source_prompt")
    if not isinstance(edited, str) or not edited.strip():
        raise ValueError("metadata item requires edited_prompt or target_prompt")
    return {"original_prompt": original.strip(), "edited_prompt": edited.strip()}


def run(metadata_path: Path, save_path: Path, unieditbench_repo: Path, port: int) -> list[dict[str, Any]]:
    sys.path.insert(0, str(unieditbench_repo))
    from openai import OpenAI
    from utils import EVAL_VIDEO_PROMPT

    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("metadata must be a JSON list")
    client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{port}/v1")
    models = client.models.list().data
    if not models:
        raise RuntimeError("judge server returned no models")
    model = models[0].id
    results: list[dict[str, Any]] = []
    for index, item in enumerate(rows, 1):
        source_path = str(Path(item["path"]).resolve())
        edited_path = str(Path(item["edit_path"]).resolve())
        for media_path in (source_path, edited_path):
            if not Path(media_path).is_file():
                raise FileNotFoundError(media_path)
        prompt = EVAL_VIDEO_PROMPT.format(**prompt_fields(item))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": source_path},
                        {"type": "video", "video": edited_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=2048,
            temperature=0,
        )
        result = dict(item)
        result.update({"judge_model": model, "prompt": prompt, "response": response.choices[0].message.content})
        results.append(result)
        print(f"[{index}/{len(rows)}] {item.get('id', index)}", flush=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--unieditbench_repo", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8005)
    args = parser.parse_args()
    run(args.metadata, args.save, args.unieditbench_repo, args.port)


if __name__ == "__main__":
    main()

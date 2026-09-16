"""Run resumable batched vLLM teacher inference for planner-SFT cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(line)["bench_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def make_record(case: dict[str, Any], plan: dict[str, Any], raw: str) -> dict[str, Any]:
    return {
        **case,
        "plan": plan,
        "agent_raw": raw,
        "search": {"agent_query": plan.get("image_search")},
        "mask": {"phrase": plan.get("mask")},
        "assets": {},
        "final_payload": {
            "refined_text_instruction": plan["refined_text_instruction"],
            "subtask": plan["subtask"],
            "search_image": False,
            "object_mask": False,
        },
    }


def main() -> None:
    from aurora.agent import PreparedVideo, load_custom_cases, sample_video_frames
    from aurora.agent_vllm import AgentVLMvLLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--merged-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--video-frames", type=int, default=6)
    parser.add_argument("--frame-max-side", type=int, default=448)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    cases = load_custom_cases(args.cases)
    done = completed_ids(args.out)
    pending = [case for case in cases if str(case["bench_id"]) not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"total": len(cases), "completed": len(done), "pending": len(pending)}), flush=True)
    agent = AgentVLMvLLM(args.merged_model, max_images=args.video_frames + 2)
    with args.out.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            items = []
            for case in batch:
                frames, fps = sample_video_frames(case["video_path"], args.video_frames, args.frame_max_side)
                items.append((case["prompt"], PreparedVideo(frames=frames, fps=fps), None))
            try:
                predictions = agent.plan_batch(items)
            except Exception as exc:
                print(f"batch {start // args.batch_size + 1} failed, retrying one by one: {exc!r}", flush=True)
                predictions = [agent.plan(instruction, video=video) for instruction, video, _ in items]
            for case, (plan, raw) in zip(batch, predictions):
                handle.write(json.dumps(make_record(case, plan, raw), ensure_ascii=False) + "\n")
            handle.flush()
            print(f"completed {len(done) + min(start + len(batch), len(pending))}/{len(cases)}", flush=True)


if __name__ == "__main__":
    main()

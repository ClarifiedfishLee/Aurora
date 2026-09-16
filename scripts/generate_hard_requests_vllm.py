"""Generate resumable hard-case user requests from clean planner examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GENERAL_CATEGORIES = (
    "entity_ambiguity",
    "pronoun_grounding",
    "implicit_local_edit",
    "over_search_negative",
    "multiple_constraints",
    "rewrite_preservation",
)

CATEGORY_GUIDANCE = {
    "entity_ambiguity": "Use a naturally ambiguous short noun, but retain enough visible attributes to resolve it from the video.",
    "pronoun_grounding": "Use a pronoun such as it, this, that one, or the one on the side; keep any crucial color, count, or location constraint.",
    "implicit_local_edit": "Phrase the request casually without saying 'local edit'; make it clear that only the intended object or region changes.",
    "over_search_negative": "Refer only to objects already visible or ordinary descriptive concepts. Do not introduce a brand, IP, celebrity, or landmark.",
    "mask_granularity": "Describe exactly the removable whole object or part, including location or appearance needed for a precise mask.",
    "multiple_constraints": "Compress the request while preserving every color, count, size, spatial, and preservation constraint.",
    "compositional_edit": "Express all component edits in one realistic user request without dropping either operation.",
    "rewrite_preservation": "Use colloquial wording while retaining every factual modifier and explicit preservation instruction.",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def choose_category(row: dict[str, Any], index: int) -> str:
    subtask = row["target_plan"]["subtask"]
    if subtask == "remove_object" and index % 2 == 0:
        return "mask_granularity"
    if subtask == "combined_tasks":
        return "compositional_edit"
    return GENERAL_CATEGORIES[index % len(GENERAL_CATEGORIES)]


def build_prompt(row: dict[str, Any], category: str) -> str:
    clean = row["target_plan"]["refined_text_instruction"]
    plan = json.dumps(row["target_plan"], ensure_ascii=False, separators=(",", ":"))
    return f"""Rewrite a clean video-editing instruction as one realistic, terse user request.
Hard-case category: {category}
Category rule: {CATEGORY_GUIDANCE[category]}

Requirements:
- Preserve the complete edit intent. Never invent a new edit, object, brand, color, count, or location.
- Keep all constraints that would change the correct edited result.
- The planner will also see the source video, so natural pronouns and visual references are allowed.
- Output one JSON object only: {{"raw_user_request":"..."}}

Clean instruction: {clean}
Target plan: {plan}"""


def parse_request(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1]).get("raw_user_request")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(value, str):
        return None
    value = " ".join(value.split()).strip()
    if not 4 <= len(value) <= 320:
        return None
    return value


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["sample_id"]) for row in load_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    rows = load_jsonl(args.canonical)
    done = completed_ids(args.out)
    pending = [(index, row) for index, row in enumerate(rows) if str(row["sample_id"]) not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"total": len(rows), "completed": len(done), "pending": len(pending)}), flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=str(args.model),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=128, seed=42)
    with args.out.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = []
            categories = []
            for index, row in batch:
                category = choose_category(row, index)
                categories.append(category)
                messages = [{"role": "user", "content": build_prompt(row, category)}]
                prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            outputs = llm.generate(prompts, sampling)
            for (index, row), category, output in zip(batch, categories, outputs):
                raw_output = output.outputs[0].text.strip()
                request = parse_request(raw_output)
                generated = request is not None
                if request is None:
                    request = row["raw_user_request"]
                record = {
                    "sample_id": row["sample_id"],
                    "category": category,
                    "raw_user_request": request,
                    "generated": generated,
                    "source_index": index,
                    "raw_output": raw_output,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"completed {len(done) + min(start + len(batch), len(pending))}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()

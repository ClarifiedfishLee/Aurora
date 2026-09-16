"""Generate resumable hard-case user requests from clean planner examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, TypeVar


_T = TypeVar("_T")


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
    "under_search": "Keep the exact external brand, IP, product, or landmark identity that requires an image search; do not replace it with a generic noun.",
    "mask_granularity": "Describe exactly the removable whole object or part, including location or appearance needed for a precise mask.",
    "multiple_constraints": "Compress the request while preserving every color, count, size, spatial, and preservation constraint.",
    "compositional_edit": "Express all component edits in one realistic user request without dropping either operation.",
    "rewrite_preservation": "Use colloquial wording while retaining every factual modifier and explicit preservation instruction.",
}

PROTECTED_TERM_GROUPS = (
    {
        "black",
        "blue",
        "brown",
        "cyan",
        "gold",
        "golden",
        "gray",
        "grey",
        "green",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "teal",
        "white",
        "yellow",
    },
    {
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    },
    {
        "above",
        "below",
        "behind",
        "beside",
        "between",
        "bottom",
        "center",
        "centre",
        "foreground",
        "left",
        "right",
        "top",
    },
    {
        "large",
        "long",
        "narrow",
        "short",
        "small",
        "tall",
        "tiny",
        "vertical",
        "wide",
    },
    {
        "brick",
        "cardboard",
        "chrome",
        "fabric",
        "glass",
        "leather",
        "metal",
        "plastic",
        "stone",
        "wooden",
    },
)

ACTION_TERMS = {
    "add_object": {"add", "attach", "hang", "insert", "mount", "place", "put"},
    "remove_object": {"clear", "delete", "erase", "get rid", "remove", "take away"},
    "replace_object": {"change", "make", "replace", "swap", "turn"},
    "change_background": {"background", "move", "scene", "set"},
    "change_color": {"color", "colour", "make", "turn"},
    "global_style": {"look", "render", "style"},
    "change_weather": {"fog", "rain", "snow", "weather"},
    "add_effect": {"add", "effect", "sparkle"},
    "camera_edit": {"camera", "pan", "zoom"},
}

SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "bag",
    "bottle",
    "camera",
    "car",
    "chair",
    "console",
    "cup",
    "for",
    "image",
    "object",
    "of",
    "reference",
    "shoe",
    "sneaker",
    "statue",
    "the",
    "toy",
    "tree",
    "with",
}

COMBINED_ACTION_GROUPS = {
    "add": {"add", "attach", "hang", "insert", "mount", "place", "put"},
    "remove": {"clear", "delete", "erase", "get rid", "remove", "take away"},
    "replace": {"replace", "swap", "substitute"},
    "transform": {"change", "make", "paint", "recolor", "turn"},
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_resumable_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read generated JSONL and repair only a truncated final write."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if not data:
        return []
    lines = data.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    valid_bytes = 0
    for index, raw_line in enumerate(lines):
        try:
            text = raw_line.decode("utf-8").strip()
            if text:
                records.append(json.loads(text))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_truncated_tail = index == len(lines) - 1 and not data.endswith(b"\n")
            if not is_truncated_tail:
                raise ValueError(f"invalid JSONL record {index + 1} in {path}") from exc
            path.write_bytes(data[:valid_bytes])
            break
        valid_bytes += len(raw_line)
    return records


def _enabled(value: Any) -> bool:
    return value not in (False, None, "", "false", "False")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _contains_term(normalized_text: str, term: str) -> bool:
    return f" {term} " in f" {normalized_text} "


def constraint_terms(text: str) -> set[str]:
    tokens = _tokens(text)
    protected = set().union(*PROTECTED_TERM_GROUPS)
    terms = tokens & protected
    terms.update(re.findall(r"\b\d+(?:\.\d+)?\b", text.lower()))
    return terms


def constraint_group_count(text: str) -> int:
    tokens = _tokens(text)
    count = sum(bool(tokens & group) for group in PROTECTED_TERM_GROUPS)
    return count + int(bool(re.search(r"\b\d+(?:\.\d+)?\b", text)))


def choose_category(row: dict[str, Any], index: int) -> str:
    plan = row["target_plan"]
    subtask = plan["subtask"]
    if _enabled(plan.get("image_search")):
        return "under_search"
    if subtask == "combined_tasks":
        return "compositional_edit"
    if _enabled(plan.get("mask")):
        return "mask_granularity"
    candidates = list(GENERAL_CATEGORIES)
    if constraint_group_count(str(row.get("raw_user_request", ""))) < 2:
        candidates.remove("multiple_constraints")
    stable_key = str(row.get("sample_id", index)).encode()
    choice = int(hashlib.sha256(stable_key).hexdigest()[:8], 16)
    return candidates[choice % len(candidates)]


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
    if not 4 <= len(value) <= 800:
        return None
    return value


def validate_request(row: dict[str, Any], category: str, request: str) -> list[str]:
    source = " ".join(str(row["raw_user_request"]).split()).strip()
    source_normalized = re.sub(r"[^a-z0-9]+", " ", source.lower()).strip()
    request_normalized = re.sub(r"[^a-z0-9]+", " ", request.lower()).strip()
    errors: list[str] = []
    if source_normalized == request_normalized:
        errors.append("unchanged_from_base")
    source_constraints = constraint_terms(source)
    missing_constraints = sorted(source_constraints - constraint_terms(request))
    if missing_constraints:
        errors.append(f"missing_constraints:{','.join(missing_constraints)}")

    plan = row["target_plan"]
    subtask = str(plan["subtask"])
    if subtask != "combined_tasks":
        action_terms = ACTION_TERMS.get(subtask)
        if action_terms and not any(_contains_term(request_normalized, term) for term in action_terms):
            errors.append(f"missing_action:{subtask}")
    else:
        for action, terms in COMBINED_ACTION_GROUPS.items():
            if any(_contains_term(source_normalized, term) for term in terms) and not any(
                _contains_term(request_normalized, term) for term in terms
            ):
                errors.append(f"missing_combined_action:{action}")

    search = plan.get("image_search")
    if _enabled(search):
        anchors = _tokens(str(search)) - SEARCH_STOPWORDS
        if anchors and not (_tokens(request) & anchors):
            errors.append("missing_search_entity")
        if category != "under_search":
            errors.append("search_category_mismatch")
    elif category == "under_search":
        errors.append("search_category_mismatch")
    if category == "over_search_negative" and _enabled(search):
        errors.append("over_search_category_mismatch")
    if category == "multiple_constraints" and constraint_group_count(source) < 2:
        errors.append("insufficient_source_constraints")
    if category == "mask_granularity" and not _enabled(plan.get("mask")):
        errors.append("mask_category_mismatch")
    if category == "compositional_edit" and subtask != "combined_tasks":
        errors.append("composition_category_mismatch")
    return errors


def completed_ids(path: Path) -> set[str]:
    return {
        str(row["sample_id"])
        for row in load_resumable_jsonl(path)
        if row.get("accepted") is True
    }


def _evenly_spaced(items: list[_T], count: int) -> list[_T]:
    if count >= len(items):
        return list(items)
    return [items[((2 * index + 1) * len(items)) // (2 * count)] for index in range(count)]


def stratified_rows(
    rows: list[dict[str, Any]], max_samples: int | None
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(rows))
    if max_samples is None or max_samples >= len(indexed):
        return indexed
    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in indexed:
        buckets.setdefault(choose_category(row, index), []).append((index, row))
    categories = [category for category in CATEGORY_GUIDANCE if buckets.get(category)]
    quotas = {category: max_samples // len(categories) for category in categories}
    for category in categories[: max_samples % len(categories)]:
        quotas[category] += 1
    selected = [
        item
        for category in categories
        for item in _evenly_spaced(buckets[category], quotas[category])
    ]
    return sorted(selected)


def select_pending(
    rows: list[dict[str, Any]], done_ids: set[str], max_samples: int | None
) -> tuple[list[tuple[int, dict[str, Any]]], set[str], list[tuple[int, dict[str, Any]]]]:
    selected = stratified_rows(rows, max_samples)
    selected_ids = {str(row["sample_id"]) for _, row in selected}
    done = done_ids & selected_ids
    pending = [
        (index, row)
        for index, row in selected
        if str(row["sample_id"]) not in done
    ]
    return selected, done, pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--rejected-out",
        type=Path,
        help="Audit log for rejected candidates (defaults beside --out).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Only process the first N canonical records (useful for quality pilots).",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.candidates < 1:
        raise ValueError("candidates must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max samples must be positive")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    rows, done, pending = select_pending(
        load_jsonl(args.canonical), completed_ids(args.out), args.max_samples
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rejected_out = args.rejected_out or args.out.with_name(
        f"{args.out.stem}.rejected{args.out.suffix}"
    )
    rejected_out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"total": len(rows), "completed": len(done), "pending": len(pending)}), flush=True)
    if not pending:
        return
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=str(args.model),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    sampling = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=256,
        seed=args.seed,
        n=args.candidates,
    )
    with (
        args.out.open("a", encoding="utf-8") as accepted_handle,
        rejected_out.open("a", encoding="utf-8") as rejected_handle,
    ):
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
                candidate_outputs = [candidate.text.strip() for candidate in output.outputs]
                parsed = [
                    (raw_output, request)
                    for raw_output in candidate_outputs
                    if (request := parse_request(raw_output)) is not None
                ]
                accepted_candidates = [
                    (raw_output, request)
                    for raw_output, request in parsed
                    if not validate_request(row, category, request)
                ]
                accepted = bool(accepted_candidates)
                if accepted:
                    raw_output, request = accepted_candidates[0]
                elif parsed:
                    raw_output, request = parsed[0]
                else:
                    raw_output, request = candidate_outputs[0], row["raw_user_request"]
                validation_errors = validate_request(row, category, request)
                record = {
                    "sample_id": row["sample_id"],
                    "category": category,
                    "raw_user_request": request,
                    "generated": bool(parsed),
                    "accepted": accepted,
                    "validation_errors": validation_errors,
                    "source_index": index,
                    "raw_output": raw_output,
                    "candidate_outputs": candidate_outputs,
                }
                destination = accepted_handle if accepted else rejected_handle
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            accepted_handle.flush()
            rejected_handle.flush()
            print(f"completed {len(done) + min(start + len(batch), len(pending))}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()

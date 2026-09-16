"""Post-seal isolation audit for the recipe2 primary and fresh-384 set.

The fresh-384 policy was sealed before candidate inference, but its original
construction inputs did not include the subsequently registered recipe2
train/eval JSONL files.  This audit closes that provenance gap without
rewriting the sealed policy or any candidate-selection artifact.

The output is deliberately labelled ``supplemental_post_seal``.  Exact prompt,
concept, constraint, strict-template, and source overlaps are blocking.  Broad
prompt skeletons belonging to routing/mask controls and paired prompt/target
4-gram matches are preserved as diagnostics only; they never silently relax a
blocking category.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evaluation.agent_only_score import normalize
from scripts.build_interpolation_validation import (
    GENERIC_AXIS_CATEGORIES,
    _template_signature,
    _template_signature_matches_prompt,
)
from scripts.build_oversearch_refresh import ParsedRow, load_jsonl, parse_rows
from scripts.select_lora_interpolation import (
    SelectionValidationError,
    load_json_object,
    sha256_file,
    validate_policy,
    validate_validation_bundle,
)

SCHEMA_VERSION = 1
AUDIT_ROLE = "supplemental_post_seal"
EXPECTED_RECIPE_TRAIN_ROWS = 1_024
EXPECTED_RECIPE_EVAL_ROWS = 256
PAIRED_NGRAM_N = 4
BLOCKING_OVERLAP_KEYS = (
    "exact_prompt",
    "exact_indexed_concept",
    "concept_phrase",
    "constraint_phrase",
    "exact_template_signature_strict",
    "wildcard_template_signature_strict",
    "source_path",
    "source_basename",
    "source_sample_id",
    "recipe_paths_missing_from_locked_v2",
)


class IsolationAuditError(RuntimeError):
    """Raised when the supplemental audit cannot be completed safely."""


def _require_regular_file(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise IsolationAuditError(f"{label} must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise IsolationAuditError(f"{label} is missing or empty: {resolved}")
    return resolved


def _normal_phrase_in(value: str, texts: Sequence[str]) -> bool:
    phrase = normalize(value)
    if not phrase:
        return False
    padded = f" {phrase} "
    return any(padded in f" {text} " for text in texts)


def _ngrams(text: str, n: int = PAIRED_NGRAM_N) -> set[str]:
    tokens = normalize(text).split()
    return {
        " ".join(tokens[index : index + n])
        for index in range(len(tokens) - n + 1)
    }


def _video_path(row: dict[str, Any], label: str) -> str:
    videos = row.get("videos")
    if (
        not isinstance(videos, list)
        or len(videos) != 1
        or not isinstance(videos[0], str)
        or not videos[0].strip()
    ):
        raise IsolationAuditError(f"{label} must contain exactly one non-empty video path")
    return videos[0]


def _parse_recipe_rows(
    train_rows: Sequence[dict[str, Any]], eval_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(train_rows) != EXPECTED_RECIPE_TRAIN_ROWS:
        raise IsolationAuditError(
            f"recipe2 train has {len(train_rows)} rows; expected {EXPECTED_RECIPE_TRAIN_ROWS}"
        )
    if len(eval_rows) != EXPECTED_RECIPE_EVAL_ROWS:
        raise IsolationAuditError(
            f"recipe2 eval has {len(eval_rows)} rows; expected {EXPECTED_RECIPE_EVAL_ROWS}"
        )
    parsed: list[dict[str, Any]] = []
    for split, rows in (("train", train_rows), ("eval", eval_rows)):
        try:
            validated: list[ParsedRow] = parse_rows(rows, f"recipe2 {split}")
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            raise IsolationAuditError(f"cannot parse recipe2 {split}: {error}") from error
        for index, row in enumerate(validated, 1):
            parsed.append(
                {
                    "split": split,
                    "index": index,
                    "prompt": row.prompt,
                    "plan": row.plan,
                    "video": row.video,
                }
            )
    return parsed


def _fresh_template_records(cases: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row in cases:
        catalog = row.get("catalog")
        if not isinstance(catalog, dict):
            raise IsolationAuditError(f"fresh case {row.get('bench_id')!r} lacks catalog metadata")
        signature = catalog.get("template_signature")
        if not isinstance(signature, str) or not signature.strip():
            raise IsolationAuditError(
                f"fresh case {row.get('bench_id')!r} lacks template_signature"
            )
        records.append(
            {
                "bench_id": str(row["bench_id"]),
                "category": str(row["category"]),
                "subtype": str(row["subtype"]),
                "signature": signature,
            }
        )
    return records


def _recipe_template_signature(row: dict[str, Any]) -> str:
    plan = row["plan"]
    slots = [
        str(plan[key])
        for key in ("image_search", "mask")
        if isinstance(plan.get(key), str)
    ]
    return _template_signature(str(row["prompt"]), slots)


def _template_hits(
    cases: Sequence[dict[str, Any]], recipe: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strict: list[dict[str, Any]] = []
    generic: list[dict[str, Any]] = []
    recipe_signatures = [(_recipe_template_signature(row), row) for row in recipe]
    for record in _fresh_template_records(cases):
        exact = [
            row for signature, row in recipe_signatures
            if signature == record["signature"]
        ]
        wildcard = [
            row for row in recipe
            if _template_signature_matches_prompt(
                record["signature"], normalize(str(row["prompt"]))
            )
        ]
        for match_type, rows in (("exact", exact), ("wildcard", wildcard)):
            for row in rows:
                hit = {
                    **record,
                    "match_type": match_type,
                    "recipe_split": row["split"],
                    "recipe_index": row["index"],
                    "recipe_prompt": row["prompt"],
                }
                target = generic if record["category"] in GENERIC_AXIS_CATEGORIES else strict
                target.append(hit)
    return strict, generic


def _paired_ngram_diagnostics(
    gold: Sequence[dict[str, Any]], recipe: Sequence[dict[str, Any]], *, limit: int = 20
) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    for fresh in gold:
        gold_plan = fresh.get("gold_plan")
        if not isinstance(gold_plan, dict):
            raise IsolationAuditError(f"fresh gold {fresh.get('bench_id')!r} lacks gold_plan")
        fresh_prompt_ngrams = _ngrams(str(fresh["prompt"]))
        fresh_target_ngrams = _ngrams(str(gold_plan["refined_text_instruction"]))
        for recipe_row in recipe:
            recipe_prompt_ngrams = _ngrams(str(recipe_row["prompt"]))
            recipe_target_ngrams = _ngrams(
                str(recipe_row["plan"]["refined_text_instruction"])
            )
            prompt_overlap = sorted(fresh_prompt_ngrams & recipe_prompt_ngrams)
            target_overlap = sorted(fresh_target_ngrams & recipe_target_ngrams)
            if not prompt_overlap or not target_overlap:
                continue
            hits.append(
                {
                    "bench_id": fresh["bench_id"],
                    "category": fresh["category"],
                    "subtype": fresh["subtype"],
                    "recipe_split": recipe_row["split"],
                    "recipe_index": recipe_row["index"],
                    "prompt_ngrams": prompt_overlap,
                    "target_ngrams": target_overlap,
                }
            )
    return {
        "n": PAIRED_NGRAM_N,
        "blocking": False,
        "pair_count": len(hits),
        "fresh_case_count": len({str(hit["bench_id"]) for hit in hits}),
        "recipe_row_count": len(
            {(str(hit["recipe_split"]), int(hit["recipe_index"])) for hit in hits}
        ),
        "by_fresh_category": dict(
            sorted(Counter(str(hit["category"]) for hit in hits).items())
        ),
        "examples": hits[:limit],
    }


def _value_ngram_diagnostics(
    gold: Sequence[dict[str, Any]], recipe_texts: Sequence[str], *, limit: int = 20
) -> dict[str, Any]:
    recipe_ngrams = set().union(*(_ngrams(text) for text in recipe_texts))
    hits: list[dict[str, Any]] = []
    for row in gold:
        catalog = row.get("catalog")
        if not isinstance(catalog, dict) or not isinstance(catalog.get("concept_id"), str):
            raise IsolationAuditError(f"fresh gold {row.get('bench_id')!r} lacks concept_id")
        values: list[tuple[str, str]] = [("concept", str(catalog["concept_id"]))]
        constraints = row.get("constraints", [])
        if not isinstance(constraints, list):
            raise IsolationAuditError(f"fresh gold {row.get('bench_id')!r} has invalid constraints")
        values.extend(
            (str(item.get("type", "constraint")), str(item["value"]))
            for item in constraints
            if isinstance(item, dict) and isinstance(item.get("value"), str)
        )
        seen: set[tuple[str, str]] = set()
        for value_type, value in values:
            key = value_type, normalize(value)
            if key in seen:
                continue
            seen.add(key)
            overlap = sorted(_ngrams(value) & recipe_ngrams)
            if overlap:
                hits.append(
                    {
                        "bench_id": row["bench_id"],
                        "category": row["category"],
                        "value_type": value_type,
                        "value": value,
                        "ngrams": overlap,
                    }
                )
    return {
        "n": PAIRED_NGRAM_N,
        "blocking": False,
        "hit_count": len(hits),
        "fresh_case_count": len({str(hit["bench_id"]) for hit in hits}),
        "by_category": dict(
            sorted(Counter(str(hit["category"]) for hit in hits).items())
        ),
        "examples": hits[:limit],
    }


def analyze_isolation(
    *,
    cases: Sequence[dict[str, Any]],
    gold: Sequence[dict[str, Any]],
    recipe: Sequence[dict[str, Any]],
    locked_v2_video_paths: set[str],
    fresh_leakage_audit: dict[str, Any],
    example_limit: int = 20,
) -> tuple[dict[str, int], dict[str, Any], dict[str, Any]]:
    """Return blocking counts, diagnostics, and bounded blocking examples."""

    recipe_prompts = {normalize(str(row["prompt"])) for row in recipe}
    recipe_concepts = {
        normalize(str(row["plan"][key]))
        for row in recipe
        for key in ("image_search", "mask")
        if isinstance(row["plan"].get(key), str)
    }
    recipe_texts = tuple(
        normalize(value)
        for row in recipe
        for value in (
            str(row["prompt"]),
            str(row["plan"]["refined_text_instruction"]),
            *(str(row["plan"][key]) for key in ("image_search", "mask") if isinstance(row["plan"].get(key), str)),
        )
    )

    exact_prompt_hits = [
        {"bench_id": row["bench_id"], "prompt": row["prompt"]}
        for row in cases
        if normalize(str(row["prompt"])) in recipe_prompts
    ]
    exact_concept_hits: list[dict[str, Any]] = []
    concept_phrase_hits: list[dict[str, Any]] = []
    constraint_phrase_hits: list[dict[str, Any]] = []
    for row in gold:
        catalog = row.get("catalog")
        if not isinstance(catalog, dict) or not isinstance(catalog.get("concept_id"), str):
            raise IsolationAuditError(f"fresh gold {row.get('bench_id')!r} lacks concept_id")
        concept = str(catalog["concept_id"])
        common = {
            "bench_id": row["bench_id"],
            "category": row["category"],
            "subtype": row["subtype"],
            "value": concept,
        }
        if normalize(concept) in recipe_concepts:
            exact_concept_hits.append(common)
        if _normal_phrase_in(concept, recipe_texts):
            concept_phrase_hits.append(common)
        constraints = row.get("constraints", [])
        if not isinstance(constraints, list):
            raise IsolationAuditError(f"fresh gold {row.get('bench_id')!r} has invalid constraints")
        for constraint in constraints:
            if not isinstance(constraint, dict) or not isinstance(constraint.get("value"), str):
                raise IsolationAuditError(
                    f"fresh gold {row.get('bench_id')!r} has an invalid constraint"
                )
            value = str(constraint["value"])
            if _normal_phrase_in(value, recipe_texts):
                constraint_phrase_hits.append(
                    {
                        "bench_id": row["bench_id"],
                        "category": row["category"],
                        "constraint_type": constraint.get("type"),
                        "value": value,
                    }
                )

    strict_template_hits, generic_template_hits = _template_hits(cases, recipe)
    strict_exact = [hit for hit in strict_template_hits if hit["match_type"] == "exact"]
    strict_wildcard = [
        hit for hit in strict_template_hits if hit["match_type"] == "wildcard"
    ]

    fresh_paths = {str(row["video_path"]) for row in cases}
    fresh_basenames = {Path(path).name for path in fresh_paths}
    fresh_sample_ids = {str(row["source"]["sample_id"]) for row in cases}
    recipe_paths = {str(row["video"]) for row in recipe}
    recipe_basenames = {Path(path).name for path in recipe_paths}
    recipe_sample_ids = {Path(path).stem for path in recipe_paths}
    source_path_hits = sorted(fresh_paths & recipe_paths)
    source_basename_hits = sorted(fresh_basenames & recipe_basenames)
    source_sample_hits = sorted(fresh_sample_ids & recipe_sample_ids)
    missing_from_v2 = sorted(recipe_paths - locked_v2_video_paths)

    blocking = {
        "exact_prompt": len(exact_prompt_hits),
        "exact_indexed_concept": len(exact_concept_hits),
        "concept_phrase": len(concept_phrase_hits),
        "constraint_phrase": len(constraint_phrase_hits),
        "exact_template_signature_strict": len(strict_exact),
        "wildcard_template_signature_strict": len(strict_wildcard),
        "source_path": len(source_path_hits),
        "source_basename": len(source_basename_hits),
        "source_sample_id": len(source_sample_hits),
        "recipe_paths_missing_from_locked_v2": len(missing_from_v2),
    }
    if tuple(blocking) != BLOCKING_OVERLAP_KEYS:
        raise AssertionError("blocking overlap schema drift")

    isolation = fresh_leakage_audit.get("isolation")
    if not isinstance(isolation, dict):
        raise IsolationAuditError("fresh leakage audit lacks isolation metadata")
    prior_video_overlap = isolation.get("prior_video_sha256_overlap")
    if prior_video_overlap != 0:
        raise IsolationAuditError(
            "fresh leakage audit does not prove zero prior video-SHA overlap"
        )
    diagnostics = {
        "counts": {
            "fresh_cases": len(cases),
            "fresh_gold": len(gold),
            "recipe2_rows": len(recipe),
            "recipe2_train_rows": sum(row["split"] == "train" for row in recipe),
            "recipe2_eval_rows": sum(row["split"] == "eval" for row in recipe),
        },
        "generic_template_overlaps": {
            "blocking": False,
            "allowed_categories": sorted(GENERIC_AXIS_CATEGORIES),
            "exact_hit_count": sum(hit["match_type"] == "exact" for hit in generic_template_hits),
            "wildcard_hit_count": sum(hit["match_type"] == "wildcard" for hit in generic_template_hits),
            "unique_signatures": sorted({str(hit["signature"]) for hit in generic_template_hits}),
            "hits": generic_template_hits[:example_limit],
        },
        "paired_prompt_target_4gram": _paired_ngram_diagnostics(
            gold, recipe, limit=example_limit
        ),
        "concept_constraint_value_4gram": _value_ngram_diagnostics(
            gold, recipe_texts, limit=example_limit
        ),
        "source_proof": {
            "recipe_unique_paths": len(recipe_paths),
            "locked_v2_unique_paths": len(locked_v2_video_paths),
            "recipe_paths_subset_locked_v2": not missing_from_v2,
            "fresh_prior_video_sha256_overlap": prior_video_overlap,
        },
    }
    examples = {
        "exact_prompt": exact_prompt_hits[:example_limit],
        "exact_indexed_concept": exact_concept_hits[:example_limit],
        "concept_phrase": concept_phrase_hits[:example_limit],
        "constraint_phrase": constraint_phrase_hits[:example_limit],
        "exact_template_signature_strict": strict_exact[:example_limit],
        "wildcard_template_signature_strict": strict_wildcard[:example_limit],
        "source_path": source_path_hits[:example_limit],
        "source_basename": source_basename_hits[:example_limit],
        "source_sample_id": source_sample_hits[:example_limit],
        "recipe_paths_missing_from_locked_v2": missing_from_v2[:example_limit],
    }
    return blocking, diagnostics, examples


def build_supplemental_audit(
    *,
    cases_path: Path,
    gold_path: Path,
    leakage_audit_path: Path,
    policy_path: Path,
    recipe2_train_path: Path,
    recipe2_eval_path: Path,
    v2_train_path: Path,
    v2_eval_path: Path,
) -> dict[str, Any]:
    paths = {
        "cases": _require_regular_file(cases_path, "fresh cases"),
        "gold": _require_regular_file(gold_path, "fresh gold"),
        "leakage_audit": _require_regular_file(leakage_audit_path, "fresh leakage audit"),
        "selection_policy": _require_regular_file(policy_path, "fresh selection policy"),
        "recipe2_train": _require_regular_file(recipe2_train_path, "recipe2 train"),
        "recipe2_eval": _require_regular_file(recipe2_eval_path, "recipe2 eval"),
        "v2_train": _require_regular_file(v2_train_path, "locked v2 train"),
        "v2_eval": _require_regular_file(v2_eval_path, "locked v2 eval"),
    }
    policy = load_json_object(paths["selection_policy"], "selection policy")
    normalized_policy = validate_policy(policy)
    cases, gold, fresh_audit = validate_validation_bundle(
        paths["cases"],
        paths["gold"],
        paths["leakage_audit"],
        policy,
        normalized_policy,
    )
    locked_inputs = policy["validation_artifacts"]["input_sha256"]
    for key in ("v2_train", "v2_eval"):
        observed = sha256_file(paths[key])
        if locked_inputs.get(key) != observed:
            raise IsolationAuditError(
                f"{key} hash differs from the sealed fresh-384 policy: {observed}"
            )

    recipe_train_rows = load_jsonl(paths["recipe2_train"])
    recipe_eval_rows = load_jsonl(paths["recipe2_eval"])
    recipe = _parse_recipe_rows(recipe_train_rows, recipe_eval_rows)
    locked_v2_rows = [*load_jsonl(paths["v2_train"]), *load_jsonl(paths["v2_eval"])]
    locked_v2_paths = {
        _video_path(row, f"locked v2 row {index}")
        for index, row in enumerate(locked_v2_rows, 1)
    }
    blocking, diagnostics, examples = analyze_isolation(
        cases=cases,
        gold=gold,
        recipe=recipe,
        locked_v2_video_paths=locked_v2_paths,
        fresh_leakage_audit=fresh_audit,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_role": AUDIT_ROLE,
        "passed": all(value == 0 for value in blocking.values()),
        "sealed_artifacts": {
            "cases_sha256": sha256_file(paths["cases"]),
            "gold_sha256": sha256_file(paths["gold"]),
            "selection_policy_sha256": sha256_file(paths["selection_policy"]),
        },
        "recipe2_artifacts": {
            "train_sha256": sha256_file(paths["recipe2_train"]),
            "eval_sha256": sha256_file(paths["recipe2_eval"]),
        },
        "method": {
            "normalization": "evaluation.agent_only_score.normalize",
            "template_policy": {
                "strict_categories": sorted(
                    set(normalized_policy["category_counts"])
                    - set(GENERIC_AXIS_CATEGORIES)
                ),
                "audit_only_categories": sorted(GENERIC_AXIS_CATEGORIES),
            },
            "source_byte_identity_proof": (
                "recipe2 video paths must be a subset of the v2 splits hashed by the "
                "sealed policy; the sealed leakage audit must report zero prior video-SHA overlap"
            ),
            "paired_ngram_role": "diagnostic_only",
        },
        "blocking_overlaps": blocking,
        "blocking_examples": examples,
        "diagnostics": diagnostics,
    }


def write_fresh_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise IsolationAuditError(f"refusing to overwrite audit output: {path}") from error


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--recipe2-train", type=Path, required=True)
    parser.add_argument("--recipe2-eval", type=Path, required=True)
    parser.add_argument("--v2-train", type=Path, required=True)
    parser.add_argument("--v2-eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = args.out.expanduser().resolve()
    inputs = {
        value.expanduser().resolve()
        for value in (
            args.cases,
            args.gold,
            args.leakage_audit,
            args.policy,
            args.recipe2_train,
            args.recipe2_eval,
            args.v2_train,
            args.v2_eval,
        )
    }
    if output in inputs:
        print("isolation audit aborted: output must differ from every input", file=sys.stderr)
        return 1
    if output.exists() or output.is_symlink():
        print(f"isolation audit aborted: refusing to overwrite {output}", file=sys.stderr)
        return 1
    try:
        audit = build_supplemental_audit(
            cases_path=args.cases,
            gold_path=args.gold,
            leakage_audit_path=args.leakage_audit,
            policy_path=args.policy,
            recipe2_train_path=args.recipe2_train,
            recipe2_eval_path=args.recipe2_eval,
            v2_train_path=args.v2_train,
            v2_eval_path=args.v2_eval,
        )
        write_fresh_json(output, audit)
    except (IsolationAuditError, SelectionValidationError, OSError, ValueError) as error:
        print(f"isolation audit aborted: {error}", file=sys.stderr)
        return 1
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

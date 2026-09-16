#!/usr/bin/env python3
"""Build a fresh, leakage-audited validation set for adapter interpolation.

The builder consumes newly streamed source videos and rejects any source that
can be linked to Week-2 data by sample id, file basename, or encoded-video
SHA-256.  It then assigns exactly one synthetic planner case to each of 384
distinct videos.  Catalog values and prompt templates are also checked against
the previous v2/refresh corpora and the Day-14 development gate.

This script only creates evaluation metadata.  It neither runs inference nor
looks at candidate-model outputs.  The emitted policy pre-registers the exact
parameter-delta interpolation grid and selection rule before inference starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from evaluation.agent_only_score import (
        PLAN_FIELDS,
        constraint_is_retained,
        normalize,
        valid_plan,
    )
except ModuleNotFoundError:  # direct ``python scripts/...py`` execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.agent_only_score import (
        PLAN_FIELDS,
        constraint_is_retained,
        normalize,
        valid_plan,
    )


TOTAL_CASES = 384
REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORY_COUNTS = {
    "no_search_negative": 128,
    "true_search_positive": 64,
    "routing_control": 64,
    "mask_control": 64,
    "rewrite_retention": 64,
}
NO_SEARCH_SUBTYPE_COUNTS = {
    "generic_style": 64,
    "generic_background": 32,
    "ordinary_target": 32,
}
MASK_TRIGGER_COUNTS = {"triggered": 32, "not_triggered": 32}
LAMBDA_GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_DRAWS = 10_000
EXPLICIT_FORBIDDEN_PHRASES = (
    "clearly visible",
    "surrounding scene",
    "sunset waterfront",
)
# Routing and mask controls measure a categorical/trigger decision rather than
# recall of the noun phrase itself. For those two categories, an exact indexed
# concept is still excluded, while an incidental phrase occurrence inside an
# older instruction is recorded rather than treated as leakage. Search
# entities, rewrite values, and no-search concepts keep the stricter phrase
# audit because their identity or wording contributes to the measured result.
# The same split applies to broad prompt-template signatures: exact prompts
# and exact template ids are forbidden everywhere, but shapes such as
# "erase the <slot>" are audit-only for routing/mask controls.
GENERIC_AXIS_CATEGORIES = frozenset({"routing_control", "mask_control"})
SUBTASK_ORDER = (
    "global_style",
    "remove_object",
    "add_object",
    "replace_object",
    "change_background",
    "change_color",
    "change_weather",
    "add_effect",
    "customization",
    "combined_tasks",
    "camera_edit",
)
ROUTING_CONTROL_COUNTS = {
    subtask: 6 if index < 9 else 5
    for index, subtask in enumerate(SUBTASK_ORDER)
}


@dataclass(frozen=True)
class PriorIndex:
    identifiers: frozenset[str]
    basenames: frozenset[str]
    video_sha256: frozenset[str]
    prompts: frozenset[str]
    concepts: frozenset[str]
    template_ids: frozenset[str]
    template_signatures: frozenset[str]
    constraint_values: frozenset[str]
    normalized_texts: tuple[str, ...]
    video_paths: int


@dataclass(frozen=True)
class SourceVideo:
    sample_id: str
    video_path: Path
    video_sha256: str
    manifest_row: dict[str, Any]


@dataclass(frozen=True)
class CaseSpec:
    category: str
    subtype: str
    axis: str
    prompt: str
    plan: dict[str, Any]
    constraints: tuple[dict[str, Any], ...]
    concept_id: str
    template_id: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text_exclusive(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _write_text_exclusive(path: Path, value: str) -> None:
    """Create one artifact without ever replacing bytes already at its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    claimed = False
    try:
        with path.open("x", encoding="utf-8") as stream:
            claimed = True
            stream.write(value)
    except FileExistsError as error:
        raise ValueError(
            f"refusing to overwrite existing output artifact: {path}"
        ) from error
    except Exception:
        # The path did not exist before this invocation and was exclusively
        # claimed above, so removing an incomplete write cannot erase an
        # earlier sealed artifact.
        if claimed:
            path.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _identifier(value: Any) -> str:
    return str(value).strip().casefold()


def _path_identifiers(path: Path) -> set[str]:
    return {_identifier(path.name), _identifier(path.stem)}


def _extract_plan(row: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("gold_plan", "target_plan", "plan"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            try:
                value = json.loads(str(message.get("content", "")))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def _row_video_paths(row: dict[str, Any], dataset_path: Path) -> list[Path]:
    raw_paths: list[str] = []
    videos = row.get("videos")
    if isinstance(videos, list):
        raw_paths.extend(str(value) for value in videos if isinstance(value, str))
    for key in ("video_path", "source_video"):
        value = row.get(key)
        if isinstance(value, str):
            raw_paths.append(value)
    resolved = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_absolute():
            resolved.append(path.resolve())
            continue
        candidates = {
            candidate.resolve()
            for candidate in (dataset_path.parent / path, REPO_ROOT / path)
            if candidate.is_file()
        }
        if not candidates:
            attempted = ", ".join(
                str(candidate.resolve())
                for candidate in (dataset_path.parent / path, REPO_ROOT / path)
            )
            raise FileNotFoundError(
                f"cannot resolve relative video path {raw!r} from {dataset_path}; tried {attempted}"
            )
        if len(candidates) != 1:
            raise ValueError(
                f"ambiguous relative video path {raw!r} from {dataset_path}: "
                + ", ".join(str(candidate) for candidate in sorted(candidates, key=str))
            )
        resolved.append(next(iter(candidates)))
    return resolved


def _row_prompts(row: dict[str, Any]) -> list[str]:
    prompts = []
    for key in ("prompt", "raw_user_request", "clean_instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            prompts.append(value.strip())
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                prompts.append(content.replace("<video>", "", 1).strip())
    return prompts


def _row_relevant_texts(row: dict[str, Any]) -> list[str]:
    values = _row_prompts(row)
    plan = _extract_plan(row)
    if isinstance(plan, dict):
        values.extend(
            str(plan[key])
            for key in ("refined_text_instruction", "image_search", "mask")
            if isinstance(plan.get(key), str)
        )
    return [value for value in values if value.strip()]


def _template_signature(prompt: str, slot_values: Iterable[str]) -> str:
    """Return a normalized prompt skeleton with known semantic slots removed."""
    signature = f" {normalize(prompt)} "
    normalized_values = sorted(
        {normalize(value) for value in slot_values if normalize(value)},
        key=lambda value: (-len(value.split()), -len(value), value),
    )
    for value in normalized_values:
        signature = signature.replace(f" {value} ", " __slot__ ")
    return " ".join(signature.split())


def _template_signature_matches_prompt(signature: str, prompt: str) -> bool:
    """Match a known-slot template against a normalized prompt from older data."""
    parts = signature.split("__slot__")
    if len(parts) == 1:
        return signature == prompt
    pattern = "^" + ".+?".join(re.escape(part) for part in parts) + "$"
    return re.fullmatch(pattern, prompt) is not None


def _template_seen_in_prior(signature: str, prior: PriorIndex) -> bool:
    return signature in prior.template_signatures or any(
        _template_signature_matches_prompt(signature, prompt)
        for prompt in prior.prompts
    )


def _row_template_slot_values(row: dict[str, Any]) -> list[str]:
    values = [*_metadata_values(row, "concept_id"), *_constraint_values(row)]
    plan = _extract_plan(row)
    if isinstance(plan, dict):
        values.extend(
            str(plan[key])
            for key in ("image_search", "mask")
            if isinstance(plan.get(key), str)
        )
    return values


def _metadata_values(row: dict[str, Any], key: str) -> list[str]:
    values: list[str] = []
    direct = row.get(key)
    if isinstance(direct, str):
        values.append(direct)
    catalog = row.get("catalog")
    if isinstance(catalog, dict) and isinstance(catalog.get(key), str):
        values.append(str(catalog[key]))
    return values


def _constraint_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    constraints = row.get("constraints")
    if not isinstance(constraints, list):
        return values
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        value = constraint.get("value")
        if isinstance(value, str):
            values.append(value)
        aliases = constraint.get("aliases")
        if isinstance(aliases, list):
            values.extend(str(alias) for alias in aliases if isinstance(alias, str))
    return values


def build_prior_index(dataset_paths: Sequence[Path]) -> PriorIndex:
    identifiers: set[str] = set()
    basenames: set[str] = set()
    video_hashes: set[str] = set()
    prompts: set[str] = set()
    concepts: set[str] = set()
    template_ids: set[str] = set()
    template_signatures: set[str] = set()
    constraint_values: set[str] = set()
    texts: list[str] = []
    hash_cache: dict[Path, str] = {}
    seen_video_paths: set[Path] = set()

    for dataset_path in dataset_paths:
        for row in load_jsonl(dataset_path):
            for key in ("sample_id", "bench_id", "source_video_id"):
                value = row.get(key)
                if isinstance(value, (str, int)) and str(value).strip():
                    identifiers.add(_identifier(value))
            source = row.get("source")
            if isinstance(source, dict):
                for key in ("sample_id", "source_id", "source_video_id"):
                    value = source.get(key)
                    if isinstance(value, (str, int)) and str(value).strip():
                        identifiers.add(_identifier(value))
            for path in _row_video_paths(row, dataset_path):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"cannot complete prior-video SHA audit; missing {path} from {dataset_path}"
                    )
                resolved_path = path.resolve()
                seen_video_paths.add(resolved_path)
                basenames.update(_path_identifiers(resolved_path))
                if resolved_path not in hash_cache:
                    hash_cache[resolved_path] = sha256_file(resolved_path)
                video_hashes.add(hash_cache[resolved_path])
            row_prompts = _row_prompts(row)
            prompts.update(normalize(value) for value in row_prompts if normalize(value))
            slot_values = _row_template_slot_values(row)
            template_signatures.update(
                _template_signature(value, slot_values) for value in row_prompts
            )
            for value in _metadata_values(row, "concept_id"):
                if normalize(value):
                    concepts.add(normalize(value))
            for value in _metadata_values(row, "template_id"):
                if normalize(value):
                    template_ids.add(normalize(value))
            for value in _constraint_values(row):
                if normalize(value):
                    constraint_values.add(normalize(value))
            plan = _extract_plan(row)
            if isinstance(plan, dict):
                for key in ("image_search", "mask"):
                    value = plan.get(key)
                    if isinstance(value, str) and normalize(value):
                        concepts.add(normalize(value))
            texts.extend(normalize(value) for value in _row_relevant_texts(row) if normalize(value))

    return PriorIndex(
        identifiers=frozenset(identifiers),
        basenames=frozenset(basenames),
        video_sha256=frozenset(video_hashes),
        prompts=frozenset(prompts),
        concepts=frozenset(concepts),
        template_ids=frozenset(template_ids),
        template_signatures=frozenset(template_signatures),
        constraint_values=frozenset(constraint_values),
        normalized_texts=tuple(texts),
        video_paths=len(seen_video_paths),
    )


def _phrase_in_prior(value: str, prior: PriorIndex) -> bool:
    phrase = normalize(value)
    if not phrase:
        return False
    padded = f" {phrase} "
    return any(padded in f" {text} " for text in prior.normalized_texts)


def select_fresh_sources(
    manifest_rows: Sequence[dict[str, Any]], source_root: Path, prior: PriorIndex
) -> tuple[list[SourceVideo], dict[str, Any]]:
    eligible: list[SourceVideo] = []
    rejection_counts: Counter[str] = Counter()
    rejection_examples: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    seen_basenames: set[str] = set()
    seen_hashes: set[str] = set()

    def reject(reason: str, sample_id: str) -> None:
        rejection_counts[reason] += 1
        rejection_examples.setdefault(reason, [])
        if len(rejection_examples[reason]) < 5:
            rejection_examples[reason].append(sample_id)

    resolved_source_root = source_root.resolve()
    for index, row in enumerate(manifest_rows, 1):
        sample_id = str(row.get("sample_id", "")).strip()
        video_value = row.get("video_path")
        if not sample_id or not isinstance(video_value, str) or not video_value.strip():
            raise ValueError(f"source manifest row {index}: sample_id and video_path are required")
        id_key = _identifier(sample_id)
        if id_key in seen_ids:
            raise ValueError(f"duplicate source sample_id: {sample_id}")
        seen_ids.add(id_key)
        path = Path(video_value)
        path = path.resolve() if path.is_absolute() else (resolved_source_root / path).resolve()
        try:
            path.relative_to(resolved_source_root)
        except ValueError as exc:
            raise ValueError(
                f"source manifest row {index}: video path escapes source_root: {video_value!r}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        path_keys = _path_identifiers(path)
        digest = sha256_file(path)
        reason: str | None = None
        if id_key in prior.identifiers or id_key in prior.basenames:
            reason = "prior_sample_id"
        elif path_keys & (set(prior.identifiers) | set(prior.basenames)):
            reason = "prior_basename"
        elif digest in prior.video_sha256:
            reason = "prior_video_sha256"
        elif path_keys & seen_basenames:
            reason = "duplicate_fresh_basename"
        elif digest in seen_hashes:
            reason = "duplicate_fresh_video_sha256"
        if reason is not None:
            reject(reason, sample_id)
            continue
        seen_basenames.update(path_keys)
        seen_hashes.add(digest)
        eligible.append(SourceVideo(sample_id, path, digest, dict(row)))

    eligible.sort(key=lambda item: _stable_hash("interpolation-validation", item.sample_id, item.video_sha256))
    if len(eligible) < TOTAL_CASES:
        raise ValueError(
            f"only {len(eligible)} fresh unique videos remain after leakage filtering; need {TOTAL_CASES}"
        )
    selected = eligible[:TOTAL_CASES]
    audit = {
        "manifest_rows": len(manifest_rows),
        "eligible_unique_videos": len(eligible),
        "selected_unique_videos": len(selected),
        "unused_eligible_videos": len(eligible) - len(selected),
        "rejections_by_reason": dict(sorted(rejection_counts.items())),
        "rejection_examples": dict(sorted(rejection_examples.items())),
    }
    return selected, audit


def _plan(
    refined: str,
    subtask: str,
    *,
    search: str | bool = False,
    mask: str | bool = False,
) -> dict[str, Any]:
    value = {
        "refined_text_instruction": refined,
        "subtask": subtask,
        "image_search": search,
        "mask": mask,
    }
    if set(value) != PLAN_FIELDS or not valid_plan(value):
        raise AssertionError(value)
    if subtask == "remove_object":
        if not isinstance(mask, str):
            raise AssertionError("remove_object controls require a mask target")
    elif mask is not False:
        raise AssertionError("only remove_object controls may trigger a mask")
    return value


def _constraint(kind: str, value: str) -> dict[str, Any]:
    return {"type": kind, "value": value}


def _no_search_specs() -> dict[str, list[CaseSpec]]:
    palettes = (
        "verdigris and cream", "umber and pearl", "saffron and slate",
        "indigo and sand", "coral and graphite", "moss and linen",
        "plum and parchment", "cerulean and ochre", "russet and flax",
        "malachite and chalk",
    )
    media = (
        "block-print", "gouache-poster", "cut-felt-collage", "risograph",
        "charcoal-wash", "paper-marquetry", "tempera-mural", "ink-and-wax",
        "drypoint-poster", "woven-paper",
    )
    styles: list[CaseSpec] = []
    style_templates = (
        (
            "recast the whole clip with a {concept}",
            "Restyle the complete clip with a {concept} while retaining the original action and layout.",
        ),
        (
            "give every frame a {concept}",
            "Apply a {concept} to every frame while retaining the original action and layout.",
        ),
        (
            "translate the sequence into a {concept}",
            "Translate the full sequence into a {concept} while retaining the original action and layout.",
        ),
        (
            "carry a {concept} across the full video",
            "Carry a {concept} across the full video while retaining the original action and layout.",
        ),
    )
    for index, (palette, medium) in enumerate((p, m) for p in palettes for m in media):
        concept = f"{palette} {medium} aesthetic"
        template_index = index % len(style_templates)
        prompt_template, refined_template = style_templates[template_index]
        template = f"interp_nosearch_style_{template_index}"
        prompt = prompt_template.format(concept=concept)
        refined = refined_template.format(concept=concept)
        styles.append(
            CaseSpec("no_search_negative", "generic_style", "search", prompt,
                     _plan(refined, "global_style"), (), concept, template)
        )

    modifiers = (
        "reed-fringed", "limestone-terraced", "willow-shaded", "copper-roofed",
        "heather-bordered", "lantern-lined", "clay-paved", "vine-draped",
        "granite-edged", "cedar-screened",
    )
    settings = (
        "canal quarter", "orchard lane", "clifftop footpath",
        "market arcade", "courtyard passage", "riverside terrace",
    )
    backgrounds: list[CaseSpec] = []
    background_templates = (
        ("use {concept} as the setting", "Change the setting to {concept} while leaving the foreground subjects untouched."),
        ("move the action into {concept}", "Move the action into {concept} while leaving the foreground subjects untouched."),
        ("make the environment resemble {concept}", "Make the environment resemble {concept} while leaving the foreground subjects untouched."),
        ("set the clip within {concept}", "Set the clip within {concept} while leaving the foreground subjects untouched."),
    )
    for index, (modifier, setting) in enumerate((a, b) for a in modifiers for b in settings):
        concept = f"an ordinary {modifier} {setting}"
        template_index = index % len(background_templates)
        prompt_template, refined_template = background_templates[template_index]
        template = f"interp_nosearch_background_{template_index}"
        prompt = prompt_template.format(concept=concept)
        refined = refined_template.format(concept=concept)
        backgrounds.append(
            CaseSpec("no_search_negative", "generic_background", "search", prompt,
                     _plan(refined, "change_background"), (), concept, template)
        )

    finishes = (
        "flecked stoneware", "brushed pewter", "woven seagrass", "matte cork",
        "glazed terracotta", "ribbed canvas", "smoked acrylic", "hammered tin",
        "pressed bamboo", "mottled enamel",
    )
    objects = (
        "storage tin", "desk caddy", "plant pot", "serving tray",
        "tool cup", "letter holder",
    )
    ordinary: list[CaseSpec] = []
    ordinary_templates = (
        ("swap the nearest loose prop for {concept}", "Replace the nearest loose prop with {concept} at the same position and scale."),
        ("make the closest small object {concept}", "Replace the closest small object with {concept} at the same position and scale."),
        ("exchange the foreground accessory for {concept}", "Exchange the foreground accessory for {concept} at the same position and scale."),
        ("turn the side prop into {concept}", "Turn the side prop into {concept} at the same position and scale."),
    )
    for index, (finish, obj) in enumerate((a, b) for a in finishes for b in objects):
        concept = f"a plain {finish} {obj}"
        template_index = index % len(ordinary_templates)
        prompt_template, refined_template = ordinary_templates[template_index]
        template = f"interp_nosearch_ordinary_{template_index}"
        prompt = prompt_template.format(concept=concept)
        refined = refined_template.format(concept=concept)
        ordinary.append(
            CaseSpec("no_search_negative", "ordinary_target", "search", prompt,
                     _plan(refined, "replace_object"), (), concept, template)
        )
    return {"generic_style": styles, "generic_background": backgrounds, "ordinary_target": ordinary}


SEARCH_ENTITIES: dict[str, tuple[str, ...]] = {
    "brand_product": (
        "Alessi Juicy Salif citrus squeezer", "Anglepoise Original 1227 lamp",
        "Bang & Olufsen Beogram 4000 turntable", "Bowers & Wilkins Zeppelin speaker",
        "Brompton C Line folding bicycle", "De'Longhi La Specialista coffee machine",
        "Herman Miller Aeron chair", "Le Creuset Signature Dutch oven",
        "Moccamaster KBGV coffee brewer", "Muji wall-mounted CD player",
        "Naim Mu-so 2nd Generation speaker", "Olympus OM-1 film camera",
        "Pelican 1510 protector case", "Rancilio Silvia espresso machine",
        "Technics SL-1200MK7 turntable", "Victorinox Swiss Champ knife",
        "Dualit Classic 4 Slice toaster", "Fellow Stagg EKG kettle",
        "Flos Arco floor lamp", "Hay About A Chair AAC22",
        "Kartell Componibili storage unit", "Lodge Cast Iron Combo Cooker",
        "Marantz Model 30 amplifier", "Vitra Panton Chair",
    ),
    "ip_character": (
        "Arthur the Aardvark character", "Babar the Elephant character",
        "Bluey Heeler character", "Calvin from Calvin and Hobbes",
        "Carmen Sandiego character", "Charlie Brown character",
        "Gromit character", "Inspector Gadget character",
        "Lara Croft character", "Popeye the Sailor character",
        "Samus Aran character", "Shaun the Sheep character",
        "The Pink Panther character", "Wallace from Wallace and Gromit",
        "Wile E. Coyote character", "Yoshi from Super Mario",
        "Aang from Avatar The Last Airbender", "Betty Boop character",
        "Daria Morgendorffer character", "Scrooge McDuck character",
        "Felix the Cat character", "Johnny Bravo character",
        "Moomintroll character", "Princess Mononoke character",
    ),
    "landmark": (
        "Atomium Brussels", "Chichen Itza El Castillo pyramid",
        "Clifton Suspension Bridge", "Florence Cathedral dome",
        "Gherkin building London", "Hallgrimskirkja Reykjavik",
        "Himeji Castle", "Lotus Temple Delhi", "Metropol Parasol Seville",
        "National Congress of Brazil building", "Osaka Castle",
        "Pont du Gard aqueduct", "Quebec Chateau Frontenac",
        "Royal Pavilion Brighton", "Semperoper Dresden", "Turning Torso Malmo",
        "Casa Mila Barcelona", "Dancing House Prague", "Fallingwater house",
        "Gateshead Millennium Bridge", "Kunsthaus Graz",
        "Milwaukee Art Museum", "Rila Monastery Bulgaria", "Torre Glories Barcelona",
    ),
    "cultural_artifact": (
        "Ashanti goldweight figurine", "Baule portrait mask",
        "Coptic woven textile panel", "Dogon granary door",
        "Etruscan bucchero chalice", "Finnish kuksa wooden cup",
        "Georgian qvevri wine vessel", "Hawaiian ipu heke drum",
        "Inuit soapstone ulu model", "Javanese kris dagger",
        "Korean celadon maebyeong vase", "Moche stirrup-spout vessel",
        "Nigerian adire textile", "Oaxacan barro negro pot",
        "Quechua chullo hat", "Sami carved antler cup",
        "Berber fibula brooch", "Chimu blackware vessel",
        "Danish bentwood tine box", "Edo period inro case",
        "Fijian tabua pendant", "Guatemalan huipil blouse",
        "Hungarian Miska jug", "Icelandic drinking horn",
    ),
}


def _search_specs() -> list[CaseSpec]:
    specs: list[CaseSpec] = []
    add_templates = (
        ("introduce {entity} beside the foremost subject", "Add {entity} beside the foremost subject with coherent scale and illumination."),
        ("place {entity} in the open area near the subject", "Add {entity} in the open area near the subject with coherent scale and illumination."),
        ("show {entity} just behind the lead subject", "Add {entity} just behind the lead subject with coherent scale and illumination."),
        ("set {entity} along the free side of the frame", "Add {entity} along the free side of the frame with coherent scale and illumination."),
    )
    replace_templates = (
        ("turn the nearest display object into {entity}", "Replace the nearest display object with {entity} without moving the other elements."),
        ("swap the central exhibit for {entity}", "Replace the central exhibit with {entity} without moving the other elements."),
        ("make the frontmost prop unmistakably {entity}", "Replace the frontmost prop with {entity} without moving the other elements."),
        ("exchange the featured object for {entity}", "Replace the featured object with {entity} without moving the other elements."),
    )
    for group, entities in SEARCH_ENTITIES.items():
        if len(entities) < 24:
            raise AssertionError(f"search group {group} needs at least 24 concepts")
        for index, entity in enumerate(entities):
            subtask = "add_object" if index < len(entities) // 2 else "replace_object"
            template_index = index % 4
            if subtask == "add_object":
                prompt_template, refined_template = add_templates[template_index]
            else:
                prompt_template, refined_template = replace_templates[template_index]
            prompt = prompt_template.format(entity=entity)
            refined = refined_template.format(entity=entity)
            specs.append(
                CaseSpec(
                    "true_search_positive", f"{group}_{subtask}", "search", prompt,
                    _plan(refined, subtask, search=entity),
                    (_constraint("identity", entity),), entity,
                    f"interp_search_{group}_{subtask}_{template_index}",
                )
            )
    return specs


ROUTING_VALUES: dict[str, tuple[str, ...]] = {
    "global_style": (
        "folded-paper relief", "wax-resist illustration", "linocut tapestry",
        "salt-print photograph", "layered vellum collage", "sgraffito mural",
        "cyanotype textile", "plaster intaglio",
    ),
    "remove_object": (
        "frayed cord by the stool", "empty carton below the shelf",
        "loose tag on the gate", "crumpled wrapper near the curb",
        "small cone beside the planter", "torn notice on the post",
        "bent sign behind the barrel", "discarded cap under the railing",
    ),
    "add_object": (
        "a coiled jute rope", "a squat sandstone bowl", "a folded felt mat",
        "a narrow willow basket", "a small zinc watering can", "a linen tool roll",
        "a shallow soapstone dish", "a braided hemp strap",
    ),
    "replace_object": (
        "a frosted resin prism", "a turned beechwood cup", "a glazed oval tile",
        "a stitched canvas pouch", "a cork display block", "a pewter desk weight",
        "a cast plaster cone", "a polished horn cylinder",
    ),
    "change_background": (
        "a quiet slate quarry", "a modest riverside depot", "a broad dune boardwalk",
        "a shaded citrus grove", "an open brick kiln yard", "a simple ferry landing",
        "a low chalk escarpment", "a sparse flax drying field",
    ),
    "change_color": (
        "muted vermilion red", "powdered lapis blue", "pale celadon green",
        "smoky amethyst purple", "burnished sienna brown", "soft alabaster white",
        "dull orpiment yellow", "deep hematite maroon",
    ),
    "change_weather": (
        "fine sea mist at dawn", "a brief dry snow flurry", "broken clouds after rain",
        "a mild amber dust haze", "thin frost under sunlight", "a calm late-day drizzle",
        "soft graupel at midday", "a clearing veil of fog",
    ),
    "add_effect": (
        "slow drifting kapok fibers", "small prismatic edge glints",
        "faint reflected ripple bands", "sparse floating mica flecks",
        "soft moving leaf shadows", "subtle analog light leaks",
        "gentle airborne thistledown", "thin refracted color bands",
    ),
    "customization": (
        "a needle-felt desk mascot", "a carved tagua-nut keepsake",
        "a stitched linen emblem", "a painted papier-mache miniature",
        "a woven raffia figurine", "a glazed stoneware token",
        "a braided rush charm", "a stamped leather miniature",
    ),
    "combined_tasks": (
        "mulberry lacquer with a reed-bed setting",
        "pewter gray with a tiled workshop setting",
        "copper green with a chalk-cliff setting",
        "porcelain blue with a flax-field setting",
        "amber brown with a canal-lock setting",
        "madder red with a pumice-yard setting",
        "lichen gray with an alder-grove setting",
    ),
    "camera_edit": (
        "a measured leftward arc", "a shallow forward pedestal move",
        "a gentle descending crane", "a steady reverse tracking move",
        "a restrained clockwise orbit", "a level rightward truck",
        "a gradual upward tilt",
    ),
}


def _routing_specs() -> list[CaseSpec]:
    specs: list[CaseSpec] = []
    for subtask in SUBTASK_ORDER:
        for index, value in enumerate(ROUTING_VALUES[subtask]):
            template = f"interp_routing_{subtask}"
            if subtask == "global_style":
                prompt, refined, mask = f"render the full sequence as {value}", f"Restyle the full sequence as {value} without altering its motion.", False
            elif subtask == "remove_object":
                prompt, refined, mask = f"erase the {value}", f"Remove the {value} and reconstruct the exposed area across all frames.", value
            elif subtask == "add_object":
                prompt, refined, mask = f"set {value} in the free lower corner", f"Add {value} in the free lower corner without covering existing objects.", False
            elif subtask == "replace_object":
                prompt, refined, mask = f"exchange the nearest prop for {value}", f"Replace the nearest prop with {value} while maintaining its trajectory.", False
            elif subtask == "change_background":
                prompt, refined, mask = f"relocate the setting to {value}", f"Change the setting to {value} while keeping the foreground action intact.", False
            elif subtask == "change_color":
                prompt, refined, mask = f"recolor the center item {value}", f"Change only the center item to {value}; retain every unrelated color.", False
            elif subtask == "change_weather":
                prompt, refined, mask = f"shift the weather to {value}", f"Change the weather to {value} without obscuring the ongoing action.", False
            elif subtask == "add_effect":
                prompt, refined, mask = f"layer {value} around the action", f"Add {value} around the action without modifying the set.", False
            elif subtask == "customization":
                prompt, refined, mask = f"personalize the lead figure as {value}", f"Customize the lead figure as {value} while retaining its movement.", False
            elif subtask == "combined_tasks":
                color, setting = value.split(" with ", 1)
                prompt, refined, mask = f"make the center item {color} and use {setting}", f"Change the center item to {color} and change the setting to {setting}.", False
            else:
                prompt, refined, mask = f"use {value} as the camera move", f"Apply {value} while leaving the recorded action unchanged.", False
            specs.append(
                CaseSpec("routing_control", subtask, "routing", prompt,
                         _plan(refined, subtask, mask=mask), (), value, template)
            )
    return specs


def _mask_specs() -> list[CaseSpec]:
    objects = ("canvas satchel", "ceramic pitcher", "wicker hamper", "rubber boot", "paper parcel", "metal toolbox", "wooden crate", "fabric cushion", "glass carafe", "stone planter")
    positions = ("beside the rear step", "under the narrow counter", "near the far doorway", "at the lower frame edge", "behind the short bench", "next to the side railing")
    triggered: list[CaseSpec] = []
    for index, (obj, position) in enumerate((a, b) for a in objects for b in positions):
        target = f"{obj} {position}"
        triggered.append(
            CaseSpec("mask_control", "triggered", "mask", f"remove the {target}",
                     _plan(f"Remove the {target} and restore the newly exposed region consistently.", "remove_object", mask=target),
                     (), target, "interp_mask_trigger")
        )
    colors = ("deep madder", "pale woad", "soft lichen", "dull brass", "cool graphite", "warm pumice", "dark mulberry", "light flax", "muted jade", "smoky coral")
    targets = ("center container", "foreground cushion", "nearest sign", "side panel", "main parcel", "front railing")
    not_triggered: list[CaseSpec] = []
    for index, (color, target) in enumerate((a, b) for a in colors for b in targets):
        concept = f"{target} {color}"
        not_triggered.append(
            CaseSpec("mask_control", "not_triggered", "mask", f"make the {target} {color}",
                     _plan(f"Change only the {target} to {color} while retaining its texture.", "change_color"),
                     (), concept, "interp_mask_nontrigger")
        )
    return {"triggered": triggered, "not_triggered": not_triggered}


def _retention_specs() -> list[CaseSpec]:
    colors = ("muted carnelian red", "dusty woad blue", "pale lichen green", "soft iris violet", "warm pumice beige", "dark damson purple", "cool pewter gray", "light flax cream", "deep malachite green", "smoky coral orange")
    objects = ("ribbed linen pouch", "speckled clay flask", "woven bast basket", "brushed zinc canister", "pressed cork case", "glazed stoneware cup", "stitched felt roll", "turned alder box", "mottled resin tile", "hammered pewter bowl")
    spatial = ("along the rear-left rail", "beneath the narrow side ledge", "beside the far-right post", "inside the open lower corner", "beyond the center floor mark", "against the back-left partition", "near the outer-right boundary", "below the raised rear platform")
    protected = ("the rear wall pattern entirely intact", "the side-window reflections wholly unchanged", "the floor markings exactly undisturbed", "the distant foliage fully untouched", "the overhead fixtures completely unaltered", "the left-hand props precisely preserved", "the right-hand shadows entirely retained", "the doorway trim wholly unmodified")
    specs: list[CaseSpec] = []
    for index, (color, obj) in enumerate((a, b) for a in colors for b in objects):
        location = spatial[index % len(spatial)]
        preservation = protected[(index // len(spatial) + index) % len(protected)]
        prompt = f"make the central item {color}, add {obj} {location}, and keep {preservation}"
        refined = f"Change the central item to {color}; add {obj} {location}; keep {preservation}."
        constraints = (
            _constraint("color", color),
            _constraint("identity", obj),
            _constraint("spatial", location),
            _constraint("preservation", preservation),
        )
        specs.append(
            CaseSpec("rewrite_retention", "lexical_copy", "rewrite", prompt,
                     _plan(refined, "combined_tasks"), constraints,
                     f"{color}|{obj}", "interp_retention_lexical_copy")
        )
    return specs


def _spec_has_forbidden_phrase(spec: CaseSpec) -> bool:
    rendered = normalize(
        " ".join(
            [spec.prompt, spec.concept_id, spec.template_id,
             str(spec.plan["refined_text_instruction"]),
             *(str(item["value"]) for item in spec.constraints)]
        )
    )
    return any(f" {normalize(phrase)} " in f" {rendered} " for phrase in EXPLICIT_FORBIDDEN_PHRASES)


def _spec_template_signature(spec: CaseSpec) -> str:
    slots = [spec.concept_id, *(str(item["value"]) for item in spec.constraints)]
    for key in ("image_search", "mask"):
        value = spec.plan.get(key)
        if isinstance(value, str):
            slots.append(value)
    return _template_signature(spec.prompt, slots)


def _eligible_spec(spec: CaseSpec, prior: PriorIndex) -> tuple[bool, str | None]:
    if _spec_has_forbidden_phrase(spec):
        return False, "explicit_forbidden_phrase"
    if normalize(spec.prompt) in prior.prompts:
        return False, "prior_exact_prompt"
    normalized_concept = normalize(spec.concept_id)
    if normalized_concept in prior.concepts:
        return False, "prior_concept_exact"
    if (
        spec.category not in GENERIC_AXIS_CATEGORIES
        and _phrase_in_prior(spec.concept_id, prior)
    ):
        return False, "prior_concept_phrase"
    if normalize(spec.template_id) in prior.template_ids:
        return False, "prior_template_id"
    if (
        spec.category not in GENERIC_AXIS_CATEGORIES
        and _template_seen_in_prior(_spec_template_signature(spec), prior)
    ):
        return False, "prior_template_signature"
    for constraint in spec.constraints:
        value = str(constraint["value"])
        if normalize(value) in prior.constraint_values or _phrase_in_prior(value, prior):
            return False, "prior_constraint_value"
    return True, None


def _select_specs(
    candidates: Sequence[CaseSpec], count: int, prior: PriorIndex, label: str
) -> tuple[list[CaseSpec], Counter[str]]:
    selected: list[CaseSpec] = []
    rejections: Counter[str] = Counter()
    rejection_examples: dict[str, list[str]] = {}
    for spec in candidates:
        eligible, reason = _eligible_spec(spec, prior)
        if not eligible:
            reason = str(reason)
            rejections[reason] += 1
            examples = rejection_examples.setdefault(reason, [])
            if len(examples) < 3:
                examples.append(spec.concept_id)
            continue
        selected.append(spec)
        if len(selected) == count:
            return selected, rejections
    raise ValueError(
        f"{label}: only {len(selected)} of {len(candidates)} fresh catalog entries "
        f"remain; need {count}; rejections={dict(sorted(rejections.items()))}; "
        f"examples={dict(sorted(rejection_examples.items()))}"
    )


def build_specs(prior: PriorIndex) -> tuple[list[CaseSpec], dict[str, int]]:
    all_specs: list[CaseSpec] = []
    rejection_counts: Counter[str] = Counter()
    no_search = _no_search_specs()
    for subtype, count in NO_SEARCH_SUBTYPE_COUNTS.items():
        selected, rejected = _select_specs(no_search[subtype], count, prior, subtype)
        all_specs.extend(selected)
        rejection_counts.update(rejected)
    search_candidates = _search_specs()
    for group in SEARCH_ENTITIES:
        for subtask in ("add_object", "replace_object"):
            subtype = f"{group}_{subtask}"
            selected, rejected = _select_specs(
                [spec for spec in search_candidates if spec.subtype == subtype],
                8,
                prior,
                f"search_{subtype}",
            )
            all_specs.extend(selected)
            rejection_counts.update(rejected)
    routing_candidates = _routing_specs()
    for subtask, count in ROUTING_CONTROL_COUNTS.items():
        selected, rejected = _select_specs(
            [spec for spec in routing_candidates if spec.subtype == subtask],
            count,
            prior,
            f"routing_{subtask}",
        )
        all_specs.extend(selected)
        rejection_counts.update(rejected)
    masks = _mask_specs()
    for subtype, count in MASK_TRIGGER_COUNTS.items():
        selected, rejected = _select_specs(masks[subtype], count, prior, f"mask_{subtype}")
        all_specs.extend(selected)
        rejection_counts.update(rejected)
    selected, rejected = _select_specs(_retention_specs(), 64, prior, "rewrite_retention")
    all_specs.extend(selected)
    rejection_counts.update(rejected)
    if len(all_specs) != TOTAL_CASES:
        raise AssertionError(len(all_specs))
    return all_specs, dict(sorted(rejection_counts.items()))


def _case_and_gold(index: int, source: SourceVideo, spec: CaseSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    bench_id = f"interp_{index:04d}"
    source_meta = {
        "sample_id": source.sample_id,
        "subset": source.manifest_row.get("subset"),
        "source_dataset": source.manifest_row.get("source_dataset"),
        "license": source.manifest_row.get("license"),
        "video_sha256": source.video_sha256,
    }
    case = {
        "bench_id": bench_id,
        "video_path": str(source.video_path),
        "prompt": spec.prompt,
        "edit_type": spec.plan["subtask"],
        "axis": spec.axis,
        "category": spec.category,
        "subtype": spec.subtype,
        "source": source_meta,
        "catalog": {
            "concept_id": spec.concept_id,
            "template_id": spec.template_id,
            "template_signature": _spec_template_signature(spec),
        },
    }
    gold = {
        **case,
        "gold_plan": spec.plan,
        "constraints": [dict(item) for item in spec.constraints],
        "source_entities": [] if isinstance(spec.plan["image_search"], str) else ["existing source video"],
    }
    if isinstance(spec.plan["image_search"], str):
        gold["search_query_aliases"] = [spec.plan["image_search"]]
    return case, gold


def _assert_output_contract(cases: Sequence[dict[str, Any]], gold: Sequence[dict[str, Any]]) -> None:
    if len(cases) != TOTAL_CASES or len(gold) != TOTAL_CASES:
        raise AssertionError("interpolation validation must contain exactly 384 cases and gold rows")
    if len({row["bench_id"] for row in cases}) != TOTAL_CASES:
        raise AssertionError("duplicate bench_id")
    if len({row["video_path"] for row in cases}) != TOTAL_CASES:
        raise AssertionError("each interpolation case must use a unique video")
    if len({row["source"]["video_sha256"] for row in cases}) != TOTAL_CASES:
        raise AssertionError("selected videos are not byte-unique")
    if len({row["source"]["sample_id"] for row in cases}) != TOTAL_CASES:
        raise AssertionError("selected source sample ids are not unique")
    if len({normalize(row["prompt"]) for row in cases}) != TOTAL_CASES:
        raise AssertionError("interpolation prompts are not unique")
    if len({normalize(row["catalog"]["concept_id"]) for row in cases}) != TOTAL_CASES:
        raise AssertionError("interpolation concepts are not unique")
    if Counter(row["category"] for row in cases) != Counter(CATEGORY_COUNTS):
        raise AssertionError("unexpected category distribution")
    if Counter(row["subtype"] for row in cases if row["category"] == "no_search_negative") != Counter(NO_SEARCH_SUBTYPE_COUNTS):
        raise AssertionError("unexpected no-search subtype distribution")
    mask_rows = [row for row in gold if row["category"] == "mask_control"]
    mask_counts = Counter(
        "triggered" if isinstance(row["gold_plan"]["mask"], str) else "not_triggered"
        for row in mask_rows
    )
    if mask_counts != Counter(MASK_TRIGGER_COUNTS):
        raise AssertionError("mask controls are not balanced")
    for case, gold_row in zip(cases, gold):
        if case["bench_id"] != gold_row["bench_id"] or not valid_plan(gold_row["gold_plan"]):
            raise AssertionError("case/gold alignment or plan contract failure")
        search_triggered = isinstance(gold_row["gold_plan"]["image_search"], str)
        if (gold_row["category"] == "true_search_positive") != search_triggered:
            raise AssertionError(
                "all and only true_search_positive controls must trigger image search"
            )
        if search_triggered:
            aliases = gold_row.get("search_query_aliases")
            if aliases != [gold_row["gold_plan"]["image_search"]]:
                raise AssertionError("search-positive query alias must exactly match gold query")
        elif "search_query_aliases" in gold_row:
            raise AssertionError("no-search controls must not define search query aliases")
        if gold_row["category"] == "mask_control":
            mask_triggered = isinstance(gold_row["gold_plan"]["mask"], str)
            if (gold_row["subtype"] == "triggered") != mask_triggered:
                raise AssertionError("mask-control subtype and mask trigger must agree")
        if gold_row["category"] == "rewrite_retention":
            refined = str(gold_row["gold_plan"]["refined_text_instruction"])
            request = str(gold_row["prompt"])
            for constraint in gold_row["constraints"]:
                if not constraint_is_retained(constraint, request):
                    raise AssertionError(f"constraint absent from request: {constraint}")
                if not constraint_is_retained(constraint, refined):
                    raise AssertionError(f"constraint absent from refined target: {constraint}")


def build_interpolation_validation(
    manifest_rows: Sequence[dict[str, Any]],
    source_root: Path,
    prior: PriorIndex,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sources, source_audit = select_fresh_sources(manifest_rows, source_root, prior)
    specs, catalog_rejections = build_specs(prior)
    cases: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for index, (source, spec) in enumerate(zip(sources, specs), 1):
        case, gold_row = _case_and_gold(index, source, spec)
        cases.append(case)
        gold.append(gold_row)
    _assert_output_contract(cases, gold)

    selected_ids = {_identifier(row["source"]["sample_id"]) for row in cases}
    selected_basenames = {
        key for row in cases for key in _path_identifiers(Path(row["video_path"]))
    }
    selected_hashes = {row["source"]["video_sha256"] for row in cases}
    prompts = {normalize(row["prompt"]) for row in cases}
    concepts = {normalize(row["catalog"]["concept_id"]) for row in cases}
    concept_records = [
        {
            "category": str(row["category"]),
            "subtype": str(row["subtype"]),
            "concept_id": normalize(row["catalog"]["concept_id"]),
        }
        for row in cases
    ]
    templates = {normalize(row["catalog"]["template_id"]) for row in cases}
    template_signatures = {
        str(row["catalog"]["template_signature"]) for row in cases
    }
    constraint_values = {
        normalize(item["value"])
        for row in gold
        for item in row.get("constraints", [])
    }
    rendered = normalize(json.dumps({"cases": cases, "gold": gold}, ensure_ascii=False))
    forbidden_hits = [
        phrase for phrase in EXPLICIT_FORBIDDEN_PHRASES
        if f" {normalize(phrase)} " in f" {rendered} "
    ]
    prior_concept_phrase_hits = sorted({
        record["concept_id"]
        for record in concept_records
        if record["category"] not in GENERIC_AXIS_CATEGORIES
        and _phrase_in_prior(record["concept_id"], prior)
    })
    allowed_generic_concept_phrase_hits = sorted(
        (
            {
                "category": record["category"],
                "subtype": record["subtype"],
                "concept_id": record["concept_id"],
            }
            for record in concept_records
            if record["category"] in GENERIC_AXIS_CATEGORIES
            and _phrase_in_prior(record["concept_id"], prior)
        ),
        key=lambda item: (item["category"], item["subtype"], item["concept_id"]),
    )
    prior_constraint_phrase_hits = sorted(
        value for value in constraint_values if _phrase_in_prior(value, prior)
    )
    template_records = [
        {
            "category": str(row["category"]),
            "subtype": str(row["subtype"]),
            "template_signature": str(row["catalog"]["template_signature"]),
        }
        for row in cases
    ]
    prior_template_signature_hits = sorted({
        record["template_signature"]
        for record in template_records
        if record["category"] not in GENERIC_AXIS_CATEGORIES
        and _template_seen_in_prior(record["template_signature"], prior)
    })
    allowed_generic_template_signature_hits = sorted(
        (
            {
                "category": record["category"],
                "subtype": record["subtype"],
                "template_signature": record["template_signature"],
            }
            for record in template_records
            if record["category"] in GENERIC_AXIS_CATEGORIES
            and _template_seen_in_prior(record["template_signature"], prior)
        ),
        key=lambda item: (
            item["category"], item["subtype"], item["template_signature"]
        ),
    )
    overlap = {
        "prior_identifier_overlap": len(selected_ids & set(prior.identifiers)),
        "prior_basename_overlap": len(selected_basenames & set(prior.basenames)),
        "prior_video_sha256_overlap": len(selected_hashes & set(prior.video_sha256)),
        "prior_exact_prompt_overlap": len(prompts & set(prior.prompts)),
        "prior_concept_overlap": len(concepts & set(prior.concepts)),
        "prior_template_id_overlap": len(templates & set(prior.template_ids)),
        "prior_template_signature_overlap": len(prior_template_signature_hits),
        "prior_constraint_value_overlap": len(constraint_values & set(prior.constraint_values)),
        "prior_concept_phrase_hit_count": len(prior_concept_phrase_hits),
        "prior_constraint_phrase_hit_count": len(prior_constraint_phrase_hits),
        "allowed_generic_concept_phrase_hit_count": len(allowed_generic_concept_phrase_hits),
        "allowed_generic_template_signature_hit_count": len(
            allowed_generic_template_signature_hits
        ),
        "explicit_forbidden_phrase_hits": forbidden_hits,
    }
    blocking_overlap_keys = {
        "prior_identifier_overlap",
        "prior_basename_overlap",
        "prior_video_sha256_overlap",
        "prior_exact_prompt_overlap",
        "prior_concept_overlap",
        "prior_template_id_overlap",
        "prior_template_signature_overlap",
        "prior_constraint_value_overlap",
        "prior_concept_phrase_hit_count",
        "prior_constraint_phrase_hit_count",
    }
    if any(overlap[key] for key in blocking_overlap_keys) or forbidden_hits:
        raise AssertionError(f"leakage audit failed: {overlap}")

    retention_rows = [row for row in gold if row["category"] == "rewrite_retention"]
    retention_constraints = [item for row in retention_rows for item in row["constraints"]]
    video_set_digest = hashlib.sha256(
        "\n".join(
            sorted(f"{row['source']['sample_id']} {row['source']['video_sha256']}" for row in cases)
        ).encode("utf-8")
    ).hexdigest()
    audit = {
        "recipe_version": 2,
        "counts": {
            "cases": len(cases),
            "gold": len(gold),
            "by_category": dict(sorted(Counter(row["category"] for row in cases).items())),
            "no_search_by_subtype": dict(sorted(Counter(
                row["subtype"] for row in cases if row["category"] == "no_search_negative"
            ).items())),
            "mask_trigger": dict(sorted(Counter(
                "triggered" if isinstance(row["gold_plan"]["mask"], str) else "not_triggered"
                for row in gold if row["category"] == "mask_control"
            ).items())),
            "search_triggered": sum(isinstance(row["gold_plan"]["image_search"], str) for row in gold),
            "retention_rows": len(retention_rows),
            "retention_constraints": len(retention_constraints),
            "retention_constraints_by_type": dict(sorted(Counter(
                str(item["type"]) for item in retention_constraints
            ).items())),
        },
        "source_filter": source_audit,
        "prior_index": {
            "identifiers": len(prior.identifiers),
            "basenames": len(prior.basenames),
            "unique_video_paths": prior.video_paths,
            "unique_video_sha256": len(prior.video_sha256),
            "prompts": len(prior.prompts),
            "concepts": len(prior.concepts),
            "template_ids": len(prior.template_ids),
            "template_signatures": len(prior.template_signatures),
            "constraint_values": len(prior.constraint_values),
        },
        "catalog_rejections_by_reason": catalog_rejections,
        "isolation": {
            **overlap,
            "prior_concept_phrase_hits": prior_concept_phrase_hits,
            "allowed_generic_concept_phrase_hits": allowed_generic_concept_phrase_hits,
            "concept_phrase_policy": {
                "exact_concept_overlap_forbidden_for_all_categories": True,
                "phrase_containment_forbidden_categories": sorted(
                    set(CATEGORY_COUNTS) - set(GENERIC_AXIS_CATEGORIES)
                ),
                "phrase_containment_audit_only_categories": sorted(
                    GENERIC_AXIS_CATEGORIES
                ),
            },
            "prior_constraint_phrase_hits": prior_constraint_phrase_hits,
            "prior_template_signature_hits": prior_template_signature_hits,
            "allowed_generic_template_signature_hits": (
                allowed_generic_template_signature_hits
            ),
            "template_signature_policy": {
                "exact_prompt_and_template_id_overlap_forbidden_for_all_categories": True,
                "wildcard_signature_overlap_forbidden_categories": sorted(
                    set(CATEGORY_COUNTS) - set(GENERIC_AXIS_CATEGORIES)
                ),
                "wildcard_signature_overlap_audit_only_categories": sorted(
                    GENERIC_AXIS_CATEGORIES
                ),
            },
            "selected_sample_ids": len(selected_ids),
            "selected_video_paths": len({row["video_path"] for row in cases}),
            "selected_video_sha256": len(selected_hashes),
            "unique_prompts": len(prompts),
            "unique_concepts": len(concepts),
            "unique_template_ids": len(templates),
            "unique_template_signatures": len(template_signatures),
            "selected_video_set_digest_sha256": video_set_digest,
        },
        "lexical_retention": {
            "all_values_in_request": all(
                constraint_is_retained(item, row["prompt"])
                for row in retention_rows for item in row["constraints"]
            ),
            "all_values_in_refined_instruction": all(
                constraint_is_retained(item, row["gold_plan"]["refined_text_instruction"])
                for row in retention_rows for item in row["constraints"]
            ),
        },
    }
    return cases, gold, audit


def _validate_adapter_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return normalized


def make_selection_policy(
    *,
    input_hashes: dict[str, str],
    cases_sha256: str,
    gold_sha256: str,
    audit_sha256: str,
    v2_adapter_sha256: str,
    refresh1_adapter_sha256: str,
    primary_adapter_sha256: str | None = None,
) -> dict[str, Any]:
    policy = {
        "version": 2,
        "created_before_candidate_inference": True,
        "candidate_rule": {
            "method": "exact_parameter_delta_interpolation",
            "formula": "theta(lambda)=theta_v2+lambda*(theta_refresh1-theta_v2)",
            "lambda_grid": LAMBDA_GRID,
            "candidate_count": len(LAMBDA_GRID),
            "v2_adapter_model_sha256": _validate_adapter_sha256(v2_adapter_sha256, "v2 adapter hash"),
            "refresh1_adapter_model_sha256": _validate_adapter_sha256(refresh1_adapter_sha256, "refresh1 adapter hash"),
            "additional_training_allowed": False,
            "day14_outputs_or_metrics_allowed_for_generation_or_selection": False,
            "day14_case_content_use": "exclusion-only during validation-set construction",
            "day14_use": "at most one post-selection adaptive regression run; not confirmatory or held-out",
            "confirmation_requirement": "confirmatory claims require a separate sealed evaluation set",
        },
        "validation_artifacts": {
            "cases_sha256": cases_sha256,
            "gold_sha256": gold_sha256,
            "leakage_audit_sha256": audit_sha256,
            "input_sha256": dict(sorted(input_hashes.items())),
            "expected_cases": TOTAL_CASES,
            "expected_unique_videos": TOTAL_CASES,
            "expected_category_counts": dict(CATEGORY_COUNTS),
            "expected_no_search_subtype_counts": dict(NO_SEARCH_SUBTYPE_COUNTS),
            "expected_mask_trigger_counts": dict(MASK_TRIGGER_COUNTS),
        },
        "eligibility_thresholds": {
            "complete_prediction_rows": TOTAL_CASES,
            "strict_raw_json_validity_min": 1.0,
            "subtask_accuracy_min": 0.95,
            "no_search_specificity_min": 0.95,
            "true_search_trigger_recall_min": 0.95,
            "search_query_end_to_end_recall_min": 0.95,
            "mask_trigger_f1_min": 0.95,
            "rewrite_constraint_retention_min": 0.85,
        },
        "eligibility_threshold_scopes": {
            "strict_raw_json_validity": "all 384 cases",
            "subtask_accuracy": "64 routing_control cases only",
            "no_search_specificity": "128 no_search_negative cases only",
            "true_search_trigger_recall": "64 true_search_positive cases only",
            "search_query_end_to_end_recall": "64 true_search_positive cases only",
            "mask_trigger_f1": "64 mask_control cases only",
            "rewrite_constraint_retention": "64 rewrite_retention cases only",
        },
        "utility": {
            "formula": "0.5*no_search_specificity+0.5*rewrite_constraint_retention",
            "no_search_specificity_scope": "128 no_search_negative cases only",
            "rewrite_constraint_retention_scope": "64 rewrite_retention cases only",
            "higher_is_better": True,
        },
        "bootstrap": {
            "method": "paired stratified case bootstrap over the two utility strata",
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "strata": {
                "no_search_negative": 128,
                "rewrite_retention": 64,
            },
        },
        "selection": {
            "rule": "one_standard_error_then_smallest_lambda",
            "reference": "eligible candidate with highest point-estimate utility; smallest lambda resolves a point-estimate tie for SE(best)",
            "reference_tie_breaker": "smallest lambda",
            "one_se_set": "eligible candidates with utility >= best utility - bootstrap SE(best)",
            "tie_breaker": "smallest lambda",
            "day14_metrics_used": False,
        },
    }
    if primary_adapter_sha256 is not None:
        policy["final_decision"] = {
            "rule": "primary_if_eligible_else_grid",
            "primary_candidate": {
                "candidate_id": "recipe2",
                "adapter_model_sha256": _validate_adapter_sha256(
                    primary_adapter_sha256, "primary adapter hash"
                ),
                "directory_name": "recipe2",
            },
        }
    return policy


def write_bundle(
    *,
    source_manifest: Path,
    source_root: Path,
    v2_train: Path,
    v2_eval: Path,
    refresh1_train: Path,
    refresh1_eval: Path,
    day14_cases: Path,
    cases_out: Path,
    gold_out: Path,
    audit_out: Path,
    policy_out: Path,
    v2_adapter_sha256: str,
    refresh1_adapter_sha256: str,
    primary_adapter_sha256: str | None = None,
) -> dict[str, Any]:
    # Validate and lock the endpoint identities before writing any artifact.
    v2_adapter_sha256 = _validate_adapter_sha256(
        v2_adapter_sha256, "v2 adapter hash"
    )
    refresh1_adapter_sha256 = _validate_adapter_sha256(
        refresh1_adapter_sha256, "refresh1 adapter hash"
    )
    if primary_adapter_sha256 is not None:
        primary_adapter_sha256 = _validate_adapter_sha256(
            primary_adapter_sha256, "primary adapter hash"
        )
    if v2_adapter_sha256 == refresh1_adapter_sha256:
        raise ValueError("v2 and refresh1 adapter hashes must identify distinct endpoints")
    input_paths = {
        "source_manifest": source_manifest.resolve(),
        "v2_train": v2_train.resolve(),
        "v2_eval": v2_eval.resolve(),
        "refresh1_train": refresh1_train.resolve(),
        "refresh1_eval": refresh1_eval.resolve(),
        "day14_forbidden_cases": day14_cases.resolve(),
    }
    declared_output_paths = {
        "cases_out": cases_out.expanduser(),
        "gold_out": gold_out.expanduser(),
        "audit_out": audit_out.expanduser(),
        "policy_out": policy_out.expanduser(),
    }
    output_paths = {
        name: path.resolve() for name, path in declared_output_paths.items()
    }
    by_path: dict[Path, list[str]] = {}
    for label, path in {**input_paths, **output_paths}.items():
        by_path.setdefault(path, []).append(label)
    aliases = {path: labels for path, labels in by_path.items() if len(labels) > 1}
    if aliases:
        rendered = "; ".join(
            f"{path}: {', '.join(labels)}"
            for path, labels in sorted(aliases.items(), key=lambda item: str(item[0]))
        )
        raise ValueError(f"input/output artifact paths must be distinct: {rendered}")
    existing_declared_outputs = sorted(
        (
            path
            for path in declared_output_paths.values()
            if path.exists() or path.is_symlink()
        ),
        key=str,
    )
    if existing_declared_outputs:
        rendered = ", ".join(str(path) for path in existing_declared_outputs)
        raise ValueError(
            "refusing to overwrite existing output artifact(s); use a fresh bundle "
            f"destination: {rendered}"
        )
    existing_outputs = sorted(
        (path for path in output_paths.values() if path.exists() or path.is_symlink()),
        key=str,
    )
    if existing_outputs:
        rendered = ", ".join(str(path) for path in existing_outputs)
        raise ValueError(
            "refusing to overwrite existing output artifact(s); use a fresh bundle "
            f"destination: {rendered}"
        )
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    prior_paths = [
        input_paths["v2_train"], input_paths["v2_eval"],
        input_paths["refresh1_train"], input_paths["refresh1_eval"],
        input_paths["day14_forbidden_cases"],
    ]
    prior = build_prior_index(prior_paths)
    cases, gold, audit = build_interpolation_validation(
        load_jsonl(input_paths["source_manifest"]), source_root.resolve(), prior
    )
    cases_out = output_paths["cases_out"]
    gold_out = output_paths["gold_out"]
    audit_out = output_paths["audit_out"]
    policy_out = output_paths["policy_out"]
    created_outputs: list[Path] = []
    try:
        write_jsonl(cases_out, cases)
        created_outputs.append(cases_out)
        write_jsonl(gold_out, gold)
        created_outputs.append(gold_out)
        audit["artifacts"] = {
            "input_sha256": dict(sorted(input_hashes.items())),
            "cases_sha256": sha256_file(cases_out),
            "gold_sha256": sha256_file(gold_out),
        }
        _write_text_exclusive(
            audit_out,
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        )
        created_outputs.append(audit_out)
        policy = make_selection_policy(
            input_hashes=input_hashes,
            cases_sha256=sha256_file(cases_out),
            gold_sha256=sha256_file(gold_out),
            audit_sha256=sha256_file(audit_out),
            v2_adapter_sha256=v2_adapter_sha256,
            refresh1_adapter_sha256=refresh1_adapter_sha256,
            primary_adapter_sha256=primary_adapter_sha256,
        )
        _write_text_exclusive(
            policy_out,
            json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        )
        created_outputs.append(policy_out)
    except Exception:
        for path in reversed(created_outputs):
            path.unlink(missing_ok=True)
        raise
    return {"audit": audit, "policy": policy, "cases": cases, "gold": gold}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--v2-train", type=Path, required=True)
    parser.add_argument("--v2-eval", type=Path, required=True)
    parser.add_argument("--refresh1-train", type=Path, required=True)
    parser.add_argument("--refresh1-eval", type=Path, required=True)
    parser.add_argument("--day14-cases", type=Path, required=True)
    parser.add_argument("--cases-out", type=Path, required=True)
    parser.add_argument("--gold-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--policy-out", type=Path, required=True)
    parser.add_argument("--v2-adapter-sha256", required=True)
    parser.add_argument("--refresh1-adapter-sha256", required=True)
    parser.add_argument("--primary-adapter-sha256")
    args = parser.parse_args()
    result = write_bundle(
        source_manifest=args.source_manifest,
        source_root=args.source_root,
        v2_train=args.v2_train,
        v2_eval=args.v2_eval,
        refresh1_train=args.refresh1_train,
        refresh1_eval=args.refresh1_eval,
        day14_cases=args.day14_cases,
        cases_out=args.cases_out,
        gold_out=args.gold_out,
        audit_out=args.audit_out,
        policy_out=args.policy_out,
        v2_adapter_sha256=args.v2_adapter_sha256,
        refresh1_adapter_sha256=args.refresh1_adapter_sha256,
        primary_adapter_sha256=args.primary_adapter_sha256,
    )
    print(json.dumps({
        "cases": len(result["cases"]),
        "gold": len(result["gold"]),
        "cases_out": str(args.cases_out),
        "gold_out": str(args.gold_out),
        "audit_out": str(args.audit_out),
        "policy_out": str(args.policy_out),
    }, indent=2))


if __name__ == "__main__":
    main()

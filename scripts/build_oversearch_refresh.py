#!/usr/bin/env python3
"""Build a leakage-controlled Week-2 refresh set for over-search errors.

The refresh deliberately separates the original grouped train/eval splits:
training examples only borrow videos from the original training split, while
validation examples only borrow videos from the original evaluation split.
Synthetic concepts and prompt templates are also split before construction.

The fixed v2 output recipe is:

* train (1,024): 256 no-search counterexamples + 768 replay examples;
* validation (256): 64 no-search targets + 64 true-search targets +
  64 routing/constraint targets + 64 original-eval replay examples.

All rows use LLaMA-Factory's ShareGPT multimodal format.  Separate validation
case and gold files make it possible to run ``aurora.agent`` and the existing
agent-only scorer without deriving labels from the training file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.agent_only_score import PLAN_FIELDS, normalize, valid_plan
from scripts.augment_planner_sft import EXTERNAL_ENTITY_GROUPS, ROUTING_TEMPLATES


TRAIN_SIZE = 1_024
VALIDATION_SIZE = 256
RECIPE_VERSION = 2
DEFAULT_FORBIDDEN_NGRAM_N = 4
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORBIDDEN_CASE_PATH = REPO_ROOT / "data/week1/planner_100.jsonl"

TRAIN_CATEGORY_COUNTS = {
    "named_style_negative": 128,
    "regional_background_negative": 80,
    "ordinary_negative": 48,
    "true_search_replay": 256,
    "route_replay": 256,
    "original_replay": 256,
}
VALIDATION_CATEGORY_COUNTS = {
    "target_negative": 64,
    "true_search": 64,
    "routing_constraints": 64,
    "original_eval_replay": 64,
}

# These are the named targets that appear in the Week-1 regression gate, plus
# the two false-positive concepts that motivated this refresh.  They must not
# occur anywhere in a generated artifact.  CLI callers may add more concepts
# by passing a forbidden-case JSONL file.
DEFAULT_FORBIDDEN_CONCEPTS = (
    "Snoopy",
    "Stanley Quencher",
    "Batmobile",
    "Starbucks holiday cup",
    "Eiffel Tower",
    "Spider-Man",
    "Burj Al Arab",
    "Trek Madone racing bicycle",
    "Pikachu",
    "Japanese cherry blossom tree",
    "Van Gogh",
    "Vincent",
    "Vincent van Gogh",
    "Japanese garden",
)

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
SEARCHABLE_SUBTASKS = {"add_object", "replace_object", "change_background", "customization"}


# Sixty-four train-only concepts, each rendered with two different templates.
# All are deceased artists, so a named style reference remains a style command
# rather than an instruction to retrieve an external reference image.
TRAIN_HISTORICAL_STYLES = (
    "Leonardo da Vinci", "Michelangelo Buonarroti", "Raphael Sanzio", "Titian",
    "Sandro Botticelli", "Caravaggio", "Rembrandt", "Johannes Vermeer",
    "Diego Velazquez", "Francisco Goya", "J. M. W. Turner", "John Constable",
    "Eugene Delacroix", "Caspar David Friedrich", "Jean-Auguste-Dominique Ingres",
    "Gustave Courbet", "Edouard Manet", "Claude Monet", "Pierre-Auguste Renoir",
    "Edgar Degas", "Berthe Morisot", "Mary Cassatt", "Paul Cezanne",
    "Paul Gauguin", "Georges Seurat", "Henri de Toulouse-Lautrec", "Gustav Klimt",
    "Egon Schiele", "Edvard Munch", "Wassily Kandinsky", "Kazimir Malevich",
    "Piet Mondrian", "Paul Klee", "Joan Miro", "Salvador Dali", "Rene Magritte",
    "Giorgio de Chirico", "Edward Hopper", "Georgia O'Keeffe", "Grant Wood",
    "Frida Kahlo", "Diego Rivera", "Tamara de Lempicka", "Amedeo Modigliani",
    "Marc Chagall", "Henri Rousseau", "Katsushika Hokusai", "Utagawa Hiroshige",
    "Sesshu Toyo", "Qi Baishi", "Zhang Daqian", "Raja Ravi Varma",
    "Abanindranath Tagore", "Amrita Sher-Gil", "Artemisia Gentileschi",
    "Hilma af Klint", "Rosa Bonheur", "Elisabeth Vigee Le Brun",
    "Thomas Gainsborough", "Joshua Reynolds", "Albrecht Durer",
    "Hans Holbein the Younger", "Pieter Bruegel the Elder", "Jan van Eyck",
)

VALIDATION_HISTORICAL_STYLES = (
    "Fra Angelico", "Giotto di Bondone", "Masaccio", "Piero della Francesca",
    "Andrea Mantegna", "Tintoretto", "Peter Paul Rubens", "Anthony van Dyck",
    "Nicolas Poussin", "Jean-Honore Fragonard", "William Blake", "Thomas Cole",
    "Frederic Edwin Church", "Ivan Aivazovsky", "Ilya Repin", "Camille Pissarro",
    "Alfred Sisley", "Odilon Redon", "Henri Matisse", "Georges Braque",
    "Fernand Leger", "Umberto Boccioni", "Natalia Goncharova", "Lyubov Popova",
    "Tarsila do Amaral", "Joaquin Sorolla", "Joaquin Torres-Garcia", "Wifredo Lam",
    "Aaron Douglas", "Jacob Lawrence", "Arthur Rackham", "Aubrey Beardsley",
)

TRAIN_REGIONAL_BACKGROUNDS = (
    "Andalusian whitewashed courtyard", "Tuscan hill village",
    "Provencal lavender valley", "Bavarian alpine hamlet",
    "Icelandic black-sand coast", "Scottish Highland glen",
    "Irish coastal meadow", "Dutch canal-side neighborhood",
    "Flemish market square", "Swiss lakeside village",
    "Austrian baroque town square", "Greek island harbor",
    "Portuguese tiled courtyard", "Anatolian stone village",
    "Cappadocian cave valley", "Levantine olive grove",
    "Persian garden pavilion", "Moroccan riad courtyard",
    "Saharan oasis settlement", "Nile-side reed village",
    "Nubian desert village", "Ethiopian highland plateau",
    "Swahili coastal courtyard", "Maasai savanna camp",
    "Cape Dutch farmstead", "Rajasthani palace courtyard",
    "Himalayan mountain village", "Bengali riverside village",
    "Sri Lankan tea-country hillside", "Balinese rice terrace",
    "Javanese volcanic village", "Thai floating market",
    "Vietnamese limestone bay", "Korean hanok village",
    "Chinese water town", "Mongolian steppe camp",
    "Patagonian mountain valley", "Andean terrace village",
    "Mexican colonial plaza", "Caribbean fishing village",
)

VALIDATION_REGIONAL_BACKGROUNDS = (
    "Norwegian fjord village", "Finnish lakeside cabin clearing",
    "Danish coastal dune settlement", "Breton fishing harbor",
    "Catalan stone village", "Croatian Adriatic port",
    "Georgian Caucasus mountain village", "Armenian tuff-stone square",
    "Jordanian desert canyon camp", "Tunisian medina courtyard",
    "Ghanaian coastal fort town", "Kenyan acacia plain",
    "Goan Portuguese quarter", "Bhutanese valley fortress town",
    "Burmese teak monastery village", "Taiwanese mountain tea settlement",
    "Filipino seaside barangay", "Canadian prairie homestead",
    "Brazilian sertao village", "Chilean Pacific cliff town",
)

TRAIN_ORDINARY_TARGETS: dict[str, tuple[str, ...]] = {
    "change_color": (
        "aubergine", "deep teal", "warm ivory", "charcoal gray", "coral pink",
        "mustard yellow", "forest green", "lavender purple", "burnt orange", "slate blue",
    ),
    "change_weather": (
        "gentle spring rain", "thin coastal fog", "brief hail shower", "overcast skies",
        "soft drifting snow", "humid summer haze", "clearing storm clouds",
        "dry autumn wind", "a light sunshower", "low evening mist",
    ),
    "add_effect": (
        "floating dandelion seeds", "a soft lens flare", "tiny soap bubbles",
        "scattered paper confetti", "faint dust motes", "a subtle rainbow prism",
        "glowing fireflies", "light film grain", "small drifting feathers",
        "gentle water reflections",
    ),
    "add_object": (
        "a folded blue blanket", "a plain ceramic bowl", "a wicker basket",
        "a small potted fern", "a wooden toy boat", "a yellow raincoat",
        "a canvas tote bag", "a red paper lantern", "a cork coaster",
    ),
    "replace_object": (
        "a smooth river stone", "a plain glass jar", "a wooden cube",
        "a woven storage box", "a brass desk bell", "a folded cotton towel",
        "a white porcelain plate", "a cork bulletin board", "a clay flowerpot",
    ),
}

VALIDATION_ORDINARY_TARGETS: dict[str, tuple[str, ...]] = {
    "change_color": ("peacock blue", "soft apricot", "graphite black"),
    "change_weather": ("patchy morning frost", "a passing coastal drizzle", "high thin clouds"),
    "add_effect": ("slowly falling maple leaves", "subtle window-light caustics"),
    "add_object": ("a plain linen cushion", "a small bamboo tray"),
    "replace_object": ("an unpainted wooden cylinder", "a simple enamel mug"),
}

# Validation-only web-search concepts.  None occurs in the released calibration
# catalog, which makes the positive validation concepts independent of replay.
VALIDATION_SEARCH_ENTITIES: dict[str, tuple[str, ...]] = {
    "brand_product": (
        "Polaroid SX-70 camera", "KitchenAid Artisan mixer", "Rimowa Original Cabin suitcase",
        "Smeg retro refrigerator", "Marshall Stanmore III speaker", "Fujifilm X100VI camera",
        "Bialetti Moka Express", "Fjallraven Kanken backpack",
    ),
    "ip_character": (
        "Kermit the Frog character", "Bugs Bunny character", "Woody from Toy Story",
        "Shrek character", "Asterix character", "Tintin character",
        "Pusheen character", "Miffy character",
    ),
    "landmark": (
        "Brooklyn Bridge", "Tower Bridge London", "Mont Saint-Michel",
        "Palace of Westminster", "Gateway Arch St Louis", "Guggenheim Museum Bilbao",
        "Acropolis of Athens", "Taipei 101 tower",
    ),
    "cultural_artifact": (
        "Terracotta warrior statue", "Sutton Hoo helmet", "Ming dynasty cloisonne vase",
        "Hopi kachina doll", "Kerala kathakali mask", "Congolese nkisi figure",
        "Maori koru carving", "Czech Bohemian crystal vase",
    ),
}


@dataclass(frozen=True)
class ParsedRow:
    row: dict[str, Any]
    prompt: str
    plan: dict[str, Any]
    video: str
    fingerprint: str
    contract_violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuiltRow:
    row: dict[str, Any]
    category: str
    concept_id: str
    template_id: str
    constraints: tuple[dict[str, Any], ...] = ()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _prompt_from_user(content: Any) -> str:
    if not isinstance(content, str) or content.count("<video>") != 1:
        raise ValueError("expected exactly one <video> token in user content")
    before, after = content.split("<video>", 1)
    if before.strip() or not after.strip():
        raise ValueError("user content must be '<video>' followed by a non-empty request")
    return after.strip()


def _contract_violations(plan: dict[str, Any]) -> tuple[str, ...]:
    violations: list[str] = []
    if isinstance(plan["image_search"], str) and plan["subtask"] not in SEARCHABLE_SUBTASKS:
        violations.append("image_search_not_allowed_for_subtask")
    if plan["subtask"] == "remove_object":
        if not isinstance(plan["mask"], str):
            violations.append("remove_object_missing_mask")
    elif plan["mask"] is not False:
        violations.append("mask_not_allowed_for_subtask")
    return tuple(violations)


def parse_rows(
    rows: Sequence[dict[str, Any]],
    label: str,
    *,
    allow_contract_violations: bool = False,
) -> list[ParsedRow]:
    parsed: list[ParsedRow] = []
    for index, row in enumerate(rows, 1):
        system = row.get("system")
        messages = row.get("messages")
        videos = row.get("videos")
        if not isinstance(system, str) or not system.strip():
            raise ValueError(f"{label} row {index}: missing system prompt")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"{label} row {index}: expected exactly two messages")
        if [message.get("role") for message in messages if isinstance(message, dict)] != [
            "user", "assistant"
        ]:
            raise ValueError(f"{label} row {index}: expected user/assistant roles")
        if not isinstance(videos, list) or len(videos) != 1 or not isinstance(videos[0], str):
            raise ValueError(f"{label} row {index}: expected exactly one video path")
        prompt = _prompt_from_user(messages[0].get("content"))
        try:
            plan = json.loads(messages[1].get("content", ""))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{label} row {index}: invalid assistant JSON") from exc
        if not valid_plan(plan):
            raise ValueError(f"{label} row {index}: invalid planner contract")
        violations = _contract_violations(plan)
        if violations and not allow_contract_violations:
            raise ValueError(
                f"{label} row {index}: current planner contract violation(s): "
                + ", ".join(violations)
            )
        canonical_plan = json.dumps(plan, ensure_ascii=False, sort_keys=True)
        fingerprint = _stable_hash(videos[0], normalize(prompt), canonical_plan)
        parsed.append(
            ParsedRow(
                row=row,
                prompt=prompt,
                plan=plan,
                video=videos[0],
                fingerprint=fingerprint,
                contract_violations=violations,
            )
        )
    return parsed


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {normalize(text)} "
    normalized_phrase = normalize(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_text


def _row_text(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _forbidden(row: dict[str, Any], forbidden_concepts: Sequence[str]) -> bool:
    text = _row_text(row)
    return any(_contains_phrase(text, phrase) for phrase in forbidden_concepts)


def _make_record(source: ParsedRow, request: str, plan: dict[str, Any]) -> dict[str, Any]:
    if set(plan) != PLAN_FIELDS or not valid_plan(plan) or _contract_violations(plan):
        raise ValueError(f"attempted to build an invalid plan: {plan}")
    return {
        "system": source.row["system"],
        "messages": [
            {"role": "user", "content": f"<video>\n{request}"},
            {
                "role": "assistant",
                "content": json.dumps(plan, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "videos": [source.video],
    }


def _source_cycle(rows: Sequence[ParsedRow], count: int, salt: str) -> list[ParsedRow]:
    by_video: dict[str, ParsedRow] = {}
    for row in rows:
        by_video.setdefault(row.video, row)
    ordered = sorted(by_video.values(), key=lambda item: _stable_hash(salt, item.video))
    if not ordered:
        raise ValueError("cannot assign synthetic rows without source videos")
    return [ordered[index % len(ordered)] for index in range(count)]


def _plan(refined: str, subtask: str, *, search: str | bool = False, mask: str | bool = False) -> dict[str, Any]:
    return {
        "refined_text_instruction": refined,
        "subtask": subtask,
        "image_search": search,
        "mask": mask,
    }


def _build_style_negatives(sources: Sequence[ParsedRow], *, validation: bool) -> list[BuiltRow]:
    concepts = VALIDATION_HISTORICAL_STYLES if validation else TRAIN_HISTORICAL_STYLES
    count = 32 if validation else 128
    assigned = _source_cycle(sources, count, "validation-style" if validation else "train-style")
    train_templates = (
        ("restyle the entire clip with the portrait lighting and brushwork associated with {concept}",
         "Render the entire video with portrait lighting and brushwork associated with {concept} while preserving every subject and motion."),
        ("give every frame the color rhythm and painted texture of {concept}",
         "Restyle every frame with the color rhythm and painted texture associated with {concept} while preserving composition and timing."),
        ("make this whole video feel like a gallery painting by {concept}",
         "Render the complete video as a gallery painting in the style associated with {concept}, preserving all scene content."),
        ("apply the visual language of {concept} across the full sequence", 
         "Apply the visual language associated with {concept} to the full sequence while preserving subjects, layout, and motion."),
    )
    validation_templates = (
        ("translate the full sequence into the characteristic painted aesthetic of {concept}",
         "Translate the full sequence into the characteristic painted aesthetic of {concept} while retaining all subjects and actions."),
        ("render every shot through the recognizable artistic vocabulary of {concept}",
         "Render every shot through the recognizable artistic vocabulary of {concept}, preserving content, framing, and movement."),
    )
    templates = validation_templates if validation else train_templates
    built = []
    for index in range(count):
        concept_index = index % len(concepts)
        round_index = index // len(concepts)
        template_index = (concept_index + round_index) % len(templates)
        request_template, refined_template = templates[template_index]
        concept = concepts[concept_index]
        request = request_template.format(concept=concept)
        refined = refined_template.format(concept=concept)
        built.append(
            BuiltRow(
                _make_record(assigned[index], request, _plan(refined, "global_style")),
                "target_negative" if validation else "named_style_negative",
                normalize(concept),
                f"{'val' if validation else 'train'}_style_{template_index}",
                ({"type": "style", "value": concept},),
            )
        )
    return built


def _build_background_negatives(sources: Sequence[ParsedRow], *, validation: bool) -> list[BuiltRow]:
    concepts = VALIDATION_REGIONAL_BACKGROUNDS if validation else TRAIN_REGIONAL_BACKGROUNDS
    count = 20 if validation else 80
    assigned = _source_cycle(sources, count, "validation-background" if validation else "train-background")
    train_templates = (
        ("move the scene to a generic {concept}", "Change the background to a generic {concept} while preserving the foreground subject and motion."),
        ("make the setting resemble an everyday {concept}", "Change the setting to an everyday {concept} while preserving all foreground content."),
        ("put the subject in a typical {concept}", "Place the subject in a typical {concept}, preserving the subject's appearance, pose, and movement."),
        ("change only the backdrop into a broad {concept} setting", "Change only the backdrop to a broadly representative {concept} setting while preserving the foreground."),
    )
    validation_templates = (
        ("recast the surroundings as a non-specific {concept}", "Recast the surroundings as a non-specific {concept} while keeping the foreground action unchanged."),
        ("use an ordinary {concept} as the new environment", "Use an ordinary {concept} as the new environment while preserving every foreground subject."),
    )
    templates = validation_templates if validation else train_templates
    built = []
    for index in range(count):
        concept_index = index % len(concepts)
        round_index = index // len(concepts)
        template_index = (concept_index + round_index) % len(templates)
        request_template, refined_template = templates[template_index]
        concept = concepts[concept_index]
        built.append(
            BuiltRow(
                _make_record(
                    assigned[index],
                    request_template.format(concept=concept),
                    _plan(refined_template.format(concept=concept), "change_background"),
                ),
                "target_negative" if validation else "regional_background_negative",
                normalize(concept),
                f"{'val' if validation else 'train'}_background_{template_index}",
                ({"type": "background", "value": concept},),
            )
        )
    return built


def _ordinary_text(subtask: str, target: str, *, validation: bool) -> tuple[str, str]:
    if subtask == "change_color":
        return (
            f"recolor only the main object {target}" if validation else f"make only the main object {target}",
            f"Change only the main object to {target} while preserving its material, the background, and all motion.",
        )
    if subtask == "change_weather":
        if validation:
            return (
                f"bring in {target} weather without veiling the activity",
                f"Introduce {target} weather. Keep the ongoing activity unobstructed from view.",
            )
        return (
            f"dial in {target} conditions, leaving the action unobscured",
            f"Establish {target} atmospheric conditions. Maintain an unobstructed view of the ongoing action.",
        )
    if subtask == "add_effect":
        return (
            f"introduce {target} around the subject" if validation else f"add {target} around the main subject",
            f"Add {target} around the main subject without obscuring or changing existing scene content.",
        )
    if subtask == "add_object":
        return (
            f"set {target} in the unused foreground space" if validation else f"put {target} beside the main subject",
            f"Add {target} beside the main subject while preserving all existing objects and motion.",
        )
    if subtask == "replace_object":
        return (
            f"swap the nearest plain object for {target}" if validation else f"replace the central everyday object with {target}",
            f"Replace the central everyday object with {target} while preserving its placement and the surrounding scene.",
        )
    raise ValueError(f"unsupported ordinary subtask: {subtask}")


def _build_ordinary_negatives(sources: Sequence[ParsedRow], *, validation: bool) -> list[BuiltRow]:
    catalog = VALIDATION_ORDINARY_TARGETS if validation else TRAIN_ORDINARY_TARGETS
    flattened = [(subtask, target) for subtask, targets in catalog.items() for target in targets]
    expected = 12 if validation else 48
    if len(flattened) != expected:
        raise AssertionError(f"ordinary catalog has {len(flattened)} rows, expected {expected}")
    assigned = _source_cycle(sources, expected, "validation-ordinary" if validation else "train-ordinary")
    built = []
    for index, (subtask, target) in enumerate(flattened):
        request, refined = _ordinary_text(subtask, target, validation=validation)
        built.append(
            BuiltRow(
                _make_record(assigned[index], request, _plan(refined, subtask)),
                "target_negative" if validation else "ordinary_negative",
                normalize(target),
                f"{'val' if validation else 'train'}_ordinary_{subtask}",
                ({"type": "target", "value": target},),
            )
        )
    return built


def _external_entity_group(entity: str) -> str | None:
    normalized = normalize(entity)
    for group, entities in EXTERNAL_ENTITY_GROUPS.items():
        if any(normalize(candidate) == normalized for candidate in entities):
            return group
    return None


def _ranked(rows: Iterable[ParsedRow], salt: str) -> list[ParsedRow]:
    return sorted(rows, key=lambda row: _stable_hash(salt, row.fingerprint))


def _select_exact(
    rows: Sequence[ParsedRow],
    count: int,
    *,
    salt: str,
    used_fingerprints: set[str],
    predicate: Callable[[ParsedRow], bool] = lambda _row: True,
    used_prompts: set[str] | None = None,
    preserve_order: bool = False,
) -> list[ParsedRow]:
    selected: list[ParsedRow] = []
    candidates = list(rows) if preserve_order else _ranked(rows, salt)
    for row in candidates:
        prompt_key = normalize(row.prompt)
        if row.fingerprint in used_fingerprints or not predicate(row):
            continue
        if used_prompts is not None and prompt_key in used_prompts:
            continue
        selected.append(row)
        used_fingerprints.add(row.fingerprint)
        if used_prompts is not None:
            used_prompts.add(prompt_key)
        if len(selected) == count:
            return selected
    raise ValueError(f"{salt}: only found {len(selected)} eligible rows; need {count}")


def _row_as_built(row: ParsedRow, category: str) -> BuiltRow:
    search = row.plan["image_search"]
    if isinstance(search, str):
        concept = normalize(search)
    else:
        concept = normalize(str(row.plan["refined_text_instruction"]))
    return BuiltRow(
        row=row.row,
        category=category,
        concept_id=concept,
        template_id=f"prompt_{_stable_hash(normalize(row.prompt))[:16]}",
    )


def _balanced_counts(total: int, keys: Sequence[str]) -> dict[str, int]:
    quotient, remainder = divmod(total, len(keys))
    return {key: quotient + int(index < remainder) for index, key in enumerate(keys)}


def _select_train_search_replay(
    rows: Sequence[ParsedRow],
    forbidden_concepts: Sequence[str],
    forbidden_prompts: set[str],
    used: set[str],
) -> list[BuiltRow]:
    selected: list[BuiltRow] = []
    used_prompts: set[str] = set()
    for group in EXTERNAL_ENTITY_GROUPS:
        for subtask in ("add_object", "replace_object"):
            candidates = [
                row
                for row in rows
                if not row.contract_violations
                and row.plan["subtask"] == subtask
                and isinstance(row.plan["image_search"], str)
                and _external_entity_group(str(row.plan["image_search"])) == group
                and not _forbidden(row.row, forbidden_concepts)
                and normalize(row.prompt) not in forbidden_prompts
            ]
            # Prefer distinct search concepts before taking another template for
            # the same concept, which keeps all eight matrix cells informative.
            candidates = sorted(
                candidates,
                key=lambda row: (
                    _stable_hash("search-concept", normalize(str(row.plan["image_search"]))),
                    _stable_hash("search-row", row.fingerprint),
                ),
            )
            distinct: list[ParsedRow] = []
            repeated: list[ParsedRow] = []
            seen_concepts: set[str] = set()
            for row in candidates:
                concept = normalize(str(row.plan["image_search"]))
                (distinct if concept not in seen_concepts else repeated).append(row)
                seen_concepts.add(concept)
            chosen = _select_exact(
                [*distinct, *repeated],
                32,
                salt=f"search-{group}-{subtask}",
                used_fingerprints=used,
                used_prompts=used_prompts,
                preserve_order=True,
            )
            selected.extend(_row_as_built(row, "true_search_replay") for row in chosen)
    if len(selected) != 256:
        raise AssertionError(f"search replay has {len(selected)} rows")
    return selected


def _select_train_route_replay(
    rows: Sequence[ParsedRow],
    forbidden_concepts: Sequence[str],
    forbidden_prompts: set[str],
    used: set[str],
) -> list[BuiltRow]:
    selected: list[BuiltRow] = []
    for subtask, quota in _balanced_counts(256, SUBTASK_ORDER).items():
        chosen = _select_exact(
            rows,
            quota,
            salt=f"route-{subtask}",
            used_fingerprints=used,
            predicate=lambda row, expected=subtask: (
                not row.contract_violations
                and row.plan["subtask"] == expected
                and row.plan["image_search"] is False
                and not _forbidden(row.row, forbidden_concepts)
                and normalize(row.prompt) not in forbidden_prompts
            ),
        )
        selected.extend(_row_as_built(row, "route_replay") for row in chosen)
    return selected


_ROUTING_CALIBRATION_PROMPTS = {normalize(item[0]) for item in ROUTING_TEMPLATES}


def _is_original_like(row: ParsedRow) -> bool:
    return (
        not row.contract_violations
        and not isinstance(row.plan["image_search"], str)
        and normalize(row.prompt) not in _ROUTING_CALIBRATION_PROMPTS
    )


def _select_diverse_original_replay(
    rows: Sequence[ParsedRow],
    count: int,
    *,
    category: str,
    salt: str,
    forbidden_concepts: Sequence[str],
    used: set[str],
    blocked_prompts: set[str] | None = None,
    blocked_concepts: set[str] | None = None,
    blocked_text_phrases: Sequence[str] = (),
) -> list[BuiltRow]:
    blocked_prompts = blocked_prompts or set()
    blocked_concepts = blocked_concepts or set()
    pools: dict[str, list[ParsedRow]] = defaultdict(list)
    for row in rows:
        concept = normalize(str(row.plan["refined_text_instruction"]))
        if (
            row.fingerprint not in used
            and _is_original_like(row)
            and not _forbidden(row.row, forbidden_concepts)
            and normalize(row.prompt) not in blocked_prompts
            and concept not in blocked_concepts
            and not any(
                _contains_phrase(_row_text(row.row), value) for value in blocked_text_phrases
            )
        ):
            pools[str(row.plan["subtask"])].append(row)
    for subtask in pools:
        pools[subtask] = _ranked(pools[subtask], f"{salt}-{subtask}")

    selected: list[ParsedRow] = []
    seen_videos: set[str] = set()
    # Round-robin over routes, first preferring unseen source videos.
    for prefer_new_video in (True, False):
        progress = True
        while len(selected) < count and progress:
            progress = False
            for subtask in SUBTASK_ORDER:
                pool = pools.get(subtask, [])
                match_index = next(
                    (
                        index
                        for index, row in enumerate(pool)
                        if not prefer_new_video or row.video not in seen_videos
                    ),
                    None,
                )
                if match_index is None:
                    continue
                row = pool.pop(match_index)
                selected.append(row)
                used.add(row.fingerprint)
                seen_videos.add(row.video)
                progress = True
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise ValueError(f"{salt}: only found {len(selected)} diverse original rows; need {count}")
    return [_row_as_built(row, category) for row in selected]


def _build_validation_search(sources: Sequence[ParsedRow]) -> list[BuiltRow]:
    specs = [
        (group, entity, subtask)
        for group, entities in VALIDATION_SEARCH_ENTITIES.items()
        for entity in entities
        for subtask in ("add_object", "replace_object")
    ]
    assigned = _source_cycle(sources, len(specs), "validation-search")
    built = []
    for index, (group, entity, subtask) in enumerate(specs):
        if subtask == "add_object":
            request = f"introduce {entity} in the clear area beside the subject"
            refined = f"Add {entity} in the clear area beside the subject with scene-consistent scale and lighting."
        else:
            request = f"transform the foremost object so it is unmistakably {entity}"
            refined = f"Replace the foremost object with {entity} while preserving placement, surroundings, and motion."
        built.append(
            BuiltRow(
                _make_record(assigned[index], request, _plan(refined, subtask, search=entity)),
                "true_search",
                normalize(entity),
                f"val_search_{group}_{subtask}",
                ({"type": "identity", "value": entity},),
            )
        )
    if len(built) != 64:
        raise AssertionError(f"validation search catalog has {len(built)} rows")
    return built


VALIDATION_ROUTE_VALUES: dict[str, tuple[str, ...]] = {
    "global_style": (
        "stained-glass mosaic", "layered cut-paper collage", "monochrome woodcut",
        "pastel chalk illustration", "enameled folk-art panel", "cyanotype print",
    ),
    "remove_object": (
        "striped cushion near the doorway", "loose cable below the table",
        "small carton at the frame edge", "empty bottle behind the subject",
        "fallen branch in the foreground", "paper notice on the wall",
    ),
    "add_object": (
        "a folded green scarf", "a small terracotta saucer", "a plain wooden whistle",
        "a blue fabric pouch", "a short beeswax candle", "a woven reed mat",
    ),
    "replace_object": (
        "a frosted glass block", "a plain copper cup", "a carved wooden oval",
        "a folded wool cap", "a matte ceramic cylinder", "a small leather notebook",
    ),
    "change_background": (
        "a quiet salt-marsh boardwalk", "an ordinary brick workshop",
        "a broad grassy riverbank", "a simple greenhouse interior",
        "a modest stone train platform", "an open wheat-field path",
    ),
    "change_color": (
        "cerulean blue", "dusty rose", "olive green", "pearl white", "copper brown", "plum purple",
    ),
    "change_weather": (
        "a calm after-rain clearing", "light sleet", "a bright cloudless noon",
        "a cool dawn haze", "a mild evening shower", "wind-blown high clouds",
    ),
    "add_effect": (
        "slow floating seed fluff", "a restrained edge glow", "small reflected light flecks",
        "fine airborne pollen", "a faint double-exposure trail", "soft rippling highlights",
    ),
    "customization": (
        "a hand-sewn felt keepsake", "a painted tin collectible", "a knitted miniature mascot",
        "a carved soap figurine", "a quilted fabric avatar", "a glazed clay souvenir",
    ),
    "combined_tasks": (
        "indigo with a quiet marsh backdrop", "cream with a red-brick arcade backdrop",
        "bronze with a misty pasture backdrop", "jade green with a dry canyon backdrop",
        "silver with a dim warehouse backdrop", "scarlet with a calm lakeshore backdrop",
    ),
    "camera_edit": (
        "a slow upward crane", "a gentle counterclockwise orbit", "a short backward dolly",
        "a steady diagonal pan", "a gradual rack-focus shift", "a subtle handheld drift",
    ),
}


def _validation_route_text(subtask: str, value: str) -> tuple[str, str, str | bool]:
    if subtask == "global_style":
        return f"reinterpret every frame as {value} artwork", f"Restyle the complete video as {value} artwork while retaining composition and motion.", False
    if subtask == "remove_object":
        return f"cleanly erase the {value}", f"Remove the {value} and reconstruct the revealed area consistently across frames.", value
    if subtask == "add_object":
        return f"place {value} near the lower-right edge", f"Add {value} near the lower-right edge without covering any existing content.", False
    if subtask == "replace_object":
        return f"exchange the closest prop for {value}", f"Replace the closest prop with {value} while retaining its location and motion.", False
    if subtask == "change_background":
        return f"relocate the scene to {value}", f"Change the background to {value} while leaving all foreground subjects unchanged.", False
    if subtask == "change_color":
        return f"recolor the central item {value} only", f"Change only the central item to {value} while preserving every other color.", False
    if subtask == "change_weather":
        return f"set the weather to {value}", f"Change the weather to {value} while preserving visibility and the original action.", False
    if subtask == "add_effect":
        return f"layer in {value} around the action", f"Add {value} around the action without obscuring subjects or changing the scene structure.", False
    if subtask == "customization":
        return f"personalize the main subject as {value}", f"Customize the main subject as {value} while retaining its recognizable features and motion.", False
    if subtask == "combined_tasks":
        color, background = value.split(" with ", 1)
        return f"make the main object {color} and use {background}", f"Change the main object to {color} and replace the background with {background}.", False
    if subtask == "camera_edit":
        return f"apply {value} while the action continues", f"Apply {value} while preserving all scene content and the original action.", False
    raise ValueError(subtask)


def _build_validation_routes(sources: Sequence[ParsedRow]) -> list[BuiltRow]:
    quotas = _balanced_counts(64, SUBTASK_ORDER)
    specs = [
        (subtask, value)
        for subtask in SUBTASK_ORDER
        for value in VALIDATION_ROUTE_VALUES[subtask][: quotas[subtask]]
    ]
    assigned = _source_cycle(sources, len(specs), "validation-routes")
    built = []
    for index, (subtask, value) in enumerate(specs):
        request, refined, mask = _validation_route_text(subtask, value)
        if subtask == "combined_tasks":
            color, background = value.split(" with ", 1)
            constraints = (
                {"type": "color", "value": color},
                {"type": "background", "value": background},
            )
        else:
            constraints = ({"type": "target", "value": value},)
        built.append(
            BuiltRow(
                _make_record(assigned[index], request, _plan(refined, subtask, mask=mask)),
                "routing_constraints",
                normalize(value),
                f"val_route_{subtask}",
                constraints,
            )
        )
    if len(built) != 64:
        raise AssertionError(f"validation route catalog has {len(built)} rows")
    return built


def _text_ngrams(text: str, n: int) -> set[str]:
    if n < 1:
        raise ValueError(f"n-gram size must be positive, got {n}")
    tokens = normalize(text).split()
    return {
        " ".join(tokens[index : index + n])
        for index in range(len(tokens) - n + 1)
    }


def paired_prompt_target_ngram_overlap(
    generated_prompt: str,
    generated_target: str,
    forbidden_prompt: str,
    forbidden_target: str,
    *,
    n: int = DEFAULT_FORBIDDEN_NGRAM_N,
) -> dict[str, list[str]]:
    """Return aligned prompt/target overlaps for one generated/gate pair.

    A near-duplicate template is actionable only when both its input prompt and
    its output target overlap the same forbidden case.  Requiring the paired
    signal avoids rejecting harmless shared planner boilerplate such as
    ``change the background to`` when the user-facing templates are unrelated.
    """

    return {
        "prompt_ngrams": sorted(
            _text_ngrams(generated_prompt, n) & _text_ngrams(forbidden_prompt, n)
        ),
        "target_ngrams": sorted(
            _text_ngrams(generated_target, n) & _text_ngrams(forbidden_target, n)
        ),
    }


def _forbidden_synthetic_ngram_audit(
    generated_by_split: Sequence[tuple[str, Sequence[BuiltRow]]],
    forbidden_case_rows: Sequence[dict[str, Any]],
    *,
    n: int,
) -> dict[str, Any]:
    """Audit generated synthetic rows only; inherited replay is excluded."""

    if n < 1:
        raise ValueError(f"forbidden n-gram size must be positive, got {n}")

    forbidden: list[dict[str, Any]] = []
    for index, row in enumerate(forbidden_case_rows, 1):
        prompt = row.get("prompt")
        gold_plan = row.get("gold_plan")
        target = gold_plan.get("refined_text_instruction") if isinstance(gold_plan, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"forbidden case {index}: missing raw prompt")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"forbidden case {index}: missing gold refined instruction")
        forbidden.append(
            {
                "index": index,
                "bench_id": str(row.get("bench_id", f"forbidden_{index:04d}")),
                "prompt": prompt,
                "target": target,
                "prompt_ngrams": _text_ngrams(prompt, n),
                "target_ngrams": _text_ngrams(target, n),
            }
        )

    hits: list[dict[str, Any]] = []
    generated_count = 0
    for split, items in generated_by_split:
        for generated_index, item in enumerate(items, 1):
            generated_count += 1
            parsed = parse_rows([item.row], f"{split} synthetic n-gram audit")[0]
            generated_target = str(parsed.plan["refined_text_instruction"])
            generated_prompt_ngrams = _text_ngrams(parsed.prompt, n)
            generated_target_ngrams = _text_ngrams(generated_target, n)
            for gate in forbidden:
                overlap = {
                    "prompt_ngrams": sorted(
                        generated_prompt_ngrams & gate["prompt_ngrams"]
                    ),
                    "target_ngrams": sorted(
                        generated_target_ngrams & gate["target_ngrams"]
                    ),
                }
                if not overlap["prompt_ngrams"] or not overlap["target_ngrams"]:
                    continue
                hits.append(
                    {
                        "generated_split": split,
                        "generated_index": generated_index,
                        "generated_category": item.category,
                        "generated_concept_id": item.concept_id,
                        "generated_template_id": item.template_id,
                        "forbidden_index": gate["index"],
                        "forbidden_bench_id": gate["bench_id"],
                        **overlap,
                        "generated_prompt": parsed.prompt,
                        "generated_target": generated_target,
                        "forbidden_prompt": gate["prompt"],
                        "forbidden_target": gate["target"],
                    }
                )

    hits.sort(
        key=lambda hit: (
            hit["generated_split"],
            hit["generated_index"],
            hit["forbidden_index"],
        )
    )
    return {
        "n": n,
        "generated_rows_checked": generated_count,
        "forbidden_cases_checked": len(forbidden),
        "hit_count": len(hits),
        "hits": hits,
    }


def _forbidden_from_cases(rows: Sequence[dict[str, Any]]) -> tuple[set[str], set[str]]:
    prompts: set[str] = set()
    concepts = set(DEFAULT_FORBIDDEN_CONCEPTS)
    for row in rows:
        prompt = row.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            prompts.add(normalize(prompt))
        plan = row.get("gold_plan")
        if isinstance(plan, dict) and isinstance(plan.get("image_search"), str):
            concepts.add(str(plan["image_search"]))
    return prompts, concepts


def _to_validation_files(rows: Sequence[BuiltRow]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for index, item in enumerate(rows, 1):
        parsed = parse_rows([item.row], f"validation output {index}")[0]
        bench_id = f"w2_refresh_{index:04d}"
        axis = "search" if item.category in {"target_negative", "true_search"} else (
            "routing" if item.category == "routing_constraints" else "replay"
        )
        case = {
            "bench_id": bench_id,
            "video_path": parsed.video,
            "prompt": parsed.prompt,
            "edit_type": parsed.plan["subtask"],
            "axis": axis,
        }
        gold_row = {
            **case,
            "gold_plan": parsed.plan,
            "constraints": [dict(value) for value in item.constraints],
            "source_entities": [] if isinstance(parsed.plan["image_search"], str) else ["existing source video"],
        }
        if isinstance(parsed.plan["image_search"], str):
            gold_row["search_query_aliases"] = [parsed.plan["image_search"]]
        cases.append(case)
        gold.append(gold_row)
    return cases, gold


def _distribution(items: Sequence[BuiltRow], field: str) -> dict[str, int]:
    values: Counter[str] = Counter()
    for item in items:
        parsed = parse_rows([item.row], "audit")[0]
        if field == "category":
            values[item.category] += 1
        elif field == "subtask":
            values[str(parsed.plan["subtask"])] += 1
        elif field == "search":
            values["triggered" if isinstance(parsed.plan["image_search"], str) else "not_triggered"] += 1
        elif field == "mask":
            values["triggered" if isinstance(parsed.plan["mask"], str) else "not_triggered"] += 1
        else:
            raise ValueError(field)
    return dict(sorted(values.items()))


def _assert_expected_counts(train: Sequence[BuiltRow], validation: Sequence[BuiltRow]) -> None:
    if len(train) != TRAIN_SIZE or len(validation) != VALIDATION_SIZE:
        raise AssertionError(f"unexpected refresh sizes: train={len(train)}, validation={len(validation)}")
    if Counter(item.category for item in train) != Counter(TRAIN_CATEGORY_COUNTS):
        raise AssertionError(f"unexpected train category counts: {Counter(item.category for item in train)}")
    if Counter(item.category for item in validation) != Counter(VALIDATION_CATEGORY_COUNTS):
        raise AssertionError(
            f"unexpected validation category counts: {Counter(item.category for item in validation)}"
        )


def _contract_exclusion_audit(rows: Sequence[ParsedRow]) -> dict[str, Any]:
    excluded = [row for row in rows if row.contract_violations]
    by_violation = Counter(
        violation for row in excluded for violation in row.contract_violations
    )
    by_subtask = Counter(str(row.plan["subtask"]) for row in excluded)
    by_violation_and_subtask: dict[str, Counter[str]] = defaultdict(Counter)
    for row in excluded:
        for violation in row.contract_violations:
            by_violation_and_subtask[violation][str(row.plan["subtask"])] += 1
    return {
        "rows": len(excluded),
        "replay_eligible_rows": len(rows) - len(excluded),
        "by_violation_type": dict(sorted(by_violation.items())),
        "by_subtask": dict(sorted(by_subtask.items())),
        "by_violation_and_subtask": {
            violation: dict(sorted(counts.items()))
            for violation, counts in sorted(by_violation_and_subtask.items())
        },
    }


def build_refresh(
    train_rows: Sequence[dict[str, Any]],
    eval_rows: Sequence[dict[str, Any]],
    *,
    forbidden_case_rows: Sequence[dict[str, Any]] | None = None,
    forbidden_ngram_n: int = DEFAULT_FORBIDDEN_NGRAM_N,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build refresh train/validation rows, validation cases/gold, and an audit summary."""
    if forbidden_case_rows is None:
        if not DEFAULT_FORBIDDEN_CASE_PATH.is_file():
            raise FileNotFoundError(
                "default forbidden-case file is unavailable; pass forbidden_case_rows explicitly: "
                f"{DEFAULT_FORBIDDEN_CASE_PATH}"
            )
        forbidden_case_rows = load_jsonl(DEFAULT_FORBIDDEN_CASE_PATH)
    # Historical v2 labels are parsed without rewriting.  Rows that violate
    # the current type1 contract remain available only as video/system sources
    # for newly generated examples and are excluded from every replay pool.
    train_source = parse_rows(
        train_rows, "base train", allow_contract_violations=True
    )
    eval_source = parse_rows(eval_rows, "base eval", allow_contract_violations=True)
    input_contract_exclusions = {
        "train": _contract_exclusion_audit(train_source),
        "eval": _contract_exclusion_audit(eval_source),
    }
    train_videos = {row.video for row in train_source}
    eval_videos = {row.video for row in eval_source}
    input_video_overlap = train_videos & eval_videos
    if input_video_overlap:
        raise ValueError(f"base train/eval video overlap: {len(input_video_overlap)}")

    forbidden_prompts, forbidden_concept_set = _forbidden_from_cases(forbidden_case_rows)
    forbidden_concepts = tuple(sorted(forbidden_concept_set, key=normalize))

    train_synthetic = [
        *_build_style_negatives(train_source, validation=False),
        *_build_background_negatives(train_source, validation=False),
        *_build_ordinary_negatives(train_source, validation=False),
    ]
    validation_synthetic = [
        *_build_style_negatives(eval_source, validation=True),
        *_build_background_negatives(eval_source, validation=True),
        *_build_ordinary_negatives(eval_source, validation=True),
    ]
    validation_search = _build_validation_search(eval_source)
    validation_routes = _build_validation_routes(eval_source)
    synthetic_ngram_audit = _forbidden_synthetic_ngram_audit(
        (
            ("train", train_synthetic),
            (
                "validation",
                [*validation_synthetic, *validation_search, *validation_routes],
            ),
        ),
        forbidden_case_rows,
        n=forbidden_ngram_n,
    )
    if synthetic_ngram_audit["hit_count"]:
        raise AssertionError(
            "forbidden synthetic prompt/target n-gram overlap: "
            + json.dumps(
                synthetic_ngram_audit["hits"][:5],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    validation_seed = [*validation_synthetic, *validation_search, *validation_routes]
    validation_seed_concepts = {item.concept_id for item in validation_seed}
    replay_exclusions = (*forbidden_concepts, *sorted(validation_seed_concepts))

    used: set[str] = set()
    train_search = _select_train_search_replay(
        train_source, replay_exclusions, forbidden_prompts, used
    )
    train_routes = _select_train_route_replay(
        train_source, replay_exclusions, forbidden_prompts, used
    )
    train_original = _select_diverse_original_replay(
        train_source,
        256,
        category="original_replay",
        salt="train-original",
        forbidden_concepts=replay_exclusions,
        used=used,
        blocked_prompts=forbidden_prompts,
    )
    train = [*train_synthetic, *train_search, *train_routes, *train_original]

    train_prompt_set = {normalize(parse_rows([item.row], "train output")[0].prompt) for item in train}
    train_concept_set = {item.concept_id for item in train}
    eval_used: set[str] = set()
    validation_original = _select_diverse_original_replay(
        eval_source,
        64,
        category="original_eval_replay",
        salt="validation-original",
        forbidden_concepts=forbidden_concepts,
        used=eval_used,
        blocked_prompts=train_prompt_set | forbidden_prompts,
        blocked_concepts=train_concept_set | validation_seed_concepts,
        blocked_text_phrases=tuple(sorted(train_concept_set)),
    )
    validation = [*validation_seed, *validation_original]
    _assert_expected_counts(train, validation)

    train_output = [item.row for item in train]
    validation_output = [item.row for item in validation]
    cases, gold = _to_validation_files(validation)

    validation_prompt_set = {
        normalize(parse_rows([item.row], "validation output")[0].prompt) for item in validation
    }
    validation_concept_set = {item.concept_id for item in validation}
    train_template_set = {item.template_id for item in train}
    validation_template_set = {item.template_id for item in validation}
    output_train_videos = {str(row["videos"][0]) for row in train_output}
    output_validation_videos = {str(row["videos"][0]) for row in validation_output}

    prompt_overlap = train_prompt_set & validation_prompt_set
    concept_overlap = train_concept_set & validation_concept_set
    template_overlap = train_template_set & validation_template_set
    forbidden_prompt_hits = (train_prompt_set | validation_prompt_set) & forbidden_prompts
    forbidden_phrase_hits = [
        (split, index)
        for split, rows in (("train", train_output), ("validation", validation_output), ("cases", cases), ("gold", gold))
        for index, row in enumerate(rows, 1)
        if _forbidden(row, forbidden_concepts)
    ]
    output_video_overlap = output_train_videos & output_validation_videos
    train_text = _row_text({"rows": train_output})
    validation_text = _row_text({"rows": validation_output})
    cross_concept_phrase_hits = {
        "validation_concepts_in_train": sum(
            _contains_phrase(train_text, concept) for concept in validation_concept_set
        ),
        "train_concepts_in_validation": sum(
            _contains_phrase(validation_text, concept) for concept in train_concept_set
        ),
    }
    if prompt_overlap:
        raise AssertionError(f"normalized train/validation prompt overlap: {len(prompt_overlap)}")
    if concept_overlap:
        raise AssertionError(f"train/validation concept overlap: {len(concept_overlap)}")
    if any(cross_concept_phrase_hits.values()):
        raise AssertionError(f"cross-split concept phrase hits: {cross_concept_phrase_hits}")
    if template_overlap:
        raise AssertionError(f"train/validation template overlap: {len(template_overlap)}")
    if forbidden_prompt_hits:
        raise AssertionError(f"forbidden exact prompt overlap: {len(forbidden_prompt_hits)}")
    if forbidden_phrase_hits:
        raise AssertionError(f"forbidden phrase hits in output: {forbidden_phrase_hits[:5]}")
    if output_video_overlap:
        raise AssertionError(f"refresh train/validation video overlap: {len(output_video_overlap)}")

    search_matrix = Counter()
    search_unique_concepts: dict[str, set[str]] = defaultdict(set)
    for item in train_search:
        parsed = parse_rows([item.row], "search matrix")[0]
        matrix_key = f"{_external_entity_group(str(parsed.plan['image_search']))}:{parsed.plan['subtask']}"
        search_matrix[matrix_key] += 1
        search_unique_concepts[matrix_key].add(normalize(str(parsed.plan["image_search"])))

    validation_search_matrix = Counter()
    for item in validation_search:
        parsed = parse_rows([item.row], "validation search matrix")[0]
        entity = str(parsed.plan["image_search"])
        group = next(
            group
            for group, entities in VALIDATION_SEARCH_ENTITIES.items()
            if entity in entities
        )
        validation_search_matrix[f"{group}:{parsed.plan['subtask']}"] += 1

    subtask_by_category: dict[str, dict[str, int]] = {}
    for category in sorted({item.category for item in [*train, *validation]}):
        matching = [item for item in [*train, *validation] if item.category == category]
        subtask_by_category[category] = _distribution(matching, "subtask")

    digest = hashlib.sha256(
        "\n".join(sorted(normalize(value) for value in forbidden_concepts)).encode("utf-8")
    ).hexdigest()
    summary = {
        "recipe_version": RECIPE_VERSION,
        "input": {
            "train_rows": len(train_source),
            "eval_rows": len(eval_source),
            "train_videos": len(train_videos),
            "eval_videos": len(eval_videos),
            "video_overlap": 0,
        },
        "input_contract_exclusions": input_contract_exclusions,
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "validation_cases": len(cases),
            "validation_gold": len(gold),
            "train_by_category": _distribution(train, "category"),
            "validation_by_category": _distribution(validation, "category"),
        },
        "train_search_replay_matrix": dict(sorted(search_matrix.items())),
        "train_search_unique_concepts_by_matrix": {
            key: len(values) for key, values in sorted(search_unique_concepts.items())
        },
        "validation_search_matrix": dict(sorted(validation_search_matrix.items())),
        "search_distribution": {
            "train": _distribution(train, "search"),
            "validation": _distribution(validation, "search"),
        },
        "mask_distribution": {
            "train": _distribution(train, "mask"),
            "validation": _distribution(validation, "mask"),
            "by_category": {
                category: _distribution(
                    [item for item in [*train, *validation] if item.category == category],
                    "mask",
                )
                for category in sorted({item.category for item in [*train, *validation]})
            },
        },
        "subtask_distribution": {
            "train": _distribution(train, "subtask"),
            "validation": _distribution(validation, "subtask"),
            "by_category": subtask_by_category,
        },
        "isolation": {
            "train_unique_videos": len(output_train_videos),
            "validation_unique_videos": len(output_validation_videos),
            "video_overlap": 0,
            "train_unique_prompts": len(train_prompt_set),
            "validation_unique_prompts": len(validation_prompt_set),
            "normalized_prompt_overlap": 0,
            "train_unique_concepts": len(train_concept_set),
            "validation_unique_concepts": len(validation_concept_set),
            "concept_overlap": 0,
            "cross_split_concept_phrase_hits": cross_concept_phrase_hits,
            "train_unique_templates": len(train_template_set),
            "validation_unique_templates": len(validation_template_set),
            "template_overlap": 0,
        },
        "forbidden_audit": {
            "concept_count": len(forbidden_concepts),
            "concept_digest_sha256": digest,
            "exact_prompt_count": len(forbidden_prompts),
            "exact_prompt_overlap": 0,
            "phrase_hit_count": 0,
            "synthetic_prompt_target_ngram_overlap": synthetic_ngram_audit,
        },
    }
    return train_output, validation_output, cases, gold, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-eval", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--validation-out", type=Path, required=True)
    parser.add_argument("--cases-out", type=Path, required=True)
    parser.add_argument("--gold-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument(
        "--forbidden-cases",
        type=Path,
        default=DEFAULT_FORBIDDEN_CASE_PATH,
        help="Optional gate/case JSONL whose exact prompts and search entities must be excluded.",
    )
    parser.add_argument(
        "--forbidden-ngram-n",
        type=int,
        default=DEFAULT_FORBIDDEN_NGRAM_N,
        help=(
            "Reject generated synthetic rows whose prompt and target both share "
            "an aligned n-gram with one forbidden case (default: 4)."
        ),
    )
    args = parser.parse_args()

    forbidden_rows = load_jsonl(args.forbidden_cases)
    train, validation, cases, gold, summary = build_refresh(
        load_jsonl(args.base_train),
        load_jsonl(args.base_eval),
        forbidden_case_rows=forbidden_rows,
        forbidden_ngram_n=args.forbidden_ngram_n,
    )
    write_jsonl(args.train_out, train)
    write_jsonl(args.validation_out, validation)
    write_jsonl(args.cases_out, cases)
    write_jsonl(args.gold_out, gold)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

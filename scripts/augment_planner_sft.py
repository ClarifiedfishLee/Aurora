"""Add diverse search and routing calibration examples to planner SFT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# Concrete, image-searchable entities grouped for coverage audits.
EXTERNAL_ENTITY_GROUPS: dict[str, list[str]] = {
    "brand_product": [
        "Adidas Samba OG sneaker",
        "Apple Vision Pro headset",
        "Bang & Olufsen Beosound A9 speaker",
        "Barbour Bedale wax jacket",
        "Birkenstock Arizona sandal",
        "Bose QuietComfort Ultra headphones",
        "Burberry check trench coat",
        "Chanel No. 5 perfume bottle",
        "Coca-Cola contour glass bottle",
        "Converse Chuck Taylor All Star high-top",
        "DJI Mavic 3 Pro drone",
        "Dyson Airwrap Multi-Styler",
        "Ferrari F40 sports car",
        "Fender Stratocaster electric guitar",
        "Gibson Les Paul Standard guitar",
        "GoPro HERO12 Black camera",
        "Gucci Jackie 1961 handbag",
        "Hermes Birkin 30 handbag",
        "IKEA POANG armchair",
        "Instax Mini 12 camera",
        "Leica M11 camera",
        "LEGO Millennium Falcon set",
        "Louis Vuitton Neverfull MM tote",
        "McDonald's Happy Meal box",
        "Microsoft Xbox Series X console",
        "Nintendo Switch OLED console",
        "Nike Air Jordan 1 Chicago sneaker",
        "Nikon Z8 camera",
        "Patagonia Retro-X fleece jacket",
        "Porsche 911 GT3 RS",
        "Ray-Ban Wayfarer sunglasses",
        "Rolex Submariner watch",
        "Samsung Galaxy Z Flip6 phone",
        "Sony PlayStation 5 console",
        "Stanley Quencher H2.0 tumbler",
        "Starbucks holiday cup",
        "Tesla Cybertruck",
        "Vespa Primavera scooter",
        "Yamaha C7 grand piano",
        "Zippo classic chrome lighter",
    ],
    "ip_character": [
        "Astro Boy character",
        "Batman character",
        "Baymax character",
        "Buzz Lightyear character",
        "Captain America shield",
        "Darth Vader character",
        "Disney Stitch character",
        "Doraemon character",
        "Elsa from Frozen",
        "Eevee Pokemon character",
        "Gandalf character",
        "Godzilla character",
        "Grogu character",
        "Harry Potter character",
        "Hello Kitty character",
        "Iron Man character",
        "Kirby character",
        "Lightning McQueen character",
        "Link from The Legend of Zelda",
        "Mario character",
        "Mickey Mouse character",
        "Minion character",
        "Monkey D. Luffy character",
        "Naruto Uzumaki character",
        "Optimus Prime character",
        "Paddington Bear character",
        "Pikachu character",
        "R2-D2 droid",
        "Sailor Moon character",
        "Scooby-Doo character",
        "Snoopy character",
        "Sonic the Hedgehog character",
        "Spider-Man character",
        "SpongeBob SquarePants character",
        "Superman character",
        "Thomas the Tank Engine character",
        "Totoro character",
        "WALL-E robot",
        "Winnie the Pooh character",
        "Wonder Woman character",
    ],
    "landmark": [
        "Angkor Wat temple",
        "Arc de Triomphe monument",
        "Big Ben and Elizabeth Tower",
        "Brandenburg Gate",
        "Burj Khalifa skyscraper",
        "Christ the Redeemer statue",
        "CN Tower",
        "Colosseum in Rome",
        "Eiffel Tower",
        "Empire State Building",
        "Forbidden City Meridian Gate",
        "Golden Gate Bridge",
        "Great Pyramid of Giza",
        "Great Sphinx of Giza",
        "Hagia Sophia",
        "Hollywood Sign",
        "Japanese cherry blossom tree",
        "Leaning Tower of Pisa",
        "Louvre Pyramid",
        "Machu Picchu citadel",
        "Marina Bay Sands",
        "Matterhorn mountain",
        "Moai statues of Easter Island",
        "Mount Fuji",
        "Mount Rushmore",
        "Neuschwanstein Castle",
        "Petronas Twin Towers",
        "Sagrada Familia basilica",
        "Seattle Space Needle",
        "St. Basil's Cathedral",
        "Statue of Liberty",
        "Stonehenge",
        "Sydney Opera House",
        "Taj Mahal",
        "Temple of Heaven Beijing",
        "The Shard London",
        "Tokyo Tower",
        "Trevi Fountain",
        "Uluru sandstone monolith",
        "White House Washington DC",
    ],
    "cultural_artifact": [
        "Aztec Sun Stone",
        "Balinese Barong mask",
        "Benin Bronze plaque",
        "Chinese blue-and-white porcelain vase",
        "Chinese dragon dance head",
        "Chinese guzheng zither",
        "Daruma doll",
        "Dutch Delftware tulip vase",
        "Egyptian canopic jar",
        "Faberge egg",
        "Greek red-figure amphora",
        "Hokusai The Great Wave off Kanagawa print",
        "Indian sitar",
        "Indonesian wayang kulit puppet",
        "Irish Celtic harp",
        "Japanese kabuto samurai helmet",
        "Japanese kokeshi doll",
        "Japanese maneki-neko cat figurine",
        "Korean moon jar",
        "Maasai beaded necklace",
        "Matryoshka nesting doll",
        "Mexican papel picado banner",
        "Moroccan brass lantern",
        "Navajo woven rug",
        "New Zealand Maori hei-tiki pendant",
        "Norwegian rosemaling wooden box",
        "Ottoman Iznik ceramic plate",
        "Persian carpet",
        "Polynesian tapa cloth",
        "Roman gladiator helmet",
        "Rosetta Stone",
        "Scottish Great Highland bagpipes",
        "Swiss cuckoo clock",
        "Thai khon mask",
        "Tibetan singing bowl",
        "Tutankhamun funerary mask",
        "Ukrainian pysanka egg",
        "Venetian carnival mask",
        "Venus de Milo statue",
        "Vietnamese Dong Son bronze drum",
    ],
}

EXTERNAL_ENTITY_SPECS = [
    (category, entity)
    for category, entities in EXTERNAL_ENTITY_GROUPS.items()
    for entity in entities
]
EXTERNAL_ENTITIES = [entity for _, entity in EXTERNAL_ENTITY_SPECS]


# (stable id, raw request template, refined instruction template, subtask).
# Article-free templates remain fluent when filled with proper names.
SEARCH_TEMPLATES_BY_CATEGORY: dict[str, list[tuple[str, str, str, str]]] = {
    "brand_product": [
        (
            "product_add_right",
            "put {entity} on the right side of the main subject",
            "Add {entity} to the right of the main subject while preserving all existing scene content.",
            "add_object",
        ),
        (
            "product_add_left_foreground",
            "add {entity} in the open foreground area on the left",
            "Add {entity} in the open foreground area on the left while preserving the main subject.",
            "add_object",
        ),
        (
            "product_add_table",
            "place {entity} on the nearest clear surface",
            "Add {entity} on the nearest clear surface with scale and lighting consistent with the scene.",
            "add_object",
        ),
        (
            "product_add_beside",
            "set {entity} beside the main visible object",
            "Add {entity} beside the main visible object without removing or covering existing content.",
            "add_object",
        ),
        (
            "product_replace_center",
            "replace the main object in the center with {entity}",
            "Replace the main object in the center with {entity} while preserving the background, lighting, and motion.",
            "replace_object",
        ),
        (
            "product_replace_foreground",
            "swap the foreground object for {entity}",
            "Replace the foreground object with {entity} while preserving its position and the surrounding scene.",
            "replace_object",
        ),
        (
            "product_replace_held",
            "make the item being held look exactly like {entity}",
            "Replace the held item with {entity} while preserving the hand, pose, background, and motion.",
            "replace_object",
        ),
        (
            "product_replace_nearest",
            "turn the object closest to the camera into {entity}",
            "Replace the object closest to the camera with {entity} while preserving the rest of the video.",
            "replace_object",
        ),
    ],
    "ip_character": [
        (
            "character_add_right",
            "add {entity} standing to the right of the main subject",
            "Add {entity} standing to the right of the main subject while preserving all existing people and objects.",
            "add_object",
        ),
        (
            "character_add_left",
            "put {entity} in the empty space on the left",
            "Add {entity} in the empty space on the left at a scale and perspective consistent with the scene.",
            "add_object",
        ),
        (
            "character_add_behind",
            "have {entity} appear just behind the main subject",
            "Add {entity} just behind the main subject without obscuring the subject or changing the background.",
            "add_object",
        ),
        (
            "character_add_beside",
            "place {entity} beside the main subject as a companion",
            "Add {entity} beside the main subject as a companion while preserving the original action and framing.",
            "add_object",
        ),
        (
            "character_replace_subject",
            "replace the main subject with {entity}",
            "Replace the main subject with {entity} while preserving the subject's pose, motion, and surroundings.",
            "replace_object",
        ),
        (
            "character_replace_center",
            "make the central figure look like {entity}",
            "Replace the central figure with {entity} while retaining the original composition and movement.",
            "replace_object",
        ),
        (
            "character_replace_toy",
            "turn the visible toy into {entity}",
            "Replace the visible toy with {entity} while preserving its location, size, and motion.",
            "replace_object",
        ),
        (
            "character_replace_nearest",
            "swap the figure nearest the camera for {entity}",
            "Replace the figure nearest the camera with {entity} while leaving every other subject unchanged.",
            "replace_object",
        ),
    ],
    "landmark": [
        (
            "landmark_add_horizon",
            "add {entity} on the distant horizon behind the subject",
            "Add {entity} on the distant horizon behind the main subject with realistic scale and perspective.",
            "add_object",
        ),
        (
            "landmark_add_right_background",
            "put {entity} in the far background on the right",
            "Add {entity} in the far background on the right while preserving the foreground subject.",
            "add_object",
        ),
        (
            "landmark_add_left_background",
            "show {entity} in the background to the left of the main subject",
            "Add {entity} in the background to the left of the main subject with scene-consistent lighting.",
            "add_object",
        ),
        (
            "landmark_add_center_background",
            "place {entity} behind the main subject as a distant landmark",
            "Add {entity} behind the main subject as a distant landmark without covering any foreground content.",
            "add_object",
        ),
        (
            "landmark_replace_structure",
            "replace the large structure in the background with {entity}",
            "Replace the large background structure with {entity} while preserving the foreground and sky.",
            "replace_object",
        ),
        (
            "landmark_replace_skyline",
            "swap the landmark on the skyline for {entity}",
            "Replace the landmark on the skyline with {entity} while preserving the camera motion and foreground.",
            "replace_object",
        ),
        (
            "landmark_replace_distant_building",
            "make the distant building look like {entity}",
            "Replace the distant building with {entity} at a realistic scale while preserving the rest of the scene.",
            "replace_object",
        ),
        (
            "landmark_replace_backdrop_feature",
            "turn the main background feature into {entity}",
            "Replace the main background feature with {entity} while keeping all foreground subjects unchanged.",
            "replace_object",
        ),
    ],
    "cultural_artifact": [
        (
            "artifact_add_right",
            "place {entity} to the right of the main object",
            "Add {entity} to the right of the main object with realistic scale, lighting, and contact.",
            "add_object",
        ),
        (
            "artifact_add_left",
            "add {entity} in the open space on the left",
            "Add {entity} in the open space on the left without moving or hiding existing objects.",
            "add_object",
        ),
        (
            "artifact_add_surface",
            "put {entity} on the nearest visible surface",
            "Add {entity} on the nearest visible surface while matching the scene perspective and lighting.",
            "add_object",
        ),
        (
            "artifact_add_beside_subject",
            "display {entity} beside the main subject",
            "Add {entity} beside the main subject as a clearly visible display object while preserving the action.",
            "add_object",
        ),
        (
            "artifact_replace_decoration",
            "replace the decorative object with {entity}",
            "Replace the decorative object with {entity} while preserving its placement and the surrounding scene.",
            "replace_object",
        ),
        (
            "artifact_replace_center",
            "swap the object in the center for {entity}",
            "Replace the object in the center with {entity} while preserving the background and motion.",
            "replace_object",
        ),
        (
            "artifact_replace_display",
            "make the displayed item look like {entity}",
            "Replace the displayed item with {entity} while preserving the display surface and lighting.",
            "replace_object",
        ),
        (
            "artifact_replace_foreground",
            "turn the foreground prop into {entity}",
            "Replace the foreground prop with {entity} while leaving every other object unchanged.",
            "replace_object",
        ),
    ],
}


# Every route has three distinct request/target pairs. The flattened order is
# interleaved by variant so an 11-row diagnostic covers the full taxonomy.
ROUTING_TEMPLATE_GROUPS: dict[str, list[tuple[str, str, str, str | bool]]] = {
    "global_style": [
        ("make the whole clip look like a soft watercolor painting", "Render the entire video in a soft watercolor-painting style while preserving its composition and motion.", "global_style", False),
        ("turn every frame into hand-drawn charcoal animation", "Restyle the entire video as hand-drawn charcoal animation while preserving every subject and action.", "global_style", False),
        ("give the complete video a warm claymation look", "Render the complete video in a warm claymation style while preserving its scene layout and timing.", "global_style", False),
    ],
    "remove_object": [
        ("get rid of the main object in the center", "Remove the main object in the center and naturally fill the revealed region while preserving the surrounding scene.", "remove_object", "main object in the center"),
        ("erase the small item just left of the main subject", "Remove the small item just left of the main subject and reconstruct the occluded background naturally.", "remove_object", "small item just left of the main subject"),
        ("take out the object closest to the camera", "Remove the object closest to the camera and fill its former region consistently across all frames.", "remove_object", "object closest to the camera"),
    ],
    "add_object": [
        ("put a small red balloon next to the main subject", "Add a small red balloon next to the main subject while preserving every existing object.", "add_object", False),
        ("add a potted sunflower in the left foreground", "Add a potted sunflower in the left foreground while preserving the subject, background, and motion.", "add_object", False),
        ("place a wooden stool behind the main subject", "Add a wooden stool behind the main subject without obscuring or moving any existing content.", "add_object", False),
    ],
    "replace_object": [
        ("swap the main visible object for a wooden sculpture", "Replace the main visible object with a wooden sculpture while preserving the background, lighting, and motion.", "replace_object", False),
        ("turn the object closest to the camera into a clear glass sphere", "Replace the object closest to the camera with a clear glass sphere while preserving all other scene content.", "replace_object", False),
        ("replace the central prop with a folded paper crane", "Replace the central prop with a folded paper crane while preserving its location and the surrounding scene.", "replace_object", False),
    ],
    "change_background": [
        ("move the scene to a quiet alpine meadow", "Change the background to a quiet alpine meadow while preserving the main foreground subject.", "change_background", False),
        ("put the subject inside a sunlit old library", "Change the background to a sunlit old library while preserving the foreground subject, pose, and motion.", "change_background", False),
        ("make the setting a foggy pine forest", "Change the background to a foggy pine forest while leaving the main subject unchanged.", "change_background", False),
    ],
    "change_color": [
        ("make only the main subject cobalt blue", "Change only the main subject to cobalt blue while preserving the background and every other object.", "change_color", False),
        ("color the main foreground object matte crimson", "Change only the main foreground object to matte crimson while preserving its material details and surroundings.", "change_color", False),
        ("turn the prominent accessory bright yellow", "Change only the prominent accessory to bright yellow while keeping all other colors unchanged.", "change_color", False),
    ],
    "change_weather": [
        ("make it lightly snow without hiding anything", "Change the weather to light snowfall while keeping every subject clearly visible.", "change_weather", False),
        ("change the weather to a gentle rain shower", "Change the weather to gentle rain while preserving subject visibility, lighting coherence, and motion.", "change_weather", False),
        ("add hazy morning mist across the scene", "Change the weather to a hazy, misty morning while keeping the foreground subjects clear.", "change_weather", False),
    ],
    "add_effect": [
        ("add a few subtle golden sparkles around the subject", "Add a few subtle golden sparkles around the main subject without obscuring any existing content.", "add_effect", False),
        ("let soft soap bubbles drift through the shot", "Add soft translucent soap bubbles drifting through the shot while preserving the original subjects and action.", "add_effect", False),
        ("add faint rays of light from the upper left", "Add faint volumetric light rays from the upper left without changing the scene structure or subjects.", "add_effect", False),
    ],
    "customization": [
        ("turn the main subject into a personalized clay figurine version of itself", "Customize the main subject as a personalized clay figurine while preserving its recognizable features, pose, and motion.", "customization", False),
        ("make a custom plush mascot based on the main subject", "Customize the main subject as a plush mascot that retains its recognizable identity and original movement.", "customization", False),
        ("redesign the main subject as a hand-painted toy avatar", "Customize the main subject as a hand-painted toy avatar while retaining its distinctive features and the scene composition.", "customization", False),
    ],
    "combined_tasks": [
        ("make the main subject emerald green and put it in a desert at sunset", "Change the main subject to emerald green and replace the background with a desert at sunset.", "combined_tasks", False),
        ("turn the central object gold and move the scene to a city at night", "Change the central object to gold and replace the background with a city at night.", "combined_tasks", False),
        ("add a red umbrella next to the subject and make it rain", "Add a red umbrella next to the main subject and change the weather to gentle rain while preserving the rest of the scene.", "combined_tasks", False),
    ],
    "camera_edit": [
        ("slowly zoom the camera in but keep the scene unchanged", "Apply a slow camera zoom-in while preserving all scene content and motion.", "camera_edit", False),
        ("make the camera pan gently from right to left", "Apply a gentle right-to-left camera pan while preserving every subject and the original action.", "camera_edit", False),
        ("orbit the camera slowly clockwise around the main subject", "Apply a slow clockwise camera orbit around the main subject without changing the subject or setting.", "camera_edit", False),
    ],
}

ROUTING_SUBTASKS = tuple(ROUTING_TEMPLATE_GROUPS)
ROUTING_TEMPLATES = [
    ROUTING_TEMPLATE_GROUPS[subtask][variant_index]
    for variant_index in range(3)
    for subtask in ROUTING_SUBTASKS
]
ALLOWED_SUBTASKS = frozenset(ROUTING_TEMPLATE_GROUPS)
SEARCHABLE_SUBTASKS = frozenset({"add_object", "replace_object", "change_background", "customization"})


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_record(source: dict[str, Any], request: str, plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": source["system"],
        "messages": [
            {"role": "user", "content": f"<video>\n{request}"},
            {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False, separators=(",", ":"))},
        ],
        "videos": source["videos"],
    }


def uniform_source_indices(population_size: int, count: int) -> list[int]:
    """Select deterministic, evenly spaced source rows across the full population."""
    if population_size < 1:
        raise ValueError("source population must be positive")
    if count < 0:
        raise ValueError("augmentation count cannot be negative")
    if count == 0:
        return []
    return [
        min(population_size - 1, ((2 * index + 1) * population_size) // (2 * count))
        for index in range(count)
    ]


def _validate_plan(plan: dict[str, Any]) -> None:
    subtask = plan["subtask"]
    image_search = plan["image_search"]
    mask = plan["mask"]
    if subtask not in ALLOWED_SUBTASKS:
        raise ValueError(f"invalid calibration subtask: {subtask}")
    if image_search is not False:
        if subtask not in SEARCHABLE_SUBTASKS or not isinstance(image_search, str) or not image_search.strip():
            raise ValueError(f"invalid image_search for calibration subtask {subtask}")
    if subtask == "remove_object":
        if not isinstance(mask, str) or not mask.strip():
            raise ValueError("remove_object calibration plans require a mask noun phrase")
    elif mask is not False:
        raise ValueError(f"mask must be false for calibration subtask {subtask}")


def build_augmented(
    base_rows: list[dict[str, Any]], search_count: int, routing_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not base_rows:
        raise ValueError("base SFT data is empty")
    if search_count < 0 or routing_count < 0:
        raise ValueError("augmentation counts cannot be negative")
    llama_rows = list(base_rows)
    metadata: list[dict[str, Any]] = []
    search_source_indices = uniform_source_indices(len(base_rows), search_count)
    routing_source_indices = uniform_source_indices(len(base_rows), routing_count)
    for index, source_index in enumerate(search_source_indices):
        source = base_rows[source_index]
        entity_index = index % len(EXTERNAL_ENTITY_SPECS)
        entity_round = index // len(EXTERNAL_ENTITY_SPECS)
        entity_category, entity = EXTERNAL_ENTITY_SPECS[entity_index]
        category_templates = SEARCH_TEMPLATES_BY_CATEGORY[entity_category]
        template_id, request_template, refined_template, subtask = category_templates[
            (entity_index + entity_round) % len(category_templates)
        ]
        request = request_template.format(entity=entity)
        refined = refined_template.format(entity=entity)
        plan = {
            "refined_text_instruction": refined,
            "subtask": subtask,
            "image_search": entity,
            "mask": False,
        }
        _validate_plan(plan)
        llama_rows.append(make_record(source, request, plan))
        metadata.append(
            {
                "category": "under_search",
                "source_index": source_index,
                "entity_category": entity_category,
                "template_id": template_id,
                "request": request,
                "target_plan": plan,
            }
        )
    for index, source_index in enumerate(routing_source_indices):
        source = base_rows[source_index]
        template_index = index % len(ROUTING_TEMPLATES)
        request, refined, subtask, mask = ROUTING_TEMPLATES[template_index]
        plan = {
            "refined_text_instruction": refined,
            "subtask": subtask,
            "image_search": False,
            "mask": mask,
        }
        _validate_plan(plan)
        llama_rows.append(make_record(source, request, plan))
        metadata.append(
            {
                "category": "routing_calibration",
                "source_index": source_index,
                "template_id": f"{subtask}_{template_index // len(ROUTING_SUBTASKS) + 1}",
                "request": request,
                "target_plan": plan,
            }
        )
    return llama_rows, metadata


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--search-count", type=int, default=500)
    parser.add_argument("--routing-count", type=int, default=500)
    args = parser.parse_args()
    rows, metadata = build_augmented(load_jsonl(args.base), args.search_count, args.routing_count)
    write_jsonl(args.out, rows)
    write_jsonl(args.metadata_out, metadata)
    print(json.dumps({"base": len(rows) - len(metadata), "augmented": len(metadata), "total": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

"""Add balanced search and routing calibration examples to planner SFT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXTERNAL_ENTITIES = [
    "Adidas Samba OG sneaker",
    "Apple Vision Pro headset",
    "Barbie Dreamhouse",
    "Boeing 787 Dreamliner",
    "Burberry check handbag",
    "Chanel No. 5 perfume bottle",
    "Christ the Redeemer statue",
    "Coca-Cola contour bottle",
    "DJI Mavic 3 drone",
    "Darth Vader helmet",
    "Disney Cinderella Castle",
    "Dyson Airwrap styler",
    "Ferrari F40 sports car",
    "Gibson Les Paul guitar",
    "Golden Gate Bridge",
    "Gucci Jackie handbag",
    "Hello Kitty plush toy",
    "Hermes Birkin handbag",
    "IKEA POANG chair",
    "KAWS Companion figure",
    "LEGO Millennium Falcon",
    "Leica M11 camera",
    "London Big Ben clock tower",
    "Louis Vuitton Neverfull bag",
    "McDonald's Happy Meal box",
    "Microsoft Xbox Series X",
    "Mini Cooper classic car",
    "Mount Rushmore monument",
    "Nike Air Jordan 1 Chicago sneaker",
    "Nintendo Switch OLED console",
    "Porsche 911 GT3 RS",
    "Rolex Submariner watch",
    "Samsung Galaxy Z Flip phone",
    "Sydney Opera House",
    "Tesla Cybertruck",
    "The Great Sphinx of Giza",
    "Totoro character plush",
    "Vespa Primavera scooter",
    "Yamaha grand piano",
    "Yellow Submarine cartoon vessel",
]


ROUTING_TEMPLATES: list[tuple[str, str, str, str | bool]] = [
    (
        "make the whole clip look like a soft watercolor painting",
        "Render the entire video in a soft watercolor-painting style while preserving its composition and motion.",
        "global_style",
        False,
    ),
    (
        "get rid of the main object in the center",
        "Remove the main object in the center and naturally fill the revealed region while preserving the surrounding scene.",
        "remove_object",
        "main object in the center",
    ),
    (
        "put a small red balloon next to the main subject",
        "Add a small red balloon next to the main subject while preserving every existing object.",
        "add_object",
        False,
    ),
    (
        "swap the main visible object for a wooden sculpture",
        "Replace the main visible object with a wooden sculpture while preserving the background, lighting, and motion.",
        "replace_object",
        False,
    ),
    (
        "move the scene to a quiet alpine meadow",
        "Change the background to a quiet alpine meadow while preserving the main foreground subject.",
        "change_background",
        False,
    ),
    (
        "make only the main subject cobalt blue",
        "Change only the main subject to cobalt blue while preserving the background and every other object.",
        "change_color",
        False,
    ),
    (
        "make it lightly snow without hiding anything",
        "Change the weather to light snowfall while keeping every subject clearly visible.",
        "change_weather",
        False,
    ),
    (
        "add a few subtle golden sparkles around the subject",
        "Add a few subtle golden sparkles around the main subject without obscuring any existing content.",
        "add_effect",
        False,
    ),
    (
        "make the main subject emerald green and put it in a desert at sunset",
        "Change the main subject to emerald green and replace the background with a desert at sunset.",
        "combined_tasks",
        False,
    ),
    (
        "slowly zoom the camera in but keep the scene unchanged",
        "Apply a slow camera zoom-in while preserving all scene content and motion.",
        "camera_edit",
        False,
    ),
]


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


def build_augmented(
    base_rows: list[dict[str, Any]], search_count: int, routing_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not base_rows:
        raise ValueError("base SFT data is empty")
    llama_rows = list(base_rows)
    metadata: list[dict[str, Any]] = []
    for index in range(search_count):
        source = base_rows[index % len(base_rows)]
        entity = EXTERNAL_ENTITIES[index % len(EXTERNAL_ENTITIES)]
        if index % 2 == 0:
            request = f"put a {entity} to the right of the main subject"
            refined = f"Add a {entity} to the right of the main subject while preserving all existing scene content."
            subtask = "add_object"
        else:
            request = f"make the main visible object look like a {entity}"
            refined = f"Replace the main visible object with a {entity} while preserving the background, lighting, and motion."
            subtask = "replace_object"
        plan = {
            "refined_text_instruction": refined,
            "subtask": subtask,
            "image_search": entity,
            "mask": False,
        }
        llama_rows.append(make_record(source, request, plan))
        metadata.append({"category": "under_search", "source_index": index % len(base_rows), "request": request, "target_plan": plan})
    for index in range(routing_count):
        source = base_rows[(search_count + index) % len(base_rows)]
        request, refined, subtask, mask = ROUTING_TEMPLATES[index % len(ROUTING_TEMPLATES)]
        plan = {
            "refined_text_instruction": refined,
            "subtask": subtask,
            "image_search": False,
            "mask": mask,
        }
        llama_rows.append(make_record(source, request, plan))
        metadata.append({"category": "routing_calibration", "source_index": (search_count + index) % len(base_rows), "request": request, "target_plan": plan})
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

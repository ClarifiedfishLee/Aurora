"""Build the deterministic 100-case Week-1 planner regression suite.

This suite reuses smoke-test media and is therefore not the frozen
Mini-AgentEdit benchmark. It is intended for fast Base-vs-LoRA comparisons and
for exercising the four planner decision axes before held-out media arrives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCES = [
    ("dog", "brown dog", "Snoopy", "snowy park", "red ball", "golden"),
    ("bottle", "clear plastic bottle", "rose quartz Stanley Quencher tumbler", "modern kitchen", "lemon", "cobalt blue"),
    ("car", "silver car", "the Batmobile", "mountain road", "traffic cone", "bright red"),
    ("cup", "cup", "Starbucks holiday cup", "wooden café table", "silver spoon", "emerald green"),
    ("city", "city skyline", "the Eiffel Tower", "sunset waterfront", "hot-air balloon", "warm orange"),
    ("person", "person", "Spider-Man", "library interior", "blue backpack", "monochrome blue"),
    ("beach", "beach", "Burj Al Arab", "tropical lagoon", "yellow beach umbrella", "turquoise"),
    ("bicycle", "bicycle", "Trek Madone racing bicycle", "forest trail", "white helmet", "matte black"),
    ("cat", "gray and white cat", "Pikachu", "cozy bedroom", "green cushion", "black"),
    ("tree", "tree", "Japanese cherry blossom tree", "Japanese garden", "wooden bench", "pink"),
]


def _constraint(kind: str, value: str, aliases: list[str] | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"type": kind, "value": value}
    if aliases:
        item["aliases"] = aliases
    return item


def _pluralize_phrase(phrase: str) -> str:
    words = phrase.split()
    irregular = {"bench": "benches"}
    words[-1] = irregular.get(words[-1], f"{words[-1]}s")
    return " ".join(words)


def _tasks(
    entity: str,
    external: str,
    background: str,
    companion: str,
    color: str,
    *,
    external_subtask: str = "replace_object",
) -> list[dict[str, Any]]:
    if external_subtask == "add_object":
        external_prompt = f"add {external} in the distant background"
        external_rewrite = f"Add {external} in the distant background while preserving the foreground scene."
    else:
        external_prompt = f"make the {entity} look like {external}"
        external_rewrite = f"Replace the {entity} with {external} while preserving the rest of the scene."
    companions = _pluralize_phrase(companion)
    return [
        {
            "prompt": external_prompt,
            "subtask": external_subtask,
            "search": external,
            "mask": False,
            "rewrite": external_rewrite,
            "constraints": [_constraint("identity", external), _constraint("preservation", "rest of the scene")],
        },
        {
            "prompt": f"make the {entity} {color}, don't change anything else",
            "subtask": "change_color",
            "search": False,
            "mask": False,
            "rewrite": f"Change the {entity} to {color} while preserving everything else.",
            "constraints": [_constraint("color", color), _constraint("preservation", "everything else")],
        },
        {
            "prompt": f"get rid of the {entity}",
            "subtask": "remove_object",
            "search": False,
            "mask": entity,
            "rewrite": f"Remove the {entity} and naturally fill the revealed area while preserving the surrounding scene.",
            "constraints": [_constraint("preservation", "surrounding scene")],
        },
        {
            "prompt": "make this look like a Van Gogh painting",
            "subtask": "global_style",
            "search": False,
            "mask": False,
            "rewrite": "Render the entire video in a vivid Van Gogh oil-painting style while preserving its composition and motion.",
            "constraints": [_constraint("identity", "Van Gogh"), _constraint("preservation", "composition")],
        },
        {
            "prompt": f"put a {companion} next to the {entity}",
            "subtask": "add_object",
            "search": False,
            "mask": False,
            "rewrite": f"Add a {companion} next to the {entity} while preserving all existing objects.",
            "constraints": [_constraint("identity", companion), _constraint("spatial", f"next to the {entity}")],
        },
        {
            "prompt": f"change the background to a {background}",
            "subtask": "change_background",
            "search": False,
            "mask": False,
            "rewrite": f"Change the background to a {background} while preserving the main foreground subject.",
            "constraints": [_constraint("identity", background), _constraint("preservation", "foreground subject")],
        },
        {
            "prompt": "make it lightly snowing but keep everything visible",
            "subtask": "change_weather",
            "search": False,
            "mask": False,
            "rewrite": "Change the weather to light snowfall while keeping every subject clearly visible.",
            "constraints": [_constraint("other", "light snowfall", ["lightly snowing"]), _constraint("preservation", "clearly visible")],
        },
        {
            "prompt": f"turn the {entity} {color} and keep the background exactly the same",
            "subtask": "change_color",
            "search": False,
            "mask": False,
            "rewrite": f"Change only the {entity} to {color}; preserve the background exactly as it is.",
            "constraints": [_constraint("color", color), _constraint("preservation", "background")],
        },
        {
            "prompt": f"add three small {companions} to the left of the {entity}",
            "subtask": "add_object",
            "search": False,
            "mask": False,
            "rewrite": f"Add three small {companions} to the left of the {entity} while preserving the original scene.",
            "constraints": [
                _constraint("count", "three", ["3"]),
                _constraint("spatial", f"left of the {entity}"),
                _constraint("identity", companion),
            ],
        },
        {
            "prompt": f"make the {entity} {color} and move the scene to a {background}",
            "subtask": "combined_tasks",
            "search": False,
            "mask": False,
            "rewrite": f"Change the {entity} to {color} and replace the background with a {background}.",
            "constraints": [_constraint("color", color), _constraint("identity", background)],
        },
    ]


def build() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, (slug, entity, external, background, companion, color) in enumerate(SOURCES):
        external_subtask = "add_object" if slug in {"city", "beach"} else "replace_object"
        tasks = _tasks(entity, external, background, companion, color, external_subtask=external_subtask)
        if source_index in {2, 6, 9}:
            tasks[3] = {
                "prompt": "add subtle golden sparkles across the scene",
                "subtask": "add_effect",
                "search": False,
                "mask": False,
                "rewrite": "Add subtle golden sparkling particles across the scene while preserving all subjects and motion.",
                "constraints": [_constraint("other", "golden sparkles"), _constraint("preservation", "all subjects")],
            }
        elif source_index in {3, 7}:
            tasks[3] = {
                "prompt": "make the camera slowly zoom in",
                "subtask": "camera_edit",
                "search": False,
                "mask": False,
                "rewrite": "Apply a slow, smooth camera zoom toward the main subject without changing the scene content.",
                "constraints": [_constraint("other", "slow"), _constraint("preservation", "scene content")],
            }
        elif source_index == 5:
            tasks[3] = {
                "prompt": "replace the main subject with the subject from the reference image",
                "subtask": "customization",
                "search": False,
                "mask": False,
                "rewrite": "Replace the main subject with the identity shown in the supplied reference image while preserving the scene.",
                "constraints": [_constraint("identity", "reference image"), _constraint("preservation", "scene")],
                "ref_image_path": "data/smoke/source_images/smoke_009_cat.jpg",
            }
        axes = ["search", "search", "mask", "mask", "routing", "routing", "routing", "rewrite", "rewrite", "routing"]
        if source_index >= 5:
            axes[4] = "search"
        if source_index < 5:
            axes[5] = "mask"
            axes[6] = "rewrite"
        for task_index, (task, axis) in enumerate(zip(tasks, axes), 1):
            case_number = source_index * 10 + task_index
            row = {
                "bench_id": f"w1_{case_number:04d}",
                "video_path": f"data/smoke/videos/smoke_{source_index + 1:03d}_{slug}.mp4",
                "prompt": task["prompt"],
                "edit_type": task["subtask"],
                "axis": axis,
                "source": {"dataset": "smoke_fixture", "source_id": f"smoke_{slug}", "license": "local test fixture"},
                "gold_plan": {
                    "refined_text_instruction": task["rewrite"],
                    "subtask": task["subtask"],
                    "image_search": task["search"],
                    "mask": task["mask"],
                },
                "constraints": task["constraints"],
                "source_entities": [entity],
            }
            if task.get("ref_image_path"):
                row["ref_image_path"] = task["ref_image_path"]
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/week1/planner_100.jsonl"))
    args = parser.parse_args()
    rows = build()
    axes = {axis: sum(row["axis"] == axis for row in rows) for axis in ("search", "mask", "routing", "rewrite")}
    if len(rows) != 100 or set(axes.values()) != {25}:
        raise RuntimeError(f"unexpected suite balance: rows={len(rows)}, axes={axes}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "num_cases": len(rows), "axes": axes}, indent=2))


if __name__ == "__main__":
    main()

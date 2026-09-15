"""Build the Week-1 single-axis A/B rendering pilot.

The script has two stages:

1. ``prepare-tools`` selects the mask cases that Aurora must execute once to
   materialize segmentation masks.  Search pairs reuse the already-audited
   Stanley, Eiffel Tower, and Pikachu smoke-search assets.
2. ``build-records`` converts those resolved tool records plus deterministic
   gold plans into paired editor records.  Each pair changes exactly one
   planner decision axis.

Routing pairs are intentional negative controls: the current editor bridge
does not consume ``plan.subtask``, so their rendered outputs should match.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


SEARCH_SPECS = [
    ("w1_0011", "under", "runs/smoke_search_agent_v2/search_images/smoke_search_stanley-6aab2c7914fa980bafcab216de597660.jpg", "Stanley Quencher H2.0"),
    ("w1_0041", "under", "runs/smoke_search_agent_v2/search_images/smoke_search_eiffel-258c32770c1940ad253c731ee556006b.jpg", "Eiffel Tower"),
    ("w1_0081", "under", "runs/smoke_search_agent_v2/search_images/smoke_search_pikachu-f23a1d44dd48c936e241d2be775faa60.jpg", "Pikachu"),
    ("w1_0002", "over", "runs/smoke_search_agent_v2/search_images/smoke_search_pikachu-f23a1d44dd48c936e241d2be775faa60.jpg", "golden character reference"),
    ("w1_0022", "over", "runs/smoke_search_agent_v2/search_images/smoke_search_eiffel-258c32770c1940ad253c731ee556006b.jpg", "bright red landmark reference"),
    ("w1_0032", "over", "runs/smoke_search_agent_v2/search_images/smoke_search_stanley-6aab2c7914fa980bafcab216de597660.jpg", "emerald green cup reference"),
    ("w1_0052", "over", "runs/smoke_search_agent_v2/search_images/smoke_search_pikachu-f23a1d44dd48c936e241d2be775faa60.jpg", "monochrome blue character reference"),
    ("w1_0062", "over", "runs/smoke_search_agent_v2/search_images/smoke_search_eiffel-258c32770c1940ad253c731ee556006b.jpg", "turquoise travel reference"),
    ("w1_0072", "over", "runs/smoke_search_agent_v2/search_images/smoke_search_stanley-6aab2c7914fa980bafcab216de597660.jpg", "matte black product reference"),
    ("w1_0092", "over", "runs/smoke_search_agent_v2/search_images/smoke_search_pikachu-f23a1d44dd48c936e241d2be775faa60.jpg", "pink character reference"),
]
MASK_IDS = [f"w1_{index:04d}" for index in range(3, 101, 10)]
REWRITE_IDS = [f"w1_{index:04d}" for source in range(8) for index in (source * 10 + 8, source * 10 + 9)]
ROUTING_IDS = [f"w1_{index:04d}" for index in (5, 15, 25, 35)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def prepare_tool_cases(planner_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = set(MASK_IDS)
    selected = [row for row in planner_rows if row["bench_id"] in wanted]
    if len(selected) != len(wanted):
        found = {row["bench_id"] for row in selected}
        raise ValueError(f"missing planner cases: {sorted(wanted - found)}")
    return selected


def _base_record(row: dict[str, Any], variant_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "bench_id": variant_id,
        "edit_type": row["edit_type"],
        "prompt": row["prompt"],
        "video_path": row["video_path"],
        "ref_image_path": row.get("ref_image_path", ""),
        "plan": copy.deepcopy(plan),
        "final_payload": {
            "refined_text_instruction": plan["refined_text_instruction"],
            "subtask": plan["subtask"],
            "search_image": False,
            "object_mask": False,
        },
    }


def _weak_rewrite(row: dict[str, Any]) -> str:
    constraints = row.get("constraints", [])
    by_type = {item["type"]: item["value"] for item in constraints}
    gold = row["gold_plan"]
    if "color" in by_type:
        entity = row["source_entities"][0]
        return f"Change the {entity} to {by_type['color']}."
    if "count" in by_type or "spatial" in by_type:
        identity = by_type.get("identity", "object")
        return f"Add a {identity} somewhere in the scene."
    return row["prompt"] or gold["refined_text_instruction"]


def _resolved_asset(record: dict[str, Any], axis: str) -> str:
    if axis == "search":
        path = record.get("search", {}).get("selected_path") or record.get("final_payload", {}).get("search_image")
    else:
        path = record.get("mask", {}).get("mask_path") or record.get("final_payload", {}).get("object_mask")
    if not path or path is False:
        raise ValueError(f"{record.get('bench_id')}: missing resolved {axis} asset")
    return str(path)


def build_records(
    planner_rows: list[dict[str, Any]], resolved_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    planner = {row["bench_id"]: row for row in planner_rows}
    resolved = {row["bench_id"]: row for row in resolved_rows}
    records: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    def add_pair(row: dict[str, Any], axis: str, good_plan: dict[str, Any], bad_plan: dict[str, Any]) -> None:
        pair_index = len(pairs) + 1
        pair_id = f"w1_ab_{pair_index:03d}"
        changed = [key for key in good_plan if good_plan[key] != bad_plan[key]]
        expected = {"search": "image_search", "mask": "mask", "rewrite": "refined_text_instruction", "routing": "subtask"}[axis]
        if changed != [expected]:
            raise ValueError(f"{pair_id}: expected only {expected} to change, got {changed}")
        good = _base_record(row, f"{pair_id}_good", good_plan)
        bad = _base_record(row, f"{pair_id}_bad", bad_plan)

        if axis == "search":
            source = resolved[row["bench_id"]]
            asset = _resolved_asset(source, axis)
            good["search"] = copy.deepcopy(source.get("search", {}))
            good["final_payload"]["search_image"] = asset
        elif axis == "mask":
            source = resolved[row["bench_id"]]
            asset = _resolved_asset(source, axis)
            good["mask"] = copy.deepcopy(source.get("mask", {}))
            good["final_payload"]["object_mask"] = asset
            bad["mask"] = {"phrase": None, "mask_path": None, "overlay_path": None, "meta": None}

        for record, quality in ((good, "good"), (bad, "bad")):
            record["counterfactual"] = {
                "pair_id": pair_id,
                "axis": axis,
                "quality": quality,
                "changed_field": expected,
                "source_case_id": row["bench_id"],
            }
            records.append(record)
        pairs.append(
            {
                "pair_id": pair_id,
                "axis": axis,
                "source_case_id": row["bench_id"],
                "instruction": row["prompt"],
                "source_video": row["video_path"],
                "good_record_id": good["bench_id"],
                "bad_record_id": bad["bench_id"],
                "changed_field": expected,
                "negative_control": axis == "routing",
            }
        )

    for case_id, mode, asset, bad_query in SEARCH_SPECS:
        row = planner[case_id]
        good = copy.deepcopy(row["gold_plan"])
        bad = copy.deepcopy(good)
        if mode == "under":
            bad["image_search"] = False
        else:
            bad["image_search"] = bad_query

        pair_index = len(pairs) + 1
        pair_id = f"w1_ab_{pair_index:03d}"
        changed = [key for key in good if good[key] != bad[key]]
        if changed != ["image_search"]:
            raise ValueError(f"{pair_id}: expected only image_search to change, got {changed}")
        good_record = _base_record(row, f"{pair_id}_good", good)
        bad_record = _base_record(row, f"{pair_id}_bad", bad)
        searched_record = good_record if mode == "under" else bad_record
        searched_record["search"] = {"query": good["image_search"] if mode == "under" else bad_query, "selected_path": asset}
        searched_record["final_payload"]["search_image"] = asset
        for record, quality in ((good_record, "good"), (bad_record, "bad")):
            record["counterfactual"] = {
                "pair_id": pair_id,
                "axis": "search",
                "quality": quality,
                "changed_field": "image_search",
                "source_case_id": row["bench_id"],
                "search_error_type": "under_search" if mode == "under" else "over_search",
            }
            records.append(record)
        pairs.append(
            {
                "pair_id": pair_id,
                "axis": "search",
                "source_case_id": row["bench_id"],
                "instruction": row["prompt"],
                "source_video": row["video_path"],
                "good_record_id": good_record["bench_id"],
                "bad_record_id": bad_record["bench_id"],
                "changed_field": "image_search",
                "search_error_type": "under_search" if mode == "under" else "over_search",
                "negative_control": False,
            }
        )

    for case_id in MASK_IDS:
        row = planner[case_id]
        good = copy.deepcopy(row["gold_plan"])
        bad = copy.deepcopy(good)
        bad["mask"] = False
        add_pair(row, "mask", good, bad)

    for case_id in REWRITE_IDS:
        row = planner[case_id]
        good = copy.deepcopy(row["gold_plan"])
        bad = copy.deepcopy(good)
        bad["refined_text_instruction"] = _weak_rewrite(row)
        add_pair(row, "rewrite", good, bad)

    for case_id in ROUTING_IDS:
        row = planner[case_id]
        good = copy.deepcopy(row["gold_plan"])
        bad = copy.deepcopy(good)
        bad["subtask"] = "global_style" if good["subtask"] != "global_style" else "add_object"
        add_pair(row, "routing", good, bad)

    if len(pairs) != 40 or len(records) != 80:
        raise RuntimeError(f"unexpected pilot size: pairs={len(pairs)}, records={len(records)}")
    return records, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-tools")
    prepare.add_argument("--planner", type=Path, default=Path("data/week1/planner_100.jsonl"))
    prepare.add_argument("--out", type=Path, default=Path("data/week1/ab_tool_cases.jsonl"))

    build = subparsers.add_parser("build-records")
    build.add_argument("--planner", type=Path, default=Path("data/week1/planner_100.jsonl"))
    build.add_argument("--resolved", type=Path, required=True)
    build.add_argument("--records-out", type=Path, default=Path("data/week1/ab_editor_records.jsonl"))
    build.add_argument("--pairs-out", type=Path, default=Path("data/week1/ab_pairs.jsonl"))
    args = parser.parse_args()

    planner_rows = load_jsonl(args.planner)
    if args.command == "prepare-tools":
        rows = prepare_tool_cases(planner_rows)
        write_jsonl(args.out, rows)
        print(json.dumps({"out": str(args.out), "num_cases": len(rows)}, indent=2))
        return

    records, pairs = build_records(planner_rows, load_jsonl(args.resolved))
    write_jsonl(args.records_out, records)
    write_jsonl(args.pairs_out, pairs)
    print(
        json.dumps(
            {
                "records_out": str(args.records_out),
                "pairs_out": str(args.pairs_out),
                "num_pairs": len(pairs),
                "axes": {axis: sum(pair["axis"] == axis for pair in pairs) for axis in ("search", "mask", "rewrite", "routing")},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

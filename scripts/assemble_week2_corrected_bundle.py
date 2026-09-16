"""Assemble one immutable, portable Week-2 corrected-selection bundle.

The runtime artifacts live in several independently sealed directories and
use names that differ from the portable bundle contract.  This utility maps
them into either the complete ``corrected-full`` profile or the durable
``corrected-thin`` profile, emits adapter identity proofs, copies the locked
Day-14 gold file, and writes the checksum manifest last.

The destination is atomically claimed and must not already exist.  Source
artifacts are never modified.  On failure, only the destination directory
created by this invocation is removed.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from scripts.verify_week2_bundle import (
    CORRECTED_FULL_PROFILE,
    CORRECTED_LAMBDA_GRID,
    CORRECTED_THIN_PROFILE,
    audit_corrected_full_bundle,
    audit_corrected_thin_bundle,
    sha256,
    verify_manifest,
    write_manifest,
)

HARDLINK_MIN_BYTES = 64 * 1024 * 1024
RECIPE_METADATA_FILES = (
    "refresh_train.jsonl",
    "refresh_eval.jsonl",
    "refresh_generation_audit.json",
    "train_refresh.yaml",
    "refresh_selection_policy.json",
    "refresh_selection.json",
)
THIN_RECIPE_ROOT_FILES = (
    "train.log",
    "trainer_log.jsonl",
    "trainer_state.json",
    "exit_code",
)
EVAL_FILES = (
    "agent_pipeline_records.jsonl",
    "planner.log",
    "metrics.json",
    "scorer.log",
    "run_manifest.json",
)
ADAPTIVE_FILES = (
    "agent_pipeline_records.jsonl",
    "planner.log",
    "metrics.json",
    "scorer.log",
    "gate_summary.json",
)
RECIPE_CHECKPOINT_STEPS = (32, 64, 96, 128)


class BundleAssemblyError(RuntimeError):
    """Raised when a corrected bundle cannot be assembled safely."""


def _candidate_id(value: float) -> str:
    return f"lambda_{round(value * 1000):04d}"


CANDIDATE_IDS = (*(_candidate_id(value) for value in CORRECTED_LAMBDA_GRID), "recipe2")


def _require_directory(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise BundleAssemblyError(f"{label} is a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise BundleAssemblyError(f"{label} is missing: {resolved}")
    return resolved


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise BundleAssemblyError(f"{label} is missing, empty, or a symlink: {path}")
    return path.resolve()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    path = _require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleAssemblyError(f"cannot parse {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise BundleAssemblyError(f"{label} must contain a JSON object: {path}")
    return value


def _assert_disjoint_destination(out_root: Path, sources: Sequence[Path]) -> None:
    out_resolved = out_root.expanduser().resolve()
    for source in sources:
        source_resolved = source.expanduser().resolve()
        if (
            out_resolved == source_resolved
            or out_resolved.is_relative_to(source_resolved)
            or source_resolved.is_relative_to(out_resolved)
        ):
            raise BundleAssemblyError(
                f"bundle destination overlaps source path: {out_resolved} / {source_resolved}"
            )


def _copy_file(
    source: Path,
    destination: Path,
    *,
    prefer_hardlink: bool,
    stats: dict[str, int],
) -> None:
    source = _require_file(source, "copy source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise BundleAssemblyError(f"refusing to overwrite bundle artifact: {destination}")

    if prefer_hardlink and source.stat().st_size >= HARDLINK_MIN_BYTES:
        try:
            os.link(source, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise BundleAssemblyError(
                f"refusing to overwrite bundle artifact: {destination}"
            ) from error
        except OSError as error:
            fallback_errors = {
                errno.EXDEV,
                errno.EPERM,
                errno.EACCES,
                errno.EMLINK,
                getattr(errno, "ENOTSUP", -1),
                getattr(errno, "EOPNOTSUPP", -1),
            }
            if error.errno not in fallback_errors:
                raise
        else:
            stats["hardlinked_files"] += 1
            return

    claimed = False
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            claimed = True
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        shutil.copystat(source, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise BundleAssemblyError(
            f"refusing to overwrite bundle artifact: {destination}"
        ) from error
    except Exception:
        if claimed:
            destination.unlink(missing_ok=True)
        raise
    stats["copied_files"] += 1


def _copy_tree(
    source: Path,
    destination: Path,
    *,
    prefer_hardlink: bool,
    stats: dict[str, int],
    include: Callable[[Path], bool] | None = None,
) -> None:
    source = _require_directory(source, "copy tree")
    if destination.exists() or destination.is_symlink():
        raise BundleAssemblyError(f"refusing to merge bundle directory: {destination}")
    destination.mkdir(parents=True)
    for item in sorted(source.rglob("*"), key=lambda path: path.as_posix()):
        relative = item.relative_to(source)
        if item.is_symlink():
            raise BundleAssemblyError(f"source tree contains a symlink: {item}")
        if include is not None and not include(relative):
            continue
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            _copy_file(
                item,
                target,
                prefer_hardlink=prefer_hardlink,
                stats=stats,
            )
        else:
            raise BundleAssemblyError(f"source tree contains a special file: {item}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise BundleAssemblyError(f"refusing to overwrite bundle artifact: {path}") from error


def _validate_source_inventory(
    *,
    profile: str,
    recipe2_root: Path,
    fresh_root: Path,
    candidates_root: Path,
    adaptive_root: Path,
    day14_gold: Path,
    adapters: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, str]]]:
    for name in RECIPE_METADATA_FILES:
        _require_file(recipe2_root / name, f"recipe2 metadata {name}")
    recipe_output = _require_directory(recipe2_root / "lora-refresh", "recipe2 trainer output")
    if profile == CORRECTED_FULL_PROFILE:
        _require_directory(
            recipe2_root / "lora-refresh-selected", "recipe2 selected adapter"
        )
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            _require_file(recipe_output / name, f"recipe2 trainer root {name}")
    for name in THIN_RECIPE_ROOT_FILES:
        _require_file(recipe_output / name, f"recipe2 trainer root {name}")
    for step in RECIPE_CHECKPOINT_STEPS:
        checkpoint = _require_directory(
            recipe_output / f"checkpoint-{step}", f"recipe2 checkpoint-{step}"
        )
        required = (
            (
                "adapter_config.json",
                "adapter_model.safetensors",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "trainer_state.json",
            )
            if profile == CORRECTED_FULL_PROFILE
            else ("trainer_state.json",)
        )
        for name in required:
            _require_file(checkpoint / name, f"recipe2 checkpoint-{step} {name}")

    fresh_names = (
        "cases.jsonl",
        "gold.jsonl",
        "audit.json",
        "policy.json",
        "recipe2_cross_audit.json",
        "comparison.json",
        "selection.json",
    )
    for name in fresh_names:
        _require_file(fresh_root / name, f"fresh384 {name}")
    policy = _load_json(fresh_root / "policy.json", "fresh384 policy")
    selection = _load_json(fresh_root / "selection.json", "final selection")
    selected_id = selection.get("selected_candidate_id")
    if selection.get("selected") is not True or selected_id not in CANDIDATE_IDS:
        raise BundleAssemblyError("selection.json does not contain one registered selection")

    actual_candidate_dirs = {
        path.name for path in candidates_root.iterdir() if path.is_dir()
    }
    if actual_candidate_dirs != set(CANDIDATE_IDS):
        raise BundleAssemblyError(
            "candidate directory set differs from the locked ten candidates: "
            f"{sorted(actual_candidate_dirs)}"
        )
    for candidate_id in CANDIDATE_IDS:
        candidate = _require_directory(
            candidates_root / candidate_id, f"candidate {candidate_id}"
        )
        adapter = _require_directory(candidate / "adapter", f"candidate {candidate_id} adapter")
        _require_file(adapter / "adapter_config.json", f"candidate {candidate_id} config")
        if profile == CORRECTED_FULL_PROFILE or candidate_id == selected_id:
            _require_file(adapter / "adapter_model.safetensors", f"candidate {candidate_id} weights")
        if candidate_id != "recipe2":
            _require_file(
                adapter / "interpolation_provenance.json",
                f"candidate {candidate_id} provenance",
            )
        eval_dir = _require_directory(candidate / "eval", f"candidate {candidate_id} eval")
        for name in EVAL_FILES:
            _require_file(eval_dir / name, f"candidate {candidate_id} eval {name}")

    for name in ADAPTIVE_FILES:
        _require_file(adaptive_root / name, f"adaptive Day-14 {name}")
    _require_file(day14_gold, "locked Day-14 gold")

    identities: dict[str, dict[str, str]] = {}
    for name, adapter in adapters.items():
        config = _require_file(adapter / "adapter_config.json", f"{name} adapter config")
        model = _require_file(adapter / "adapter_model.safetensors", f"{name} adapter weights")
        identities[name] = {
            "adapter_model_sha256": sha256(model),
            "adapter_config_sha256": sha256(config),
        }

    candidate_rule = policy.get("candidate_rule")
    final_decision = policy.get("final_decision")
    primary = (
        final_decision.get("primary_candidate")
        if isinstance(final_decision, dict)
        else None
    )
    expected_models = {
        "v2": candidate_rule.get("v2_adapter_model_sha256")
        if isinstance(candidate_rule, dict)
        else None,
        "refresh1": candidate_rule.get("refresh1_adapter_model_sha256")
        if isinstance(candidate_rule, dict)
        else None,
        "recipe2": primary.get("adapter_model_sha256")
        if isinstance(primary, dict)
        else None,
    }
    for name, expected in expected_models.items():
        if identities[name]["adapter_model_sha256"] != expected:
            raise BundleAssemblyError(
                f"{name} adapter weights differ from the sealed selection policy"
            )
    locked_day14 = (
        policy.get("validation_artifacts", {})
        .get("input_sha256", {})
        .get("day14_forbidden_cases")
    )
    if sha256(day14_gold) != locked_day14:
        raise BundleAssemblyError("Day-14 gold differs from the exclusion-locked suite")
    return policy, selection, identities


def assemble_bundle(
    *,
    profile: str,
    recipe2_root: Path,
    fresh_root: Path,
    candidates_root: Path,
    adaptive_day14_root: Path,
    day14_gold: Path,
    v2_adapter: Path,
    refresh1_adapter: Path,
    out_root: Path,
) -> dict[str, Any]:
    if profile not in {CORRECTED_FULL_PROFILE, CORRECTED_THIN_PROFILE}:
        raise BundleAssemblyError(f"unsupported corrected bundle profile: {profile}")
    recipe2_root = _require_directory(recipe2_root, "recipe2 root")
    fresh_root = _require_directory(fresh_root, "fresh384 root")
    candidates_root = _require_directory(candidates_root, "candidate root")
    adaptive_day14_root = _require_directory(
        adaptive_day14_root, "adaptive Day-14 root"
    )
    day14_gold = _require_file(day14_gold, "locked Day-14 gold")
    adapters = {
        "v2": _require_directory(v2_adapter, "v2 adapter"),
        "refresh1": _require_directory(refresh1_adapter, "refresh1 adapter"),
        "recipe2": _require_directory(
            recipe2_root / "lora-refresh-selected", "recipe2 selected adapter"
        ),
    }
    requested_out_root = out_root.expanduser()
    # Check the caller's path before resolving it.  ``Path.resolve()`` follows
    # a dangling final symlink and would otherwise make that already-existing
    # directory entry look like a fresh destination at the symlink target.
    if requested_out_root.exists() or requested_out_root.is_symlink():
        raise BundleAssemblyError(
            f"bundle destination must not already exist: {requested_out_root}"
        )
    out_root = requested_out_root.resolve()
    _assert_disjoint_destination(
        out_root,
        (
            recipe2_root,
            fresh_root,
            candidates_root,
            adaptive_day14_root,
            day14_gold,
            *adapters.values(),
        ),
    )
    _, selection, identities = _validate_source_inventory(
        profile=profile,
        recipe2_root=recipe2_root,
        fresh_root=fresh_root,
        candidates_root=candidates_root,
        adaptive_root=adaptive_day14_root,
        day14_gold=day14_gold,
        adapters=adapters,
    )
    selected_id = str(selection["selected_candidate_id"])
    stats = {"copied_files": 0, "hardlinked_files": 0}

    out_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_root.mkdir()
    except FileExistsError as error:
        raise BundleAssemblyError(
            f"bundle destination was claimed concurrently: {out_root}"
        ) from error

    try:
        recipe_destination = out_root / "recipe2"
        recipe_destination.mkdir()
        if profile == CORRECTED_FULL_PROFILE:
            _copy_tree(
                recipe2_root / "lora-refresh",
                recipe_destination / "lora-refresh",
                prefer_hardlink=True,
                stats=stats,
            )
            _copy_tree(
                recipe2_root / "lora-refresh-selected",
                recipe_destination / "lora-refresh-selected",
                prefer_hardlink=True,
                stats=stats,
            )
        else:
            thin_output = recipe_destination / "lora-refresh"
            thin_output.mkdir()
            for name in THIN_RECIPE_ROOT_FILES:
                _copy_file(
                    recipe2_root / "lora-refresh" / name,
                    thin_output / name,
                    prefer_hardlink=False,
                    stats=stats,
                )
            for step in RECIPE_CHECKPOINT_STEPS:
                checkpoint = thin_output / f"checkpoint-{step}"
                checkpoint.mkdir()
                _copy_file(
                    recipe2_root
                    / "lora-refresh"
                    / f"checkpoint-{step}"
                    / "trainer_state.json",
                    checkpoint / "trainer_state.json",
                    prefer_hardlink=False,
                    stats=stats,
                )

        recipe_metadata = recipe_destination / "metadata"
        recipe_metadata.mkdir()
        for name in RECIPE_METADATA_FILES:
            _copy_file(
                recipe2_root / name,
                recipe_metadata / name,
                prefer_hardlink=False,
                stats=stats,
            )
        recipe_selection = _load_json(
            recipe2_root / "refresh_selection.json", "recipe2 selection"
        )
        selected_step = recipe_selection.get("selected_step")
        if isinstance(selected_step, bool) or not isinstance(selected_step, int):
            raise BundleAssemblyError("recipe2 selection has no integer selected_step")
        if profile == CORRECTED_THIN_PROFILE:
            _write_json(
                recipe_metadata / "selected_adapter_identity.json",
                {
                    "schema_version": 1,
                    "selected_step": selected_step,
                    **identities["recipe2"],
                },
            )

        fresh_destination = out_root / "fresh384"
        fresh_destination.mkdir()
        fresh_mapping = {
            "cases.jsonl": "cases.jsonl",
            "gold.jsonl": "gold.jsonl",
            "audit.json": "leakage_audit.json",
            "policy.json": "selection_policy.json",
            "recipe2_cross_audit.json": "recipe2_cross_audit.json",
        }
        for source_name, destination_name in fresh_mapping.items():
            _copy_file(
                fresh_root / source_name,
                fresh_destination / destination_name,
                prefer_hardlink=False,
                stats=stats,
            )

        selection_destination = out_root / "selection"
        selection_destination.mkdir()
        for name in ("comparison.json", "selection.json"):
            _copy_file(
                fresh_root / name,
                selection_destination / name,
                prefer_hardlink=False,
                stats=stats,
            )
        endpoint_destination = selection_destination / "endpoint_configs"
        endpoint_destination.mkdir()
        for name, adapter in adapters.items():
            _copy_file(
                adapter / "adapter_config.json",
                endpoint_destination / f"{name}.adapter_config.json",
                prefer_hardlink=False,
                stats=stats,
            )
        _write_json(
            selection_destination / "adapter_identities.json",
            {"schema_version": 1, **identities},
        )

        candidate_destination = out_root / "candidates"
        candidate_destination.mkdir()
        for candidate_id in CANDIDATE_IDS:
            source_candidate = candidates_root / candidate_id
            destination_candidate = candidate_destination / candidate_id
            destination_candidate.mkdir()
            keep_weight = (
                profile == CORRECTED_FULL_PROFILE or candidate_id == selected_id
            )
            _copy_tree(
                source_candidate / "adapter",
                destination_candidate / "adapter",
                prefer_hardlink=True,
                stats=stats,
                include=lambda relative, keep_weight=keep_weight: (
                    keep_weight or relative.name != "adapter_model.safetensors"
                ),
            )
            eval_destination = destination_candidate / "eval"
            eval_destination.mkdir()
            for name in EVAL_FILES:
                _copy_file(
                    source_candidate / "eval" / name,
                    eval_destination / name,
                    prefer_hardlink=False,
                    stats=stats,
                )

        adaptive_destination = out_root / "adaptive_day14"
        adaptive_destination.mkdir()
        _copy_file(
            day14_gold,
            adaptive_destination / "gold.jsonl",
            prefer_hardlink=False,
            stats=stats,
        )
        for name in ADAPTIVE_FILES:
            _copy_file(
                adaptive_day14_root / name,
                adaptive_destination / name,
                prefer_hardlink=False,
                stats=stats,
            )

        manifest = out_root / "checksums.sha256"
        manifest_files = write_manifest(out_root, manifest)
        manifest_errors = verify_manifest(out_root, manifest)
        if manifest_errors:
            raise BundleAssemblyError(
                "new checksum manifest failed verification: " + "; ".join(manifest_errors)
            )
        audit = (
            audit_corrected_full_bundle(out_root)
            if profile == CORRECTED_FULL_PROFILE
            else audit_corrected_thin_bundle(out_root)
        )
        if not audit["ok"]:
            preview = "; ".join(str(value) for value in audit["errors"][:10])
            raise BundleAssemblyError(f"assembled bundle failed semantic audit: {preview}")
        return {
            "profile": profile,
            "out_root": str(out_root),
            "selected_candidate_id": selected_id,
            "manifest_files": manifest_files,
            **stats,
        }
    except BaseException:
        shutil.rmtree(out_root)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(CORRECTED_FULL_PROFILE, CORRECTED_THIN_PROFILE),
        required=True,
    )
    parser.add_argument("--recipe2-root", type=Path, required=True)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--adaptive-day14-root", type=Path, required=True)
    parser.add_argument("--day14-gold", type=Path, required=True)
    parser.add_argument("--v2-adapter", type=Path, required=True)
    parser.add_argument("--refresh1-adapter", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = assemble_bundle(
            profile=args.profile,
            recipe2_root=args.recipe2_root,
            fresh_root=args.fresh_root,
            candidates_root=args.candidates_root,
            adaptive_day14_root=args.adaptive_day14_root,
            day14_gold=args.day14_gold,
            v2_adapter=args.v2_adapter,
            refresh1_adapter=args.refresh1_adapter,
            out_root=args.out_root,
        )
    except (BundleAssemblyError, OSError, ValueError) as error:
        print(f"bundle assembly aborted: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create one exact delta-space interpolation of two compatible LoRA adapters.

For two rank-``r`` adapters with the same LoRA scaling ``s = alpha / r``, this
utility writes a rank-``2r`` adapter whose effective update is

``s * ((1 - lambda_b) * B_a @ A_a + lambda_b * B_b @ A_b)``.

It does so without loading the base model by concatenating the factors:

``A_out = [A_a; A_b]`` and
``B_out = [(1 - lambda_b) * B_a, lambda_b * B_b]``.

The output LoRA alpha is doubled together with the rank, preserving the input
scaling exactly.  One coefficient is produced per invocation so that a model
selection grid consists of independently hashed, auditable artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
PROVENANCE_FILE = "interpolation_provenance.json"
METHOD = "exact_delta_space_rank_concat"

# These files can affect tokenizer/processor behaviour at inference time.  We
# deliberately do not copy Trainer state, optimizer state, or README files.
INFERENCE_METADATA_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
    "image_processor_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    "vocab.txt",
)


class InterpolationError(ValueError):
    """Raised when an adapter cannot be combined safely and reproducibly."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise InterpolationError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InterpolationError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise InterpolationError(f"{path}: expected a JSON object")
    return value


def _require_regular_source(path: Path, label: str) -> None:
    if path.is_symlink():
        raise InterpolationError(f"{label} must not be a symlink: {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise InterpolationError(f"{label} is missing or empty: {path}")


def _validate_coefficient(lambda_b: float) -> float:
    if isinstance(lambda_b, bool) or not isinstance(lambda_b, (int, float)):
        raise InterpolationError("lambda_b must be a finite number in [0, 1]")
    value = float(lambda_b)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise InterpolationError("lambda_b must be a finite number in [0, 1]")
    return value


def _validate_pure_lora_config(
    config: dict[str, Any], path: Path
) -> tuple[int, int | float]:
    if config.get("peft_type") != "LORA":
        raise InterpolationError(f"{path}: peft_type must be 'LORA'")
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise InterpolationError(f"{path}: r must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise InterpolationError(f"{path}: lora_alpha must be a positive finite number")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise InterpolationError(f"{path}: lora_alpha must be a positive finite number")
    if (
        not isinstance(config.get("base_model_name_or_path"), str)
        or not config["base_model_name_or_path"].strip()
    ):
        raise InterpolationError(f"{path}: base_model_name_or_path must be non-empty")
    targets = config.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(item, str) and item for item in targets)
    ):
        raise InterpolationError(
            f"{path}: target_modules must be a non-empty string list"
        )
    if len(set(targets)) != len(targets):
        raise InterpolationError(f"{path}: target_modules contains duplicates")

    unsupported_truthy = (
        "use_dora",
        "use_rslora",
        "use_qalora",
        "lora_bias",
        "ensure_weight_tying",
    )
    for field in unsupported_truthy:
        if config.get(field):
            raise InterpolationError(
                f"{path}: unsupported pure-LoRA option {field}=true"
            )
    if config.get("bias", "none") != "none":
        raise InterpolationError(f"{path}: only bias='none' is supported")

    unsupported_payloads = (
        "alpha_pattern",
        "rank_pattern",
        "modules_to_save",
        "target_parameters",
        "trainable_token_indices",
        "layer_replication",
    )
    for field in unsupported_payloads:
        if config.get(field) not in (None, {}, []):
            raise InterpolationError(f"{path}: unsupported non-empty field {field}")
    return rank, alpha


def _canonical_compatibility_config(config: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize only fields whose ordering is semantically irrelevant."""

    canonical = dict(config)
    canonical["target_modules"] = sorted(config["target_modules"])
    return canonical


def _load_weights(path: Path) -> dict[str, np.ndarray]:
    try:
        from safetensors.numpy import load_file
    except ModuleNotFoundError as error:  # pragma: no cover - project dependency guard
        raise RuntimeError(
            "safetensors is required to interpolate LoRA adapters"
        ) from error
    try:
        return load_file(path)
    except Exception as error:
        raise InterpolationError(
            f"cannot load safetensors weights {path}: {error}"
        ) from error


def _paired_factors(
    weights: dict[str, np.ndarray],
    *,
    rank: int,
    target_modules: set[str],
    label: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    factor_map: dict[str, dict[str, np.ndarray]] = {}
    for key, tensor in weights.items():
        if key.endswith(".lora_A.weight"):
            root = key[: -len(".lora_A.weight")]
            factor = "A"
        elif key.endswith(".lora_B.weight"):
            root = key[: -len(".lora_B.weight")]
            factor = "B"
        else:
            raise InterpolationError(f"{label}: unsupported non-LoRA tensor {key!r}")
        if factor in factor_map.setdefault(root, {}):
            raise InterpolationError(f"{label}: duplicate {factor} factor for {root!r}")
        factor_map[root][factor] = tensor

    if not factor_map:
        raise InterpolationError(f"{label}: adapter contains no LoRA factors")
    paired: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for root, factors in factor_map.items():
        if set(factors) != {"A", "B"}:
            raise InterpolationError(f"{label}: unpaired LoRA factors for {root!r}")
        module_name = root.rsplit(".", 1)[-1]
        if module_name not in target_modules:
            raise InterpolationError(
                f"{label}: tensor module {module_name!r} is absent from target_modules"
            )
        factor_a, factor_b = factors["A"], factors["B"]
        if factor_a.ndim != 2 or factor_b.ndim != 2:
            raise InterpolationError(
                f"{label}: LoRA factors for {root!r} must be matrices"
            )
        if factor_a.shape[0] != rank or factor_b.shape[1] != rank:
            raise InterpolationError(
                f"{label}: rank mismatch for {root!r}: A={factor_a.shape}, B={factor_b.shape}, r={rank}"
            )
        if factor_a.dtype != np.float32 or factor_b.dtype != np.float32:
            raise InterpolationError(
                f"{label}: only float32 factors are supported; {root!r} has "
                f"A={factor_a.dtype}, B={factor_b.dtype}"
            )
        if not np.isfinite(factor_a).all() or not np.isfinite(factor_b).all():
            raise InterpolationError(f"{label}: non-finite LoRA values for {root!r}")
        paired[root] = (factor_a, factor_b)
    return paired


def _plan_metadata_copy(adapter_a: Path, adapter_b: Path) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    for name in INFERENCE_METADATA_FILES:
        candidates = [root / name for root in (adapter_a, adapter_b)]
        present = [path for path in candidates if path.is_file()]
        dangling = [
            path for path in candidates if path.is_symlink() and not path.is_file()
        ]
        if dangling:
            raise InterpolationError(
                f"broken inference metadata symlink: {dangling[0]}"
            )
        if not present:
            continue
        hashes = {_sha256(path) for path in present}
        if len(hashes) != 1:
            raise InterpolationError(
                f"inference metadata differs between adapters and cannot be chosen safely: {name}"
            )
        # Prefer adapter B when both have the same file because it is normally
        # the later checkpoint, while byte identity prevents silent drift.
        source = candidates[1] if candidates[1].is_file() else candidates[0]
        planned.append(
            {
                "name": name,
                "source": source,
                "source_adapter": "both"
                if len(present) == 2
                else ("b" if source == candidates[1] else "a"),
                "sha256": next(iter(hashes)),
            }
        )
    return planned


def _validate_output_location(
    adapter_a: Path, adapter_b: Path, output_dir: Path
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise InterpolationError(
            f"output directory must not already exist: {output_dir}"
        )
    for source in (adapter_a, adapter_b):
        if output_dir == source or output_dir.is_relative_to(source):
            raise InterpolationError(
                f"output directory must not be inside a source adapter: {output_dir}"
            )


def interpolate_adapters(
    adapter_a: Path,
    adapter_b: Path,
    output_dir: Path,
    lambda_b: float,
) -> dict[str, Any]:
    """Write one exact delta-space interpolation and return its provenance."""

    lambda_b = _validate_coefficient(lambda_b)
    adapter_a = adapter_a.expanduser().resolve()
    adapter_b = adapter_b.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if adapter_a == adapter_b:
        raise InterpolationError("adapter A and adapter B must be distinct directories")
    for adapter, label in ((adapter_a, "adapter A"), (adapter_b, "adapter B")):
        if not adapter.is_dir():
            raise InterpolationError(f"{label} directory is missing: {adapter}")
        _require_regular_source(adapter / ADAPTER_CONFIG, f"{label} config")
        _require_regular_source(adapter / ADAPTER_WEIGHTS, f"{label} weights")
    _validate_output_location(adapter_a, adapter_b, output_dir)

    config_a_path = adapter_a / ADAPTER_CONFIG
    config_b_path = adapter_b / ADAPTER_CONFIG
    weights_a_path = adapter_a / ADAPTER_WEIGHTS
    weights_b_path = adapter_b / ADAPTER_WEIGHTS
    source_hashes = {
        "adapter_a": {
            "adapter_config_sha256": _sha256(config_a_path),
            "adapter_model_sha256": _sha256(weights_a_path),
        },
        "adapter_b": {
            "adapter_config_sha256": _sha256(config_b_path),
            "adapter_model_sha256": _sha256(weights_b_path),
        },
    }
    config_a = _load_json_object(config_a_path)
    config_b = _load_json_object(config_b_path)
    rank, alpha = _validate_pure_lora_config(config_a, config_a_path)
    _validate_pure_lora_config(config_b, config_b_path)
    canonical_a = _canonical_compatibility_config(config_a)
    canonical_b = _canonical_compatibility_config(config_b)
    if canonical_a != canonical_b:
        differing = sorted(
            key
            for key in set(canonical_a) | set(canonical_b)
            if canonical_a.get(key) != canonical_b.get(key)
        )
        raise InterpolationError(
            "adapter configs must be semantically identical; differing fields: "
            + ", ".join(differing)
        )
    targets = set(config_a["target_modules"])

    weights_a = _load_weights(weights_a_path)
    weights_b = _load_weights(weights_b_path)
    if set(weights_a) != set(weights_b):
        missing_from_b = sorted(set(weights_a) - set(weights_b))
        missing_from_a = sorted(set(weights_b) - set(weights_a))
        raise InterpolationError(
            "adapter tensor keys differ: "
            f"missing_from_b={missing_from_b[:3]}, missing_from_a={missing_from_a[:3]}"
        )
    factors_a = _paired_factors(
        weights_a, rank=rank, target_modules=targets, label="adapter A"
    )
    factors_b = _paired_factors(
        weights_b, rank=rank, target_modules=targets, label="adapter B"
    )
    if set(factors_a) != set(factors_b):
        raise InterpolationError("adapter LoRA module sets differ")

    output_weights: dict[str, np.ndarray] = {}
    weight_a = 1.0 - lambda_b
    for root in sorted(factors_a):
        factor_a_a, factor_b_a = factors_a[root]
        factor_a_b, factor_b_b = factors_b[root]
        if factor_a_a.shape != factor_a_b.shape or factor_b_a.shape != factor_b_b.shape:
            raise InterpolationError(
                f"adapter factor shapes differ for {root!r}: "
                f"A={factor_a_a.shape}/{factor_a_b.shape}, "
                f"B={factor_b_a.shape}/{factor_b_b.shape}"
            )
        concatenated_a = np.concatenate((factor_a_a, factor_a_b), axis=0)
        if lambda_b == 0.0:
            scaled_b_a = factor_b_a.copy()
            scaled_b_b = np.zeros_like(factor_b_b)
        elif lambda_b == 1.0:
            scaled_b_a = np.zeros_like(factor_b_a)
            scaled_b_b = factor_b_b.copy()
        else:
            scaled_b_a = np.multiply(factor_b_a, np.float32(weight_a), dtype=np.float32)
            scaled_b_b = np.multiply(factor_b_b, np.float32(lambda_b), dtype=np.float32)
        concatenated_b = np.concatenate((scaled_b_a, scaled_b_b), axis=1)
        output_weights[f"{root}.lora_A.weight"] = np.ascontiguousarray(concatenated_a)
        output_weights[f"{root}.lora_B.weight"] = np.ascontiguousarray(concatenated_b)

    output_config = dict(config_a)
    output_config["r"] = rank * 2
    output_config["lora_alpha"] = alpha * 2
    output_config["target_modules"] = sorted(config_a["target_modules"])
    metadata_plan = _plan_metadata_copy(adapter_a, adapter_b)

    # Detect source mutation between validation/loading and artifact creation.
    # Without this check, provenance could identify bytes other than those
    # actually used to construct the output tensors.
    current_hashes = {
        "adapter_a": {
            "adapter_config_sha256": _sha256(config_a_path),
            "adapter_model_sha256": _sha256(weights_a_path),
        },
        "adapter_b": {
            "adapter_config_sha256": _sha256(config_b_path),
            "adapter_model_sha256": _sha256(weights_b_path),
        },
    }
    if current_hashes != source_hashes:
        raise InterpolationError("source adapter changed while it was being read")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # mkdir is the atomic fresh-directory claim.  Never merge into or replace
    # an existing directory, even if it is empty.
    output_dir.mkdir()
    try:
        output_config_path = output_dir / ADAPTER_CONFIG
        output_weights_path = output_dir / ADAPTER_WEIGHTS
        output_config_path.write_text(
            json.dumps(output_config, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        try:
            from safetensors.numpy import save_file
        except (
            ModuleNotFoundError
        ) as error:  # pragma: no cover - project dependency guard
            raise RuntimeError(
                "safetensors is required to interpolate LoRA adapters"
            ) from error
        save_file(
            output_weights,
            output_weights_path,
            metadata={"format": "pt", "interpolation_method": METHOD},
        )

        copied_metadata: list[dict[str, Any]] = []
        for item in metadata_plan:
            destination = output_dir / item["name"]
            shutil.copyfile(item["source"], destination, follow_symlinks=True)
            if destination.is_symlink() or not destination.is_file():
                raise InterpolationError(
                    f"metadata output is not a regular file: {destination}"
                )
            copied_hash = _sha256(destination)
            if copied_hash != item["sha256"]:
                raise InterpolationError(
                    f"metadata copy checksum mismatch: {destination}"
                )
            copied_metadata.append(
                {
                    "name": item["name"],
                    "sha256": copied_hash,
                    "source_adapter": item["source_adapter"],
                }
            )

        provenance = {
            "schema_version": 1,
            "method": METHOD,
            "equation": "delta_out=(1-lambda_b)*delta_a+lambda_b*delta_b",
            "coefficient": {
                "lambda_b": lambda_b,
                "adapter_a_weight": weight_a,
                "adapter_b_weight": lambda_b,
            },
            "sources": {
                "adapter_a": {
                    "path": str(adapter_a),
                    **source_hashes["adapter_a"],
                },
                "adapter_b": {
                    "path": str(adapter_b),
                    **source_hashes["adapter_b"],
                },
            },
            "lora": {
                "input_rank": rank,
                "input_lora_alpha": alpha,
                "input_scaling": float(alpha) / rank,
                "output_rank": rank * 2,
                "output_lora_alpha": alpha * 2,
                "output_scaling": float(alpha * 2) / (rank * 2),
                "module_count": len(factors_a),
                "tensor_count": len(output_weights),
                "dtype": "float32",
            },
            "output": {
                "path": str(output_dir),
                "adapter_config_sha256": _sha256(output_config_path),
                "adapter_model_sha256": _sha256(output_weights_path),
                "copied_inference_metadata": copied_metadata,
            },
        }
        (output_dir / PROVENANCE_FILE).write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if any(path.is_symlink() for path in output_dir.iterdir()):
            raise InterpolationError("output adapter must not contain symlinks")
        return provenance
    except Exception:
        # This directory was atomically claimed by this invocation and was
        # known not to exist beforehand, so cleanup cannot remove user data.
        shutil.rmtree(output_dir)
        raise


def _unit_interval(text: str) -> float:
    try:
        value = float(text)
        return _validate_coefficient(value)
    except (ValueError, InterpolationError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-a", type=Path, required=True, help="lambda=0 endpoint"
    )
    parser.add_argument(
        "--adapter-b", type=Path, required=True, help="lambda=1 endpoint"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="fresh output directory"
    )
    coefficient = parser.add_mutually_exclusive_group(required=True)
    coefficient.add_argument(
        "--lambda-b",
        "--lambda-refresh",
        dest="lambda_b",
        type=_unit_interval,
        help="weight of adapter B; produce one grid value per invocation",
    )
    coefficient.add_argument(
        "--alpha",
        dest="interpolation_alpha",
        type=_unit_interval,
        help="alias for the interpolation coefficient (not LoRA alpha)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    lambda_b = args.lambda_b if args.lambda_b is not None else args.interpolation_alpha
    result = interpolate_adapters(
        args.adapter_a, args.adapter_b, args.out_dir, lambda_b
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

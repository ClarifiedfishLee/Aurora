import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from scripts.interpolate_lora_adapters import (
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    InterpolationError,
    build_arg_parser,
    interpolate_adapters,
)

ROOT = "base_model.model.model.language_model.layers.0.self_attn.q_proj"
A_KEY = f"{ROOT}.lora_A.weight"
B_KEY = f"{ROOT}.lora_B.weight"


def _config(**updates) -> dict:
    value = {
        "base_model_name_or_path": "/models/Qwen3-VL-8B-Instruct",
        "bias": "none",
        "ensure_weight_tying": False,
        "inference_mode": True,
        "lora_alpha": 4,
        "lora_bias": False,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 2,
        "rank_pattern": {},
        "alpha_pattern": {},
        "target_modules": ["q_proj"],
        "target_parameters": None,
        "task_type": "CAUSAL_LM",
        "trainable_token_indices": None,
        "layer_replication": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }
    value.update(updates)
    return value


def _factors(which: str) -> dict[str, np.ndarray]:
    if which == "a":
        factor_a = np.array([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]], dtype=np.float32)
        factor_b = np.array([[0.25, -1.0], [2.0, 0.5]], dtype=np.float32)
    else:
        factor_a = np.array([[-2.0, 0.25, 1.5], [1.0, 3.0, -0.5]], dtype=np.float32)
        factor_b = np.array([[1.5, 0.75], [-0.25, 2.0]], dtype=np.float32)
    return {A_KEY: factor_a, B_KEY: factor_b}


def _write_adapter(
    path: Path,
    which: str,
    *,
    config: dict | None = None,
    weights: dict[str, np.ndarray] | None = None,
) -> None:
    path.mkdir()
    (path / ADAPTER_CONFIG).write_text(
        json.dumps(config or _config(), indent=2) + "\n", encoding="utf-8"
    )
    save_file(
        weights or _factors(which), path / ADAPTER_WEIGHTS, metadata={"format": "pt"}
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _effective_delta(adapter: Path) -> np.ndarray:
    config = json.loads((adapter / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    weights = load_file(adapter / ADAPTER_WEIGHTS)
    return (config["lora_alpha"] / config["r"]) * (weights[B_KEY] @ weights[A_KEY])


class InterpolateLoraAdaptersTest(unittest.TestCase):
    def _parents(self, root: Path) -> tuple[Path, Path]:
        adapter_a = root / "adapter-a"
        adapter_b = root / "adapter-b"
        _write_adapter(adapter_a, "a")
        _write_adapter(adapter_b, "b")
        return adapter_a, adapter_b

    def test_endpoints_have_exact_effective_parent_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter_a, adapter_b = self._parents(root)
            expected = {
                0.0: _effective_delta(adapter_a),
                1.0: _effective_delta(adapter_b),
            }

            for coefficient in (0.0, 1.0):
                with self.subTest(coefficient=coefficient):
                    output = root / f"out-{coefficient}"
                    interpolate_adapters(adapter_a, adapter_b, output, coefficient)
                    np.testing.assert_allclose(
                        _effective_delta(output),
                        expected[coefficient],
                        rtol=1e-7,
                        atol=1e-7,
                    )

                    weights = load_file(output / ADAPTER_WEIGHTS)
                    rank = 2
                    if coefficient == 0.0:
                        np.testing.assert_array_equal(
                            weights[B_KEY][:, :rank], _factors("a")[B_KEY]
                        )
                        np.testing.assert_array_equal(weights[B_KEY][:, rank:], 0.0)
                    else:
                        np.testing.assert_array_equal(weights[B_KEY][:, :rank], 0.0)
                        np.testing.assert_array_equal(
                            weights[B_KEY][:, rank:], _factors("b")[B_KEY]
                        )

    def test_midpoint_is_exact_delta_space_average_and_doubles_rank_and_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter_a, adapter_b = self._parents(root)
            output = root / "midpoint"

            provenance = interpolate_adapters(adapter_a, adapter_b, output, 0.5)

            expected = 0.5 * _effective_delta(adapter_a) + 0.5 * _effective_delta(
                adapter_b
            )
            np.testing.assert_allclose(
                _effective_delta(output), expected, rtol=1e-7, atol=1e-7
            )
            output_config = json.loads(
                (output / ADAPTER_CONFIG).read_text(encoding="utf-8")
            )
            self.assertEqual(output_config["r"], 4)
            self.assertEqual(output_config["lora_alpha"], 8)
            self.assertEqual(provenance["lora"]["input_scaling"], 2.0)
            self.assertEqual(provenance["lora"]["output_scaling"], 2.0)
            self.assertEqual(provenance["method"], "exact_delta_space_rank_concat")

    def test_writes_hashed_provenance_and_dereferences_inference_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter_a, adapter_b = self._parents(root)
            tokenizer = '{"tokenizer_class":"Qwen2TokenizerFast"}\n'
            for adapter in (adapter_a, adapter_b):
                (adapter / "tokenizer_config.json").write_text(
                    tokenizer, encoding="utf-8"
                )
            metadata_target = root / "template-source.jinja"
            metadata_target.write_text("{{ messages }}\n", encoding="utf-8")
            (adapter_b / "chat_template.jinja").symlink_to(metadata_target)
            output = root / "out"

            source_hashes = {
                "a": _sha256(adapter_a / ADAPTER_WEIGHTS),
                "b": _sha256(adapter_b / ADAPTER_WEIGHTS),
            }
            returned = interpolate_adapters(adapter_a, adapter_b, output, 0.25)
            stored = json.loads((output / "interpolation_provenance.json").read_text())

            self.assertEqual(stored, returned)
            self.assertEqual(
                stored["sources"]["adapter_a"]["adapter_model_sha256"],
                source_hashes["a"],
            )
            self.assertEqual(
                stored["sources"]["adapter_b"]["adapter_model_sha256"],
                source_hashes["b"],
            )
            self.assertEqual(
                stored["output"]["adapter_model_sha256"],
                _sha256(output / ADAPTER_WEIGHTS),
            )
            self.assertEqual(
                (output / "chat_template.jinja").read_text(), "{{ messages }}\n"
            )
            self.assertFalse((output / "chat_template.jinja").is_symlink())
            self.assertFalse(any(path.is_symlink() for path in output.iterdir()))

    def test_refuses_existing_or_nested_output_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter_a, adapter_b = self._parents(root)
            source_hashes = (
                _sha256(adapter_a / ADAPTER_WEIGHTS),
                _sha256(adapter_b / ADAPTER_WEIGHTS),
            )
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(InterpolationError, "must not already exist"):
                interpolate_adapters(adapter_a, adapter_b, existing, 0.5)
            with self.assertRaisesRegex(InterpolationError, "must not be inside"):
                interpolate_adapters(adapter_a, adapter_b, adapter_a / "child", 0.5)
            self.assertEqual(
                source_hashes,
                (
                    _sha256(adapter_a / ADAPTER_WEIGHTS),
                    _sha256(adapter_b / ADAPTER_WEIGHTS),
                ),
            )

    def test_rejects_config_or_tensor_incompatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter_a = root / "adapter-a"
            adapter_b = root / "adapter-b"
            _write_adapter(adapter_a, "a")
            _write_adapter(
                adapter_b, "b", config=_config(base_model_name_or_path="/wrong/base")
            )
            with self.assertRaisesRegex(
                InterpolationError, "differing fields.*base_model"
            ):
                interpolate_adapters(
                    adapter_a, adapter_b, root / "config-mismatch", 0.5
                )

            adapter_c = root / "adapter-c"
            _write_adapter(adapter_c, "b", weights={A_KEY: _factors("b")[A_KEY]})
            with self.assertRaisesRegex(InterpolationError, "tensor keys differ"):
                interpolate_adapters(
                    adapter_a, adapter_c, root / "tensor-mismatch", 0.5
                )

    def test_target_module_order_is_compatible_but_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter_a = root / "adapter-a"
            adapter_b = root / "adapter-b"
            _write_adapter(
                adapter_a,
                "a",
                config=_config(target_modules=["q_proj", "k_proj"]),
            )
            _write_adapter(
                adapter_b,
                "b",
                config=_config(target_modules=["k_proj", "q_proj"]),
            )

            output = root / "reordered-output"
            interpolate_adapters(adapter_a, adapter_b, output, 0.5)
            output_config = json.loads(
                (output / ADAPTER_CONFIG).read_text(encoding="utf-8")
            )
            self.assertEqual(output_config["target_modules"], ["k_proj", "q_proj"])

            duplicate_a = root / "duplicate-a"
            duplicate_b = root / "duplicate-b"
            duplicate_config = _config(target_modules=["q_proj", "q_proj"])
            _write_adapter(duplicate_a, "a", config=duplicate_config)
            _write_adapter(duplicate_b, "b", config=duplicate_config)
            with self.assertRaisesRegex(InterpolationError, "contains duplicates"):
                interpolate_adapters(
                    duplicate_a, duplicate_b, root / "duplicate-output", 0.5
                )

    def test_rejects_non_pure_lora_and_invalid_coefficients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter_a = root / "adapter-a"
            adapter_b = root / "adapter-b"
            _write_adapter(adapter_a, "a", config=_config(use_dora=True))
            _write_adapter(adapter_b, "b", config=_config(use_dora=True))
            with self.assertRaisesRegex(InterpolationError, "use_dora"):
                interpolate_adapters(adapter_a, adapter_b, root / "out", 0.5)
            for coefficient in (-0.01, 1.01, float("nan"), float("inf")):
                with (
                    self.subTest(coefficient=coefficient),
                    self.assertRaisesRegex(InterpolationError, "finite number in"),
                ):
                    interpolate_adapters(
                        adapter_a, adapter_b, root / "unused", coefficient
                    )

    def test_cli_accepts_one_lambda_or_alpha_alias_not_both(self):
        parser = build_arg_parser()
        base = ["--adapter-a", "a", "--adapter-b", "b", "--out-dir", "out"]
        self.assertEqual(
            parser.parse_args([*base, "--lambda-b", "0.375"]).lambda_b, 0.375
        )
        self.assertEqual(
            parser.parse_args([*base, "--alpha", "0.625"]).interpolation_alpha, 0.625
        )
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([*base, "--lambda-b", "0.5", "--alpha", "0.5"])


if __name__ == "__main__":
    unittest.main()

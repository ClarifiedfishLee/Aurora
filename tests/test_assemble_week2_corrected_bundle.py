import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.assemble_week2_corrected_bundle as assembler
from scripts.verify_week2_bundle import verify_manifest


def _write(path: Path, value: bytes | str = b"x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_bytes(value)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AssembleCorrectedBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.recipe = self.root / "recipe2"
        self.fresh = self.root / "fresh"
        self.candidates = self.root / "candidates"
        self.adaptive = self.root / "adaptive"
        self.day14 = self.root / "day14.jsonl"
        self.v2 = self.root / "v2-adapter"
        self.refresh1 = self.root / "refresh1-adapter"

        for name, adapter in (
            ("v2", self.v2),
            ("refresh1", self.refresh1),
            ("recipe2", self.recipe / "lora-refresh-selected"),
        ):
            _write(
                adapter / "adapter_config.json",
                json.dumps({"base_model_name_or_path": "/model", "name": name}),
            )
            _write(adapter / "adapter_model.safetensors", f"weights-{name}".encode())

        recipe_output = self.recipe / "lora-refresh"
        for name in assembler.THIN_RECIPE_ROOT_FILES:
            _write(recipe_output / name, "0\n" if name == "exit_code" else "x\n")
        _write(recipe_output / "adapter_config.json", b"recipe-root-config")
        _write(recipe_output / "adapter_model.safetensors", b"recipe-root-weights")
        for step in assembler.RECIPE_CHECKPOINT_STEPS:
            checkpoint = recipe_output / f"checkpoint-{step}"
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "trainer_state.json",
            ):
                _write(checkpoint / name, f"{name}-{step}".encode())

        for name in assembler.RECIPE_METADATA_FILES:
            if name == "refresh_selection.json":
                value = json.dumps({"selected_step": 96})
            else:
                value = "{}\n"
            _write(self.recipe / name, value)

        _write(self.day14, '{"bench_id":"day14"}\n')
        identities = {
            "v2": _digest(self.v2 / "adapter_model.safetensors"),
            "refresh1": _digest(self.refresh1 / "adapter_model.safetensors"),
            "recipe2": _digest(
                self.recipe / "lora-refresh-selected/adapter_model.safetensors"
            ),
        }
        policy = {
            "candidate_rule": {
                "v2_adapter_model_sha256": identities["v2"],
                "refresh1_adapter_model_sha256": identities["refresh1"],
            },
            "final_decision": {
                "primary_candidate": {
                    "adapter_model_sha256": identities["recipe2"]
                }
            },
            "validation_artifacts": {
                "input_sha256": {"day14_forbidden_cases": _digest(self.day14)}
            },
        }
        _write(self.fresh / "policy.json", json.dumps(policy))
        _write(
            self.fresh / "selection.json",
            json.dumps({"selected": True, "selected_candidate_id": "recipe2"}),
        )
        for name in (
            "cases.jsonl",
            "gold.jsonl",
            "audit.json",
            "recipe2_cross_audit.json",
            "comparison.json",
        ):
            _write(self.fresh / name, "{}\n")

        for candidate_id in assembler.CANDIDATE_IDS:
            candidate = self.candidates / candidate_id
            _write(candidate / "adapter/adapter_config.json", "{}\n")
            source_weights = (
                self.recipe / "lora-refresh-selected/adapter_model.safetensors"
                if candidate_id == "recipe2"
                else None
            )
            _write(
                candidate / "adapter/adapter_model.safetensors",
                source_weights.read_bytes()
                if source_weights is not None
                else f"candidate-{candidate_id}".encode(),
            )
            if candidate_id != "recipe2":
                _write(candidate / "adapter/interpolation_provenance.json", "{}\n")
            for name in assembler.EVAL_FILES:
                _write(candidate / "eval" / name, "{}\n")
            _write(candidate / "eval/keyframes/frame-0001.png", b"not portable")

        for name in assembler.ADAPTIVE_FILES:
            _write(self.adaptive / name, "{}\n")
        _write(self.adaptive / "keyframes/frame-0001.png", b"not portable")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _assemble(self, profile: str, destination: Path) -> dict:
        audit = {"ok": True, "errors": [], "observations": {}}
        with (
            patch.object(assembler, "HARDLINK_MIN_BYTES", 1),
            patch.object(
                assembler, "audit_corrected_full_bundle", return_value=audit
            ),
            patch.object(
                assembler, "audit_corrected_thin_bundle", return_value=audit
            ),
        ):
            return assembler.assemble_bundle(
                profile=profile,
                recipe2_root=self.recipe,
                fresh_root=self.fresh,
                candidates_root=self.candidates,
                adaptive_day14_root=self.adaptive,
                day14_gold=self.day14,
                v2_adapter=self.v2,
                refresh1_adapter=self.refresh1,
                out_root=destination,
            )

    def test_full_maps_paths_hashes_identities_and_hardlinks_weights(self) -> None:
        destination = self.root / "bundle-full"
        result = self._assemble("corrected-full", destination)

        self.assertEqual(result["selected_candidate_id"], "recipe2")
        self.assertGreater(result["hardlinked_files"], 0)
        self.assertEqual(
            (destination / "fresh384/leakage_audit.json").read_bytes(),
            (self.fresh / "audit.json").read_bytes(),
        )
        self.assertEqual(
            (destination / "fresh384/selection_policy.json").read_bytes(),
            (self.fresh / "policy.json").read_bytes(),
        )
        self.assertEqual(
            (destination / "adaptive_day14/gold.jsonl").read_bytes(),
            self.day14.read_bytes(),
        )
        self.assertFalse(
            (destination / "candidates/lambda_0000/eval/keyframes").exists()
        )
        source_weight = self.candidates / "lambda_0000/adapter/adapter_model.safetensors"
        copied_weight = (
            destination / "candidates/lambda_0000/adapter/adapter_model.safetensors"
        )
        self.assertEqual(source_weight.stat().st_ino, copied_weight.stat().st_ino)
        identities = json.loads(
            (destination / "selection/adapter_identities.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(identities["v2"]["adapter_model_sha256"], _digest(self.v2 / "adapter_model.safetensors"))
        self.assertEqual(
            verify_manifest(destination, destination / "checksums.sha256"), []
        )

    def test_thin_keeps_only_selected_candidate_weight_and_identity(self) -> None:
        destination = self.root / "bundle-thin"
        self._assemble("corrected-thin", destination)

        self.assertTrue(
            (destination / "candidates/recipe2/adapter/adapter_model.safetensors").is_file()
        )
        for candidate_id in assembler.CANDIDATE_IDS:
            if candidate_id != "recipe2":
                self.assertFalse(
                    (
                        destination
                        / f"candidates/{candidate_id}/adapter/adapter_model.safetensors"
                    ).exists()
                )
        self.assertFalse((destination / "recipe2/lora-refresh-selected").exists())
        self.assertFalse(
            (destination / "recipe2/lora-refresh/adapter_model.safetensors").exists()
        )
        for step in assembler.RECIPE_CHECKPOINT_STEPS:
            checkpoint = destination / f"recipe2/lora-refresh/checkpoint-{step}"
            self.assertEqual(
                sorted(path.name for path in checkpoint.iterdir()),
                ["trainer_state.json"],
            )
        identity = json.loads(
            (
                destination
                / "recipe2/metadata/selected_adapter_identity.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(identity["schema_version"], 1)
        self.assertEqual(identity["selected_step"], 96)
        self.assertEqual(
            verify_manifest(destination, destination / "checksums.sha256"), []
        )

    def test_existing_destination_is_preserved(self) -> None:
        destination = self.root / "existing"
        destination.mkdir()
        sentinel = destination / "sentinel.txt"
        sentinel.write_text("keep\n", encoding="utf-8")

        with self.assertRaisesRegex(
            assembler.BundleAssemblyError, "must not already exist"
        ):
            self._assemble("corrected-full", destination)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

    def test_dangling_destination_symlink_is_preserved_and_rejected(self) -> None:
        target = self.root / "missing-target"
        destination = self.root / "dangling-output"
        destination.symlink_to(target, target_is_directory=True)

        with self.assertRaisesRegex(
            assembler.BundleAssemblyError, "must not already exist"
        ):
            self._assemble("corrected-full", destination)

        self.assertTrue(destination.is_symlink())
        self.assertFalse(target.exists())

    def test_failed_semantic_audit_cleans_only_claimed_destination(self) -> None:
        destination = self.root / "failed"
        audit = {"ok": False, "errors": ["fixture failure"], "observations": {}}
        with patch.object(
            assembler, "audit_corrected_full_bundle", return_value=audit
        ), self.assertRaisesRegex(
            assembler.BundleAssemblyError, "failed semantic audit"
        ):
            assembler.assemble_bundle(
                profile="corrected-full",
                recipe2_root=self.recipe,
                fresh_root=self.fresh,
                candidates_root=self.candidates,
                adaptive_day14_root=self.adaptive,
                day14_gold=self.day14,
                v2_adapter=self.v2,
                refresh1_adapter=self.refresh1,
                out_root=destination,
            )
        self.assertFalse(destination.exists())
        self.assertTrue((self.recipe / "refresh_selection.json").is_file())


if __name__ == "__main__":
    unittest.main()

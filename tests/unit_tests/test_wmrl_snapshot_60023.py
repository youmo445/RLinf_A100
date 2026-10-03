"""Dependency-light consistency tests for the archived 60023 WM-RL snapshot."""

import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/wmrl_snapshot_60023"


class SnapshotConsistencyTest(unittest.TestCase):
    """Check provenance, import closure, and retained feature definitions."""

    def test_copied_source_hashes(self):
        manifest = json.loads((DOCS / "source_manifest.json").read_text())
        for item in manifest["copied_source_files"]:
            with self.subTest(path=item["path"]):
                self.assertEqual(hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest(), item["sha256"])

    def test_local_rlinf_import_targets_exist(self):
        manifest = json.loads((DOCS / "source_manifest.json").read_text())
        for item in manifest["copied_source_files"]:
            path = ROOT / item["path"]
            if path.suffix != ".py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("rlinf."):
                    module = ROOT.joinpath(*node.module.split("."))
                    with self.subTest(source=item["path"], module=node.module):
                        self.assertTrue(module.with_suffix(".py").is_file() or (module / "__init__.py").is_file())

    def test_instruction_routing_and_stage_reward_classes_preserved(self):
        for filename, expected in [
            ("world_model_wan_multi_view_env.py", {"InstructionRoutedResnetRewModel", "InstructionRoutedGripperTintResnetRewModel", "WanmultiviewEnv"}),
            ("world_model_wan_multi_view_stage_reward_env.py", {"OrderedStageResnetRewardRouter", "WanmultiviewStageRewardEnv"}),
        ]:
            tree = ast.parse((ROOT / "rlinf/envs/world_model" / filename).read_text())
            self.assertTrue(expected.issubset({node.name for node in tree.body if isinstance(node, ast.ClassDef)}))

    def test_dependency_patches_present(self):
        deps = json.loads((DOCS / "dependency_versions.json").read_text())
        for name in ["openpi", "robotwin"]:
            self.assertTrue((DOCS / deps[name]["patch"]).is_file())
        self.assertTrue(all(item["tracked"] for item in deps["diffsynth"]["files"]))


if __name__ == "__main__":
    unittest.main()

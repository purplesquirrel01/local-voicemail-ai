import importlib
import io
from pathlib import Path
import tarfile
import tempfile
import tomllib
import unittest

from tools.audit_release_artifacts import audit_archive


class PackagingConfigurationTests(unittest.TestCase):
    def test_litert_agent_constraints_is_included_in_installed_modules(self):
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "pyproject.toml").open("rb") as handle:
            configuration = tomllib.load(handle)

        modules = configuration["tool"]["setuptools"]["py-modules"]

        self.assertIn("agent_constraints", modules)

    def test_unsupported_components_are_excluded_from_public_package(self):
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "pyproject.toml").open("rb") as handle:
            configuration = tomllib.load(handle)

        packages = configuration["tool"]["setuptools"]["packages"]
        modules = configuration["tool"]["setuptools"]["py-modules"]

        self.assertNotIn("workflow_insights", packages)
        self.assertNotIn("voicemail_demo", packages)
        self.assertNotIn("switchboard", packages)
        self.assertNotIn("switchboard_service", modules)

    def test_public_metadata_and_service_entry_points(self):
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "pyproject.toml").open("rb") as handle:
            configuration = tomllib.load(handle)

        project = configuration["project"]
        self.assertEqual(project["version"], "1.4.0")
        self.assertEqual(project["license"], {"text": "Apache-2.0"})
        self.assertEqual(
            set(project["scripts"]),
            {
                "lvt-watcher",
                "lvt-portal",
                "lvt-whisper-api",
                "lvt-parakeet-api",
                "lvt-gemma-api",
            },
        )
        for target in project["scripts"].values():
            module_name, function_name = target.split(":", 1)
            module = importlib.import_module(module_name)
            self.assertTrue(callable(getattr(module, function_name)))

    def test_candidate_agent_prompts_are_included_in_installed_package(self):
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "pyproject.toml").open("rb") as handle:
            configuration = tomllib.load(handle)

        packages = configuration["tool"]["setuptools"]["packages"]
        package_data = configuration["tool"]["setuptools"]["package-data"]

        self.assertIn("prompts", packages)
        self.assertIn("*.md", package_data["prompts"])
        self.assertTrue((project_root / "prompts" / "__init__.py").is_file())
        for filename in (
            "candidate_scout_agent.md",
            "numbers_agent.md",
            "subject_name_agent.md",
            "dob_agent.md",
            "name_agent.md",
            "caller_name_fallback_agent.md",
        ):
            self.assertTrue((project_root / "prompts" / filename).is_file(), filename)

    def test_artifact_audit_rejects_models_logs_and_excluded_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                for name in (
                    "release/models/model.litertlm",
                    "release/problem.txt",
                    "release/switchboard/config.py",
                ):
                    payload = b"synthetic\n"
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    handle.addfile(info, io.BytesIO(payload))

            messages = [finding.message for finding in audit_archive(archive)]
            self.assertIn("forbidden generated/model data is present", messages)
            self.assertIn("excluded path is present", messages)

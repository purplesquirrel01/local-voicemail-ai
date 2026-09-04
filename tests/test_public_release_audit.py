import json
import tempfile
import unittest
from pathlib import Path

from tools.public_release_audit import scan_tree


class PublicReleaseAuditTests(unittest.TestCase):
    def test_generated_wheel_virtual_environments_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = root / ".tmp-wheel-check" / "Lib" / "site-packages"
            generated.mkdir(parents=True)
            (generated / "third_party.py").write_text(
                "support_number = '" + "555" + "-444-0100'\n",
                encoding="utf-8",
            )

            self.assertEqual(scan_tree(root), [])

    def test_generated_release_and_test_workspaces_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in (
                ".pytest-orchestrator/fixture.json",
                ".build-tmp/fixture.json",
                "local-test-release-synthetic/release-manifest.json",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    '{"digest":"' + "555" + "444" + "0100" + '"}\n',
                    encoding="utf-8",
                )

            self.assertEqual(scan_tree(root), [])

    def test_audit_reports_nonfictional_phone_and_excluded_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.txt").write_text(
                "CALLBACK=" + "-".join(("555", "444", "0100")) + "\n",
                encoding="utf-8",
            )
            (root / "problem.txt").write_text("production log\n", encoding="utf-8")
            findings = scan_tree(root)
            messages = "\n".join(finding.message for finding in findings)
            self.assertIn("fictional 555", messages)
            self.assertIn("excluded path", messages)

    def test_audit_accepts_documentation_networks_and_synthetic_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "example.txt").write_text(
                "AI_URL=http://192.0.2.20\nUSER=synthetic-admin@example.invalid\n",
                encoding="utf-8",
            )
            self.assertEqual(scan_tree(root), [])

    def test_audit_requires_declared_synthetic_fixture_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = root / "eval"
            fixtures.mkdir()
            (fixtures / "cases.jsonl").write_text(
                json.dumps(
                    {
                        "id": "synthetic-case",
                        "metadata": {"callerid": "SYNTHETIC"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            messages = "\n".join(finding.message for finding in scan_tree(root))

            self.assertIn("synthetic fixture provenance document is missing", messages)
            self.assertIn("synthetic fixture provenance marker is missing", messages)

    def test_audit_accepts_declared_synthetic_fixture_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = root / "eval"
            fixtures.mkdir()
            (fixtures / "SYNTHETIC_DATA_PROVENANCE.md").write_text(
                "# Synthetic data provenance\n",
                encoding="utf-8",
            )
            (fixtures / "cases.jsonl").write_text(
                json.dumps(
                    {
                        "id": "synthetic-case",
                        "metadata": {
                            "callerid": "SYNTHETIC",
                            "synthetic": True,
                            "provenance": "hand-authored",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(scan_tree(root), [])


if __name__ == "__main__":
    unittest.main()

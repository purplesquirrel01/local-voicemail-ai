"""Synthetic negative controls for the public-source checks."""

import ipaddress
from pathlib import Path
import tempfile
import unittest

from tools.audit_secrets import scan_for_secrets
from tools.public_release_audit import scan_tree


class PublicationChecksTests(unittest.TestCase):
    def test_phone_audit_checks_exchange_and_sentence_punctuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Unassigned area code: this deliberately invalid control is not dialable.
            invalid = "-".join(("555", "444", "0100"))
            (root / "example.txt").write_text(f"Call {invalid}.\n", encoding="utf-8")
            findings = scan_tree(root)
            self.assertEqual(len(findings), 1)
            self.assertNotIn(invalid, findings[0].message)
            self.assertIn("fictional 555", findings[0].message)

    def test_reserved_phone_with_punctuation_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "example.txt").write_text("Call 202-555-0142.\n", encoding="utf-8")
            self.assertEqual(scan_tree(root), [])

    def test_contacts_are_checked_in_unrecognized_text_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_email = "synthetic" + "@" + "invalid.invalid"
            invalid_ip = str(ipaddress.IPv4Address(1))
            (root / "example.custom").write_text(
                f"{invalid_email}\n{invalid_ip}\n", encoding="utf-8"
            )
            findings = scan_tree(root)
            self.assertEqual(len(findings), 2)
            self.assertTrue(all(invalid_email not in item.message for item in findings))
            self.assertTrue(all(invalid_ip not in item.message for item in findings))

    def test_runtime_data_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("state.sqlite3", "recording.WAV", "model.gguf", ".env"):
                (root / name).write_bytes(b"")
            self.assertEqual(len(scan_tree(root)), 4)

    def test_private_key_marker_is_detected_beyond_two_megabytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = "-----BEGIN RSA " + "PRIVATE KEY-----"
            (root / "large.txt").write_text("x" * (2 * 1024 * 1024) + marker, encoding="utf-8")
            self.assertEqual([item.kind for item in scan_for_secrets(root)], ["private-key"])

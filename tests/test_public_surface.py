import builtins
import importlib
import os
import sys
import unittest
from unittest.mock import patch
from unittest import mock


class PublicSurfaceTests(unittest.TestCase):
    def test_watcher_does_not_depend_on_excluded_demo_package(self):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "voicemail_demo" or name.startswith("voicemail_demo."):
                raise AssertionError("public watcher imported excluded demo code")
            return real_import(name, *args, **kwargs)

        original = sys.modules.pop("watcher", None)
        try:
            with mock.patch("builtins.__import__", side_effect=guarded_import):
                module = importlib.import_module("watcher")
        finally:
            if original is not None:
                sys.modules["watcher"] = original

        self.assertTrue(callable(module.main))

    def test_watcher_defaults_are_local_and_email_is_opt_in(self):
        import watcher

        with mock.patch.dict(os.environ, {}, clear=True):
            settings = watcher.Settings.from_env()

        self.assertEqual(
            settings.whisper_url,
            "http://127.0.0.1:8765/transcribe/voicemail",
        )
        self.assertEqual(settings.smtp_host, "")
        self.assertEqual(settings.from_address, "")
        self.assertFalse(settings.email_enabled)
        self.assertFalse(settings.gemma_log_raw_response)
        self.assertFalse(settings.verification_apply_resolved_values)
        self.assertFalse(settings.transcript_lattice_apply_enabled)
        self.assertEqual(
            settings.state_db,
            "/var/lib/local-voicemail-transcription/pbx/state.sqlite3",
        )

    def test_portal_has_no_insights_dependency_and_forwarding_is_opt_in(self):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "workflow_insights" or name.startswith("workflow_insights."):
                raise AssertionError("public portal imported excluded insights code")
            return real_import(name, *args, **kwargs)

        original = sys.modules.pop("voicemail_portal", None)
        try:
            with mock.patch("builtins.__import__", side_effect=guarded_import):
                portal = importlib.import_module("voicemail_portal")
        finally:
            if original is not None:
                sys.modules["voicemail_portal"] = original

        with mock.patch.dict(os.environ, {}, clear=True):
            settings = portal.Settings.from_env()

        self.assertFalse(settings.forward_email_enabled)
        self.assertFalse(settings.forward_enabled)
        original_settings = portal.SETTINGS
        portal.SETTINGS = settings
        try:
            page = portal.portal_page(
                portal.PortalUser("100", "100", "", "Synthetic User"),
                "synthetic-csrf",
            ).body.decode("utf-8")
        finally:
            portal.SETTINGS = original_settings
        self.assertIn("const forwardingEnabled = false;", page)
        self.assertRegex(page, r'id="directoryBtn"[^>]* hidden>')
        self.assertEqual(settings.smtp_host, "")
        self.assertEqual(settings.from_address, "")
        self.assertEqual(
            settings.state_db,
            "/var/lib/local-voicemail-transcription/pbx/state.sqlite3",
        )
        page = portal.login_page().body.decode("utf-8")
        self.assertIn("Local Voicemail Transcription", page)
        private_brand = "".join(("O", "C", "I"))
        self.assertNotIn(private_brand, page)

    def test_portal_branding_and_organization_vocabulary_are_configurable(self):
        import verification
        import voicemail_portal as portal

        original_settings = portal.SETTINGS
        try:
            with patch.dict(
                os.environ,
                {
                    "VOICEMAIL_PORTAL_BRAND_NAME": "Example Voice Review",
                    "VOICEMAIL_PORTAL_BRAND_TAGLINE": "Synthetic test portal",
                    "VOICEMAIL_ORGANIZATION_TERMS": "Example Health Network,Sample Billing",
                },
                clear=False,
            ):
                portal.SETTINGS = portal.Settings.from_env()
                page = portal.login_page().body.decode("utf-8")
                self.assertIn("Example Voice Review", page)
                self.assertIn("Synthetic test portal", page)
                self.assertFalse(verification.person_like_name("Example Health Network"))
                self.assertFalse(verification.person_like_name("Sample Billing"))
        finally:
            portal.SETTINGS = original_settings


if __name__ == "__main__":
    unittest.main()

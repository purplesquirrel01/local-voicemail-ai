import importlib
import tempfile
import unittest
from pathlib import Path

import verification
import watcher


class RefactorPackageSurfaceTests(unittest.TestCase):
    def test_compatibility_entry_points_export_public_symbols(self):
        import voicemail_portal

        for symbol in ("Settings", "VoicemailStore", "VoicemailProcessor", "main"):
            self.assertTrue(hasattr(watcher, symbol), symbol)
        for symbol in (
            "Settings",
            "PortalStore",
            "hash_password",
            "verify_password",
            "sign_session",
            "read_session",
            "app",
            "main",
        ):
            self.assertTrue(hasattr(voicemail_portal, symbol), symbol)
        for symbol in ("parse_gemma_response", "resolve_name_field", "resolve_phone_field", "resolve_dob_field"):
            self.assertTrue(hasattr(verification, symbol), symbol)

    def test_refactor_package_skeletons_import(self):
        for package_name in ("voicemail_common", "voicemail_watcher", "voicemail_portal_app", "voicemail_verification"):
            module = importlib.import_module(package_name)
            self.assertEqual(module.__name__, package_name)

    def test_common_helpers_match_existing_watcher_and_portal_behavior(self):
        import voicemail_portal
        from voicemail_common.formatting import format_duration as common_format_duration
        from voicemail_common.formatting import format_phone_number as common_format_phone_number
        from voicemail_common.keys import build_file_key as common_build_file_key
        from voicemail_common.keys import build_legacy_file_key as common_build_legacy_file_key
        from voicemail_common.spool import extract_extension as common_extract_extension
        from voicemail_common.spool import is_voicemail_txt as common_is_voicemail_txt
        from voicemail_common.spool import matching_wav_path as common_matching_wav_path
        from voicemail_common.spool import parse_txt as common_parse_txt

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inbox = Path(tmp) / "voicemail" / "default" / "154" / "INBOX"
            inbox.mkdir(parents=True)
            txt_path = inbox / "msg0002.txt"
            txt_path.write_text(
                "\n".join(
                    [
                        "origtime=1770000100",
                        'callerid="SYNTHETIC CALLER" <12175550101>',
                        "origmailbox=154",
                        "duration=75",
                    ]
                ),
                encoding="utf-8",
            )

            info = common_parse_txt(str(txt_path))
            self.assertEqual(info, watcher.parse_txt(str(txt_path)))
            self.assertEqual(info, voicemail_portal.parse_txt(str(txt_path)))
            self.assertEqual(common_extract_extension(str(txt_path)), watcher.extract_extension(str(txt_path)))
            self.assertEqual(common_extract_extension(str(txt_path)), voicemail_portal.extract_extension(str(txt_path)))
            self.assertEqual(common_is_voicemail_txt(str(txt_path)), watcher.is_voicemail_txt(str(txt_path)))
            self.assertEqual(common_is_voicemail_txt(str(txt_path)), voicemail_portal.is_voicemail_txt(str(txt_path)))
            self.assertEqual(common_matching_wav_path(str(txt_path)), watcher.matching_wav_path(str(txt_path)))
            self.assertEqual(common_matching_wav_path(str(txt_path)), voicemail_portal.matching_wav_path(str(txt_path)))
            self.assertEqual(common_build_legacy_file_key("154", info, str(txt_path)), watcher.build_legacy_file_key("154", info, str(txt_path)))
            self.assertEqual(common_build_legacy_file_key("154", info, str(txt_path)), voicemail_portal.build_legacy_file_key("154", info, str(txt_path)))
            self.assertEqual(common_build_file_key("154", info, str(txt_path)), watcher.build_file_key("154", info, str(txt_path)))
            self.assertEqual(common_build_file_key("154", info, str(txt_path)), voicemail_portal.build_file_key("154", info, str(txt_path)))
            self.assertEqual(common_format_duration("75", empty_on_invalid=False), watcher.format_duration("75"))
            self.assertEqual(common_format_duration("75"), voicemail_portal.format_duration("75"))
            self.assertEqual(common_format_phone_number("1 (217) 555-0101"), watcher.format_phone_number("1 (217) 555-0101"))
            self.assertEqual(common_format_phone_number("1 (217) 555-0101"), voicemail_portal.format_phone_number("1 (217) 555-0101"))

    def test_watcher_package_surfaces_match_compat_exports(self):
        from voicemail_watcher import gemma_client, pipeline, schema, store, transcript_corrections, verification_stage

        self.assertIs(
            watcher.apply_verified_phone_corrections_to_transcript,
            transcript_corrections.apply_verified_phone_corrections_to_transcript,
        )
        self.assertIs(gemma_client.build_gemma_input_payload, watcher.build_gemma_input_payload)
        self.assertIs(gemma_client.call_gemma_field_extraction, watcher.call_gemma_field_extraction)
        self.assertIs(verification_stage.VerificationRunResult, watcher.VerificationRunResult)
        self.assertIs(verification_stage.safe_verify_voicemail_fields, watcher.safe_verify_voicemail_fields)
        self.assertIs(verification_stage.select_entities_for_output, watcher.select_entities_for_output)
        self.assertIs(store.VoicemailStore, watcher.VoicemailStore)
        self.assertEqual(schema.STATUS_COMPLETED, watcher.STATUS_COMPLETED)
        self.assertEqual(schema.STATUS_DEAD, watcher.STATUS_DEAD)
        self.assertEqual(schema.TERMINAL_STATUSES, watcher.TERMINAL_STATUSES)
        self.assertIs(pipeline.VoicemailProcessor, watcher.VoicemailProcessor)

    def test_portal_package_surfaces_match_compat_exports(self):
        import voicemail_portal
        from voicemail_portal_app import app as portal_app
        from voicemail_portal_app import audio, auth, file_ops, mailbox_discovery, render, routes, store

        self.assertIs(render.login_page, voicemail_portal.login_page)
        self.assertIs(render.portal_page, voicemail_portal.portal_page)
        self.assertIs(portal_app.app, voicemail_portal.app)
        self.assertIs(portal_app.create_app(), voicemail_portal.app)
        self.assertIs(portal_app.main, voicemail_portal.main)
        self.assertIs(store.PortalStore, voicemail_portal.PortalStore)
        self.assertIs(store.get_store, voicemail_portal.get_store)
        self.assertIs(auth.hash_password, voicemail_portal.hash_password)
        self.assertIs(auth.verify_password, voicemail_portal.verify_password)
        self.assertIs(auth.sign_session, voicemail_portal.sign_session)
        self.assertIs(auth.read_session, voicemail_portal.read_session)
        self.assertIs(mailbox_discovery.discover_mailboxes, voicemail_portal.discover_mailboxes)
        self.assertIs(audio.stream_audio_file, voicemail_portal.stream_audio_file)
        self.assertIs(file_ops.move_message_to_trash, voicemail_portal.move_message_to_trash)
        self.assertIs(file_ops.restore_message_to_inbox, voicemail_portal.restore_message_to_inbox)
        self.assertIs(routes.list_voicemails, voicemail_portal.list_voicemails)

    def test_verification_domain_package_surfaces_match_compat_exports(self):
        from voicemail_verification import attribution, budget, dob, gemma_response, names, phone, resolvers, schema, types

        self.assertIs(types.CandidateRecord, verification.CandidateRecord)
        self.assertIs(types.FieldResolution, verification.FieldResolution)
        self.assertIs(types.AttributionResult, verification.AttributionResult)
        self.assertIs(schema.GemmaSchemaError, verification.GemmaSchemaError)
        self.assertIs(gemma_response.parse_gemma_response, verification.parse_gemma_response)
        self.assertIs(budget.VerificationBudgetExceeded, verification.VerificationBudgetExceeded)
        self.assertIs(budget.check_budget, verification.check_budget)
        self.assertIs(attribution.map_evidence_to_timestamps, verification.map_evidence_to_timestamps)
        self.assertIs(phone.normalize_phone_candidate, verification.normalize_phone_candidate)
        self.assertIs(phone.resolve_phone_field, verification.resolve_phone_field)
        self.assertIs(dob.parse_dob, verification.parse_dob)
        self.assertIs(dob.resolve_dob_field, verification.resolve_dob_field)
        self.assertIs(names.extract_subject_reference_name_candidates, verification.extract_subject_reference_name_candidates)
        self.assertIs(names.resolve_name_field, verification.resolve_name_field)
        self.assertIs(resolvers.resolve_legacy_field, verification.resolve_legacy_field)


if __name__ == "__main__":
    unittest.main()

import gc
import inspect
import os
import json
import re
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adjudication import validate_transcript_adjudication_decision
from asr_lattice import DisagreementSpan, correct_transcript_constrained
from parakeet_server import extract_numbers_from_text as extract_parakeet_numbers_from_text
from verification import (
    AttributionResult,
    CandidateRecord,
    FieldResolution,
    GemmaSchemaError,
    ParakeetResult,
    extract_compact_dob_candidates,
    extract_explicit_patient_name_candidates,
    extract_numbers_from_text,
    extract_relationship_name_candidates,
    extract_self_identification_name_candidates,
    extract_spelled_name_candidates,
    extract_subject_reference_name_candidates,
    format_dob,
    map_evidence_to_timestamps,
    normalize_parakeet_payload,
    normalize_phone_candidate,
    parse_dob,
    parse_gemma_response,
    resolve_dob_field,
    resolve_legacy_field,
    resolve_name_field,
    resolve_phone_field,
)
from watcher import (
    TranscriptionResult,
    VerificationRunResult,
    apply_verified_phone_corrections_to_transcript,
    build_gemma_input_payload,
    call_gemma_field_extraction,
    extract_word_timestamps,
    safe_verify_voicemail_fields,
    select_entities_for_output,
)
import watcher


def phone_record(
    candidate_id,
    field_name,
    whisper_numbers=None,
    parakeet_numbers=None,
    parakeet_error=None,
    attribution=None,
    gemma_number=None,
):
    attribution = attribution or AttributionResult(
        candidate_id=candidate_id,
        field_name=field_name,
        evidence_text="you can reach me at 202-555-0109",
        mapped=True,
        mapping_method="word",
        matched_text="you can reach me at 202-555-0109",
    )
    parakeet = None
    if parakeet_numbers is not None or parakeet_error is not None:
        parakeet = ParakeetResult(
            candidate_id=candidate_id,
            text="",
            normalized_numbers=parakeet_numbers or [],
            error=parakeet_error,
        )
    return CandidateRecord(
        candidate_id=candidate_id,
        field_name=field_name,
        gemma={
            "candidate_id": candidate_id,
            "evidence_text": attribution.evidence_text,
            **(
                {
                    "raw": gemma_number,
                    "normalized": normalize_phone_candidate(gemma_number).normalized,
                    "formatted": normalize_phone_candidate(gemma_number).formatted,
                }
                if gemma_number
                else {}
            ),
        },
        attribution=attribution,
        whisper_numbers=whisper_numbers or [],
        parakeet=parakeet,
    )


def name_record(candidate_id, value, raw=None, source=None, caller_id_used=None):
    return CandidateRecord(
        candidate_id=candidate_id,
        field_name="name",
        gemma={
            "candidate_id": candidate_id,
            "raw": raw or value,
            "value": value,
            "source": source,
            "caller_id_used": caller_id_used,
            "evidence_text": f"this is {raw or value}",
        },
        attribution=AttributionResult(
            candidate_id=candidate_id,
            field_name="name",
            evidence_text=f"this is {raw or value}",
            mapped=True,
            mapping_method="word",
            matched_text=f"this is {raw or value}",
        ),
    )


def spelled_name_record(candidate_id, raw, value, evidence):
    record = name_record(
        candidate_id,
        value,
        raw=raw,
        source="transcript_spelling_corrected",
    )
    record.gemma["evidence_text"] = evidence
    record.gemma["confidence"] = "high"
    record.attribution.evidence_text = evidence
    record.attribution.matched_text = evidence
    return record


def relationship_name_record(candidate_id, value, evidence):
    record = name_record(
        candidate_id,
        value,
        raw=value,
        source="relationship_subject",
    )
    record.gemma["evidence_text"] = evidence
    record.gemma["confidence"] = "high"
    record.attribution.evidence_text = evidence
    record.attribution.matched_text = evidence
    return record


def subject_reference_name_record(candidate_id, value, evidence):
    record = name_record(
        candidate_id,
        value,
        raw=value,
        source="subject_reference",
    )
    record.gemma["evidence_text"] = evidence
    record.gemma["confidence"] = "high"
    record.attribution.evidence_text = evidence
    record.attribution.matched_text = evidence
    return record


def dob_record(candidate_id, normalized, evidence=None, parakeet_text=""):
    evidence = evidence or f"date of birth is {normalized}"
    parakeet = ParakeetResult(candidate_id=candidate_id, text=parakeet_text) if parakeet_text else None
    return CandidateRecord(
        candidate_id=candidate_id,
        field_name="dob",
        gemma={
            "candidate_id": candidate_id,
            "raw": normalized,
            "normalized": normalized,
            "evidence_text": evidence,
        },
        attribution=AttributionResult(
            candidate_id=candidate_id,
            field_name="dob",
            evidence_text=evidence,
            mapped=True,
            mapping_method="word",
            matched_text=evidence,
        ),
        parakeet=parakeet,
    )


def transcription_for_text(text):
    words = [
        {"word": match.group(0), "start": index * 0.1, "end": index * 0.1 + 0.09}
        for index, match in enumerate(re.finditer(r"\S+", text))
    ]
    return TranscriptionResult(
        text=text,
        entities={"_word_timestamps": words},
        segments=[{"text": text, "start": 0.0, "end": max(0.1, len(words) * 0.1)}],
    )


@contextmanager
def portal_test_env(values):
    old_env = os.environ.copy()
    os.environ.clear()
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def write_portal_message(inbox, msg_name, duration=None, origtime="1770000000"):
    os.makedirs(inbox, exist_ok=True)
    txt_path = os.path.join(inbox, f"{msg_name}.txt")
    wav_path = os.path.join(inbox, f"{msg_name}.wav")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(
            "origmailbox=154\n"
            f"origtime={origtime}\n"
            "origdate=Wed May 06 10:00:00 AM UTC 2026\n"
            'callerid="TEST CALLER" <2175550100>\n'
        )
        if duration is not None:
            handle.write(f"duration={duration}\n")
    with open(wav_path, "wb") as handle:
        handle.write(b"not real wav but stable")
    with open(os.path.join(inbox, f"{msg_name}.gsm"), "wb") as handle:
        handle.write(b"not real gsm but stable")
    return txt_path, wav_path


class VerificationCoreTests(unittest.TestCase):
    def test_verification_fields_default_to_all_supported_fields(self):
        old_env = os.environ.copy()
        try:
            os.environ.pop("VOICEMAIL_VERIFICATION_FIELDS", None)
            self.assertEqual(
                watcher.Settings.from_env().verification_fields,
                ("name", "dob", "callback_number", "fax_number"),
            )
            os.environ["VOICEMAIL_VERIFICATION_FIELDS"] = "  "
            self.assertEqual(
                watcher.Settings.from_env().verification_fields,
                ("name", "dob", "callback_number", "fax_number"),
            )
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_verification_fields_are_normalized_deduplicated_and_ordered(self):
        self.assertEqual(
            watcher.parse_verification_fields("FAX_NUMBER,name,dob,name"),
            ("name", "dob", "fax_number"),
        )

    def test_verification_fields_reject_unknown_values(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported VOICEMAIL_VERIFICATION_FIELDS value.*address",
        ):
            watcher.parse_verification_fields("name,address")

    def test_verification_scope_only_builds_audits_for_selected_fields(self):
        original_call = watcher.call_gemma_field_extraction
        original_parakeet = watcher.run_parakeet_for_record
        parakeet_calls = []
        payload = self.valid_gemma_payload()
        payload["patient_names"] = [
            {
                "raw": "Morgan Example",
                "value": "Morgan Example",
                "evidence_text": "patient Morgan Example",
                "source": "transcript",
            }
        ]
        watcher.call_gemma_field_extraction = lambda *_args, **_kwargs: payload
        watcher.run_parakeet_for_record = lambda *_args, **_kwargs: parakeet_calls.append("parakeet")
        try:
            settings = SimpleNamespace(
                gemma_field_extraction_enabled=True,
                verification_total_timeout_seconds=100,
                gemma_fail_open=False,
                verification_apply_resolved_values=True,
                verification_fields=("name",),
            )
            result = watcher.verify_voicemail_fields(
                "/tmp/synthetic.wav",
                transcription_for_text("This message is for patient Morgan Example."),
                {},
                settings,
            )
        finally:
            watcher.call_gemma_field_extraction = original_call
            watcher.run_parakeet_for_record = original_parakeet

        self.assertEqual([row["field_name"] for row in result.audit_rows], ["name"])
        self.assertEqual(result.proposed_entities["name"], "Morgan Example")
        self.assertEqual(parakeet_calls, [])

    def test_verification_scope_preserves_unscoped_original_fields(self):
        original = {
            "name": "Original Name",
            "dob": "01/02/1980",
            "callback_number": "2175550100",
            "fax_number": "2175550199",
        }
        resolutions = [
            FieldResolution(
                field_name="name",
                final_value="Selected Name",
                normalized_value="selected name",
                status="gemma_final",
            )
        ]

        self.assertEqual(
            watcher.apply_resolutions_to_entities(original, resolutions),
            {
                "name": "Selected Name",
                "dob": "01/02/1980",
                "callback_number": "2175550100",
                "fax_number": "2175550199",
            },
        )

    def test_parakeet_runs_only_for_scoped_non_name_fields(self):
        self.assertEqual(
            watcher.parakeet_verification_fields(
                ("name", "dob", "callback_number", "fax_number")
            ),
            ("dob", "callback_number", "fax_number"),
        )

    def test_gemma_unavailable_audits_only_scoped_fields(self):
        original_call = watcher.call_gemma_field_extraction
        watcher.call_gemma_field_extraction = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic unavailable")
        )
        try:
            result = watcher.verify_voicemail_fields(
                "/tmp/synthetic.wav",
                TranscriptionResult(text="Synthetic message", entities={"dob": "01/02/1980"}),
                {},
                SimpleNamespace(
                    gemma_field_extraction_enabled=True,
                    verification_total_timeout_seconds=100,
                    gemma_fail_open=False,
                    verification_apply_resolved_values=True,
                    verification_fields=("dob",),
                ),
            )
        finally:
            watcher.call_gemma_field_extraction = original_call

        self.assertEqual([row["field_name"] for row in result.audit_rows], ["dob"])
        self.assertEqual(result.proposed_entities["dob"], "01/02/1980")

    def test_safe_verification_failure_audits_only_scoped_fields(self):
        original_verify = watcher.verify_voicemail_fields
        watcher.verify_voicemail_fields = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic unexpected failure")
        )
        try:
            result = watcher.safe_verify_voicemail_fields(
                "synthetic-key",
                "/tmp/synthetic.wav",
                TranscriptionResult(text="Synthetic message", entities={"name": "Original Name"}),
                {},
                SimpleNamespace(
                    gemma_fail_open=False,
                    verification_fields=("name",),
                ),
            )
        finally:
            watcher.verify_voicemail_fields = original_verify

        self.assertEqual([row["field_name"] for row in result.audit_rows], ["name"])
        self.assertEqual(result.proposed_entities["name"], "Original Name")

    def valid_gemma_payload(self):
        return {
            "patient_names": [],
            "name_correction_candidates": [],
            "dob_candidates": [],
            "callback_numbers": [],
            "fax_numbers": [],
            "uncertain_numbers": [],
            "possible_errors": [],
        }

    def test_store_discover_retires_stale_duplicate_rows_for_same_path(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db_path = os.path.join(tmp, "state.sqlite3")
            txt_path = os.path.abspath(os.path.join(tmp, "msg0001.txt"))
            wav_path = os.path.abspath(os.path.join(tmp, "msg0001.wav"))
            store = watcher.VoicemailStore(db_path)

            self.assertEqual(
                store.discover("old-key", "154", txt_path, wav_path),
                watcher.STATUS_DISCOVERED,
            )
            store.upsert_transcript(
                "old-key",
                "154",
                txt_path,
                wav_path,
                {"origmailbox": "154", "origtime": "1770000000", "duration": "30"},
                "old transcript",
                {"callback_number": "202-555-0109"},
            )
            store.mark_completed("old-key", 14)
            now = watcher.utc_now_iso()
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO voicemail_field_verification (
                        file_key, field_name, final_value, normalized_value, status,
                        created_utc, updated_utc
                    )
                    VALUES ('old-key', 'name', 'Synthetic Person', 'syntheticperson', 'verified', ?, ?)
                    """,
                    (now, now),
                )
                conn.execute(
                    """
                    INSERT INTO asr_runs (
                        run_id, file_key, engine, role, audio_view, created_utc
                    )
                    VALUES ('run-old', 'old-key', 'whisper', 'primary', 'canonical', ?)
                    """,
                    (now,),
                )
                conn.execute(
                    """
                    INSERT INTO asr_span_candidates (
                        span_id, file_key, field_type, source, status, created_utc, updated_utc
                    )
                    VALUES ('span-old', 'old-key', 'name', 'whisper', 'audited', ?, ?)
                    """,
                    (now, now),
                )
                conn.execute(
                    """
                    INSERT INTO transcript_corrections (
                        correction_id, file_key, span_id, decision_type, created_utc
                    )
                    VALUES ('corr-old', 'old-key', 'span-old', 'choose_primary', ?)
                    """,
                    (now,),
                )

            self.assertEqual(
                store.discover("new-key", "154", txt_path, wav_path),
                watcher.STATUS_DISCOVERED,
            )

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT file_key, status FROM voicemails ORDER BY file_key"
                ).fetchall()
                self.assertEqual(
                    [(row["file_key"], row["status"]) for row in rows],
                    [("new-key", watcher.STATUS_DISCOVERED), ("old-key", watcher.STATUS_DEAD)],
                )
                transcript = conn.execute(
                    "SELECT deleted_utc, deleted_by FROM voicemail_transcripts WHERE file_key = 'old-key'"
                ).fetchone()
                self.assertIsNotNone(transcript["deleted_utc"])
                self.assertEqual(transcript["deleted_by"], "deduped_by_current_inbox_path")
                for table in (
                    "voicemail_field_verification",
                    "asr_runs",
                    "asr_span_candidates",
                    "transcript_corrections",
                ):
                    count = conn.execute(
                        f"SELECT count(*) FROM {table} WHERE file_key = 'old-key'"
                    ).fetchone()[0]
                    self.assertEqual(count, 0, table)
            finally:
                conn.close()

    def test_portal_sync_dedupes_same_path_with_duration_drift(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            txt_path, wav_path = write_portal_message(inbox, "msg0007", duration=20, origtime="1770000007")

            with portal_test_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            info = voicemail_portal.parse_txt(txt_path)
            current_key = voicemail_portal.build_file_key("154", info, txt_path)
            old_key = "old-duration-drift-key"
            now = voicemail_portal.utc_now_iso()
            with sqlite3.connect(state_db) as conn:
                conn.execute(
                    """
                    INSERT INTO voicemail_transcripts (
                        file_key, extension, mailbox, folder, msg_name, txt_path, wav_path,
                        callerid, origtime, origdate, duration, transcript, entities_json,
                        created_utc, updated_utc, deleted_utc, deleted_by
                    )
                    VALUES (?, '154', '154', 'INBOX', 'msg0007', ?, ?, ?, ?, ?, 19,
                            'duration drift transcript', '{"callback_number":"217-555-0100"}',
                            ?, ?, NULL, NULL)
                    """,
                    (
                        old_key,
                        txt_path,
                        wav_path,
                        info["callerid"],
                        int(info["origtime"]),
                        info["origdate"],
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO voicemails (
                        file_key, status, extension, txt_path, wav_path,
                        attempts, first_seen_utc, updated_utc, emailed_utc, transcript_chars
                    )
                    VALUES (?, 'completed', '154', ?, ?, 1, ?, ?, ?, 25)
                    """,
                    (old_key, txt_path, wav_path, now, now, now),
                )

            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                conn.row_factory = sqlite3.Row
                active_rows = conn.execute(
                    """
                    SELECT file_key, transcript, entities_json
                    FROM voicemail_transcripts
                    WHERE txt_path = ?
                      AND deleted_utc IS NULL
                      AND folder = 'INBOX'
                    """,
                    (txt_path,),
                ).fetchall()
                old_row = conn.execute(
                    "SELECT deleted_utc, deleted_by FROM voicemail_transcripts WHERE file_key = ?",
                    (old_key,),
                ).fetchone()
                queue_row = conn.execute(
                    "SELECT status, transcript_chars FROM voicemails WHERE file_key = ?",
                    (current_key,),
                ).fetchone()

            self.assertEqual(len(active_rows), 1)
            self.assertEqual(active_rows[0]["file_key"], current_key)
            self.assertEqual(active_rows[0]["transcript"], "duration drift transcript")
            self.assertIn("217-555-0100", active_rows[0]["entities_json"])
            self.assertIsNotNone(old_row["deleted_utc"])
            self.assertEqual(old_row["deleted_by"], "deduped_by_current_inbox_path")
            self.assertEqual(queue_row["status"], "completed")
            self.assertEqual(queue_row["transcript_chars"], 25)

    def test_portal_sync_does_not_merge_reused_slot_with_different_origtime(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            txt_path, wav_path = write_portal_message(inbox, "msg0008", duration=20, origtime="1770000008")

            with portal_test_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            info = voicemail_portal.parse_txt(txt_path)
            current_key = voicemail_portal.build_file_key("154", info, txt_path)
            old_key = "old-reused-slot-key"
            now = voicemail_portal.utc_now_iso()
            with sqlite3.connect(state_db) as conn:
                conn.execute(
                    """
                    INSERT INTO voicemail_transcripts (
                        file_key, extension, mailbox, folder, msg_name, txt_path, wav_path,
                        callerid, origtime, origdate, duration, transcript, entities_json,
                        created_utc, updated_utc, deleted_utc, deleted_by
                    )
                    VALUES (?, '154', '154', 'INBOX', 'msg0008', ?, ?, ?, 1770009999, ?, 20,
                            'old reused slot transcript', '{"callback_number":"217-555-0100"}',
                            ?, ?, NULL, NULL)
                    """,
                    (
                        old_key,
                        txt_path,
                        wav_path,
                        info["callerid"],
                        info["origdate"],
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO voicemails (
                        file_key, status, extension, txt_path, wav_path,
                        attempts, first_seen_utc, updated_utc, emailed_utc, transcript_chars
                    )
                    VALUES (?, 'completed', '154', ?, ?, 1, ?, ?, ?, 26)
                    """,
                    (old_key, txt_path, wav_path, now, now, now),
                )

            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                conn.row_factory = sqlite3.Row
                current_row = conn.execute(
                    """
                    SELECT transcript, entities_json
                    FROM voicemail_transcripts
                    WHERE file_key = ?
                      AND deleted_utc IS NULL
                      AND folder = 'INBOX'
                    """,
                    (current_key,),
                ).fetchone()
                old_row = conn.execute(
                    "SELECT deleted_utc, deleted_by FROM voicemail_transcripts WHERE file_key = ?",
                    (old_key,),
                ).fetchone()

            self.assertIsNotNone(current_row)
            self.assertFalse((current_row["transcript"] or "").strip())
            self.assertNotIn("217-555-0100", current_row["entities_json"])
            self.assertIsNotNone(old_row["deleted_utc"])
            self.assertEqual(old_row["deleted_by"], "deduped_by_current_inbox_path_reused")

    def test_watcher_legacy_key_migration_rekeys_auxiliary_rows(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db_path = os.path.join(tmp, "state.sqlite3")
            txt_path = os.path.abspath(os.path.join(tmp, "msg0002.txt"))
            wav_path = os.path.abspath(os.path.join(tmp, "msg0002.wav"))
            store = watcher.VoicemailStore(db_path)
            store.discover("legacy-key", "154", txt_path, wav_path)
            store.upsert_transcript(
                "legacy-key",
                "154",
                txt_path,
                wav_path,
                {"origmailbox": "154", "origtime": "1770000000", "duration": "30"},
                "legacy transcript",
                {},
            )
            now = watcher.utc_now_iso()
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO voicemail_field_verification (
                        file_key, field_name, final_value, normalized_value, status,
                        created_utc, updated_utc
                    )
                    VALUES ('legacy-key', 'name', 'Synthetic Person', 'syntheticperson', 'verified', ?, ?)
                    """,
                    (now, now),
                )
                conn.execute(
                    """
                    INSERT INTO asr_runs (
                        run_id, file_key, engine, role, audio_view, created_utc
                    )
                    VALUES ('run-legacy', 'legacy-key', 'whisper', 'primary', 'canonical', ?)
                    """,
                    (now,),
                )
                conn.execute(
                    """
                    INSERT INTO asr_span_candidates (
                        span_id, file_key, field_type, source, status, created_utc, updated_utc
                    )
                    VALUES ('span-legacy', 'legacy-key', 'name', 'whisper', 'audited', ?, ?)
                    """,
                    (now, now),
                )
                conn.execute(
                    """
                    INSERT INTO transcript_corrections (
                        correction_id, file_key, span_id, decision_type, created_utc
                    )
                    VALUES ('corr-legacy', 'legacy-key', 'span-legacy', 'choose_primary', ?)
                    """,
                    (now,),
                )

            store.migrate_legacy_key("legacy-key", "current-key")

            with sqlite3.connect(db_path) as conn:
                for table in (
                    "voicemail_field_verification",
                    "asr_runs",
                    "asr_span_candidates",
                    "transcript_corrections",
                ):
                    old_count = conn.execute(
                        f"SELECT count(*) FROM {table} WHERE file_key = 'legacy-key'"
                    ).fetchone()[0]
                    new_count = conn.execute(
                        f"SELECT count(*) FROM {table} WHERE file_key = 'current-key'"
                    ).fetchone()[0]
                    self.assertEqual(old_count, 0, table)
                    self.assertEqual(new_count, 1, table)

    def test_store_marks_missing_paths_dead_without_touching_terminal_rows(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db_path = os.path.join(tmp, "state.sqlite3")
            txt_path = os.path.abspath(os.path.join(tmp, "msg0002.txt"))
            wav_path = os.path.abspath(os.path.join(tmp, "msg0002.wav"))
            store = watcher.VoicemailStore(db_path)

            store.discover("active-key", "154", txt_path, wav_path)
            store.discover("done-key", "154", os.path.join(tmp, "msg0003.txt"), os.path.join(tmp, "msg0003.wav"))
            store.mark_completed("done-key", 20)

            self.assertEqual(store.mark_dead_by_path(txt_path, "missing source"), 1)

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                active = conn.execute(
                    "SELECT status, last_error FROM voicemails WHERE file_key = 'active-key'"
                ).fetchone()
                done = conn.execute(
                    "SELECT status FROM voicemails WHERE file_key = 'done-key'"
                ).fetchone()
                self.assertEqual(active["status"], watcher.STATUS_DEAD)
                self.assertEqual(active["last_error"], "missing source")
                self.assertEqual(done["status"], watcher.STATUS_COMPLETED)
            finally:
                conn.close()

    def test_phone_normalization(self):
        self.assertEqual(
            normalize_phone_candidate("two zero two five five five zero one one one").normalized,
            "2025550111",
        )
        self.assertEqual(normalize_phone_candidate("(202) 555-0111").formatted, "202-555-0111")
        leading = normalize_phone_candidate("1-202-555-0111")
        self.assertEqual(leading.normalized, "2025550111")
        self.assertTrue(leading.leading_one_stripped)
        self.assertFalse(normalize_phone_candidate("217477662").valid)
        self.assertFalse(normalize_phone_candidate("22175550111").valid)

    def test_parakeet_explicit_phone_suppresses_sliding_digit_windows(self):
        text = (
            "Again, that was Denise Neal, date of birth, 131-1978. "
            "My phone number is 202-555-0107. Thanks."
        )

        normalized, formatted = extract_parakeet_numbers_from_text(text)

        self.assertEqual(normalized, ["2025550107"])
        self.assertEqual(formatted, ["202-555-0107"])

    def test_parakeet_extension_digits_do_not_create_sliding_phone_windows(self):
        text = "Otherwise, you can call me back at 202-555-01-31 Extension 5308."

        normalized, formatted = extract_parakeet_numbers_from_text(text)

        self.assertEqual(normalized, ["2025550131"])
        self.assertEqual(formatted, ["202-555-0131"])

    def test_normalize_parakeet_payload_prefers_text_without_extension_windows(self):
        payload = {
            "text": "Otherwise, you can call me back at 202-555-01-31 Extension 5308.",
            "normalized_numbers": [
                "2025550131",
                "2025550111",
                "2025550128",
                "2025550126",
                "1855501268",
            ],
            "formatted_numbers": [
                "202-555-0131",
                "202-555-0112",
                "202-555-0128",
                "202-555-0126",
                "186-291-5308",
            ],
        }

        result = normalize_parakeet_payload("callback_number:0", payload)

        self.assertEqual(result.normalized_numbers, ["2025550131"])
        self.assertEqual(result.formatted_numbers, ["202-555-0131"])

    def test_callback_with_extension_verifies_after_parakeet_normalization(self):
        evidence_text = "you can call me back at (202) 555-0131 extension 5308"
        parakeet = normalize_parakeet_payload(
            "callback_number:0",
            {
                "text": "Otherwise, you can call me back at 202-555-01-31 Extension 5308.",
                "normalized_numbers": [
                    "2025550131",
                    "2025550111",
                    "2025550128",
                    "2025550126",
                    "1855501268",
                ],
            },
        )
        attribution = AttributionResult(
            candidate_id="callback_number:0",
            field_name="callback_number",
            evidence_text=evidence_text,
            mapped=True,
            mapping_method="word",
            matched_text=evidence_text,
        )
        record = CandidateRecord(
            candidate_id="callback_number:0",
            field_name="callback_number",
            gemma={
                "candidate_id": "callback_number:0",
                "raw": "(202) 555-0131",
                "normalized": "2025550131",
                "formatted": "(202) 555-0131",
                "evidence_text": evidence_text,
            },
            attribution=attribution,
            whisper_numbers=["2025550131"],
            parakeet=parakeet,
        )

        resolution = resolve_phone_field("callback_number", [record])

        self.assertEqual(resolution.status, "verified")
        self.assertFalse(resolution.needs_review)
        self.assertEqual(resolution.final_value, "202-555-0131")
        self.assertNotIn("multiple_parakeet_numbers", resolution.review_reasons)

    def test_true_multiple_parakeet_numbers_remain_ambiguous_after_normalization(self):
        text = "You can call me at 202-555-0113 or 202-555-0100."
        parakeet = normalize_parakeet_payload("callback_number:0", {"text": text})
        attribution = AttributionResult(
            candidate_id="callback_number:0",
            field_name="callback_number",
            evidence_text=text,
            mapped=True,
            mapping_method="word",
            matched_text=text,
        )
        record = CandidateRecord(
            candidate_id="callback_number:0",
            field_name="callback_number",
            gemma={"candidate_id": "callback_number:0", "evidence_text": text},
            attribution=attribution,
            whisper_numbers=["2025550113"],
            parakeet=parakeet,
        )

        resolution = resolve_phone_field("callback_number", [record])

        self.assertEqual(set(parakeet.normalized_numbers), {"2025550113", "2025550100"})
        self.assertEqual(resolution.status, "ambiguous")
        self.assertTrue(resolution.needs_review)
        self.assertIn("multiple_parakeet_numbers", resolution.review_reasons)

    def test_missing_gemma_array_key_is_schema_invalid(self):
        payload = self.valid_gemma_payload()
        del payload["callback_numbers"]
        with self.assertRaises(GemmaSchemaError):
            parse_gemma_response(payload)
        legacy = resolve_legacy_field("callback_number", "202-555-0111", True, "gemma_invalid_json")
        self.assertEqual(legacy.status, "legacy_fallback")
        self.assertEqual(legacy.final_value, "202-555-0111")

    def test_compact_gemma_empty_payload_expands_to_canonical_arrays(self):
        parsed = parse_gemma_response({"n": [], "d": [], "c": [], "f": [], "u": [], "e": []})
        self.assertEqual(parsed, self.valid_gemma_payload())

    def test_legacy_six_key_gemma_payload_adds_empty_name_corrections(self):
        payload = self.valid_gemma_payload()
        del payload["name_correction_candidates"]

        parsed = parse_gemma_response(payload)

        self.assertEqual(parsed["name_correction_candidates"], [])
        self.assertEqual(list(parsed), list(self.valid_gemma_payload()))

    def test_compact_gemma_optional_name_correction_tuple_expands(self):
        parsed = parse_gemma_response(
            {
                "n": [],
                "r": [["Bailey Sampel", "Bailey Example", "this is Bailey Sampel", "EXAMPLE,BAILEY", "last_name_phonetic_match"]],
                "d": [],
                "c": [],
                "f": [],
                "u": [],
                "e": [],
            }
        )

        self.assertEqual(
            parsed["name_correction_candidates"],
            [
                {
                    "raw": "Bailey Sampel",
                    "suggested_value": "Bailey Example",
                    "evidence_text": "this is Bailey Sampel",
                    "caller_id_used": "EXAMPLE,BAILEY",
                    "reason": "last_name_phonetic_match",
                }
            ],
        )

    def test_compact_gemma_tuple_payload_expands_to_canonical_arrays(self):
        payload = {
            "n": [["Taylor Sample", "Taylor Sample", "this is Taylor Sample", "transcript", ""]],
            "d": [["June twenty fifth nineteen eighty", "06/25/1980", "DOB June twenty fifth nineteen eighty"]],
            "c": [["217-555-0100", "2175550100", "217-555-0100", "call back", "call back at 217-555-0100"]],
            "f": [["217-555-0199", "2175550199", "217-555-0199", "fax", "fax 217-555-0199"]],
            "u": [{"raw": "217 maybe"}],
            "e": ["short ambiguous date"],
        }

        parsed = parse_gemma_response(payload)

        self.assertEqual(
            parsed["patient_names"],
            [
                {
                    "raw": "Taylor Sample",
                    "value": "Taylor Sample",
                    "evidence_text": "this is Taylor Sample",
                    "source": "transcript",
                    "caller_id_used": "",
                }
            ],
        )
        self.assertEqual(
            parsed["dob_candidates"],
            [
                {
                    "raw": "June twenty fifth nineteen eighty",
                    "normalized": "06/25/1980",
                    "evidence_text": "DOB June twenty fifth nineteen eighty",
                }
            ],
        )
        self.assertEqual(
            parsed["callback_numbers"],
            [
                {
                    "raw": "217-555-0100",
                    "normalized": "2175550100",
                    "formatted": "217-555-0100",
                    "label_cue": "call back",
                    "evidence_text": "call back at 217-555-0100",
                }
            ],
        )
        self.assertEqual(
            parsed["fax_numbers"],
            [
                {
                    "raw": "217-555-0199",
                    "normalized": "2175550199",
                    "formatted": "217-555-0199",
                    "label_cue": "fax",
                    "evidence_text": "fax 217-555-0199",
                }
            ],
        )
        self.assertEqual(parsed["uncertain_numbers"], [{"raw": "217 maybe"}])
        self.assertEqual(parsed["possible_errors"], ["short ambiguous date"])

    def test_compact_gemma_payload_from_litert_response_string_parses(self):
        response = {
            "response": json.dumps(
                {
                    "n": [["Taylor Sample", "Taylor Sample", "this is Taylor Sample", "transcript", ""]],
                    "d": [],
                    "c": [],
                    "f": [],
                    "u": [],
                    "e": [],
                }
            )
        }

        parsed = parse_gemma_response(response)

        self.assertEqual(parsed["patient_names"][0]["value"], "Taylor Sample")

    def test_compact_gemma_payload_rejects_partial_schema(self):
        with self.assertRaises(GemmaSchemaError):
            parse_gemma_response({"n": [], "d": [], "c": [], "f": [], "u": []})

    def test_compact_gemma_payload_rejects_malformed_tuple_lengths(self):
        with self.assertRaises(GemmaSchemaError):
            parse_gemma_response({"n": [["Taylor Sample"]], "d": [], "c": [], "f": [], "u": [], "e": []})

        with self.assertRaises(GemmaSchemaError):
            parse_gemma_response({"n": [], "d": [["raw", "06/25/1980"]], "c": [], "f": [], "u": [], "e": []})

        with self.assertRaises(GemmaSchemaError):
            parse_gemma_response({"n": [], "d": [], "c": [["217-555-0100"]], "f": [], "u": [], "e": []})

    def test_compact_gemma_payload_rejects_candidate_limits_and_oversized_strings(self):
        too_many = {
            "n": [["Taylor Sample", "Taylor Sample", "this is Taylor Sample", "transcript", ""] for _ in range(11)],
            "d": [],
            "c": [],
            "f": [],
            "u": [],
            "e": [],
        }
        with self.assertRaises(GemmaSchemaError):
            parse_gemma_response(too_many)

        long_value = {
            "n": [["A" * 201, "Taylor Sample", "this is Taylor Sample", "transcript", ""]],
            "d": [],
            "c": [],
            "f": [],
            "u": [],
            "e": [],
        }
        with self.assertRaises(GemmaSchemaError):
            parse_gemma_response(long_value)

        long_evidence = {
            "n": [],
            "d": [],
            "c": [["217-555-0100", "2175550100", "217-555-0100", "call back", "x" * 501]],
            "f": [],
            "u": [],
            "e": [],
        }
        with self.assertRaises(GemmaSchemaError):
            parse_gemma_response(long_evidence)

    def test_empty_gemma_callback_resolves_not_included(self):
        resolution = resolve_phone_field("callback_number", [])
        self.assertEqual(resolution.status, "not_included")
        self.assertIsNone(resolution.final_value)
        self.assertIn("no_gemma_candidate", resolution.review_reasons)

    def test_possible_errors_do_not_invalidate_valid_arrays(self):
        payload = self.valid_gemma_payload()
        payload["possible_errors"] = ["Multiple phone numbers found."]
        self.assertEqual(parse_gemma_response(payload)["possible_errors"], ["Multiple phone numbers found."])

    def test_uncertain_numbers_only_never_sets_phone_field(self):
        payload = self.valid_gemma_payload()
        payload["uncertain_numbers"] = [{"raw": "217 maybe 477"}]
        parsed = parse_gemma_response(payload)
        self.assertEqual(parsed["callback_numbers"], [])
        self.assertIsNone(resolve_phone_field("callback_number", []).final_value)

    def test_multiple_same_parakeet_phone_candidates_aggregate(self):
        records = [
            phone_record("callback_number:0", "callback_number", ["2025550109"], ["2025550109"]),
            phone_record("callback_number:1", "callback_number", ["2025550109"], ["2025550109"]),
        ]
        resolution = resolve_phone_field("callback_number", records)
        self.assertEqual(resolution.status, "verified")
        self.assertEqual(resolution.final_value, "202-555-0109")
        self.assertEqual(resolution.as_audit_row()["normalized_value"], "2025550109")

    def test_multiple_different_parakeet_phone_results_are_ambiguous_with_span_fallback(self):
        records = [
            phone_record("callback_number:0", "callback_number", ["2025550109"], ["2025550109"]),
            phone_record("callback_number:1", "callback_number", ["2025550109"], ["2025550115"]),
        ]
        resolution = resolve_phone_field("callback_number", records)
        self.assertEqual(resolution.status, "ambiguous")
        self.assertTrue(resolution.needs_review)
        self.assertEqual(resolution.final_value, "202-555-0109")
        self.assertIn("multiple_parakeet_numbers", resolution.review_reasons)

    def test_parakeet_override_is_reviewed(self):
        record = phone_record("callback_number:0", "callback_number", ["2025550109"], ["2025550115"])
        resolution = resolve_phone_field("callback_number", [record])
        self.assertEqual(resolution.status, "parakeet_override")
        self.assertTrue(resolution.needs_review)
        self.assertEqual(resolution.final_value, "202-555-0115")

    def test_parakeet_and_whisper_entity_agreement_is_verified(self):
        record = phone_record("callback_number:0", "callback_number", ["2025550103"], ["2025550102"])
        resolution = resolve_phone_field(
            "callback_number",
            [record],
            legacy_value="202-555-0102",
        )
        self.assertEqual(resolution.status, "verified")
        self.assertFalse(resolution.needs_review)
        self.assertEqual(resolution.final_value, "202-555-0102")
        self.assertEqual(resolution.whisper_json["entity_number"], "2025550102")
        self.assertEqual(resolution.whisper_json["agreement_source"], "entity")

    def test_whisper_and_caller_id_agreement_blocks_parakeet_override(self):
        record = phone_record("callback_number:0", "callback_number", ["2025550119"], ["2025550122"])
        resolution = resolve_phone_field(
            "callback_number",
            [record],
            legacy_value="202-555-0119",
            caller_id_value="202-555-0119",
        )
        self.assertEqual(resolution.status, "whisper_caller_id_verified")
        self.assertFalse(resolution.needs_review)
        self.assertEqual(resolution.final_value, "202-555-0119")
        self.assertEqual(resolution.normalized_value, "2025550119")
        self.assertEqual(resolution.whisper_json["entity_number"], "2025550119")
        self.assertEqual(resolution.whisper_json["caller_id_number"], "2025550119")
        self.assertEqual(resolution.whisper_json["agreement_source"], "entity_caller_id")
        self.assertIn("parakeet_disagreed", resolution.review_reasons)

    def test_whisper_span_and_caller_id_agreement_blocks_parakeet_override(self):
        record = phone_record("callback_number:0", "callback_number", ["2025550119"], ["2025550122"])
        resolution = resolve_phone_field(
            "callback_number",
            [record],
            caller_id_value="202-555-0119",
        )
        self.assertEqual(resolution.status, "whisper_caller_id_verified")
        self.assertFalse(resolution.needs_review)
        self.assertEqual(resolution.final_value, "202-555-0119")
        self.assertEqual(resolution.whisper_json["agreement_source"], "span_caller_id")

    def test_parakeet_phone_correction_updates_transcript_and_timestamps(self):
        transcript = "Please call me. 202-555-010L. Thank you."
        entities = {
            "_word_timestamps": [
                {"word": "Please", "start": 0.0, "end": 0.2},
                {"word": "call", "start": 0.2, "end": 0.4},
                {"word": "me.", "start": 0.4, "end": 0.6},
                {"word": "202-555-010L.", "start": 0.7, "end": 1.8},
                {"word": "Thank", "start": 1.9, "end": 2.1},
                {"word": "you.", "start": 2.1, "end": 2.3},
            ]
        }
        audit_rows = [
            {
                "field_name": "callback_number",
                "final_value": "202-555-0102",
                "normalized_value": "2025550102",
                "status": "parakeet_override",
                "attribution_json": [{"field_name": "callback_number", "word_start": 3, "word_end": 3}],
                "parakeet_json": [{"normalized_numbers": ["2025550102"]}],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, "Please call me. 202-555-0102. Thank you.")
        self.assertEqual(corrected_entities["_word_timestamps"][3]["word"], "202-555-0102")
        self.assertTrue(corrected_entities["transcript_corrections"][0]["word_timestamps_updated"])

    def test_dob_parakeet_override_rewrites_only_attributed_span(self):
        transcript = "Old DOB 01/05/1980. Date of birth is 01/05/1980, thanks."
        entities = {
            "_word_timestamps": [
                {"word": "Old", "start": 0.0, "end": 0.1},
                {"word": "DOB", "start": 0.1, "end": 0.2},
                {"word": "01/05/1980.", "start": 0.2, "end": 0.6},
                {"word": "Date", "start": 0.7, "end": 0.8},
                {"word": "of", "start": 0.8, "end": 0.9},
                {"word": "birth", "start": 0.9, "end": 1.0},
                {"word": "is", "start": 1.0, "end": 1.1},
                {"word": "01/05/1980,", "start": 1.1, "end": 1.6},
                {"word": "thanks.", "start": 1.7, "end": 2.0},
            ]
        }
        audit_rows = [
            {
                "field_name": "dob",
                "final_value": "01/15/1980",
                "normalized_value": "01151980",
                "status": "parakeet_override",
                "attribution_json": [{"field_name": "dob", "word_start": 3, "word_end": 7}],
                "gemma_json": [{"normalized": "01/05/1980", "raw": "01/05/1980"}],
                "parakeet_json": [{"text": "date of birth is 01/15/1980"}],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(
            corrected,
            "Old DOB 01/05/1980. Date of birth is 01/15/1980, thanks.",
        )
        self.assertEqual(corrected_entities["_word_timestamps"][2]["word"], "01/05/1980.")
        self.assertEqual(corrected_entities["_word_timestamps"][7]["word"], "01/15/1980,")
        self.assertEqual(corrected_entities["transcript_corrections"][0]["field_name"], "dob")
        self.assertEqual(corrected_entities["transcript_corrections"][0]["value"], "01/15/1980")
        self.assertTrue(corrected_entities["transcript_corrections"][0]["word_timestamps_updated"])

    def test_dob_parakeet_override_without_word_span_does_not_rewrite_transcript(self):
        transcript = "Date of birth is 01/05/1980."
        audit_rows = [
            {
                "field_name": "dob",
                "final_value": "01/15/1980",
                "normalized_value": "01151980",
                "status": "parakeet_override",
                "attribution_json": [],
                "gemma_json": [{"normalized": "01/05/1980"}],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            {"_word_timestamps": [{"word": "01/05/1980.", "start": 0.0, "end": 0.5}]},
            audit_rows,
        )

        self.assertEqual(corrected, transcript)
        self.assertNotIn("transcript_corrections", corrected_entities)

    def test_caller_id_name_correction_updates_transcript_and_timestamps(self):
        transcript = "Hey, Casey. It's Mark Exampel again. Please call me."
        entities = {
            "_word_timestamps": [
                {"word": "Hey,", "start": 0.0, "end": 0.2},
                {"word": "Casey.", "start": 0.2, "end": 0.4},
                {"word": "It's", "start": 0.5, "end": 0.6},
                {"word": "Mark", "start": 0.6, "end": 0.8},
                {"word": "Exampel", "start": 0.8, "end": 1.1},
                {"word": "again.", "start": 1.1, "end": 1.3},
            ]
        }
        audit_rows = [
            {
                "field_name": "name",
                "final_value": "Mark Example",
                "normalized_value": "markjennings",
                "status": "caller_id_spelling_corrected",
                "attribution_json": [{"word_start": 2, "word_end": 5}],
                "gemma_json": [
                    {
                        "raw": "Mark Exampel",
                        "value": "Mark Example",
                        "source": "caller_id_corrected",
                        "evidence_text": "It's Mark Exampel again.",
                    }
                ],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, "Hey, Casey. It's Mark Example again. Please call me.")
        self.assertEqual(corrected_entities["_word_timestamps"][3]["word"], "Mark")
        self.assertEqual(corrected_entities["_word_timestamps"][4]["word"], "Example")
        self.assertTrue(corrected_entities["transcript_corrections"][0]["word_timestamps_updated"])

    def test_caller_id_name_correction_ignores_unrelated_name_candidates(self):
        transcript = "This is Bailey Example, Casey Sample's daughter."
        entities = {
            "_word_timestamps": [
                {"word": "This", "start": 0.0, "end": 0.1},
                {"word": "is", "start": 0.1, "end": 0.2},
                {"word": "Bailey", "start": 0.2, "end": 0.4},
                {"word": "Example,", "start": 0.4, "end": 0.6},
                {"word": "Casey", "start": 0.7, "end": 0.9},
                {"word": "Robin's", "start": 0.9, "end": 1.1},
                {"word": "daughter.", "start": 1.1, "end": 1.3},
            ]
        }
        audit_rows = [
            {
                "field_name": "name",
                "final_value": "Bailey Example",
                "normalized_value": "caseylane",
                "status": "caller_id_spelling_corrected",
                "attribution_json": [
                    {"field_name": "name", "word_start": 4, "word_end": 6},
                    {"field_name": "name", "word_start": 0, "word_end": 3},
                ],
                "gemma_json": [
                    {
                        "raw": "Casey Sample",
                        "value": "Casey Sample",
                        "source": "transcript",
                        "evidence_text": "Casey Sample's daughter",
                    },
                    {
                        "raw": "Bailey Example",
                        "value": "Bailey Example",
                        "source": "self_identification",
                        "evidence_text": "This is Bailey Example",
                    },
                ],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, transcript)
        self.assertNotIn("transcript_corrections", corrected_entities)

    def test_multi_token_phone_correction_merges_timestamps(self):
        transcript = "My number is (202) 555-011B. Thanks."
        entities = {
            "_word_timestamps": [
                {"word": "My", "start": 0.0, "end": 0.1},
                {"word": "number", "start": 0.1, "end": 0.3},
                {"word": "is", "start": 0.3, "end": 0.4},
                {"word": "(202)", "start": 0.4, "end": 0.8},
                {"word": "555-011B.", "start": 0.8, "end": 1.7},
                {"word": "Thanks.", "start": 1.8, "end": 2.0},
            ]
        }
        audit_rows = [
            {
                "field_name": "callback_number",
                "final_value": "202-555-0117",
                "normalized_value": "2025550117",
                "status": "verified",
                "attribution_json": [{"field_name": "callback_number", "word_start": 3, "word_end": 4}],
                "parakeet_json": [{"normalized_numbers": ["2025550117"]}],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, "My number is 202-555-0117. Thanks.")
        self.assertEqual(corrected_entities["_word_timestamps"][3]["word"], "202-555-0117")
        self.assertEqual(corrected_entities["_word_timestamps"][3]["start"], 0.4)
        self.assertEqual(corrected_entities["_word_timestamps"][3]["end"], 1.7)
        self.assertEqual(corrected_entities["_word_timestamps"][4]["word"], "Thanks.")

    def test_repeated_area_code_phone_correction_collapses_duplicate_prefix(self):
        transcript = "My number is (202) 202-555-0110. Thanks."
        entities = {
            "_word_timestamps": [
                {"word": "My", "start": 0.0, "end": 0.1},
                {"word": "number", "start": 0.1, "end": 0.3},
                {"word": "is", "start": 0.3, "end": 0.4},
                {"word": "(202)", "start": 0.4, "end": 0.8},
                {"word": "202-555-0110.", "start": 0.8, "end": 1.7},
                {"word": "Thanks.", "start": 1.8, "end": 2.0},
            ]
        }
        audit_rows = [
            {
                "field_name": "callback_number",
                "final_value": "202-555-0110",
                "normalized_value": "2025550110",
                "status": "verified",
                "attribution_json": [{"field_name": "callback_number", "word_start": 3, "word_end": 4}],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, "My number is 202-555-0110. Thanks.")
        self.assertEqual(corrected_entities["_word_timestamps"][3]["word"], "202-555-0110")
        self.assertEqual(corrected_entities["_word_timestamps"][3]["start"], 0.4)
        self.assertEqual(corrected_entities["_word_timestamps"][3]["end"], 1.7)
        self.assertEqual(corrected_entities["_word_timestamps"][4]["word"], "Thanks.")

    def test_repeated_area_code_correction_stays_in_attributed_field_span(self):
        transcript = "Callback (202) 202-555-0110. Fax (202) 202-555-0110."
        entities = {
            "_word_timestamps": [
                {"word": "Callback", "start": 0.0, "end": 0.2},
                {"word": "(202)", "start": 0.3, "end": 0.6},
                {"word": "202-555-0110.", "start": 0.7, "end": 1.5},
                {"word": "Fax", "start": 1.6, "end": 1.8},
                {"word": "(202)", "start": 1.9, "end": 2.2},
                {"word": "202-555-0110.", "start": 2.3, "end": 3.1},
            ]
        }
        audit_rows = [
            {
                "field_name": "callback_number",
                "final_value": "202-555-0110",
                "normalized_value": "2025550110",
                "status": "verified",
                "attribution_json": [{"field_name": "callback_number", "word_start": 1, "word_end": 2}],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, "Callback 202-555-0110. Fax (202) 202-555-0110.")
        self.assertEqual(corrected_entities["_word_timestamps"][1]["word"], "202-555-0110")
        self.assertEqual(corrected_entities["_word_timestamps"][2]["word"], "Fax")
        self.assertEqual(corrected_entities["_word_timestamps"][3]["word"], "(202)")

    def test_repeated_area_code_correction_rejects_mismatched_prefix(self):
        transcript = "My number is (312) 202-555-0110. Thanks."
        entities = {
            "_word_timestamps": [
                {"word": "My", "start": 0.0, "end": 0.1},
                {"word": "number", "start": 0.1, "end": 0.3},
                {"word": "is", "start": 0.3, "end": 0.4},
                {"word": "(312)", "start": 0.4, "end": 0.8},
                {"word": "202-555-0110.", "start": 0.8, "end": 1.7},
                {"word": "Thanks.", "start": 1.8, "end": 2.0},
            ]
        }
        audit_rows = [
            {
                "field_name": "callback_number",
                "final_value": "202-555-0110",
                "normalized_value": "2025550110",
                "status": "verified",
                "attribution_json": [{"field_name": "callback_number", "word_start": 3, "word_end": 4}],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, transcript)
        self.assertEqual(corrected_entities["_word_timestamps"][3]["word"], "(312)")
        self.assertEqual(corrected_entities["_word_timestamps"][4]["word"], "202-555-0110")

    def test_digit_by_digit_callback_correction_updates_transcript(self):
        transcript = "Testing. Callback numbers 2-0-2-5-5-5-0-1-2-1. Thanks."
        entities = {
            "_word_timestamps": [
                {"word": "Testing.", "start": 0.0, "end": 0.3},
                {"word": "Callback", "start": 0.4, "end": 0.7},
                {"word": "numbers", "start": 0.7, "end": 1.0},
                {"word": "2-0-2-5-5-5-0-1-2-1.", "start": 1.0, "end": 3.0},
                {"word": "Thanks.", "start": 3.1, "end": 3.4},
            ]
        }
        audit_rows = [
            {
                "field_name": "callback_number",
                "final_value": "202-555-0121",
                "normalized_value": "2025550121",
                "status": "whisper_caller_id_verified",
                "attribution_json": [{"field_name": "callback_number", "word_start": 3, "word_end": 3}],
                "parakeet_json": [],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, "Testing. Callback numbers 202-555-0121. Thanks.")
        self.assertEqual(corrected_entities["_word_timestamps"][3]["word"], "202-555-0121")
        self.assertTrue(corrected_entities["transcript_corrections"][0]["word_timestamps_updated"])

    def test_digit_by_digit_fax_correction_updates_transcript(self):
        transcript = "Good back numbers 2-0-2-5-5-5-0-1-1-3. Thanks."
        entities = {
            "_word_timestamps": [
                {"word": "Good", "start": 0.0, "end": 0.2},
                {"word": "back", "start": 0.2, "end": 0.4},
                {"word": "numbers", "start": 0.4, "end": 0.7},
                {"word": "2-0-2-5-5-5-0-1-1-3.", "start": 0.7, "end": 2.6},
                {"word": "Thanks.", "start": 2.7, "end": 3.0},
            ]
        }
        audit_rows = [
            {
                "field_name": "fax_number",
                "final_value": "202-555-0113",
                "normalized_value": "2025550113",
                "status": "parakeet_override",
                "attribution_json": [{"field_name": "fax_number", "word_start": 3, "word_end": 3}],
                "parakeet_json": [{"normalized_numbers": ["2025550113"]}],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, "Good back numbers 202-555-0113. Thanks.")
        self.assertEqual(corrected_entities["_word_timestamps"][3]["word"], "202-555-0113")

    def test_phone_correction_only_changes_its_attributed_field_span(self):
        transcript = "Callback (202) 555-012B. Fax 202-555-014B."
        entities = {
            "_word_timestamps": [
                {"word": "Callback", "start": 0.0, "end": 0.3},
                {"word": "(202)", "start": 0.4, "end": 0.8},
                {"word": "555-012B.", "start": 0.8, "end": 1.4},
                {"word": "Fax", "start": 1.5, "end": 1.7},
                {"word": "202-555-014B.", "start": 1.8, "end": 2.8},
            ]
        }
        audit_rows = [
            {
                "field_name": "callback_number",
                "final_value": "202-555-0125",
                "normalized_value": "2025550125",
                "status": "verified",
                "attribution_json": [
                    {"field_name": "callback_number", "word_start": 1, "word_end": 2}
                ],
            },
            {
                "field_name": "fax_number",
                "final_value": "202-555-0130",
                "normalized_value": "2025550130",
                "status": "verified",
                "attribution_json": [
                    {"field_name": "fax_number", "word_start": 4, "word_end": 4}
                ],
            },
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, "Callback 202-555-0125. Fax 202-555-0130.")
        self.assertEqual(corrected_entities["_word_timestamps"][1]["word"], "202-555-0125")
        self.assertEqual(corrected_entities["_word_timestamps"][3]["word"], "202-555-0130")
        self.assertCountEqual(
            [item["field_name"] for item in corrected_entities["transcript_corrections"]],
            ["callback_number", "fax_number"],
        )

    def test_fax_parakeet_correction_changes_only_fax_span(self):
        transcript = "Callback 202-555-0101. Fax 202-555-014B."
        entities = {
            "_word_timestamps": [
                {"word": "Callback", "start": 0.0, "end": 0.3},
                {"word": "202-555-0101.", "start": 0.4, "end": 1.2},
                {"word": "Fax", "start": 1.3, "end": 1.5},
                {"word": "202-555-014B.", "start": 1.6, "end": 2.4},
            ]
        }
        audit_rows = [
            {
                "field_name": "fax_number",
                "final_value": "202-555-0101",
                "normalized_value": "2025550101",
                "status": "parakeet_override",
                "attribution_json": [{"field_name": "fax_number", "word_start": 3, "word_end": 3}],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, "Callback 202-555-0101. Fax 202-555-0101.")
        self.assertEqual(corrected_entities["_word_timestamps"][1]["word"], "202-555-0101.")
        self.assertEqual(corrected_entities["_word_timestamps"][3]["word"], "202-555-0101")

    def test_fax_correction_does_not_rewrite_same_raw_callback_number(self):
        transcript = "Callback 202-555-014B. Fax 202-555-014B."
        entities = {
            "_word_timestamps": [
                {"word": "Callback", "start": 0.0, "end": 0.3},
                {"word": "202-555-014B.", "start": 0.4, "end": 1.2},
                {"word": "Fax", "start": 1.3, "end": 1.5},
                {"word": "202-555-014B.", "start": 1.6, "end": 2.4},
            ]
        }
        audit_rows = [
            {
                "field_name": "fax_number",
                "final_value": "202-555-0130",
                "normalized_value": "2025550130",
                "status": "parakeet_override",
                "attribution_json": [{"field_name": "fax_number", "word_start": 3, "word_end": 3}],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, "Callback 202-555-014B. Fax 202-555-0130.")
        self.assertEqual(corrected_entities["_word_timestamps"][1]["word"], "202-555-014B.")
        self.assertEqual(corrected_entities["_word_timestamps"][3]["word"], "202-555-0130")

    def test_phone_correction_does_not_use_global_transcript_fallback(self):
        transcript = "Callback 202-555-014B. Fax 202-555-014B."
        entities = {
            "_word_timestamps": [
                {"word": "Callback", "start": 0.0, "end": 0.3},
                {"word": "202-555-014B.", "start": 0.4, "end": 1.4},
                {"word": "Fax", "start": 1.5, "end": 1.7},
                {"word": "202-555-014B.", "start": 1.8, "end": 2.8},
            ]
        }
        audit_rows = [
            {
                "field_name": "fax_number",
                "final_value": "202-555-0130",
                "normalized_value": "2025550130",
                "status": "verified",
                "attribution_json": [
                    {
                        "field_name": "fax_number",
                        "mapping_method": "segment",
                        "matched_text": transcript,
                    }
                ],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, transcript)
        self.assertNotIn("transcript_corrections", corrected_entities)

    def test_phone_correction_ignores_other_field_attribution_span(self):
        transcript = "Callback 202-555-014B. Fax 202-555-014B."
        entities = {
            "_word_timestamps": [
                {"word": "Callback", "start": 0.0, "end": 0.3},
                {"word": "202-555-014B.", "start": 0.4, "end": 1.4},
                {"word": "Fax", "start": 1.5, "end": 1.7},
                {"word": "202-555-014B.", "start": 1.8, "end": 2.8},
            ]
        }
        audit_rows = [
            {
                "field_name": "fax_number",
                "final_value": "202-555-0130",
                "normalized_value": "2025550130",
                "status": "verified",
                "attribution_json": [
                    {"field_name": "callback_number", "word_start": 1, "word_end": 1}
                ],
            }
        ]

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            audit_rows,
        )

        self.assertEqual(corrected, transcript)
        self.assertNotIn("transcript_corrections", corrected_entities)

    def test_fax_parakeet_result_updates_only_fax_field(self):
        record = phone_record("fax_number:0", "fax_number", ["2025550111"], ["2025550111"])
        resolution = resolve_phone_field("fax_number", [record])
        self.assertEqual(resolution.field_name, "fax_number")
        self.assertEqual(resolution.final_value, "202-555-0111")

    def test_fax_candidate_is_not_replaced_by_unrelated_parakeet_number(self):
        attribution = AttributionResult(
            candidate_id="fax_number:0",
            field_name="fax_number",
            evidence_text="fax number is 202-555-0103",
            mapped=True,
            mapping_method="word",
            matched_text="fax number is 202-555-0103",
        )
        record = phone_record(
            "fax_number:0",
            "fax_number",
            ["2025550103"],
            ["2175550101"],
            attribution=attribution,
            gemma_number="202-555-0103",
        )

        resolution = resolve_phone_field("fax_number", [record])

        self.assertEqual(resolution.final_value, "202-555-0103")
        self.assertEqual(resolution.normalized_value, "2025550103")
        self.assertNotEqual(resolution.normalized_value, "2175550101")
        self.assertTrue(resolution.needs_review)
        self.assertIn("parakeet_disagreed", resolution.review_reasons)

    def test_fax_parakeet_matching_callback_preserves_callback_entity(self):
        attribution = AttributionResult(
            candidate_id="fax_number:0",
            field_name="fax_number",
            evidence_text="fax number is 202-555-0103",
            mapped=True,
            mapping_method="word",
            word_start=2,
            word_end=5,
            matched_text="fax number is 202-555-0103",
        )
        record = phone_record(
            "fax_number:0",
            "fax_number",
            ["2025550103"],
            ["2175550101"],
            attribution=attribution,
            gemma_number="202-555-0103",
        )

        resolution = resolve_phone_field("fax_number", [record])
        updated = watcher.apply_resolutions_to_entities(
            {"callback_number": "217-555-0101", "fax_number": "202-555-0103"},
            [resolution],
        )

        self.assertEqual(updated["callback_number"], "217-555-0101")
        self.assertEqual(updated["fax_number"], "217-555-0101")
        self.assertEqual(resolution.status, "parakeet_override")
        self.assertNotIn("parakeet_outside_attributed_span", resolution.review_reasons)
        self.assertIn("parakeet_disagreed", resolution.review_reasons)

    def test_segment_multiple_phone_values_blocks_parakeet_override(self):
        attribution = AttributionResult(
            candidate_id="fax_number:0",
            field_name="fax_number",
            evidence_text="my fax is in this segment",
            mapped=True,
            mapping_method="segment",
            matched_text="fax 202-555-0111 callback 202-555-0109",
            review_reasons=["segment_fallback", "multiple_phone_values_in_segment"],
        )
        record = phone_record("fax_number:0", "fax_number", [], ["2025550111"], attribution=attribution)
        resolution = resolve_phone_field("fax_number", [record])
        self.assertEqual(resolution.status, "ambiguous")
        self.assertIsNone(resolution.final_value)

    def test_multiple_patient_names_different_picks_best_with_review_flag(self):
        resolution = resolve_name_field([
            name_record("name:0", "Jordan Example"),
            name_record("name:1", "Catherine Example"),
        ])
        self.assertEqual(resolution.status, "gemma_final")
        self.assertTrue(resolution.needs_review)
        self.assertIn(resolution.final_value, {"Jordan Example", "Catherine Example"})
        self.assertIn("multiple_field_candidates", resolution.review_reasons)

    def test_caller_id_spelling_correction_accepts_guarded_same_first_last_name_match(self):
        corrected = resolve_name_field([
            name_record(
                "name:0",
                "Quinn Example",
                raw="Quinn Exampel",
                source="caller_id_corrected",
                caller_id_used="QUINN L EXAMPLE",
            )
        ], caller_id_name="QUINN L EXAMPLE")
        self.assertEqual(corrected.status, "caller_id_spelling_corrected")
        self.assertEqual(corrected.final_value, "Quinn Example")
        self.assertNotIn("caller_id_correction_disabled", corrected.review_reasons)

        rejected = resolve_name_field([
            name_record(
                "name:0",
                "Example Clinic",
                raw="Katherine Example",
                source="caller_id_corrected",
                caller_id_used="Example Clinic",
            )
        ], caller_id_name="Example Clinic")
        self.assertEqual(rejected.final_value, "Katherine Example")
        self.assertIn("caller_id_correction_disabled", rejected.review_reasons)

    def test_greeting_addressee_name_is_rejected(self):
        record = name_record("name:0", "Casey", raw="Casey")
        record.gemma["evidence_text"] = "Hey, Casey."
        record.attribution.evidence_text = "Hey, Casey."
        record.attribution.matched_text = "Hey, Casey."

        resolution = resolve_name_field([record], caller_id_name="Mark Example")

        self.assertEqual(resolution.status, "not_included")
        self.assertIsNone(resolution.final_value)
        self.assertIn("addressee_name_rejected", resolution.review_reasons)

    def test_uncertain_addressee_does_not_block_self_identified_name(self):
        addressee = name_record("name:0", "Casey", raw="Casey")
        addressee.gemma["evidence_text"] = "Casey, I think that's your name, if I heard that correctly."
        addressee.attribution.evidence_text = addressee.gemma["evidence_text"]
        addressee.attribution.matched_text = addressee.gemma["evidence_text"]

        noisy = name_record("name:1", "Avery before", raw="Avery before")
        noisy.gemma["evidence_text"] = "My name is Avery before."
        noisy.attribution.evidence_text = noisy.gemma["evidence_text"]
        noisy.attribution.matched_text = noisy.gemma["evidence_text"]

        numeric_noise = name_record("name:2", "Top4", raw="Top4")
        numeric_noise.gemma["evidence_text"] = "Again, my name is Top4."
        numeric_noise.attribution.evidence_text = numeric_noise.gemma["evidence_text"]
        numeric_noise.attribution.matched_text = numeric_noise.gemma["evidence_text"]

        corrected = name_record("name:3", "Avery Example", raw="Avery Example")
        corrected.gemma["evidence_text"] = "And my name is Avery Example."
        corrected.attribution.evidence_text = corrected.gemma["evidence_text"]
        corrected.attribution.matched_text = corrected.gemma["evidence_text"]

        resolution = resolve_name_field([addressee, noisy, numeric_noise, corrected])

        self.assertEqual(resolution.status, "gemma_final")
        self.assertEqual(resolution.final_value, "Avery Example")
        self.assertIn("addressee_name_rejected", resolution.review_reasons)
        self.assertIn("name_not_person_like", resolution.review_reasons)

    def test_caller_id_corrects_same_first_close_surname_self_identification(self):
        resolution = resolve_name_field([
            name_record(
                "name:0",
                "Mark Example",
                raw="Mark Exampel",
                source="caller_id_corrected",
                caller_id_used="MARK EXAMPLE",
            )
        ], caller_id_name="MARK EXAMPLE")

        self.assertEqual(resolution.status, "caller_id_spelling_corrected")
        self.assertEqual(resolution.final_value, "Mark Example")
        self.assertNotIn("caller_id_correction_disabled", resolution.review_reasons)

    def test_caller_id_corrects_same_first_rough_matching_last_name(self):
        record = name_record(
            "name:0",
            "Quinn Example",
            raw="Quinn Exampel",
            source="caller_id_corrected",
            caller_id_used="QUINN L EXAMPLE",
        )
        record.gemma["evidence_text"] = "Yes, this is Quinn Exampel."
        record.attribution.evidence_text = record.gemma["evidence_text"]
        record.attribution.matched_text = record.gemma["evidence_text"]

        resolution = resolve_name_field([record], caller_id_name="QUINN L EXAMPLE")

        self.assertEqual(resolution.status, "caller_id_spelling_corrected")
        self.assertEqual(resolution.final_value, "Quinn Example")

    def test_caller_id_corrects_last_name_only_when_enabled(self):
        record = name_record(
            "name:0",
            "Taylor Example",
            raw="Taylor Exampel",
            source="caller_id_corrected",
            caller_id_used="QUINN L EXAMPLE",
        )
        record.gemma["evidence_text"] = "Yes, this is Taylor Exampel."
        record.attribution.evidence_text = record.gemma["evidence_text"]
        record.attribution.matched_text = record.gemma["evidence_text"]

        resolution = resolve_name_field([record], caller_id_name="QUINN L EXAMPLE")

        self.assertEqual(resolution.status, "caller_id_spelling_corrected")
        self.assertEqual(resolution.final_value, "Taylor Example")

    def test_caller_id_corrects_close_last_first_full_name_when_selected(self):
        record = name_record(
            "name:0",
            "Morgan Sample",
            raw="Morgan Sample",
            source="caller_id_corrected",
            caller_id_used="SAMPLE MORGAN",
        )
        record.gemma["evidence_text"] = "Hi, this is Morgan Sample."
        record.attribution.evidence_text = record.gemma["evidence_text"]
        record.attribution.matched_text = record.gemma["evidence_text"]

        resolution = resolve_name_field([record], caller_id_name="SAMPLE MORGAN")

        self.assertEqual(resolution.status, "caller_id_spelling_corrected")
        self.assertEqual(resolution.final_value, "Morgan Sample")

    def test_caller_id_corrects_close_first_last_full_name_when_selected(self):
        record = name_record(
            "name:0",
            "Morgan Sample",
            raw="Morgan Sample",
            source="caller_id_corrected",
            caller_id_used="MORGAN SAMPLE",
        )
        record.gemma["evidence_text"] = "Hi, this is Morgan Sample."
        record.attribution.evidence_text = record.gemma["evidence_text"]
        record.attribution.matched_text = record.gemma["evidence_text"]

        resolution = resolve_name_field([record], caller_id_name="MORGAN SAMPLE")

        self.assertEqual(resolution.status, "caller_id_spelling_corrected")
        self.assertEqual(resolution.final_value, "Morgan Sample")

    def test_caller_id_correction_does_not_expand_complete_spoken_first_name(self):
        record = name_record(
            "name:0",
            "Caseyan Example",
            raw="Casey Example",
            source="caller_id_corrected",
            caller_id_used="EXAMPLE CASEYAN",
        )
        record.gemma["evidence_text"] = "Hi, this is Casey Example."
        record.attribution.evidence_text = record.gemma["evidence_text"]
        record.attribution.matched_text = record.gemma["evidence_text"]

        resolution = resolve_name_field([record], caller_id_name="EXAMPLE CASEYAN")

        self.assertEqual(resolution.status, "gemma_final")
        self.assertEqual(resolution.final_value, "Casey Example")
        self.assertIn("caller_id_correction_disabled", resolution.review_reasons)

    def test_caller_id_last_name_only_correction_can_be_disabled(self):
        old_value = os.environ.get("CANDIDATE_AGENT_CALLER_ID_LAST_NAME_ONLY_CORRECTION")
        os.environ["CANDIDATE_AGENT_CALLER_ID_LAST_NAME_ONLY_CORRECTION"] = "false"
        try:
            record = name_record(
                "name:0",
                "Taylor Example",
                raw="Taylor Exampel",
                source="caller_id_corrected",
                caller_id_used="QUINN L EXAMPLE",
            )
            record.gemma["evidence_text"] = "Yes, this is Taylor Exampel."
            record.attribution.evidence_text = record.gemma["evidence_text"]
            record.attribution.matched_text = record.gemma["evidence_text"]

            resolution = resolve_name_field([record], caller_id_name="QUINN L EXAMPLE")
        finally:
            if old_value is None:
                os.environ.pop("CANDIDATE_AGENT_CALLER_ID_LAST_NAME_ONLY_CORRECTION", None)
            else:
                os.environ["CANDIDATE_AGENT_CALLER_ID_LAST_NAME_ONLY_CORRECTION"] = old_value

        self.assertEqual(resolution.status, "gemma_final")
        self.assertEqual(resolution.final_value, "Taylor Exampel")
        self.assertIn("caller_id_correction_disabled", resolution.review_reasons)

    def test_caller_id_last_name_only_does_not_replace_first_name(self):
        record = name_record(
            "name:0",
            "Quinn Example",
            raw="Taylor Exampel",
            source="caller_id_corrected",
            caller_id_used="QUINN L EXAMPLE",
        )
        record.gemma["evidence_text"] = "Yes, this is Taylor Exampel."
        record.attribution.evidence_text = record.gemma["evidence_text"]
        record.attribution.matched_text = record.gemma["evidence_text"]

        resolution = resolve_name_field([record], caller_id_name="QUINN L EXAMPLE")

        self.assertEqual(resolution.status, "gemma_final")
        self.assertEqual(resolution.final_value, "Taylor Exampel")
        self.assertIn("caller_id_correction_disabled", resolution.review_reasons)

    def test_caller_id_last_comma_first_does_not_correct_spoken_name(self):
        record = name_record("name:0", "Bailey Sampel", raw="Bailey Sampel", source="transcript")
        record.gemma["evidence_text"] = "This is Bailey Sampel."
        record.attribution.evidence_text = record.gemma["evidence_text"]
        record.attribution.matched_text = record.gemma["evidence_text"]

        resolution = resolve_name_field([record], caller_id_name="SAMPLE, BAILEY")

        self.assertEqual(resolution.status, "gemma_final")
        self.assertEqual(resolution.final_value, "Bailey Sampel")

    def test_caller_id_does_not_expand_first_name_only(self):
        record = name_record("name:0", "Quinn", raw="Quinn", source="transcript")
        record.gemma["evidence_text"] = "This is Quinn."
        record.attribution.evidence_text = record.gemma["evidence_text"]
        record.attribution.matched_text = record.gemma["evidence_text"]

        resolution = resolve_name_field([record], caller_id_name="QUINN L EXAMPLE")

        self.assertEqual(resolution.status, "gemma_final")
        self.assertEqual(resolution.final_value, "Quinn")

    def test_spelled_name_fallback_extracts_compact_last_name(self):
        named = extract_spelled_name_candidates("Bailey Sampel, EXAMPLE, please call.", ["Bailey Sampel"])
        self.assertEqual(named[0]["raw"], "Bailey Sampel")
        self.assertEqual(named[0]["value"], "Bailey Example")
        self.assertEqual(named[0]["source"], "transcript_spelling_corrected")
        self.assertEqual(named[0]["evidence_text"], "Bailey Sampel, EXAMPLE")

        introduced = extract_spelled_name_candidates("This is Bailey Sampel, EXAMPLE, calling back.")
        self.assertEqual(introduced[0]["value"], "Bailey Example")

    def test_spelled_name_fallback_adds_patient_name_when_gemma_misses(self):
        payload = self.valid_gemma_payload()

        added = watcher.add_spelled_name_fallback_candidates(
            payload,
            "This is Bailey Sampel, EXAMPLE, calling back.",
        )

        self.assertEqual(added, 1)
        self.assertEqual(payload["patient_names"][0]["raw"], "Bailey Sampel")
        self.assertEqual(payload["patient_names"][0]["value"], "Bailey Example")
        self.assertEqual(payload["patient_names"][0]["source"], "transcript_spelling_corrected")

    def test_spelled_name_fallback_extracts_hyphenated_full_name_and_multipliers(self):
        candidates = extract_spelled_name_candidates(
            "This is Avery Exampel, A-V-E-R-Y, S-A-M-P-L-E.",
        )
        self.assertEqual(candidates[0]["raw"], "Avery Exampel")
        self.assertEqual(candidates[0]["value"], "Avery Sample")

        doubled = extract_spelled_name_candidates(
            "This is Avery Exampel, A-V-E-R-Y, S-A-M-P-L-E.",
        )
        self.assertEqual(doubled[0]["value"], "Avery Sample")

    def test_spelled_name_fallback_corrects_interleaved_spoken_name_spelling(self):
        candidates = extract_spelled_name_candidates(
            "A caller for Avery, A-V-E-R-Y, Exampel, E-X-A-M-P-L-E",
            ["Avery Exampel"],
        )
        self.assertEqual(candidates[0]["raw"], "Avery Exampel")
        self.assertEqual(candidates[0]["value"], "Avery Example")
        self.assertEqual(
            candidates[0]["evidence_text"],
            "Avery, A-V-E-R-Y, Exampel, E-X-A-M-P-L-E",
        )

    def test_spelled_name_fallback_accepts_repeated_first_name_with_letter_hint(self):
        candidates = extract_spelled_name_candidates(
            "Hi, this is Avery Exampel. Avery with an A, S-A-M-P-L-E, 101-70."
        )

        self.assertEqual(candidates[0]["raw"], "Avery Exampel")
        self.assertEqual(candidates[0]["value"], "Avery Sample")

    def test_spelled_name_fallback_rejects_acronyms_addressees_and_isolated_words(self):
        self.assertEqual(extract_spelled_name_candidates("NETV"), [])
        self.assertEqual(extract_spelled_name_candidates("Bailey Sampel, EXAMPLE"), [])
        self.assertEqual(extract_spelled_name_candidates("This is Bailey Sampel, MRI."), [])
        self.assertEqual(extract_spelled_name_candidates("Hey Casey, C-A-S-E-Y.", ["Casey"]), [])
        self.assertEqual(extract_spelled_name_candidates("I am a nurse, NETV."), [])

    def test_relationship_name_fallback_extracts_subject_name(self):
        candidates = extract_relationship_name_candidates(
            "This is Robin Sample. I'm calling from my husband Rick Sample.",
        )

        self.assertEqual(candidates[0]["raw"], "Rick Sample")
        self.assertEqual(candidates[0]["value"], "Rick Sample")
        self.assertEqual(candidates[0]["source"], "relationship_subject")
        self.assertEqual(candidates[0]["evidence_text"], "my husband Rick Sample")

    def test_relationship_name_fallback_extracts_reverse_possessive_subject_name(self):
        candidates = extract_relationship_name_candidates(
            "This is Bailey Example, Casey Sample's daughter.",
        )

        self.assertEqual(candidates[0]["raw"], "Casey Sample")
        self.assertEqual(candidates[0]["value"], "Casey Sample")
        self.assertEqual(candidates[0]["source"], "relationship_subject")
        self.assertEqual(candidates[0]["evidence_text"], "Casey Sample's daughter")

    def test_relationship_name_fallback_fills_when_gemma_misses_name(self):
        text = "This is Robin Sample. I'm calling from my husband Rick Sample."
        payload = self.valid_gemma_payload()
        transcription = transcription_for_text(text)

        self.assertEqual(watcher.add_relationship_name_fallback_candidates(payload, text), 1)
        records = watcher.build_candidate_records("name", payload, transcription)
        resolution = resolve_name_field(records)

        self.assertEqual(resolution.status, "relationship_subject")
        self.assertEqual(resolution.final_value, "Rick Sample")

    def test_relationship_fallback_upgrades_existing_reverse_relationship_candidate(self):
        text = "This is Bailey Example, Casey Sample's daughter."
        payload = self.valid_gemma_payload()
        payload["patient_names"] = [
            {
                "raw": "Casey Sample",
                "value": "Casey Sample",
                "evidence_text": "Casey Sample's daughter",
                "source": "transcript",
            }
        ]

        self.assertEqual(watcher.add_relationship_name_fallback_candidates(payload, text), 1)
        self.assertEqual(len(payload["patient_names"]), 1)
        self.assertEqual(payload["patient_names"][0]["source"], "relationship_subject")
        self.assertEqual(payload["patient_names"][0]["value"], "Casey Sample")

    def test_relationship_name_resolution_prefers_subject_over_speaker(self):
        resolution = resolve_name_field([
            name_record("name:0", "Robin Sample", raw="Robin Sample", source="transcript"),
            relationship_name_record("name:1", "Rick Sample", "my husband Rick Sample"),
        ])

        self.assertEqual(resolution.status, "relationship_subject")
        self.assertEqual(resolution.final_value, "Rick Sample")

    def test_reverse_relationship_name_resolution_prefers_subject_over_speaker(self):
        text = "This is Bailey Example, Casey Sample's daughter."
        payload = self.valid_gemma_payload()
        transcription = transcription_for_text(text)

        self.assertEqual(watcher.add_self_identification_name_fallback_candidates(payload, text), 1)
        self.assertEqual(watcher.add_relationship_name_fallback_candidates(payload, text), 1)
        records = watcher.build_candidate_records("name", payload, transcription)
        resolution = resolve_name_field(records)

        self.assertEqual(resolution.status, "relationship_subject")
        self.assertEqual(resolution.final_value, "Casey Sample")
        self.assertTrue(resolution.needs_review)
        self.assertIn("multiple_field_candidates", resolution.review_reasons)

    def test_matching_caller_id_does_not_promote_self_identification_to_spelling_correction(self):
        resolution = resolve_name_field([
            name_record("name:0", "Bailey Example", raw="Bailey Example", source="self_identification")
        ], caller_id_name="Bailey Example")

        self.assertEqual(resolution.final_value, "Bailey Example")
        self.assertEqual(resolution.status, "self_identification")

    def test_reverse_relationship_generic_gemma_candidate_wins_over_caller_id_match(self):
        original_call_gemma = watcher.call_gemma_field_extraction

        def fake_gemma(*_args, **_kwargs):
            payload = self.valid_gemma_payload()
            payload["patient_names"] = [
                {
                    "raw": "Casey Sample",
                    "value": "Casey Sample",
                    "evidence_text": "Casey Sample's daughter",
                    "source": "transcript",
                }
            ]
            return payload

        watcher.call_gemma_field_extraction = fake_gemma
        try:
            settings = SimpleNamespace(
                gemma_field_extraction_enabled=True,
                verification_total_timeout_seconds=100,
                gemma_fail_open=True,
                verification_apply_resolved_values=True,
            )
            result = watcher.verify_voicemail_fields(
                "/tmp/no.wav",
                transcription_for_text("Yes, this is Bailey Example, Casey Sample's daughter."),
                {"callerid": '"Bailey Example" <217-555-0101>'},
                settings,
            )
        finally:
            watcher.call_gemma_field_extraction = original_call_gemma

        self.assertEqual(result.proposed_entities.get("name"), "Casey Sample")
        name_row = next(row for row in result.audit_rows if row["field_name"] == "name")
        self.assertEqual(name_row["status"], "gemma_final")
        self.assertFalse(name_row["needs_review"])
        self.assertNotIn("multiple_field_candidates", name_row["review_reasons"])

    def test_self_identification_still_resolves_without_relationship_subject(self):
        resolution = resolve_name_field([
            name_record("name:0", "Robin Sample", raw="Robin Sample", source="transcript")
        ])

        self.assertEqual(resolution.status, "gemma_final")
        self.assertEqual(resolution.final_value, "Robin Sample")

    def test_verification_does_not_capture_name_when_gemma_misses(self):
        original_call_gemma = watcher.call_gemma_field_extraction

        def fake_gemma(*_args, **_kwargs):
            return self.valid_gemma_payload()

        watcher.call_gemma_field_extraction = fake_gemma
        try:
            settings = SimpleNamespace(
                gemma_field_extraction_enabled=True,
                verification_total_timeout_seconds=100,
                gemma_fail_open=False,
                verification_apply_resolved_values=True,
            )
            result = watcher.verify_voicemail_fields(
                "/tmp/no.wav",
                transcription_for_text("Hi, this is Casey Example again."),
                {},
                settings,
            )
        finally:
            watcher.call_gemma_field_extraction = original_call_gemma

        self.assertIsNone(result.proposed_entities.get("name"))
        self.assertEqual(result.audit_rows[0]["status"], "not_included")

    def test_verification_does_not_capture_names_when_gemma_schema_invalid(self):
        original_call_gemma = watcher.call_gemma_field_extraction

        def fake_gemma(*_args, **_kwargs):
            raise GemmaSchemaError("Gemma response missing required key 'patient_names'")

        watcher.call_gemma_field_extraction = fake_gemma
        try:
            settings = SimpleNamespace(
                gemma_field_extraction_enabled=True,
                verification_total_timeout_seconds=100,
                gemma_fail_open=True,
                verification_apply_resolved_values=True,
            )
            cases = [
                ("Hi, this is Robin Sample.", "Robin Sample"),
                ("Hi, this is for Robin Keller again.", "Robin Keller"),
            ]
            for transcript, expected_name in cases:
                with self.subTest(transcript=transcript):
                    result = watcher.verify_voicemail_fields(
                        "/tmp/no.wav",
                        transcription_for_text(transcript),
                        {},
                        settings,
                    )

                    self.assertIsNone(result.proposed_entities.get("name"))
                    name_row = next(row for row in result.audit_rows if row["field_name"] == "name")
                    self.assertEqual(name_row["status"], "legacy_fallback")
                    self.assertIn("gemma_invalid_json", name_row["review_reasons"])
        finally:
            watcher.call_gemma_field_extraction = original_call_gemma

    def test_verification_does_not_capture_subject_reference_when_gemma_misses(self):
        original_call_gemma = watcher.call_gemma_field_extraction

        def fake_gemma(*_args, **_kwargs):
            return self.valid_gemma_payload()

        watcher.call_gemma_field_extraction = fake_gemma
        try:
            settings = SimpleNamespace(
                gemma_field_extraction_enabled=True,
                verification_total_timeout_seconds=100,
                gemma_fail_open=False,
                verification_apply_resolved_values=True,
            )
            result = watcher.verify_voicemail_fields(
                "/tmp/no.wav",
                transcription_for_text(
                    "Hi, this is Jordan. I'm a nurse with Acme Benefits. "
                    "I'm calling in regards to Morgan Example."
                ),
                {},
                settings,
            )
        finally:
            watcher.call_gemma_field_extraction = original_call_gemma

        self.assertIsNone(result.proposed_entities.get("name"))
        self.assertEqual(result.audit_rows[0]["status"], "not_included")
        self.assertFalse(result.audit_rows[0]["needs_review"])
        self.assertIn("no_gemma_candidate", result.audit_rows[0]["review_reasons"])

    def test_subject_reference_alone_requires_confirmation(self):
        resolution = resolve_name_field([
            subject_reference_name_record("name:0", "Morgan Example", "calling in regards to Morgan Example")
        ])

        self.assertEqual(resolution.final_value, "Morgan Example")
        self.assertEqual(resolution.status, "subject_reference")
        self.assertTrue(resolution.needs_review)
        self.assertIn("subject_reference_unconfirmed", resolution.review_reasons)

    def test_subject_reference_with_gemma_agreement_allows_name(self):
        resolution = resolve_name_field([
            subject_reference_name_record("name:0", "Morgan Example", "calling in regards to Morgan Example"),
            name_record("name:1", "Morgan Example", raw="Morgan Example", source="transcript"),
        ])

        self.assertEqual(resolution.final_value, "Morgan Example")
        self.assertEqual(resolution.status, "gemma_final")

    def test_subject_reference_with_relationship_agreement_allows_name(self):
        resolution = resolve_name_field([
            subject_reference_name_record("name:0", "Morgan Example", "calling in regards to Morgan Example"),
            relationship_name_record("name:1", "Morgan Example", "my husband Morgan Example"),
        ])

        self.assertEqual(resolution.final_value, "Morgan Example")
        self.assertEqual(resolution.status, "relationship_subject")

    def test_self_identification_fallback_trims_calling_back(self):
        candidates = extract_self_identification_name_candidates(
            "Hi, Example. This is Taylor Sample calling back."
        )

        self.assertEqual(candidates[0]["value"], "Taylor Sample")
        self.assertEqual(candidates[0]["evidence_text"], "This is Taylor Sample")
        self.assertEqual(candidates[0]["source"], "self_identification")

    def test_self_identification_for_name_evidence_includes_full_cleaned_name(self):
        candidates = extract_self_identification_name_candidates(
            "Hi, this is for Robin Keller again."
        )

        self.assertEqual(candidates[0]["value"], "Robin Keller")
        self.assertEqual(candidates[0]["evidence_text"], "this is for Robin Keller")

    def test_self_identification_fallback_trims_spelling_confirmation(self):
        candidates = extract_self_identification_name_candidates(
            "Hi, this is Taylor Example. That's J-O-R-D-A-N. I need help."
        )

        self.assertEqual(candidates[0]["value"], "Taylor Example")
        self.assertEqual(candidates[0]["evidence_text"], "this is Taylor Example")

    def test_gemma_name_and_self_identification_spelling_confirmation_do_not_conflict(self):
        original_call_gemma = watcher.call_gemma_field_extraction

        def fake_gemma(*_args, **_kwargs):
            payload = self.valid_gemma_payload()
            payload["patient_names"] = [
                {
                    "raw": "Taylor Example",
                    "value": "Taylor Example",
                    "evidence_text": "Hi, this is Taylor Example. That's J-O-R-D-A-N.",
                }
            ]
            return payload

        watcher.call_gemma_field_extraction = fake_gemma
        try:
            settings = SimpleNamespace(
                gemma_field_extraction_enabled=True,
                verification_total_timeout_seconds=100,
                gemma_fail_open=True,
                verification_apply_resolved_values=True,
            )
            result = watcher.verify_voicemail_fields(
                "/tmp/no.wav",
                transcription_for_text("Hi, this is Taylor Example. That's J-O-R-D-A-N. I need help."),
                {},
                settings,
            )
        finally:
            watcher.call_gemma_field_extraction = original_call_gemma

        self.assertEqual(result.proposed_entities.get("name"), "Taylor Example")
        name_row = next(row for row in result.audit_rows if row["field_name"] == "name")
        self.assertNotEqual(name_row["status"], "ambiguous")
        self.assertNotIn("multiple_field_candidates", name_row["review_reasons"])

    def test_ordinary_its_sentence_does_not_create_self_identification_candidate(self):
        self.assertEqual(
            extract_self_identification_name_candidates("It's not affecting any other systems."),
            [],
        )

    def test_verification_does_not_capture_subject_name_when_gemma_empty(self):
        original_call_gemma = watcher.call_gemma_field_extraction

        def fake_gemma(*_args, **_kwargs):
            return self.valid_gemma_payload()

        watcher.call_gemma_field_extraction = fake_gemma
        try:
            settings = SimpleNamespace(
                gemma_field_extraction_enabled=True,
                verification_total_timeout_seconds=100,
                gemma_fail_open=True,
                verification_apply_resolved_values=True,
            )
            result = watcher.verify_voicemail_fields(
                "/tmp/no.wav",
                transcription_for_text(
                    "Hi, this is for Robin Keller again. It's not affecting any other systems."
                ),
                {},
                settings,
            )
        finally:
            watcher.call_gemma_field_extraction = original_call_gemma

        self.assertIsNone(result.proposed_entities.get("name"))
        name_row = next(row for row in result.audit_rows if row["field_name"] == "name")
        self.assertEqual(name_row["status"], "not_included")
        self.assertNotIn("multiple_field_candidates", name_row["review_reasons"])

    def test_name_fallbacks_match_v1_regression_examples(self):
        self.assertEqual(
            extract_self_identification_name_candidates("Hi, this is Quinn.")[0]["value"],
            "Quinn",
        )
        self.assertEqual(
            extract_self_identification_name_candidates(
                "This is Robin Sample. I just got the phone with you."
            )[0]["value"],
            "Robin Sample",
        )
        self.assertEqual(
            extract_relationship_name_candidates(
                "my husband is due for the rooster cone injection"
            ),
            [],
        )
        self.assertEqual(
            extract_relationship_name_candidates(
                "my husband Quinn Example about his elbow"
            )[0]["value"],
            "Quinn Example",
        )
        self.assertEqual(
            extract_relationship_name_candidates(
                "I'm calling on behalf of Avery Sample."
            )[0]["value"],
            "Avery Sample",
        )

        proxy_subject = resolve_name_field([
            name_record("name:0", "Morgan Sample", source="self_identification"),
            relationship_name_record("name:1", "Avery Sample", "calling on behalf of Avery Sample"),
        ])
        self.assertEqual(proxy_subject.final_value, "Avery Sample")

        relationship_subject = resolve_name_field([
            name_record("name:0", "Quinn Example", source="self_identification"),
            relationship_name_record("name:1", "Quinn Example", "my husband Quinn Example"),
        ])
        self.assertEqual(relationship_subject.final_value, "Quinn Example")

    def test_name_fallbacks_match_v15_overcapture_examples(self):
        self.assertEqual(
            extract_explicit_patient_name_candidates(
                "I'm calling on a patient, First Last, date of birth 01/24/1959."
            )[0]["value"],
            "First Last",
        )
        self.assertEqual(
            extract_relationship_name_candidates(
                "I got a call from his daughter that said we were supposed"
            ),
            [],
        )
        self.assertEqual(
            extract_self_identification_name_candidates(
                "Yes, I am calling for Bailey. My name is Avery Example."
            )[0]["value"],
            "Avery Example",
        )
        self.assertEqual(
            extract_self_identification_name_candidates(
                "Hey Avery, how are you? Good afternoon. My name is Example Example. I am with Example Health Network."
            )[0]["value"],
            "Example Example",
        )
        self.assertEqual(
            extract_relationship_name_candidates(
                "Hi, this is Casey Example calling on behalf of Example Health Network."
            ),
            [],
        )
        self.assertEqual(
            extract_self_identification_name_candidates(
                "This is First Last. It's called Child Therapy. It's on White Oaks Drive."
            ),
            [
                {
                    "raw": "First Last",
                    "value": "First Last",
                    "evidence_text": "This is First Last",
                    "source": "self_identification",
                    "caller_id_used": "",
                    "confidence": "high",
                }
            ],
        )
        self.assertEqual(
            extract_relationship_name_candidates(
                "trying to get an appointment for my mother scheduled as soon as we can"
            ),
            [],
        )
        self.assertEqual(
            extract_self_identification_name_candidates(
                "this is Example Example. I am home right now."
            ),
            [
                {
                    "raw": "Example Example",
                    "value": "Example Example",
                    "evidence_text": "this is Example Example",
                    "source": "self_identification",
                    "caller_id_used": "",
                    "confidence": "high",
                }
            ],
        )
        self.assertEqual(
            extract_self_identification_name_candidates(
                "The name is Bailey Example. I am having problems"
            ),
            [
                {
                    "raw": "Bailey Example",
                    "value": "Bailey Example",
                    "evidence_text": "The name is Bailey Example",
                    "source": "self_identification",
                    "caller_id_used": "",
                    "confidence": "high",
                }
            ],
        )
        self.assertEqual(
            extract_self_identification_name_candidates(
                "Hi, my name is Bailey Example. It is probably safest..."
            ),
            [
                {
                    "raw": "Bailey Example",
                    "value": "Bailey Example",
                    "evidence_text": "my name is Bailey Example",
                    "source": "self_identification",
                    "caller_id_used": "",
                    "confidence": "high",
                }
            ],
        )
        self.assertEqual(
            extract_self_identification_name_candidates(
                "This is Bailey Example. I already called and left a message. I am very impatient."
            ),
            [
                {
                    "raw": "Bailey Example",
                    "value": "Bailey Example",
                    "evidence_text": "This is Bailey Example",
                    "source": "self_identification",
                    "caller_id_used": "",
                    "confidence": "high",
                }
            ],
        )

    def test_name_resolution_prefers_strong_fallbacks_over_noisy_generic_names(self):
        cases = [
            (
                "I'm calling on a patient, First Last, date of birth 01/24/1959. "
                "I got a call from his daughter that said we were supposed",
                "That Said",
                "his daughter that said",
                "First Last",
            ),
            (
                "Yes, I am calling for Bailey. My name is Avery Example.",
                "Taylor",
                "calling for Taylor",
                "Avery Example",
            ),
            (
                "This is First Last. I just saw Dr. Marty. I believe the little girl's name "
                "that I was with was Shelby.",
                "Shelby",
                "Shelby",
                "First Last",
            ),
            (
                "The patient name is Bailey Example, 1-1-70, and I am their relative, Avery.",
                "Teresa",
                "her daughter, Teresa",
                "Bailey Example",
            ),
            (
                "This is Robin Example again, trying to get an appointment "
                "scheduled as soon as we can.",
                "Scheduled As Soon As",
                "my mother scheduled as soon as",
                "Robin Example",
            ),
            (
                "this is Example Example. I am returning your call. I am home right now.",
                "Home Right Now",
                "I'm home right now",
                "Example Example",
            ),
            (
                "Hey Avery, how are you? Good afternoon. My name is Example Example. I am with Example Health Network.",
                "Seth",
                "Hey Seth",
                "Example Example",
            ),
            (
                "Hi, this is Casey Example calling on behalf of Example Health Network.",
                "Example Health Network",
                "on behalf of Example Health Network",
                "Casey Example",
            ),
            (
                "I am calling on a patient, Patient Example. A family member "
                "that said we were supposed to get an order for a lumbar puncture.",
                "That Said",
                "his daughter that said",
                "Patient Example",
            ),
            (
                "Hi, my name is Bailey Example. It is probably safest...",
                "Probably Safest",
                "It's probably safest",
                "Bailey Example",
            ),
            (
                "The name is Bailey Example. I am having problems",
                "Having Problems",
                "I'm having problems",
                "Bailey Example",
            ),
            (
                "This is Bailey Example. I already called and left a message. I am very impatient.",
                "Very Impatient",
                "I'm very impatient",
                "Bailey Example",
            ),
            (
                "Yes, my name is Casey Example, and my provider is from Example Clinic, "
                "and they told me to call you guys to see if I can get into therapy. "
                "I had a stroke about three weeks ago",
                "Three Weeks Ago",
                "three weeks ago",
                "Casey Example",
            ),
        ]

        for transcript, noisy_name, noisy_evidence, expected in cases:
            with self.subTest(expected=expected):
                payload = self.valid_gemma_payload()
                payload["patient_names"] = [
                    {
                        "raw": noisy_name,
                        "value": noisy_name,
                        "evidence_text": noisy_evidence,
                        "source": "transcript",
                    }
                ]
                watcher.add_spelled_name_fallback_candidates(payload, transcript)
                watcher.add_explicit_patient_name_fallback_candidates(payload, transcript)
                watcher.add_self_identification_name_fallback_candidates(payload, transcript)
                watcher.add_relationship_name_fallback_candidates(payload, transcript)
                watcher.add_subject_reference_name_fallback_candidates(payload, transcript)
                records = watcher.build_candidate_records("name", payload, transcription_for_text(transcript))
                resolution = resolve_name_field(records)

                self.assertEqual(resolution.final_value, expected)

    def test_name_fallbacks_reject_role_and_non_person_subjects(self):
        self.assertEqual(
            extract_self_identification_name_candidates("Hi, I'm a nurse with Acme Benefits."),
            [],
        )
        self.assertEqual(
            extract_subject_reference_name_candidates("I'm calling about your appointment."),
            [],
        )
        self.assertEqual(
            extract_subject_reference_name_candidates("calling about knee surgery tomorrow"),
            [],
        )
        self.assertEqual(
            extract_subject_reference_name_candidates("calling about pre authorization"),
            [],
        )
        self.assertEqual(
            extract_subject_reference_name_candidates("calling regarding benefits eligibility"),
            [],
        )
        self.assertEqual(
            extract_subject_reference_name_candidates("calling about account balance"),
            [],
        )

    def test_spelled_name_resolution_prefers_transcript_spelling(self):
        ordinary = name_record("name:0", "Bailey Sampel", raw="Bailey Sampel")
        ordinary.gemma["evidence_text"] = "Bailey Sampel"
        ordinary.attribution.evidence_text = "Bailey Sampel"
        ordinary.attribution.matched_text = "Bailey Sampel"
        corrected = spelled_name_record("name:1", "Bailey Sampel", "Bailey Example", "Bailey Sampel, EXAMPLE")

        resolution = resolve_name_field([ordinary, corrected])

        self.assertEqual(resolution.status, "transcript_spelling_corrected")
        self.assertEqual(resolution.final_value, "Bailey Example")

    def test_spelled_name_resolution_preserves_same_value_full_name_from_gemma(self):
        corrected = spelled_name_record(
            "name:0",
            "Bailey Example",
            "Bailey Example",
            "my name is Bailey, B-A-I-L-E-Y, Example, E-X-A-M-P-L-E",
        )
        short_self_id = name_record("name:1", "Bailey", raw="Bailey", source="self_identification")
        short_self_id.gemma["evidence_text"] = "my name is Bailey"
        short_self_id.attribution.evidence_text = "my name is Bailey"
        short_self_id.attribution.matched_text = "my name is Bailey"

        resolution = resolve_name_field([corrected, short_self_id])

        self.assertEqual(resolution.status, "transcript_spelling_corrected")
        self.assertEqual(resolution.final_value, "Bailey Example")

    def test_spelled_name_conflicts_pick_best_with_review_flag(self):
        resolution = resolve_name_field([
            spelled_name_record("name:0", "Bailey Sampel", "Bailey Sample", "Bailey Sampel, S-A-M-P-L-E"),
            spelled_name_record("name:1", "Bailey Sampel", "Bailey Example", "Bailey Sampel, E-X-A-M-P-L-E"),
        ])

        self.assertEqual(resolution.status, "transcript_spelling_corrected")
        self.assertTrue(resolution.needs_review)
        self.assertIn(resolution.final_value, {"Bailey Sample", "Bailey Example"})
        self.assertIn("multiple_field_candidates", resolution.review_reasons)

    def test_spelled_name_transcript_correction_preserves_spelling_evidence(self):
        transcript = "This is Avery Exampel, A-V-E-R-Y, S-A-M-P-L-E. Please call."
        corrected, entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            {},
            [
                {
                    "field_name": "name",
                    "final_value": "Avery Sample",
                    "status": "transcript_spelling_corrected",
                    "gemma_json": [
                        {
                            "raw": "Avery Exampel",
                            "value": "Avery Sample",
                            "source": "transcript_spelling_corrected",
                            "evidence_text": "Avery Exampel, A-V-E-R-Y, S-A-M-P-L-E",
                        }
                    ],
                }
            ],
        )

        self.assertIn("Avery Sample, A-V-E-R-Y, S-A-M-P-L-E", corrected)
        self.assertIn("S-A-M-P-L-E", corrected)
        self.assertEqual(entities["transcript_corrections"][0]["status"], "transcript_spelling_corrected")

    def test_spelled_name_transcript_correction_rewrites_repeated_changed_token(self):
        transcript = (
            "This is Taylor Sampel, and I need to reschedule. "
            "Please call me back this is Taylor T-A-Y-L-O-R Sampel S-A-M-P-L-E."
        )
        entities = {
            "_word_timestamps": [
                {"word": word}
                for word in [
                    "This",
                    "is",
                    "Taylor",
                    "Sampel,",
                    "and",
                    "I",
                    "need",
                    "to",
                    "reschedule.",
                    "Please",
                    "call",
                    "me",
                    "back",
                    "this",
                    "is",
                    "Taylor",
                    "T-A-Y-L-O-R",
                    "Sampel",
                    "S-A-M-P-L-E.",
                ]
            ]
        }

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            [
                {
                    "field_name": "name",
                    "final_value": "Taylor Sample",
                    "status": "transcript_spelling_corrected",
                    "gemma_json": [
                        {
                            "raw": "Taylor Sampel",
                            "value": "Taylor Sample",
                            "source": "transcript_spelling_corrected",
                            "evidence_text": "this is Taylor T-A-Y-L-O-R Sampel S-A-M-P-L-E",
                        }
                    ],
                }
            ],
        )

        self.assertEqual(
            corrected,
            "This is Taylor Sample, and I need to reschedule. "
            "Please call me back this is Taylor T-A-Y-L-O-R Sample S-A-M-P-L-E.",
        )
        self.assertIn("T-A-Y-L-O-R Sample S-A-M-P-L-E", corrected)
        self.assertEqual(corrected_entities["transcript_corrections"][0]["transcript_replacements"], 2)
        words = corrected_entities["_word_timestamps"]
        self.assertEqual(words[3]["word"], "Sample,")
        self.assertEqual(words[17]["word"], "Sample")
        self.assertTrue(corrected_entities["transcript_corrections"][0]["word_timestamps_updated"])

    def test_unequal_name_token_count_replaces_full_phrase_not_individual_tokens(self):
        transcript = "Avery Exampel called about scheduling. Exampel was repeated later."

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            {},
            [
                {
                    "field_name": "name",
                    "final_value": "Avery Lee Sample",
                    "status": "transcript_spelling_corrected",
                    "gemma_json": [
                        {
                            "raw": "Avery Exampel",
                            "value": "Avery Lee Sample",
                            "source": "transcript_spelling_corrected",
                            "evidence_text": "Avery Exampel, A-V-E-R-Y, S-A-M-P-L-E",
                        }
                    ],
                }
            ],
        )

        self.assertEqual(
            corrected,
            "Avery Lee Sample called about scheduling. Exampel was repeated later.",
        )
        self.assertEqual(corrected_entities["transcript_corrections"][0]["transcript_replacements"], 1)

    def test_multiple_dob_candidates_different_is_ambiguous(self):
        resolution = resolve_dob_field([
            dob_record("dob:0", "01/05/1980"),
            dob_record("dob:1", "01/15/1980"),
        ], today=date(2026, 5, 2))
        self.assertEqual(resolution.status, "ambiguous")
        self.assertTrue(resolution.needs_review)
        self.assertIsNone(resolution.final_value)

    def test_dob_parakeet_disagreement_flags_review_without_override(self):
        resolution = resolve_dob_field([
            dob_record("dob:0", "01/05/1980", parakeet_text="date of birth is 01/15/1980")
        ], today=date(2026, 5, 2))
        self.assertEqual(resolution.final_value, "01/05/1980")
        self.assertEqual(resolution.normalized_value, "01051980")
        self.assertEqual(resolution.status, "gemma_final")
        self.assertTrue(resolution.needs_review)
        self.assertIn("dob_parakeet_audit_disagreement", resolution.review_reasons)

    def test_dob_parakeet_agreement_preserves_gemma_final(self):
        resolution = resolve_dob_field([
            dob_record("dob:0", "01/05/1980", parakeet_text="date of birth is 01/05/1980")
        ], today=date(2026, 5, 2))
        self.assertEqual(resolution.final_value, "01/05/1980")
        self.assertEqual(resolution.status, "gemma_final")
        self.assertFalse(resolution.needs_review)

    def test_dob_parakeet_to_1353_agrees_with_february_thirteenth(self):
        resolution = resolve_dob_field([
            dob_record("dob:0", "02/13/1953", parakeet_text="This is Casey. Date of birth to 1353."),
            dob_record("dob:1", "02/13/1953", parakeet_text="This is an on the drop roll date of birth to 1353."),
        ], today=date(2026, 5, 2))
        self.assertEqual(resolution.final_value, "02/13/1953")
        self.assertEqual(resolution.status, "gemma_final")
        self.assertFalse(resolution.needs_review)

    def test_dob_multiple_parakeet_values_do_not_override(self):
        resolution = resolve_dob_field([
            dob_record("dob:0", "01/05/1980", parakeet_text="date of birth is 01/15/1980 or 02/15/1980")
        ], today=date(2026, 5, 2))
        self.assertEqual(resolution.final_value, "01/05/1980")
        self.assertEqual(resolution.status, "gemma_final")
        self.assertTrue(resolution.needs_review)
        self.assertIn("multiple_parakeet_dobs", resolution.review_reasons)

    def test_dob_parakeet_does_not_resolve_multiple_gemma_candidates(self):
        resolution = resolve_dob_field([
            dob_record("dob:0", "01/05/1980", parakeet_text="date of birth is 01/15/1980"),
            dob_record("dob:1", "02/05/1980", parakeet_text="date of birth is 01/15/1980"),
        ], today=date(2026, 5, 2))
        self.assertIsNone(resolution.final_value)
        self.assertEqual(resolution.status, "ambiguous")
        self.assertTrue(resolution.needs_review)
        self.assertIn("multiple_field_candidates", resolution.review_reasons)
        self.assertIn("dob_parakeet_audit_disagreement", resolution.review_reasons)

    def test_dob_implausible_parakeet_value_does_not_override(self):
        resolution = resolve_dob_field([
            dob_record("dob:0", "01/05/1980", parakeet_text="date of birth is 01/15/2035")
        ], today=date(2026, 5, 2))
        self.assertEqual(resolution.final_value, "01/05/1980")
        self.assertEqual(resolution.status, "gemma_final")

    def test_clip_padding_clamped_to_segment_bounds(self):
        words = [
            {"word": "reach", "start": 1.0, "end": 1.2},
            {"word": "me", "start": 1.2, "end": 1.3},
            {"word": "at", "start": 1.3, "end": 1.4},
            {"word": "202-555-0109", "start": 1.6, "end": 2.5},
        ]
        segments = [{"text": "reach me at 202-555-0109", "start": 0.8, "end": 2.8}]
        attribution = map_evidence_to_timestamps(
            "name",
            "name:0",
            "reach me at 202-555-0109",
            words,
            segments,
            "2025550109",
        )
        self.assertTrue(attribution.mapped)
        self.assertEqual(attribution.clip_start, 0.8)
        self.assertEqual(attribution.clip_end, 2.8)

    def test_phone_clip_uses_wide_window_for_short_digit_timestamp(self):
        words = [
            {"word": "call.", "start": 21.32, "end": 21.5},
            {"word": "202-555-0102.", "start": 22.2, "end": 22.6},
            {"word": "Thank", "start": 24.34, "end": 24.6},
            {"word": "you.", "start": 24.6, "end": 24.78},
        ]
        segments = [
            {
                "text": "So please give me a call. 202-555-0102. Thank you.",
                "start": 20.06,
                "end": 24.78,
            }
        ]
        attribution = map_evidence_to_timestamps(
            "callback_number",
            "callback_number:0",
            "202-555-0102",
            words,
            segments,
            "2025550102",
        )
        self.assertTrue(attribution.mapped)
        self.assertLessEqual(attribution.clip_start, 20.2)
        self.assertGreaterEqual(attribution.clip_end, 26.6)

    def test_extract_numbers_from_text_spoken_and_numeric(self):
        values = {item.normalized for item in extract_numbers_from_text("fax two zero two five five five zero one one one")}
        self.assertEqual(values, {"2025550111"})

    def test_near_phone_fallback_adds_partial_callback_only(self):
        text = (
            "Testing, testing. 1, 2, 3, 4, 5. This is Bailey Example. "
            "Date of birth, 05/21/2001. A good callback number is 202-555-012. "
            "And a good fax number is 202-555-0113."
        )
        payload = self.valid_gemma_payload()

        added = watcher.add_near_phone_fallback_candidates(payload, text)

        self.assertEqual(added, 1)
        self.assertEqual(payload["callback_numbers"][0]["raw"], "202-555-012")
        self.assertEqual(payload["callback_numbers"][0]["source"], "near_phone_fragment")
        self.assertEqual(payload["fax_numbers"], [])
        self.assertEqual(payload["uncertain_numbers"], [])

    def test_near_phone_fallback_ignores_test_counts_dates_and_valid_numbers(self):
        payload = self.valid_gemma_payload()
        text = (
            "Testing 1, 2, 3, 4, 5. Date of birth is 05/21/2001. "
            "A good fax number is 202-555-0113."
        )

        added = watcher.add_near_phone_fallback_candidates(payload, text)

        self.assertEqual(added, 0)
        self.assertEqual(payload["callback_numbers"], [])
        self.assertEqual(payload["fax_numbers"], [])

    def test_near_phone_fallback_adds_overlong_callback(self):
        payload = self.valid_gemma_payload()
        text = "A good callback number is 202-555-01204. Thank you."

        added = watcher.add_near_phone_fallback_candidates(payload, text)

        self.assertEqual(added, 1)
        self.assertEqual(payload["callback_numbers"][0]["raw"], "202-555-01204")

    def test_near_phone_fallback_parakeet_completes_callback_span(self):
        text = (
            "Testing, testing. 1, 2, 3, 4, 5. This is a test voicemail. "
            "This is Bailey Example. Date of birth, 01/01/1970. "
            "A good callback number is 202-555-012. "
            "And a good fax number is 202-555-0113. Thank you. Bye."
        )
        original_call_gemma = watcher.call_gemma_field_extraction
        original_run_parakeet = watcher.run_parakeet_for_record

        def fake_gemma(*_args, **_kwargs):
            payload = self.valid_gemma_payload()
            payload["fax_numbers"] = [
                {
                    "raw": "202-555-0113",
                    "normalized": "2025550113",
                    "formatted": "202-555-0113",
                    "label_cue": "fax number",
                    "evidence_text": "fax number is 202-555-0113",
                }
            ]
            return payload

        def fake_run_parakeet(_wav_path, record, *_args, **_kwargs):
            numbers = ["2025550120"] if record.field_name == "callback_number" else ["2025550113"]
            record.parakeet = ParakeetResult(record.candidate_id, normalized_numbers=numbers)

        watcher.call_gemma_field_extraction = fake_gemma
        watcher.run_parakeet_for_record = fake_run_parakeet
        try:
            settings = SimpleNamespace(
                gemma_field_extraction_enabled=True,
                verification_total_timeout_seconds=100,
                gemma_fail_open=True,
                verification_apply_resolved_values=True,
            )
            transcription = transcription_for_text(text)
            result = watcher.verify_voicemail_fields("/tmp/no.wav", transcription, {}, settings)
            corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
                transcription.text,
                result.proposed_entities,
                result.audit_rows,
            )
        finally:
            watcher.call_gemma_field_extraction = original_call_gemma
            watcher.run_parakeet_for_record = original_run_parakeet

        self.assertEqual(result.proposed_entities["callback_number"], "202-555-0120")
        self.assertEqual(result.proposed_entities["fax_number"], "202-555-0113")
        self.assertIn("A good callback number is 202-555-0120.", corrected)
        self.assertIn("good fax number is 202-555-0113", corrected)
        self.assertIn("202-555-0120", [item["word"] for item in corrected_entities["_word_timestamps"]])

    def test_near_phone_fallback_parakeet_repairs_overlong_callback_span(self):
        text = "A good callback number is 202-555-01204. Thank you."
        original_call_gemma = watcher.call_gemma_field_extraction
        original_run_parakeet = watcher.run_parakeet_for_record

        watcher.call_gemma_field_extraction = lambda *_args, **_kwargs: self.valid_gemma_payload()

        def fake_run_parakeet(_wav_path, record, *_args, **_kwargs):
            record.parakeet = ParakeetResult(record.candidate_id, normalized_numbers=["2025550120"])

        watcher.run_parakeet_for_record = fake_run_parakeet
        try:
            settings = SimpleNamespace(
                gemma_field_extraction_enabled=True,
                verification_total_timeout_seconds=100,
                gemma_fail_open=True,
                verification_apply_resolved_values=True,
            )
            transcription = transcription_for_text(text)
            result = watcher.verify_voicemail_fields("/tmp/no.wav", transcription, {}, settings)
            corrected, _entities = apply_verified_phone_corrections_to_transcript(
                transcription.text,
                result.proposed_entities,
                result.audit_rows,
            )
        finally:
            watcher.call_gemma_field_extraction = original_call_gemma
            watcher.run_parakeet_for_record = original_run_parakeet

        self.assertEqual(result.proposed_entities["callback_number"], "202-555-0120")
        self.assertEqual(corrected, "A good callback number is 202-555-0120. Thank you.")

    def test_near_phone_fallback_parakeet_completes_fax_span(self):
        text = "A good fax number is 202-555-011. Thank you."
        original_call_gemma = watcher.call_gemma_field_extraction
        original_run_parakeet = watcher.run_parakeet_for_record

        watcher.call_gemma_field_extraction = lambda *_args, **_kwargs: self.valid_gemma_payload()

        def fake_run_parakeet(_wav_path, record, *_args, **_kwargs):
            record.parakeet = ParakeetResult(record.candidate_id, normalized_numbers=["2025550113"])

        watcher.run_parakeet_for_record = fake_run_parakeet
        try:
            settings = SimpleNamespace(
                gemma_field_extraction_enabled=True,
                verification_total_timeout_seconds=100,
                gemma_fail_open=True,
                verification_apply_resolved_values=True,
            )
            transcription = transcription_for_text(text)
            result = watcher.verify_voicemail_fields("/tmp/no.wav", transcription, {}, settings)
            corrected, _entities = apply_verified_phone_corrections_to_transcript(
                transcription.text,
                result.proposed_entities,
                result.audit_rows,
            )
        finally:
            watcher.call_gemma_field_extraction = original_call_gemma
            watcher.run_parakeet_for_record = original_run_parakeet

        self.assertEqual(result.proposed_entities["fax_number"], "202-555-0113")
        self.assertEqual(corrected, "A good fax number is 202-555-0113. Thank you.")


    def test_partial_parakeet_success_with_conflicting_whisper_spans_is_ambiguous(self):
        records = [
            phone_record("callback_number:0", "callback_number", ["2025550109"], ["2025550109"]),
            phone_record("callback_number:1", "callback_number", ["2025550115"], parakeet_error="unavailable"),
        ]
        resolution = resolve_phone_field("callback_number", records)
        self.assertEqual(resolution.status, "ambiguous")
        self.assertTrue(resolution.needs_review)
        self.assertIsNone(resolution.final_value)
        self.assertIn("parakeet_unavailable", resolution.review_reasons)
        self.assertIn("multiple_field_candidates", resolution.review_reasons)

    def test_caller_id_cannot_create_missing_spoken_name(self):
        record = CandidateRecord(
            candidate_id="name:0",
            field_name="name",
            gemma={
                "candidate_id": "name:0",
                "raw": None,
                "value": "Catherine Example",
                "source": "caller_id_corrected",
                "caller_id_used": "Catherine Example",
                "evidence_text": "please call me back",
            },
            attribution=AttributionResult(
                candidate_id="name:0",
                field_name="name",
                evidence_text="please call me back",
                mapped=True,
                mapping_method="word",
                matched_text="please call me back",
            ),
        )
        resolution = resolve_name_field([record], caller_id_name="Catherine Example")
        self.assertEqual(resolution.status, "not_included")
        self.assertIsNone(resolution.final_value)
        self.assertIn("caller_id_correction_disabled", resolution.review_reasons)

    def test_dob_month_name_evidence_must_support_day(self):
        resolution = resolve_dob_field([
            dob_record("dob:0", "01/05/1980", evidence="date of birth is January 6 1980")
        ], today=date(2026, 5, 2))
        self.assertEqual(resolution.status, "not_included")
        self.assertIsNone(resolution.final_value)
        self.assertIn("dob_implausible", resolution.review_reasons)

    def test_compact_dob_parser_accepts_cue_bound_short_forms(self):
        examples = {
            "5566": "05/05/1966",
            "6448": "06/04/1948",
            "625-54": "06/25/1954",
            "062554": "06/25/1954",
            "6/25/54": "06/25/1954",
            "424 of 60": "04/24/1960",
            "4 24 of 60": "04/24/1960",
            "04 24 of 60": "04/24/1960",
            "six four forty eight": "06/04/1948",
            "six twenty five fifty four": "06/25/1954",
        }
        for raw, expected in examples.items():
            self.assertEqual(format_dob(parse_dob(raw)), expected)

    def test_compact_dob_candidates_require_cue_or_patient_name(self):
        cued = extract_compact_dob_candidates("Please note date of birth 5566.")
        self.assertEqual(cued[0]["normalized"], "05/05/1966")
        self.assertEqual(cued[0]["raw"], "5566")

        spoken = extract_compact_dob_candidates("Date of birth six four forty eight.")
        self.assertEqual(spoken[0]["normalized"], "06/04/1948")

        cued_filler = extract_compact_dob_candidates("Please note date of birth 424 of 60.")
        self.assertEqual(cued_filler[0]["normalized"], "04/24/1960")
        self.assertEqual(cued_filler[0]["raw"], "424 of 60")

        named = extract_compact_dob_candidates("Jane Example, 625-54 needs a call.", ["Jane Example"])
        self.assertEqual(named[0]["normalized"], "06/25/1954")
        self.assertEqual(named[0]["evidence_text"], "Jane Example, 625-54")

        named_filler = extract_compact_dob_candidates("Jordan Sample, 424 of 60 needs a call.")
        self.assertEqual(named_filler[0]["normalized"], "04/24/1960")
        self.assertEqual(named_filler[0]["evidence_text"], "Jordan Sample, 424 of 60")

        introduced = extract_compact_dob_candidates("This is Jane Example, 625-54, calling back.")
        self.assertEqual(introduced[0]["normalized"], "06/25/1954")

        self.assertEqual(extract_compact_dob_candidates("424 of 60 needs a call."), [])
        self.assertEqual(extract_compact_dob_candidates("callback 424 of 60"), [])
        self.assertEqual(extract_compact_dob_candidates("625-54 needs a call."), [])
        self.assertEqual(extract_compact_dob_candidates("callback 625-54"), [])
        self.assertEqual(extract_compact_dob_candidates("I am a nurse, 625-54."), [])
        self.assertEqual(extract_compact_dob_candidates("Jane Example, 625-54", []), [])
        self.assertEqual(
            extract_compact_dob_candidates(
                "Bailey Example. My telephone number is (312) 444-0",
                ["Bailey Example"],
            ),
            [],
        )

    def test_compact_dob_fallback_accepts_birthdate_and_spelling_tail_examples(self):
        lisa = extract_compact_dob_candidates("Casey Example, birthdate 7763.")
        self.assertEqual(lisa[0]["normalized"], "07/07/1963")
        self.assertEqual(lisa[0]["raw"], "7763")

        jody = extract_compact_dob_candidates(
            "Hi, this is Avery Exampel. Avery with an A, S-A-M-P-L-E, 725-59.",
            ["Avery Exampel"],
        )
        self.assertEqual(jody[0]["normalized"], "07/25/1959")
        self.assertEqual(jody[0]["raw"], "725-59")

    def test_compact_dob_fallback_fills_when_gemma_misses_dob(self):
        text = "Jane Example, 625-54 needs a call."
        payload = self.valid_gemma_payload()
        payload["patient_names"] = [
            {
                "raw": "Jane Example",
                "value": "Jane Example",
                "evidence_text": "Jane Example, 625-54",
                "source": "transcript",
            }
        ]
        transcription = TranscriptionResult(
            text=text,
            entities={
                "_word_timestamps": [
                    {"word": "Jane", "start": 0.0, "end": 0.2},
                    {"word": "Example,", "start": 0.2, "end": 0.5},
                    {"word": "625-54", "start": 0.5, "end": 1.0},
                    {"word": "needs", "start": 1.0, "end": 1.2},
                ]
            },
            segments=[{"text": text, "start": 0.0, "end": 1.2}],
        )

        self.assertEqual(watcher.add_compact_dob_fallback_candidates(payload, transcription.text), 1)
        records = watcher.build_candidate_records("dob", payload, transcription)
        resolution = resolve_dob_field(records, today=date(2026, 5, 2))

        self.assertEqual(resolution.status, "gemma_final")
        self.assertEqual(resolution.final_value, "06/25/1954")

    def test_compact_dob_fallback_rejects_phone_fragment_after_name(self):
        text = "Bailey Example. My telephone number is (202) 314-0."
        payload = self.valid_gemma_payload()
        payload["patient_names"] = [
            {
                "raw": "Bailey Example",
                "value": "Bailey Example",
                "evidence_text": "Bailey Example",
                "source": "transcript",
            }
        ]
        transcription = transcription_for_text(text)

        self.assertEqual(watcher.add_compact_dob_fallback_candidates(payload, transcription.text), 0)
        self.assertEqual(watcher.build_candidate_records("dob", payload, transcription), [])

    def test_dob_resolver_rejects_compact_phone_context_candidate(self):
        text = "Bailey Example. My telephone number is (202) 314-0."
        payload = self.valid_gemma_payload()
        payload["dob_candidates"] = [
            {
                "raw": "314-0",
                "normalized": "03/01/1940",
                "evidence_text": "My telephone number is (202) 314-0",
                "source": "compact_dob_name_fallback",
            }
        ]
        transcription = transcription_for_text(text)

        records = watcher.build_candidate_records("dob", payload, transcription)
        resolution = resolve_dob_field(records, today=date(2026, 5, 2))

        self.assertEqual(resolution.status, "not_included")
        self.assertIsNone(resolution.final_value)
        self.assertIn("dob_phone_context_rejected", resolution.review_reasons)

    def test_compact_dob_fallback_fills_asr_filler_form(self):
        text = "Jordan Sample, 424 of 60 needs a call."
        payload = self.valid_gemma_payload()
        payload["patient_names"] = [
            {
                "raw": "Jordan Sample",
                "value": "Jordan Sample",
                "evidence_text": "Jordan Sample, 424 of 60",
                "source": "transcript",
            }
        ]
        transcription = transcription_for_text(text)

        self.assertEqual(watcher.add_compact_dob_fallback_candidates(payload, transcription.text), 1)
        records = watcher.build_candidate_records("dob", payload, transcription)
        resolution = resolve_dob_field(records, today=date(2026, 5, 2))

        self.assertEqual(resolution.status, "gemma_final")
        self.assertEqual(resolution.final_value, "04/24/1960")

    def test_compact_dob_fallback_conflict_requires_review(self):
        text = "Jane Example, 625-54 date of birth 01/05/1980."
        payload = self.valid_gemma_payload()
        payload["patient_names"] = [
            {
                "raw": "Jane Example",
                "value": "Jane Example",
                "evidence_text": "Jane Example, 625-54",
                "source": "transcript",
            }
        ]
        payload["dob_candidates"] = [
            {
                "raw": "01/05/1980",
                "normalized": "01/05/1980",
                "evidence_text": "date of birth 01/05/1980",
            }
        ]
        transcription = TranscriptionResult(
            text=text,
            entities={
                "_word_timestamps": [
                    {"word": "Jane", "start": 0.0, "end": 0.2},
                    {"word": "Example,", "start": 0.2, "end": 0.5},
                    {"word": "625-54", "start": 0.5, "end": 1.0},
                    {"word": "date", "start": 1.0, "end": 1.1},
                    {"word": "of", "start": 1.1, "end": 1.2},
                    {"word": "birth", "start": 1.2, "end": 1.3},
                    {"word": "01/05/1980.", "start": 1.3, "end": 1.8},
                ]
            },
            segments=[{"text": text, "start": 0.0, "end": 1.8}],
        )

        self.assertEqual(watcher.add_compact_dob_fallback_candidates(payload, transcription.text), 1)
        records = watcher.build_candidate_records("dob", payload, transcription)
        resolution = resolve_dob_field(records, today=date(2026, 5, 2))

        self.assertEqual(resolution.status, "ambiguous")
        self.assertIsNone(resolution.final_value)
        self.assertIn("multiple_field_candidates", resolution.review_reasons)

    def test_compact_dob_fallback_rejects_future_dates(self):
        text = "Date of birth 123126."
        payload = self.valid_gemma_payload()
        transcription = TranscriptionResult(
            text=text,
            entities={
                "_word_timestamps": [
                    {"word": "Date", "start": 0.0, "end": 0.1},
                    {"word": "of", "start": 0.1, "end": 0.2},
                    {"word": "birth", "start": 0.2, "end": 0.3},
                    {"word": "123126.", "start": 0.3, "end": 0.7},
                ]
            },
            segments=[{"text": text, "start": 0.0, "end": 0.7}],
        )

        self.assertEqual(watcher.add_compact_dob_fallback_candidates(payload, transcription.text), 0)
        self.assertEqual(watcher.build_candidate_records("dob", payload, transcription), [])

    def test_extract_word_timestamps_prefers_top_level_words_without_duplicates(self):
        payload = {
            "words": [{"word": "hello", "start": 0.0, "end": 0.4}],
            "segments": [
                {"text": "hello", "start": 0.0, "end": 0.4, "words": [{"word": "hello", "start": 0.0, "end": 0.4}]}
            ],
        }
        words = extract_word_timestamps(payload)
        self.assertEqual(words, [{"word": "hello", "start": 0.0, "end": 0.4}])

    def test_gemma_input_payload_excludes_timing_data(self):
        transcription = TranscriptionResult(
            text="Call me at 202-555-0102.",
            raw_text="Call me at 202-555-0103.",
            processed_text="Call me at 202-555-0102.",
            entities={
                "callback_number": "202-555-0102",
                "_word_timestamps": [{"word": "Call", "start": 0.0, "end": 0.2}],
            },
            segments=[{"text": "Call me at 202-555-0102."}],
        )
        payload = build_gemma_input_payload(
            transcription,
            {"callerid": '"MARK EXAMPLE" <2025550102>', "origmailbox": "581"},
        )
        self.assertEqual(
            payload,
            {
                "transcript": "Call me at 202-555-0102.",
                "caller_id": '"MARK EXAMPLE" <2025550102>',
                "mailbox": "581",
            },
        )
        self.assertNotIn("raw_text", payload)
        self.assertNotIn("processed_text", payload)
        self.assertNotIn("entities", payload)
        self.assertNotIn("segments", payload)
        self.assertNotIn("words", payload)

    def test_adaptive_gemma_prompt_keeps_simple_transcript_under_budget(self):
        prompt_path = Path(__file__).resolve().parents[1] / "gemma_field_prompt_litert_adaptive_core.md"
        core_prompt = prompt_path.read_text(encoding="utf-8").strip()
        transcript = "Hi, this is Taylor Sample with a quick scheduling note for Tuesday morning only today"
        self.assertEqual(len(transcript), 85)

        prompt = watcher.build_adaptive_gemma_prompt(core_prompt, transcript)
        input_payload = {"transcript": transcript, "caller_id": "", "mailbox": "190"}
        prompt_text = f"{prompt}\n\nInput JSON:\n{json.dumps(input_payload, ensure_ascii=True)}"

        self.assertLess(len(prompt_text), 3000)
        self.assertEqual(watcher.select_adaptive_gemma_prompt_capsules(transcript), [])

    def test_adaptive_gemma_prompt_selects_relevant_capsules(self):
        cases = [
            ("My name is Bailey, B-A-I-L-E-Y, Sample, S-A-M-P-L-E.", "spelled_name"),
            ("I am calling for my husband Jordan Sample.", "relationship_subject"),
            ("Patient Avery Example DOB 020370.", "compact_dob"),
            ("Hi Casey, this is Taylor Sample.", "addressee_name_exclusion"),
            ("This is Avery calling from Example Clinic about a client.", "organization_exclusion"),
            ("Please call me back at 202-555-0100 or fax 202-555-0199.", "callback_fax"),
        ]

        for transcript, expected_capsule in cases:
            with self.subTest(expected_capsule=expected_capsule):
                self.assertIn(expected_capsule, watcher.select_adaptive_gemma_prompt_capsules(transcript))

    def test_gemma_http_caller_uses_adaptive_prompt_file(self):
        class FakeResponse:
            status_code = 200

            def json(self):
                return {"response": '{"n":[],"d":[],"c":[],"f":[],"u":[],"e":[]}'}

        class FakeRequests:
            payload = None

            def post(self, _url, json=None, **_kwargs):
                FakeRequests.payload = json
                return FakeResponse()

        original_requests = watcher.requests
        watcher.requests = FakeRequests()
        try:
            transcript = "Hi, this is Taylor Sample with a quick scheduling note for Tuesday morning only today"
            settings = SimpleNamespace(
                gemma_prompt_path=str(Path(__file__).resolve().parents[1] / "gemma_field_prompt_litert_adaptive_core.md"),
                gemma_model="gemma-test",
                gemma_base_url="http://127.0.0.1:11434",
                gemma_api_mode="litert_chat",
                gemma_max_retries=0,
                gemma_timeout_seconds=1,
            )
            parsed = call_gemma_field_extraction(TranscriptionResult(text=transcript, entities={}), {}, settings, None)
        finally:
            watcher.requests = original_requests

        self.assertEqual(parsed, self.valid_gemma_payload())
        self.assertIsNotNone(FakeRequests.payload)
        message = FakeRequests.payload["message"]
        self.assertIn("ADAPTIVE_GEMMA_FIELD_PROMPT_V1", message)
        self.assertNotIn("Adaptive examples:", message)
        self.assertLess(len(message), 3000)

    def test_gemma_schema_error_preserved_by_http_caller(self):
        class FakeResponse:
            status_code = 200

            def json(self):
                return {"response": '{"patient_names": []}'}

        class FakeRequests:
            def post(self, *args, **kwargs):
                return FakeResponse()

        original_requests = watcher.requests
        watcher.requests = FakeRequests()
        try:
            settings = SimpleNamespace(
                gemma_prompt_path="",
                gemma_model="gemma-test",
                gemma_base_url="http://127.0.0.1:11434",
                gemma_max_retries=1,
                gemma_timeout_seconds=1,
            )
            transcription = TranscriptionResult(text="hello", entities={})
            with self.assertRaises(GemmaSchemaError):
                call_gemma_field_extraction(transcription, {}, settings, None)
        finally:
            watcher.requests = original_requests

    def test_gemma_response_rejects_candidate_limits_and_oversized_strings(self):
        too_many = self.valid_gemma_payload()
        too_many["patient_names"] = [
            {
                "raw": "Avery Example",
                "value": "Avery Example",
                "evidence_text": "this is Avery Example",
                "source": "transcript",
                "caller_id_used": "",
                "confidence": "candidate_only",
            }
            for _ in range(11)
        ]
        with self.assertRaises(GemmaSchemaError):
            parse_gemma_response(too_many)

        long_raw = self.valid_gemma_payload()
        long_raw["patient_names"] = [
            {
                "raw": "A" * 201,
                "value": "Avery Example",
                "evidence_text": "this is Avery Example",
                "source": "transcript",
                "caller_id_used": "",
                "confidence": "candidate_only",
            }
        ]
        with self.assertRaises(GemmaSchemaError):
            parse_gemma_response(long_raw)

        long_evidence = self.valid_gemma_payload()
        long_evidence["callback_numbers"] = [
            {
                "raw": "217-555-0100",
                "normalized": "2175550100",
                "formatted": "217-555-0100",
                "label_cue": "callback",
                "evidence_text": "x" * 501,
                "confidence": "candidate_only",
            }
        ]
        with self.assertRaises(GemmaSchemaError):
            parse_gemma_response(long_evidence)

    def test_select_entities_requires_complete_result_and_audit_success(self):
        original = {"name": "Legacy", "dob": None, "callback_number": "202-555-0109", "fax_number": None}
        proposed = dict(original)
        proposed["name"] = "Gemma Name"
        incomplete = VerificationRunResult(proposed, [], should_apply=True, complete=False)
        self.assertEqual(select_entities_for_output(original, incomplete, audit_written=True, require_audit_for_apply=True), original)

        complete = VerificationRunResult(proposed, [], should_apply=True, complete=True)
        self.assertEqual(select_entities_for_output(original, complete, audit_written=False, require_audit_for_apply=True), original)
        self.assertEqual(select_entities_for_output(original, complete, audit_written=True, require_audit_for_apply=True), proposed)


    def test_caller_id_name_correction_rewrites_repeated_changed_tokens(self):
        transcript = "Hi Casey. This is Mark Exampel. I spoke to Mark Exampel in billing."
        entities = {
            "_word_timestamps": [
                {"word": word}
                for word in [
                    "Hi",
                    "Casey.",
                    "This",
                    "is",
                    "Mark",
                    "Exampel.",
                    "I",
                    "spoke",
                    "to",
                    "Mark",
                    "Exampel",
                    "in",
                    "billing.",
                ]
            ]
        }
        audit_row = {
            "field_name": "name",
            "final_value": "Mark Example",
            "status": "caller_id_spelling_corrected",
            "gemma_json": [{"raw": "Mark Exampel", "value": "Mark Example"}],
            "attribution_json": [{"field_name": "name", "word_start": 4, "word_end": 5}],
        }

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            entities,
            [audit_row],
        )

        self.assertEqual(
            corrected,
            "Hi Casey. This is Mark Example. I spoke to Mark Example in billing.",
        )
        words = corrected_entities["_word_timestamps"]
        self.assertEqual([words[4]["word"], words[5]["word"]], ["Mark", "Example."])
        self.assertEqual([words[9]["word"], words[10]["word"]], ["Mark", "Example"])
        self.assertEqual(corrected_entities["transcript_corrections"][0]["transcript_replacements"], 2)

    def test_name_transcript_correction_rewrites_multiple_unspanned_equal_token_occurrences(self):
        transcript = "This is Mark Exampel. I spoke to Mark Exampel in billing."
        audit_row = {
            "field_name": "name",
            "final_value": "Mark Example",
            "status": "caller_id_spelling_corrected",
            "gemma_json": [{"raw": "Mark Exampel", "value": "Mark Example"}],
            "attribution_json": [],
        }

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            {},
            [audit_row],
        )

        self.assertEqual(corrected, "This is Mark Example. I spoke to Mark Example in billing.")
        self.assertEqual(corrected_entities["transcript_corrections"][0]["transcript_replacements"], 2)

    def test_caller_id_name_transcript_correction_uses_value_when_caller_id_has_middle_initial(self):
        transcript = "This is Quinn Exampel. Quinn Exampel needs a call back."
        audit_row = {
            "field_name": "name",
            "final_value": "Quinn Example",
            "status": "caller_id_spelling_corrected",
            "gemma_json": [
                {
                    "raw": "Quinn Exampel",
                    "value": "Quinn Example",
                    "source": "caller_id_corrected",
                    "caller_id_used": "QUINN L EXAMPLE",
                }
            ],
            "attribution_json": [],
        }

        corrected, corrected_entities = apply_verified_phone_corrections_to_transcript(
            transcript,
            {},
            [audit_row],
        )

        self.assertEqual(corrected, "This is Quinn Example. Quinn Example needs a call back.")
        self.assertEqual(corrected_entities["transcript_corrections"][0]["transcript_replacements"], 2)

    def test_global_timeout_after_partial_resolution_preserves_original_entities(self):
        original_call_gemma = watcher.call_gemma_field_extraction
        original_build = watcher.build_candidate_records
        original_resolve_name = watcher.resolve_name_field
        original_resolve_dob = watcher.resolve_dob_field

        def fake_gemma(*_args, **_kwargs):
            return self.valid_gemma_payload()

        def fake_build(*_args, **_kwargs):
            return []

        def fake_resolve_name(*_args, **_kwargs):
            return FieldResolution("name", "Gemma Name", "gemmaname", "gemma_final")

        def timeout_resolve_dob(*_args, **_kwargs):
            raise watcher.VerificationBudgetExceeded("timeout after partial work")

        watcher.call_gemma_field_extraction = fake_gemma
        watcher.build_candidate_records = fake_build
        watcher.resolve_name_field = fake_resolve_name
        watcher.resolve_dob_field = timeout_resolve_dob
        try:
            settings = SimpleNamespace(
                gemma_field_extraction_enabled=True,
                verification_total_timeout_seconds=100,
                gemma_fail_open=True,
                verification_apply_resolved_values=True,
            )
            transcription = TranscriptionResult(
                text="hello",
                entities={"name": "Legacy", "dob": "01/01/1980", "callback_number": "202-555-0109"},
            )
            result = watcher.verify_voicemail_fields("/tmp/no.wav", transcription, {}, settings)
            self.assertTrue(result.timed_out)
            self.assertFalse(result.complete)
            self.assertFalse(result.should_apply)
            self.assertEqual(result.proposed_entities, transcription.entities)
        finally:
            watcher.call_gemma_field_extraction = original_call_gemma
            watcher.build_candidate_records = original_build
            watcher.resolve_name_field = original_resolve_name
            watcher.resolve_dob_field = original_resolve_dob

    def test_safe_verify_failure_preserves_original_entities(self):
        original_verify = watcher.verify_voicemail_fields

        def raises(*_args, **_kwargs):
            raise RuntimeError("boom")

        watcher.verify_voicemail_fields = raises
        try:
            settings = SimpleNamespace(gemma_fail_open=True)
            transcription = TranscriptionResult(text="hello", entities={"name": "Legacy"})
            result = safe_verify_voicemail_fields("abc123", "/tmp/no.wav", transcription, {}, settings)
            self.assertFalse(result.should_apply)
            self.assertFalse(result.complete)
            self.assertEqual(result.proposed_entities, {"name": "Legacy"})
            self.assertTrue(result.audit_rows)
        finally:
            watcher.verify_voicemail_fields = original_verify


class HybridV15Tests(unittest.TestCase):
    def patched_env(self, values):
        return portal_test_env(values)

    def write_portal_message(self, inbox, msg_name, duration=None, origtime="1770000000"):
        return write_portal_message(inbox, msg_name, duration=duration, origtime=origtime)

    def test_watcher_and_portal_spool_identity_helpers_match(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inbox = Path(tmp) / "voicemail" / "default" / "154" / "INBOX"
            inbox.mkdir(parents=True)
            txt_path = inbox / "msg0001.txt"
            txt_path.write_text(
                "\n".join(
                    [
                        "origtime=1770000000",
                        'callerid="WIRELESS CALLER" <2175550100>',
                        "origmailbox=154",
                        "duration=17",
                    ]
                ),
                encoding="utf-8",
            )

            watcher_info = watcher.parse_txt(str(txt_path))
            portal_info = voicemail_portal.parse_txt(str(txt_path))

            self.assertEqual(watcher_info, portal_info)
            self.assertEqual(watcher.extract_extension(str(txt_path)), "154")
            self.assertEqual(voicemail_portal.extract_extension(str(txt_path)), "154")
            self.assertTrue(watcher.is_voicemail_txt(str(txt_path)))
            self.assertTrue(voicemail_portal.is_voicemail_txt(str(txt_path)))
            self.assertEqual(watcher.matching_wav_path(str(txt_path)), voicemail_portal.matching_wav_path(str(txt_path)))
            self.assertEqual(
                watcher.build_legacy_file_key("154", watcher_info, str(txt_path)),
                voicemail_portal.build_legacy_file_key("154", portal_info, str(txt_path)),
            )
            self.assertEqual(
                watcher.build_file_key("154", watcher_info, str(txt_path)),
                voicemail_portal.build_file_key("154", portal_info, str(txt_path)),
            )

    def test_portal_route_contract_paths_are_stable(self):
        import voicemail_portal

        route_methods = {
            (route.path, method)
            for route in voicemail_portal.app.routes
            for method in getattr(route, "methods", set())
        }
        expected = {
            ("/health", "GET"),
            ("/login", "GET"),
            ("/login", "POST"),
            ("/logout", "POST"),
            ("/voicemails", "GET"),
            ("/api/voicemails", "GET"),
            ("/api/extensions", "GET"),
            ("/api/directory", "GET"),
            ("/api/voicemails/{file_key}/audio", "GET"),
            ("/api/voicemails/{file_key}/restore", "POST"),
            ("/api/voicemails/{file_key}/comment", "POST"),
            ("/api/voicemails/{file_key}/delete", "POST"),
            ("/api/voicemails/bulk-delete", "POST"),
            ("/api/voicemails/{file_key}", "DELETE"),
        }

        self.assertTrue(expected.issubset(route_methods), sorted(expected - route_methods))

    def test_portal_directory_api_returns_full_read_only_directory_for_non_admin(self):
        from dataclasses import replace

        import voicemail_portal

        user = voicemail_portal.PortalUser("154", "154", "", "Extension 154", False)
        discovered = [
            {"extension": "154", "display_name": "Robin Example"},
            {"extension": "155", "display_name": "Avery Example"},
            {"extension": "156", "display_name": ""},
        ]
        old_current_user = voicemail_portal.current_user
        old_discover_mailboxes = voicemail_portal.discover_mailboxes
        old_settings = voicemail_portal.SETTINGS
        voicemail_portal.current_user = lambda _request: user
        voicemail_portal.discover_mailboxes = lambda: discovered
        voicemail_portal.SETTINGS = replace(old_settings, forward_enabled=True)
        try:
            directory_response = voicemail_portal.list_directory(SimpleNamespace())
            directory_payload = json.loads(directory_response.body.decode("utf-8"))
            extensions_response = voicemail_portal.list_extensions(SimpleNamespace())
            extensions_payload = json.loads(extensions_response.body.decode("utf-8"))
        finally:
            voicemail_portal.current_user = old_current_user
            voicemail_portal.discover_mailboxes = old_discover_mailboxes
            voicemail_portal.SETTINGS = old_settings

        self.assertEqual(directory_payload, discovered)
        self.assertEqual(extensions_payload, [{"extension": "154", "display_name": "Robin Example"}])

    def test_portal_forwarding_routes_are_denied_by_default(self):
        import voicemail_portal

        with self.assertRaises(voicemail_portal.HTTPException) as context:
            voicemail_portal.list_directory(SimpleNamespace())
        self.assertEqual(context.exception.status_code, 403)

        with self.assertRaises(voicemail_portal.HTTPException) as context:
            voicemail_portal.forward_voicemail(
                "synthetic-key",
                voicemail_portal.ForwardVoicemailRequest(target_extension="155"),
                SimpleNamespace(),
            )
        self.assertEqual(context.exception.status_code, 403)

    def test_portal_user_excluded_extensions_override_all_access(self):
        import voicemail_portal

        wildcard = voicemail_portal.PortalUser(
            "manager",
            "*",
            "",
            "Manager",
            False,
            excluded_extensions=("222",),
        )
        admin = voicemail_portal.PortalUser(
            "admin",
            "*",
            "",
            "Admin",
            True,
            excluded_extensions=("222",),
        )

        self.assertTrue(wildcard.can_access_extension("154"))
        self.assertFalse(wildcard.can_access_extension("222"))
        self.assertTrue(admin.can_access_extension("154"))
        self.assertFalse(admin.can_access_extension("222"))

    def test_portal_manual_users_load_excluded_extensions(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            users_file = os.path.join(tmp, "users.json")
            with open(users_file, "w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {
                            "username": "manager",
                            "extension": "*",
                            "display_name": "Manager",
                            "password_hash": "hash",
                            "excluded_extensions": ["222", "bad", "333", "222"],
                        }
                    ],
                    handle,
                )

            users = voicemail_portal.load_manual_users(SimpleNamespace(users_file=users_file, auto_users=False))

        self.assertEqual(users["manager"].excluded_extensions, ("222", "333"))

    def test_portal_wildcard_user_exclusions_filter_voicemails_and_direct_access(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            state_db = os.path.join(tmp, "state.sqlite3")
            for extension, message, origtime in [
                ("154", "msg0154", "1770000154"),
                ("155", "msg0155", "1770000155"),
                ("156", "msg0156", "1770000156"),
            ]:
                inbox = os.path.join(watch_dir, "vitalpbx-voicemail", extension, "INBOX")
                self.write_portal_message(inbox, message, duration=20, origtime=origtime)

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            user = voicemail_portal.PortalUser(
                "manager",
                "*",
                "",
                "Manager",
                False,
                excluded_extensions=("155",),
            )
            active = store.list_voicemails(user, "active")

            with sqlite3.connect(state_db) as conn:
                excluded_key = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE extension = '155'"
                ).fetchone()[0]

            with self.assertRaises(voicemail_portal.HTTPException):
                store.get_voicemail(excluded_key, user)

        self.assertEqual([item["extension"] for item in active], ["156", "154"])

    def test_portal_extensions_api_hides_excluded_extensions_for_wildcard_user(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser(
            "manager",
            "*",
            "",
            "Manager",
            False,
            excluded_extensions=("155",),
        )
        discovered = [
            {"extension": "154", "display_name": "Robin Example"},
            {"extension": "155", "display_name": "Avery Example"},
            {"extension": "156", "display_name": ""},
        ]
        old_current_user = voicemail_portal.current_user
        old_discover_mailboxes = voicemail_portal.discover_mailboxes
        voicemail_portal.current_user = lambda _request: user
        voicemail_portal.discover_mailboxes = lambda: discovered
        try:
            response = voicemail_portal.list_extensions(SimpleNamespace())
            payload = json.loads(response.body.decode("utf-8"))
        finally:
            voicemail_portal.current_user = old_current_user
            voicemail_portal.discover_mailboxes = old_discover_mailboxes

        self.assertEqual(
            payload,
            [
                {"extension": "154", "display_name": "Robin Example"},
                {"extension": "156", "display_name": ""},
            ],
        )

    def test_portal_directory_popout_button_and_inert_entries_render(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("154", "154", "", "Extension 154", False)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn('id="directoryBtn"', page_html)
        self.assertIn('class="address-book-icon"', page_html)
        self.assertIn('title="Directory"', page_html)
        self.assertIn('aria-label="Directory"', page_html)
        self.assertIn('id="directoryMenu"', page_html)
        self.assertIn('id="directorySearch"', page_html)
        self.assertIn('placeholder="Search directory"', page_html)
        self.assertIn('class="directory-entry"', page_html)
        self.assertIn("async function loadDirectory()", page_html)
        self.assertIn("function toggleDirectoryMenu(forceOpen = null)", page_html)
        self.assertIn("function renderDirectoryMenu()", page_html)
        self.assertIn("function directoryEntryLabel(item)", page_html)
        self.assertIn('return `${name} - Ext ${item.extension}`;', page_html)
        self.assertIn('return `Ext ${item.extension}`;', page_html)
        self.assertIn("function directoryMatchesSearch(item)", page_html)
        self.assertIn("namedEntries", page_html)
        self.assertIn("unnamedEntries", page_html)
        self.assertIn("localeCompare", page_html)
        self.assertIn("toggleExtensionMenu(false);", page_html)
        self.assertIn("toggleDirectoryMenu(false);", page_html)
        self.assertIn('document.getElementById("directorySearch").addEventListener("input",', page_html)
        self.assertIn('document.getElementById("directoryBtn").addEventListener("click"', page_html)
        self.assertIn("loadDirectory();", page_html)
        self.assertIn('const isAdmin = false;', page_html)
        self.assertIn('return `<div class="directory-entry" role="listitem">${label}</div>`;', page_html)
        self.assertIn("if (!isAdmin) return;", page_html)

    def test_portal_admin_directory_entries_filter_to_mailbox(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("admin", "*", "", "Admin", True)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn("const isAdmin = true;", page_html)
        self.assertIn("function selectExtensionFilter(extension)", page_html)
        self.assertIn('data-directory-extension="${escapeHtml(item.extension)}"', page_html)
        self.assertIn('document.getElementById("directoryItems").addEventListener("click",', page_html)
        self.assertIn("if (!isAdmin) return;", page_html)
        self.assertIn('const button = event.target.closest("[data-directory-extension]");', page_html)
        self.assertIn('selectExtensionFilter(button.dataset.directoryExtension || "");', page_html)
        self.assertIn('selectExtensionFilter(button.dataset.extension || "");', page_html)
        self.assertIn("toggleExtensionMenu(false);", page_html)
        self.assertIn("toggleDirectoryMenu(false);", page_html)
        self.assertIn("No voicemails found for", page_html)

    def test_portal_pages_render_shared_favicon_link(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("154", "154", "", "Extension 154", False)
        login_html = voicemail_portal.login_page().body.decode("utf-8")
        portal_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")
        expected = (
            '<link rel="icon" type="image/svg+xml" '
            f'href="{voicemail_portal.app_path("/brand/favicon")}?v=1">'
        )

        self.assertIn(expected, login_html)
        self.assertIn(expected, portal_html)

    def test_portal_mobile_detail_header_stacks_caller_id_above_actions(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("154", "154", "", "Extension 154", False)
        portal_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")
        mobile_css = portal_html.split("@media (max-width: 820px)", 1)[1].split("</style>", 1)[0]

        self.assertRegex(
            mobile_css,
            r"\.detail-header\s*\{\s*flex-direction:\s*column;\s*\}",
        )

    def test_portal_mobile_typography_restores_list_and_reduces_detail(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("154", "154", "", "Extension 154", False)
        portal_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")
        desktop_css, mobile_tail = portal_html.split("@media (max-width: 820px)", 1)
        mobile_css = mobile_tail.split("</style>", 1)[0]

        desktop_sizes = (
            (r"h1\s*\{[^}]*font-size:\s*24px;", "24px heading"),
            (r"\.mailbox-badge span\s*\{[^}]*font-size:\s*11px;", "11px label"),
            (r"\.extension-option span\s*\{[^}]*font-size:\s*12px;", "12px helper"),
            (r"\.bulk-check\s*\{[^}]*font-size:\s*13px;", "13px bulk control"),
            (r"\.directory-empty\s*\{[^}]*font-size:\s*14px;", "14px empty state"),
            (r"\.menu-search-wrap input\s*\{[^}]*font-size:\s*15px;", "15px input"),
            (r"\.mailbox-badge\s*\{[^}]*font-size:\s*16px;", "16px badge"),
            (r"\.review-progress-current\s*\{[^}]*font-size:\s*18px;", "18px review count"),
            (r"\.delete-comment-panel label\s*\{[^}]*font-size:\s*22px;", "22px comment label"),
        )
        for pattern, label in desktop_sizes:
            with self.subTest(desktop=label):
                self.assertRegex(desktop_css, pattern)

        mobile_sizes = (
            (r"body\s*\{[^}]*font-size:\s*16px;", "page base restored to 16px"),
            (r"h1\s*\{[^}]*font-size:\s*24px;", "heading restored to 24px"),
            (r"\.extension-option span\s*\{[^}]*font-size:\s*12px;", "menu helper restored to 12px"),
            (r"\.bulk-check,[^}]*\.item-meta\s*\{[^}]*font-size:\s*13px;", "list metadata restored to 13px"),
            (r"\.directory-empty,\s*\.item-preview\s*\{[^}]*font-size:\s*14px;", "list preview restored to 14px"),
            (r"\.menu-search-wrap input,\s*\.toolbar input\s*\{[^}]*font-size:\s*15px;", "search controls restored to 15px"),
            (r"\.detail\s*\{[^}]*font-size:\s*18px;", "detail base reduced to 18px"),
            (r"\.detail \.mailbox-badge span\s*\{[^}]*font-size:\s*12\.375px;", "mailbox label remains 12.375px"),
            (r"\.detail \.field span\s*\{[^}]*font-size:\s*13\.6125px;", "field label increased by 10 percent"),
            (r"\.detail \.speed-controls,[^}]*\.detail \.delete-comment-save\s*\{[^}]*font-size:\s*13\.5px;", "non-field 12px detail controls remain 13.5px"),
            (r"\.detail \.field-extra\s*\{[^}]*font-size:\s*14\.85px;", "field helper increased by 10 percent"),
            (r"\.detail \.forward-summary,[^}]*\.detail \.delete-comment-help\s*\{[^}]*font-size:\s*14\.625px;", "detail 13px to 14.625px"),
            (r"\.detail \.field,[^}]*\.detail \.transcript-box\s*\{[^}]*font-size:\s*18\.5625px;", "field and transcript text increased by 10 percent"),
            (r"\.detail \.mailbox-badge,[^}]*\.detail \.review-progress-total\s*\{[^}]*font-size:\s*18px;", "detail 16px to 18px"),
            (r"\.detail \.review-progress-current\s*\{[^}]*font-size:\s*20\.25px;", "detail 18px to 20.25px"),
            (r"\.detail \.delete-comment-panel label\s*\{[^}]*font-size:\s*24\.75px;", "detail 22px to 24.75px"),
        )
        for pattern, label in mobile_sizes:
            with self.subTest(mobile=label):
                self.assertRegex(mobile_css, pattern)

        self.assertRegex(
            mobile_css,
            r"button,\s*input,\s*textarea,\s*select\s*\{[^}]*font-size:\s*16px;",
        )
        self.assertRegex(
            mobile_css,
            r"\.detail button,[^}]*\.detail select\s*\{[^}]*font-size:\s*18px;",
        )
        self.assertNotRegex(mobile_css, r"\.item-meta[^}]*font-size:\s*19\.5px;")
        self.assertNotRegex(mobile_css, r"\.item-preview[^}]*font-size:\s*21px;")

    def test_portal_mobile_rows_wrap_and_controls_have_touch_targets(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("154", "154", "", "Extension 154", False)
        portal_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")
        mobile_css = portal_html.split("@media (max-width: 820px)", 1)[1].split("</style>", 1)[0]

        self.assertRegex(mobile_css, r"\.userbar\s*\{[^}]*flex-wrap:\s*nowrap;")
        self.assertRegex(mobile_css, r"\.detail-actions\s*\{[^}]*flex-wrap:\s*wrap;")
        self.assertRegex(
            mobile_css,
            r"button,\s*\.userbar a\.nav-link[^}]*\{[^}]*min-height:\s*44px;",
        )
        self.assertRegex(
            mobile_css,
            r"\.icon-btn,[^}]*\.delete-comment-save\s*\{"
            r"[^}]*min-width:\s*44px;[^}]*min-height:\s*44px;",
        )

    def test_portal_mobile_userbar_uses_short_admin_label_in_one_row(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser(
            "admin",
            "",
            "",
            "Voicemail Lab Administrator",
            True,
        )
        portal_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")
        desktop_css, mobile_tail = portal_html.split("@media (max-width: 820px)", 1)
        mobile_css = mobile_tail.split("</style>", 1)[0]

        self.assertIn(
            '<span class="user-label" aria-label="Voicemail Lab Administrator">',
            portal_html,
        )
        self.assertIn(
            '<span class="user-label-full">Voicemail Lab Administrator</span>',
            portal_html,
        )
        self.assertIn(
            '<span class="user-label-mobile">Administrator</span>',
            portal_html,
        )
        self.assertRegex(
            desktop_css,
            r"\.user-label-mobile\s*\{\s*display:\s*none;\s*\}",
        )
        self.assertRegex(
            mobile_css,
            r"\.userbar\s*\{[^}]*flex-wrap:\s*nowrap;[^}]*gap:\s*4px;",
        )
        self.assertRegex(
            mobile_css,
            r"\.userbar \.user-label\s*\{[^}]*flex:\s*1 1 auto;"
            r"[^}]*min-width:\s*0;[^}]*text-overflow:\s*ellipsis;"
            r"[^}]*white-space:\s*nowrap;",
        )
        self.assertRegex(
            mobile_css,
            r"\.userbar \.user-label-full\s*\{\s*display:\s*none;\s*\}",
        )
        self.assertRegex(
            mobile_css,
            r"\.userbar \.user-label-mobile\s*\{\s*display:\s*inline;\s*\}",
        )
        self.assertRegex(
            mobile_css,
            r"\.userbar button,\s*\.userbar a\.nav-link\s*\{"
            r"[^}]*font-size:\s*13px;[^}]*padding:\s*0 6px;"
            r"[^}]*white-space:\s*nowrap;",
        )

    def test_portal_mobile_safe_area_spacing_preserves_sticky_header(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("154", "154", "", "Extension 154", False)
        portal_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")
        desktop_css, mobile_tail = portal_html.split("@media (max-width: 820px)", 1)
        mobile_css = mobile_tail.split("</style>", 1)[0]

        self.assertRegex(desktop_css, r"header\s*\{[^}]*position:\s*sticky;[^}]*top:\s*0;")
        self.assertNotRegex(mobile_css, r"header\s*\{[^}]*position:\s*(?:static|relative);")
        self.assertRegex(
            mobile_css,
            r"main\s*\{[^}]*padding-bottom:\s*calc\(10px \+ env\(safe-area-inset-bottom\)\);",
        )
        self.assertRegex(
            mobile_css,
            r"\.toast-host\s*\{[^}]*bottom:\s*calc\(var\(--toast-bottom, 96px\) \+ env\(safe-area-inset-bottom\)\);",
        )

    def test_portal_favicon_route_serves_packaged_neutral_svg(self):
        import voicemail_portal
        from fastapi.testclient import TestClient

        client = TestClient(voicemail_portal.app)
        response = client.get("/brand/favicon")
        expected = (
            Path(__file__).resolve().parents[1] / "lvt_assets" / "voicemail-portal-icon.svg"
        ).read_bytes()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/svg+xml")
        self.assertEqual(response.content, expected)

    def test_portal_admin_extension_menu_search_filters_sandwich_menu(self):
        import voicemail_portal

        admin = voicemail_portal.PortalUser("admin", "*", "", "Admin", True)
        staff = voicemail_portal.PortalUser("154", "154", "", "Extension 154", False)
        admin_html = voicemail_portal.portal_page(admin, "csrf-token").body.decode("utf-8")
        staff_html = voicemail_portal.portal_page(staff, "csrf-token").body.decode("utf-8")

        self.assertIn("const isAdmin = true;", admin_html)
        self.assertIn("const isAdmin = false;", staff_html)
        self.assertIn('let extensionMenuSearchTerm = "";', admin_html)
        self.assertIn('id="extensionMenuSearch"', admin_html)
        self.assertIn('placeholder="Search extensions"', admin_html)
        self.assertIn("function extensionMenuMatchesSearch(item)", admin_html)
        self.assertIn("const extensionRows = isAdmin ? extensions.filter(extensionMenuMatchesSearch) : extensions;", admin_html)
        self.assertIn("All Extensions", admin_html)
        self.assertIn("No extensions found.", admin_html)
        self.assertIn('document.getElementById("extensionMenuSearch")', admin_html)
        self.assertIn("extensionMenuSearchTerm = event.target.value || \"\";", admin_html)
        self.assertIn("renderExtensionMenu();", admin_html)
        self.assertIn("menu-search-wrap", admin_html)
        self.assertIn("directorySearch", admin_html)
        self.assertIn("const extensionRows = isAdmin ? extensions.filter(extensionMenuMatchesSearch) : extensions;", staff_html)

    def test_portal_sandwich_menu_svg_is_larger_without_resizing_button(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("154", "154", "", "Extension 154", False)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn('id="menuBtn" class="menu-btn"', page_html)
        self.assertIn('id="directoryBtn" class="menu-btn"', page_html)
        self.assertIn(".menu-btn {", page_html)
        self.assertIn("width: 40px;", page_html)
        self.assertIn("min-width: 40px;", page_html)
        self.assertIn("height: 40px;", page_html)
        self.assertIn("min-height: 40px;", page_html)
        self.assertIn("#menuBtn svg {", page_html)
        self.assertIn("width: 28px;", page_html)
        self.assertIn("height: 28px;", page_html)
        self.assertNotIn("#menuBtn {\n      width: 48px;", page_html)

    def test_portal_directory_svg_is_slightly_larger_without_resizing_button(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("154", "154", "", "Extension 154", False)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn('id="directoryBtn" class="menu-btn"', page_html)
        self.assertIn('class="address-book-icon"', page_html)
        self.assertIn(".menu-btn {", page_html)
        self.assertIn("width: 40px;", page_html)
        self.assertIn("min-width: 40px;", page_html)
        self.assertIn("height: 40px;", page_html)
        self.assertIn("min-height: 40px;", page_html)
        self.assertIn("#directoryBtn svg {", page_html)
        self.assertIn("width: 25px;", page_html)
        self.assertIn("height: 25px;", page_html)
        self.assertNotIn("#directoryBtn {\n      width: 48px;", page_html)

    def test_portal_directory_svg_morphs_to_x_when_expanded(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("154", "154", "", "Extension 154", False)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn('id="directoryBtn" class="menu-btn"', page_html)
        self.assertIn('aria-expanded="false"', page_html)
        self.assertIn('class="directory-book-shape"', page_html)
        self.assertIn('class="directory-x-line directory-x-line-first"', page_html)
        self.assertIn('class="directory-x-line directory-x-line-second"', page_html)
        self.assertIn("#directoryBtn .directory-book-shape", page_html)
        self.assertIn("#directoryBtn .directory-x-line", page_html)
        self.assertIn('#directoryBtn[aria-expanded="true"] .directory-book-shape', page_html)
        self.assertIn('#directoryBtn[aria-expanded="true"] .directory-x-line', page_html)
        self.assertIn('button.setAttribute("aria-expanded", willOpen ? "true" : "false");', page_html)

    def test_portal_sandwich_menu_svg_morphs_to_x_when_expanded(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("154", "154", "", "Extension 154", False)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn('id="menuBtn" class="menu-btn"', page_html)
        self.assertIn('aria-expanded="false"', page_html)
        self.assertIn('class="menu-line menu-line-top"', page_html)
        self.assertIn('class="menu-line menu-line-middle"', page_html)
        self.assertIn('class="menu-line menu-line-bottom"', page_html)
        self.assertIn("#menuBtn[aria-expanded=\"true\"] .menu-line-top", page_html)
        self.assertIn("transform: translateY(5px) rotate(45deg);", page_html)
        self.assertIn("#menuBtn[aria-expanded=\"true\"] .menu-line-middle", page_html)
        self.assertIn("opacity: 0;", page_html)
        self.assertIn("#menuBtn[aria-expanded=\"true\"] .menu-line-bottom", page_html)
        self.assertIn("transform: translateY(-5px) rotate(-45deg);", page_html)
        self.assertIn('button.setAttribute("aria-expanded", willOpen ? "true" : "false");', page_html)

    def test_sqlite_store_contexts_close_connections(self):
        import voicemail_portal

        watcher_source = Path(watcher.__file__).read_text(encoding="utf-8")
        portal_source = Path(voicemail_portal.__file__).read_text(encoding="utf-8")

        self.assertIn("def _transaction", watcher_source)
        self.assertIn("def _transaction", portal_source)
        self.assertNotIn("with self._connect() as conn", watcher_source)
        self.assertNotIn("with self._connect() as conn", portal_source)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watcher_store = watcher.VoicemailStore(os.path.join(tmp, "watcher.sqlite3"))
            watcher_store.discover(
                "close-test-key",
                "154",
                os.path.join(tmp, "msg0001.txt"),
                os.path.join(tmp, "msg0001.wav"),
            )
            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": os.path.join(tmp, "portal.sqlite3"),
                    "VOICEMAIL_WATCH_DIR": tmp,
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                portal_settings = voicemail_portal.Settings.from_env()
            portal_store = voicemail_portal.PortalStore(portal_settings)
            portal_store.list_voicemails(voicemail_portal.PortalUser("154", "154", "", "154"))
            del watcher_store
            del portal_store
            gc.collect()

    def test_default_profile_keeps_v15_accuracy_features_disabled(self):
        with self.patched_env({"VOICEMAIL_EMAIL_ENABLED": "false"}):
            settings = watcher.Settings.from_env()

        self.assertEqual(settings.accuracy_profile, "default")
        self.assertFalse(settings.parakeet_full_pass_enabled)
        self.assertFalse(settings.transcript_lattice_enabled)
        self.assertFalse(settings.llm_adjudication_enabled)
        self.assertFalse(settings.transcript_lattice_llm_adjudication_enabled)
        self.assertFalse(settings.transcript_lattice_apply_enabled)

    def test_v15_max_profile_enables_only_audit_accuracy_features(self):
        with self.patched_env(
            {
                "VOICEMAIL_EMAIL_ENABLED": "false",
                "VOICEMAIL_ACCURACY_PROFILE": "v1_5_max",
            }
        ):
            settings = watcher.Settings.from_env()

        self.assertTrue(settings.parakeet_full_pass_enabled)
        self.assertTrue(settings.transcript_lattice_enabled)
        self.assertTrue(settings.llm_adjudication_enabled)
        self.assertTrue(settings.transcript_lattice_llm_adjudication_enabled)
        self.assertTrue(settings.asr_runs_enabled)
        self.assertFalse(settings.transcript_lattice_apply_enabled)

    def test_transcript_adjudication_rejects_invented_text(self):
        span = {
            "span_id": "span-1",
            "primary_text": "6448",
            "alternatives": [
                {"text": "6448", "source": "whisper:primary:canonical", "is_primary": True},
                {"text": "6488", "source": "parakeet:full_pass:canonical"},
            ],
        }

        invented = {
            "decision_type": "choose_alternative",
            "chosen_alternative_index": 1,
            "final_text": "6499",
            "source": "parakeet:full_pass:canonical",
            "confidence": 0.99,
        }
        self.assertIsNone(validate_transcript_adjudication_decision(span, invented, min_confidence=0.8))

        valid = dict(invented, final_text="6488")
        self.assertEqual(
            validate_transcript_adjudication_decision(span, valid, min_confidence=0.8)["new_text"],
            "6488",
        )

    def test_lattice_rejects_punctuation_only_transcript_candidate(self):
        primary = (
            "At this point, we're leaving it on the books, but we don't know if he'll even be able "
            "to make it at that point. But we would just like some clarification on what perhaps is "
            "going to happen. Thank you, Casey. Have a good weekend. Bye."
        )
        proposed = (
            "At this point we're leaving it on the books But we don't know if he'll even be able "
            "to make it At that point But we would just like some clarification On what perhaps is "
            "going to happen Thank you Casey Have a good weekend Bye"
        )
        span = DisagreementSpan(
            span_id="lattice:punctuation-only",
            start=None,
            end=None,
            primary_text=primary,
            alternatives=[
                {"text": primary, "source": "whisper:primary:canonical", "is_primary": True},
                {"text": proposed, "source": "parakeet:full_pass:canonical", "strong": True},
            ],
        )

        corrected, corrections = correct_transcript_constrained(primary, [span], SimpleNamespace())

        self.assertEqual(corrected, primary)
        self.assertEqual(corrections[0].decision_type, "keep_primary")
        self.assertIn("punctuation_only_candidate_rejected", corrections[0].reason_codes)

    def test_lattice_word_correction_preserves_primary_punctuation(self):
        primary = "Please note date of birth 6448."
        span = DisagreementSpan(
            span_id="lattice:dob",
            start=None,
            end=None,
            primary_text="date of birth 6448.",
            alternatives=[
                {"text": "date of birth 6448.", "source": "whisper:primary:canonical", "is_primary": True},
                {
                    "text": "date of birth 6488",
                    "source": "parakeet:full_pass:canonical",
                    "strong": True,
                    "reason_code": "verified_digit_consensus",
                },
            ],
            contains_digits=True,
            field_hint="dob",
        )

        corrected, corrections = correct_transcript_constrained(primary, [span], SimpleNamespace())

        self.assertEqual(corrected, "Please note date of birth 6488.")
        self.assertEqual(corrections[0].new_text, "date of birth 6488.")
        self.assertIn("primary_punctuation_preserved", corrections[0].reason_codes)

    def test_lattice_name_correction_preserves_span_comma(self):
        primary = "Patient is Jordan Sample, please call."
        span = DisagreementSpan(
            span_id="lattice:name",
            start=None,
            end=None,
            primary_text="Jordan Sample,",
            alternatives=[
                {"text": "Jordan Sample,", "source": "whisper:primary:canonical", "is_primary": True},
                {"text": "Jordan Exampel", "source": "parakeet:full_pass:canonical", "strong": True},
            ],
            field_hint="name",
        )

        corrected, corrections = correct_transcript_constrained(primary, [span], SimpleNamespace())

        self.assertEqual(corrected, "Patient is Jordan Exampel, please call.")
        self.assertEqual(corrections[0].new_text, "Jordan Exampel,")
        self.assertIn("primary_punctuation_preserved", corrections[0].reason_codes)

    def test_lattice_audit_does_not_change_saved_transcript_without_apply_flag(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inbox = os.path.join(tmp, "154", "INBOX")
            os.makedirs(inbox)
            txt_path = os.path.join(inbox, "msg0001.txt")
            wav_path = os.path.join(inbox, "msg0001.wav")
            with open(txt_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "origmailbox=154\n"
                    "origtime=1770000000\n"
                    "origdate=Wed May 06 10:00:00 AM UTC 2026\n"
                    "duration=20\n"
                    "callerid=\"TEST CALLER\" <2175550100>\n"
                )
            with open(wav_path, "wb") as handle:
                handle.write(b"not real wav but stable")

            with self.patched_env(
                {
                    "VOICEMAIL_WATCH_DIR": tmp,
                    "VOICEMAIL_STATE_DB": os.path.join(tmp, "state.sqlite3"),
                    "VOICEMAIL_EMAIL_ENABLED": "false",
                    "VOICEMAIL_ACCURACY_PROFILE": "v1_5_max",
                    "GEMMA_FIELD_EXTRACTION_ENABLED": "false",
                }
            ):
                settings = watcher.Settings.from_env()
            store = watcher.VoicemailStore(settings.state_db)
            processor = watcher.VoicemailProcessor(settings, store)
            processor.whisper_endpoints.least_busy_order = lambda *_args, **_kwargs: (settings.whisper_url,)

            words = [
                {"word": "date"},
                {"word": "of"},
                {"word": "birth"},
                {"word": "6448", "probability": 0.42},
            ]
            parakeet_words = [
                {"word": "date"},
                {"word": "of"},
                {"word": "birth"},
                {"word": "6488", "probability": 0.96},
            ]
            original_transcribe = watcher.transcribe
            original_verify = watcher.safe_verify_voicemail_fields
            original_get_email = watcher.get_email
            original_parakeet = watcher.call_parakeet_full_transcription
            original_adjudication = watcher.call_gemma_transcript_lattice_adjudication
            watcher.transcribe = lambda *_args, **_kwargs: TranscriptionResult(
                text="date of birth 6448",
                entities={},
                words=words,
            )
            watcher.safe_verify_voicemail_fields = lambda *_args, **_kwargs: VerificationRunResult(
                proposed_entities={},
                audit_rows=[],
                should_apply=False,
            )
            watcher.get_email = lambda *_args, **_kwargs: ["synthetic-user@example.invalid"]
            watcher.call_parakeet_full_transcription = lambda *_args, **_kwargs: {
                "run_id": "asr_parakeet_test",
                "engine": "parakeet",
                "role": "full_pass",
                "audio_view": "canonical",
                "transcript": "date of birth 6488",
                "processed_text": "date of birth 6488",
                "words": parakeet_words,
                "created_utc": watcher.utc_now_iso(),
            }
            watcher.call_gemma_transcript_lattice_adjudication = lambda span, *_args, **_kwargs: {
                "decision_type": "choose_alternative",
                "chosen_alternative_index": 1,
                "text": "6488",
                "new_text": "6488",
                "source": "parakeet:full_pass:canonical",
                "confidence": 0.98,
                "reason_code": "llm_adjudicated_grounded_alternative",
            }
            try:
                processor.process_once(txt_path)
            finally:
                watcher.transcribe = original_transcribe
                watcher.safe_verify_voicemail_fields = original_verify
                watcher.get_email = original_get_email
                watcher.call_parakeet_full_transcription = original_parakeet
                watcher.call_gemma_transcript_lattice_adjudication = original_adjudication

            conn = sqlite3.connect(settings.state_db)
            conn.row_factory = sqlite3.Row
            try:
                transcript = conn.execute("SELECT transcript, entities_json FROM voicemail_transcripts").fetchone()
                self.assertEqual(transcript["transcript"], "date of birth 6448")
                entities = json.loads(transcript["entities_json"])
                self.assertEqual(entities["_corrected_transcript"], "date of birth 6488")
                self.assertTrue(entities["_transcript_corrections"])
                self.assertEqual(conn.execute("SELECT count(*) FROM asr_runs").fetchone()[0], 2)
                self.assertGreater(conn.execute("SELECT count(*) FROM asr_span_candidates").fetchone()[0], 0)
                self.assertGreater(conn.execute("SELECT count(*) FROM transcript_corrections").fetchone()[0], 0)
            finally:
                conn.close()

    def test_field_verification_apply_gate_blocks_field_transcript_correction_but_not_lattice(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inbox = os.path.join(tmp, "154", "INBOX")
            txt_path, _ = self.write_portal_message(inbox, "msg0010", duration=20, origtime="1770000010")
            state_db = os.path.join(tmp, "state.sqlite3")
            with self.patched_env(
                {
                    "VOICEMAIL_WATCH_DIR": tmp,
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_EMAIL_ENABLED": "false",
                    "TRANSCRIPT_LATTICE_ENABLED": "true",
                    "TRANSCRIPT_LATTICE_APPLY_ENABLED": "false",
                    "GEMMA_FIELD_EXTRACTION_ENABLED": "false",
                }
            ):
                settings = watcher.Settings.from_env()

            store = watcher.VoicemailStore(settings.state_db)
            processor = watcher.VoicemailProcessor(settings, store)
            processor.whisper_endpoints.least_busy_order = lambda *_args, **_kwargs: (settings.whisper_url,)
            original_text = "Call me at 2025550101."
            captured_lattice_inputs = []
            audit_rows = [
                {
                    "field_name": "callback_number",
                    "final_value": "202-555-0100",
                    "normalized_value": "2025550100",
                    "status": "parakeet_override",
                    "attribution_json": [
                        {
                            "field_name": "callback_number",
                            "word_start": 3,
                            "word_end": 3,
                        }
                    ],
                }
            ]

            original_transcribe = watcher.transcribe
            original_verify = watcher.safe_verify_voicemail_fields
            original_lattice = watcher.run_transcript_lattice_audit
            watcher.transcribe = lambda *_args, **_kwargs: TranscriptionResult(
                text=original_text,
                entities={
                    "callback_number": "2025550113",
                    "_word_timestamps": [{"word": word} for word in original_text.split()],
                },
            )
            watcher.safe_verify_voicemail_fields = lambda *_args, **_kwargs: VerificationRunResult(
                proposed_entities={"callback_number": "202-555-0100"},
                audit_rows=audit_rows,
                should_apply=False,
                complete=True,
            )

            def fake_lattice(_file_key, _wav_path, _transcription, transcript_text, *_args, **_kwargs):
                captured_lattice_inputs.append(transcript_text)
                return transcript_text, []

            watcher.run_transcript_lattice_audit = fake_lattice
            try:
                processor.process_once(txt_path)
            finally:
                watcher.transcribe = original_transcribe
                watcher.safe_verify_voicemail_fields = original_verify
                watcher.run_transcript_lattice_audit = original_lattice

            self.assertEqual(captured_lattice_inputs, [original_text])
            with sqlite3.connect(settings.state_db) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT transcript, entities_json FROM voicemail_transcripts").fetchone()
            self.assertEqual(row["transcript"], original_text)
            entities = json.loads(row["entities_json"])
            self.assertEqual(entities["callback_number"], "2025550113")
            self.assertNotIn("transcript_corrections", entities)

    def test_audit_write_failure_blocks_field_verification_transcript_correction(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inbox = os.path.join(tmp, "154", "INBOX")
            txt_path, _ = self.write_portal_message(inbox, "msg0011", duration=20, origtime="1770000011")
            state_db = os.path.join(tmp, "state.sqlite3")
            with self.patched_env(
                {
                    "VOICEMAIL_WATCH_DIR": tmp,
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_EMAIL_ENABLED": "false",
                    "VERIFICATION_REQUIRE_AUDIT_FOR_APPLY": "true",
                    "GEMMA_FIELD_EXTRACTION_ENABLED": "false",
                    "TRANSCRIPT_LATTICE_ENABLED": "false",
                }
            ):
                settings = watcher.Settings.from_env()

            store = watcher.VoicemailStore(settings.state_db)
            processor = watcher.VoicemailProcessor(settings, store)
            processor.whisper_endpoints.least_busy_order = lambda *_args, **_kwargs: (settings.whisper_url,)
            original_text = "Call me at 2025550101."
            audit_rows = [
                {
                    "field_name": "callback_number",
                    "final_value": "202-555-0100",
                    "normalized_value": "2025550100",
                    "status": "parakeet_override",
                    "attribution_json": [
                        {
                            "field_name": "callback_number",
                            "word_start": 3,
                            "word_end": 3,
                        }
                    ],
                }
            ]

            original_transcribe = watcher.transcribe
            original_verify = watcher.safe_verify_voicemail_fields
            original_upsert = store.upsert_field_verifications
            watcher.transcribe = lambda *_args, **_kwargs: TranscriptionResult(
                text=original_text,
                entities={
                    "callback_number": "2025550113",
                    "_word_timestamps": [{"word": word} for word in original_text.split()],
                },
            )
            watcher.safe_verify_voicemail_fields = lambda *_args, **_kwargs: VerificationRunResult(
                proposed_entities={"callback_number": "202-555-0100"},
                audit_rows=audit_rows,
                should_apply=True,
                complete=True,
            )

            def fail_upsert(*_args, **_kwargs):
                raise RuntimeError("audit blocked")

            store.upsert_field_verifications = fail_upsert
            try:
                processor.process_once(txt_path)
            finally:
                watcher.transcribe = original_transcribe
                watcher.safe_verify_voicemail_fields = original_verify
                store.upsert_field_verifications = original_upsert

            with sqlite3.connect(settings.state_db) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT transcript, entities_json FROM voicemail_transcripts").fetchone()
            self.assertEqual(row["transcript"], original_text)
            entities = json.loads(row["entities_json"])
            self.assertEqual(entities["callback_number"], "2025550113")
            self.assertNotIn("transcript_corrections", entities)

    def test_lattice_apply_keeps_primary_punctuation_for_punctuation_only_candidate(self):
        primary = (
            "At this point, we're leaving it on the books, but we don't know if he'll even be able "
            "to make it at that point. But we would just like some clarification on what perhaps is "
            "going to happen. Thank you, Casey. Have a good weekend. Bye."
        )
        proposed = (
            "At this point we're leaving it on the books But we don't know if he'll even be able "
            "to make it At that point But we would just like some clarification On what perhaps is "
            "going to happen Thank you Casey Have a good weekend Bye"
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inbox = os.path.join(tmp, "154", "INBOX")
            os.makedirs(inbox)
            txt_path = os.path.join(inbox, "msg0002.txt")
            wav_path = os.path.join(inbox, "msg0002.wav")
            with open(txt_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "origmailbox=154\n"
                    "origtime=1770000001\n"
                    "origdate=Wed May 06 10:01:00 AM UTC 2026\n"
                    "duration=20\n"
                    "callerid=\"TEST CALLER\" <2175550100>\n"
                )
            with open(wav_path, "wb") as handle:
                handle.write(b"not real wav but stable")

            with self.patched_env(
                {
                    "VOICEMAIL_WATCH_DIR": tmp,
                    "VOICEMAIL_STATE_DB": os.path.join(tmp, "state.sqlite3"),
                    "VOICEMAIL_EMAIL_ENABLED": "false",
                    "VOICEMAIL_ACCURACY_PROFILE": "v1_5_max",
                    "TRANSCRIPT_LATTICE_APPLY_ENABLED": "true",
                    "LLM_ADJUDICATION_ENABLED": "false",
                    "TRANSCRIPT_LATTICE_LLM_ADJUDICATION_ENABLED": "false",
                    "GEMMA_FIELD_EXTRACTION_ENABLED": "false",
                }
            ):
                settings = watcher.Settings.from_env()
            store = watcher.VoicemailStore(settings.state_db)
            processor = watcher.VoicemailProcessor(settings, store)
            processor.whisper_endpoints.least_busy_order = lambda *_args, **_kwargs: (settings.whisper_url,)

            original_transcribe = watcher.transcribe
            original_verify = watcher.safe_verify_voicemail_fields
            original_get_email = watcher.get_email
            original_parakeet = watcher.call_parakeet_full_transcription
            original_build_spans = watcher.build_disagreement_spans
            watcher.transcribe = lambda *_args, **_kwargs: TranscriptionResult(
                text=primary,
                entities={},
                words=[{"word": token} for token in primary.split()],
            )
            watcher.safe_verify_voicemail_fields = lambda *_args, **_kwargs: VerificationRunResult(
                proposed_entities={},
                audit_rows=[],
                should_apply=False,
            )
            watcher.get_email = lambda *_args, **_kwargs: ["synthetic-user@example.invalid"]
            watcher.call_parakeet_full_transcription = lambda *_args, **_kwargs: {
                "run_id": "asr_parakeet_punctuation",
                "engine": "parakeet",
                "role": "full_pass",
                "audio_view": "canonical",
                "transcript": proposed,
                "processed_text": proposed,
                "words": [{"word": token} for token in proposed.split()],
                "created_utc": watcher.utc_now_iso(),
            }
            watcher.build_disagreement_spans = lambda *_args, **_kwargs: [
                DisagreementSpan(
                    span_id="lattice:punctuation-only",
                    start=None,
                    end=None,
                    primary_text=primary,
                    alternatives=[
                        {"text": primary, "source": "whisper:primary:canonical", "is_primary": True},
                        {"text": proposed, "source": "parakeet:full_pass:canonical", "strong": True},
                    ],
                )
            ]
            try:
                processor.process_once(txt_path)
            finally:
                watcher.transcribe = original_transcribe
                watcher.safe_verify_voicemail_fields = original_verify
                watcher.get_email = original_get_email
                watcher.call_parakeet_full_transcription = original_parakeet
                watcher.build_disagreement_spans = original_build_spans

            conn = sqlite3.connect(settings.state_db)
            conn.row_factory = sqlite3.Row
            try:
                transcript = conn.execute("SELECT transcript, entities_json FROM voicemail_transcripts").fetchone()
                self.assertEqual(transcript["transcript"], primary)
                entities = json.loads(transcript["entities_json"])
                self.assertEqual(entities["_corrected_transcript"], primary)
                rows = conn.execute("SELECT reason_code FROM transcript_corrections").fetchall()
                self.assertIn("punctuation_only_candidate_rejected", [row["reason_code"] for row in rows])
            finally:
                conn.close()

    def test_fallback_recipient_must_be_explicit_when_email_enabled(self):
        original_observer = watcher.Observer
        watcher.Observer = object
        try:
            settings = SimpleNamespace(
                watch_dir=os.getcwd(),
                local_timezone="America/Chicago",
                email_enabled=True,
                fallback_recipient="",
                whisper_api_key="test",
            )
            with self.assertRaises(RuntimeError):
                watcher.validate_startup_dependencies(settings)
        finally:
            watcher.Observer = original_observer

    def test_provisional_spelling_rules_apply_before_verification_without_mutating_input(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inbox = os.path.join(tmp, "155", "INBOX")
            txt_path, _ = self.write_portal_message(
                inbox,
                "msg0091",
                duration=20,
                origtime="1770000091",
            )
            metadata_path = Path(txt_path)
            metadata_path.write_text(
                metadata_path.read_text(encoding="utf-8").replace(
                    "origmailbox=154",
                    "origmailbox=155",
                ),
                encoding="utf-8",
            )
            rules_path = Path(tmp) / "mailbox_spelling_rules.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "155": [
                            {"from": "Mariah", "to": "Marria"},
                            {"from": "Zach", "to": "Zac"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.patched_env(
                {
                    "VOICEMAIL_WATCH_DIR": tmp,
                    "VOICEMAIL_STATE_DB": os.path.join(tmp, "state.sqlite3"),
                    "VOICEMAIL_EMAIL_ENABLED": "false",
                    "GEMMA_FIELD_EXTRACTION_ENABLED": "false",
                    "PARAKEET_FULL_PASS_ENABLED": "false",
                    "TRANSCRIPT_LATTICE_ENABLED": "false",
                    "VOICEMAIL_MAILBOX_SPELLING_RULES_ENABLED": "true",
                    "VOICEMAIL_MAILBOX_SPELLING_RULES_PATH": str(rules_path),
                }
            ):
                settings = watcher.Settings.from_env()

            store = watcher.VoicemailStore(settings.state_db)
            processor = watcher.VoicemailProcessor(settings, store)
            processor.whisper_endpoints.least_busy_order = lambda *_args, **_kwargs: (settings.whisper_url,)
            raw_transcription = TranscriptionResult(
                text="Mariah called Zach.",
                entities={"name": "Zach Example"},
            )
            observed = {}
            original_transcribe = watcher.transcribe
            original_verify = watcher.safe_verify_voicemail_fields
            watcher.transcribe = lambda *_args, **_kwargs: raw_transcription

            def synthetic_verify(_file_key, _wav_path, received_transcription, *_args, **_kwargs):
                with sqlite3.connect(settings.state_db) as connection:
                    connection.row_factory = sqlite3.Row
                    row = connection.execute(
                        "SELECT transcript, entities_json FROM voicemail_transcripts"
                    ).fetchone()
                observed["provisional_transcript"] = row["transcript"]
                observed["provisional_entities"] = json.loads(row["entities_json"])
                observed["verification_transcript"] = received_transcription.text
                observed["verification_entities"] = dict(received_transcription.entities or {})
                return VerificationRunResult(
                    proposed_entities=dict(received_transcription.entities or {}),
                    audit_rows=[],
                    should_apply=False,
                    complete=True,
                )

            watcher.safe_verify_voicemail_fields = synthetic_verify
            try:
                with self.assertLogs("voicemail_watcher", level="INFO") as captured_logs:
                    processor.process_once(txt_path)
            finally:
                watcher.transcribe = original_transcribe
                watcher.safe_verify_voicemail_fields = original_verify

            self.assertEqual(observed["provisional_transcript"], "Marria called Zac.")
            self.assertEqual(observed["provisional_entities"]["name"], "Zac Example")
            self.assertEqual(observed["verification_transcript"], "Mariah called Zach.")
            self.assertEqual(observed["verification_entities"]["name"], "Zach Example")
            self.assertEqual(raw_transcription.text, "Mariah called Zach.")
            self.assertEqual(raw_transcription.entities["name"], "Zach Example")
            self.assertTrue(any("stage=provisional" in line for line in captured_logs.output))
            self.assertTrue(any("stage=final" in line for line in captured_logs.output))

    def test_email_disabled_does_not_resolve_recipients_before_processing(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inbox = os.path.join(tmp, "154", "INBOX")
            txt_path, _ = self.write_portal_message(inbox, "msg0012", duration=20, origtime="1770000012")
            with self.patched_env(
                {
                    "VOICEMAIL_WATCH_DIR": tmp,
                    "VOICEMAIL_STATE_DB": os.path.join(tmp, "state.sqlite3"),
                    "VOICEMAIL_EMAIL_ENABLED": "false",
                    "VOICEMAIL_CONFIG": os.path.join(tmp, "missing.conf"),
                    "VOICEMAIL_FALLBACK_RECIPIENT": "",
                    "GEMMA_FIELD_EXTRACTION_ENABLED": "false",
                    "TRANSCRIPT_LATTICE_ENABLED": "false",
                }
            ):
                settings = watcher.Settings.from_env()

            store = watcher.VoicemailStore(settings.state_db)
            processor = watcher.VoicemailProcessor(settings, store)
            processor.whisper_endpoints.least_busy_order = lambda *_args, **_kwargs: (settings.whisper_url,)
            original_transcribe = watcher.transcribe
            original_verify = watcher.safe_verify_voicemail_fields
            original_get_email = watcher.get_email
            watcher.transcribe = lambda *_args, **_kwargs: TranscriptionResult(
                text="Synthetic voicemail.",
                entities={"callback_number": None},
            )
            watcher.safe_verify_voicemail_fields = lambda *_args, **_kwargs: VerificationRunResult(
                proposed_entities={"callback_number": None},
                audit_rows=[],
                should_apply=False,
                complete=False,
            )
            watcher.get_email = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("get_email should not run"))
            try:
                processor.process_once(txt_path)
            finally:
                watcher.transcribe = original_transcribe
                watcher.safe_verify_voicemail_fields = original_verify
                watcher.get_email = original_get_email

            with sqlite3.connect(settings.state_db) as conn:
                conn.row_factory = sqlite3.Row
                transcript = conn.execute("SELECT transcript FROM voicemail_transcripts").fetchone()
                queue_row = conn.execute("SELECT status FROM voicemails").fetchone()
            self.assertEqual(transcript["transcript"], "Synthetic voicemail.")
            self.assertEqual(queue_row["status"], "completed")

    def test_verification_observer_reports_gemma_and_parakeet_boundaries(self):
        with self.patched_env(
            {
                "GEMMA_FIELD_EXTRACTION_ENABLED": "true",
                "PARAKEET_VERIFICATION_ENABLED": "true",
                "VOICEMAIL_EMAIL_ENABLED": "false",
            }
        ):
            settings = watcher.Settings.from_env()
        transcription = transcription_for_text("Please call 217-555-0100.")
        observed = []
        original_call_gemma = watcher.call_gemma_field_extraction
        original_run_parakeet = watcher.run_parakeet_for_record

        def fake_gemma(*_args, **_kwargs):
            payload = watcher.empty_gemma_field_payload()
            payload["callback_numbers"] = [
                {
                    "raw": "217-555-0100",
                    "normalized": "2175550100",
                    "formatted": "217-555-0100",
                    "label_cue": "call",
                    "evidence_text": "call 217-555-0100",
                }
            ]
            return payload

        def fake_parakeet(_wav_path, record, *_args, **_kwargs):
            record.parakeet = ParakeetResult(record.candidate_id, normalized_numbers=["2175550100"])

        watcher.call_gemma_field_extraction = fake_gemma
        watcher.run_parakeet_for_record = fake_parakeet
        try:
            watcher.verify_voicemail_fields(
                "synthetic.wav",
                transcription,
                {"callerid": "Synthetic Caller <2175550100>"},
                settings,
                event_observer=lambda phase, **fields: observed.append((phase, fields)),
            )
        finally:
            watcher.call_gemma_field_extraction = original_call_gemma
            watcher.run_parakeet_for_record = original_run_parakeet

        self.assertEqual(
            [phase for phase, _fields in observed],
            ["gemma.started", "gemma.completed", "parakeet.started", "parakeet.completed"],
        )

    def test_verification_observer_does_not_claim_parakeet_without_candidates(self):
        with self.patched_env(
            {
                "GEMMA_FIELD_EXTRACTION_ENABLED": "true",
                "PARAKEET_VERIFICATION_ENABLED": "true",
                "VOICEMAIL_EMAIL_ENABLED": "false",
            }
        ):
            settings = watcher.Settings.from_env()
        observed = []
        original_call_gemma = watcher.call_gemma_field_extraction
        watcher.call_gemma_field_extraction = lambda *_args, **_kwargs: watcher.empty_gemma_field_payload()
        try:
            watcher.verify_voicemail_fields(
                "synthetic.wav",
                TranscriptionResult(text="Synthetic voicemail.", entities={}),
                {},
                settings,
                event_observer=lambda phase, **_fields: observed.append(phase),
            )
        finally:
            watcher.call_gemma_field_extraction = original_call_gemma

        self.assertEqual(observed, ["gemma.started", "gemma.completed"])

    def test_portal_login_rate_limit_helpers(self):
        import voicemail_portal

        settings = SimpleNamespace(login_rate_limit_attempts=2, login_rate_limit_window_seconds=300)
        key = "127.0.0.1:154"
        voicemail_portal.clear_login_failures(key)
        self.assertFalse(voicemail_portal.login_rate_limited(key, settings))
        voicemail_portal.record_login_failure(key, settings)
        self.assertFalse(voicemail_portal.login_rate_limited(key, settings))
        voicemail_portal.record_login_failure(key, settings)
        self.assertTrue(voicemail_portal.login_rate_limited(key, settings))
        voicemail_portal.clear_login_failures(key)
        self.assertFalse(voicemail_portal.login_rate_limited(key, settings))

    def test_portal_caller_id_digits_for_copy(self):
        import voicemail_portal

        self.assertEqual(
            voicemail_portal.caller_number_digits_from_callerid('"Caller" <(202)-555-0121>'),
            "2025550121",
        )
        self.assertEqual(
            voicemail_portal.caller_number_digits_from_callerid("+1 202-555-0121"),
            "2025550121",
        )
        self.assertEqual(voicemail_portal.caller_number_digits_from_callerid("<101>"), "")
        self.assertEqual(voicemail_portal.caller_number_digits_from_callerid("Unavailable"), "")

    def test_portal_payload_includes_caller_id_digits_for_copy(self):
        import voicemail_portal

        store = object.__new__(voicemail_portal.PortalStore)
        store.settings = SimpleNamespace(local_timezone="America/Chicago")
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT 'copy-key' AS file_key,
                       '154' AS extension,
                       '154' AS mailbox,
                       'INBOX' AS folder,
                       'msg0007' AS msg_name,
                       '' AS txt_path,
                       '' AS wav_path,
                       '"Test Caller" <2175550100>' AS callerid,
                       1770000007 AS origtime,
                       'Wed May 06 10:00:00 AM UTC 2026' AS origdate,
                       20 AS duration,
                       'Synthetic voicemail transcript.' AS transcript,
                       '{}' AS entities_json,
                       'completed' AS processing_status,
                       NULL AS deleted_utc,
                       NULL AS deleted_by
                """
            ).fetchone()
            payload = store._row_to_payload(row, include_transcript=True)
        finally:
            conn.close()

        self.assertEqual(payload["callerid_number_digits"], "2175550100")

    def test_portal_single_delete_clicks_use_undo_without_confirmation(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("staff", "6001", "", "Staff User", True)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")
        undo_delete_options = "{ confirmDelete: false, advanceAfterDelete: true, showUndo: true }"

        self.assertIn(
            f"deleteVoicemail(button.dataset.delete, {undo_delete_options})",
            page_html,
        )
        self.assertIn(
            f"deleteVoicemail(item.file_key, {undo_delete_options})",
            page_html,
        )
        self.assertIn(
            f"deleteVoicemail(selectedKey, {undo_delete_options})",
            page_html,
        )
        self.assertNotIn("deleteVoicemail(button.dataset.delete);", page_html)
        self.assertNotIn("deleteVoicemail(item.file_key);", page_html)
        self.assertIn(
            "confirm(`Delete ${keys.length} selected ${label} from the mailbox?`)",
            page_html,
        )
        self.assertNotIn("prompt(", page_html)
        self.assertIn('id="deleteCommentPanel"', page_html)
        self.assertIn('<label for="deleteCommentInline">Comment</label>', page_html)
        self.assertIn('id="deleteCommentSave"', page_html)
        self.assertIn('if (item.deleted_utc && !item.deleted_comment) return "";', page_html)
        self.assertIn('const readOnlyAttrs = item.deleted_utc ? "readonly" : "";', page_html)
        self.assertIn('${item.deleted_utc ? deleteCommentPanelHtml(item) : ""}', page_html)
        self.assertIn('${item.deleted_utc ? "" : deleteCommentPanelHtml(item)}', page_html)
        self.assertNotIn('${item.deleted_comment ? `<br>Comment:', page_html)
        self.assertNotIn("!item.delete_comment_required || item.deleted_utc", page_html)
        self.assertNotIn("Delete Comment", page_html)
        self.assertIn("function deleteCommentForKey(key, forceRequired = false)", page_html)
        self.assertIn("async function saveDeleteComment(key)", page_html)
        self.assertIn("/comment", page_html)
        self.assertNotIn("Required for your extension before deleting.", page_html)
        self.assertNotIn("Bulk delete uses this comment for selected voicemails.", page_html)

    def test_portal_single_delete_optimistically_removes_before_fetch(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("staff", "6001", "", "Staff User", True)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn("let pendingDeleteKeys = new Set();", page_html)
        self.assertIn("function optimisticallyRemoveVoicemails(keys)", page_html)
        self.assertIn("function restoreOptimisticVoicemails(snapshots)", page_html)
        self.assertIn("const optimisticSnapshots = optimisticallyRemoveVoicemails([key]);", page_html)
        self.assertLess(
            page_html.index("const optimisticSnapshots = optimisticallyRemoveVoicemails([key]);"),
            page_html.index("response = await fetch(`${basePath}/api/voicemails/${key}/delete`"),
        )
        self.assertIn("voicemails = loadedVoicemails.filter(item => !pendingDeleteKeys.has(item.file_key));", page_html)

    def test_portal_single_delete_restores_snapshot_on_failure_and_reloads_silently_on_success(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("staff", "6001", "", "Staff User", True)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn("restoreOptimisticVoicemails(optimisticSnapshots);", page_html)
        self.assertIn("alert(payload.detail || \"Delete failed.\");", page_html)
        self.assertIn("alert(\"Delete failed.\");", page_html)
        self.assertIn("if (showUndo) showUndoToast(key);", page_html)
        self.assertIn("void loadVoicemails({ silent: true });", page_html)
        self.assertNotIn("await loadVoicemails();\n      if (nextKey && voicemails.some", page_html)

    def test_portal_undo_toast_appears_before_delete_request_and_queues_early_click(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("staff", "6001", "", "Staff User", True)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn("let queuedUndoKeys = new Set();", page_html)
        self.assertLess(
            page_html.index("const optimisticSnapshots = optimisticallyRemoveVoicemails([key]);"),
            page_html.index("if (showUndo) showUndoToast(key);"),
        )
        self.assertLess(
            page_html.index("if (showUndo) showUndoToast(key);"),
            page_html.index("response = await fetch(`${basePath}/api/voicemails/${key}/delete`"),
        )
        self.assertIn("if (pendingDeleteKeys.has(restoreKey))", page_html)
        self.assertIn("queuedUndoKeys.add(restoreKey);", page_html)
        self.assertIn('button.textContent = "Restoring";', page_html)

    def test_portal_undo_toast_raised_above_delete_comment_save_button(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("staff", "6001", "", "Staff User", True)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn("bottom: var(--toast-bottom, 96px);", page_html)
        self.assertIn("function positionUndoToastAboveControls()", page_html)
        self.assertIn('document.getElementById("deleteCommentSave")', page_html)
        self.assertIn('host.style.setProperty("--toast-bottom", `${bottom}px`);', page_html)
        self.assertIn("positionUndoToastAboveControls();", page_html)

    def test_portal_queued_undo_runs_after_delete_success_and_failed_delete_hides_toast(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("staff", "6001", "", "Staff User", True)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn("const undoQueued = queuedUndoKeys.has(key);", page_html)
        self.assertIn(
            "restoreDeletedVoicemail(key, { confirmRestore: false, requireDeletedFolder: false });",
            page_html,
        )
        self.assertIn("if (undoToastKey === key) hideUndoToast();", page_html)
        self.assertIn("queuedUndoKeys.delete(key);", page_html)

    def test_portal_restore_optimistically_removes_deleted_row_before_fetch(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("staff", "6001", "", "Staff User", True)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn(
            'const restoreSnapshots = currentFolder === "deleted" ? optimisticallyRemoveVoicemails([key]) : [];',
            page_html,
        )
        self.assertLess(
            page_html.index(
                'const restoreSnapshots = currentFolder === "deleted" ? optimisticallyRemoveVoicemails([key]) : [];'
            ),
            page_html.index("response = await fetch(`${basePath}/api/voicemails/${key}/restore`"),
        )
        self.assertIn("restoreVoicemail(button.dataset.restore);", page_html)
        self.assertIn("restoreVoicemail(item.file_key)", page_html)

    def test_portal_restore_restores_snapshot_on_failure_and_preserves_undo_path(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("staff", "6001", "", "Staff User", True)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn("restoreOptimisticVoicemails(restoreSnapshots);", page_html)
        self.assertIn('alert("Restore failed.");', page_html)
        self.assertIn("alert(message);", page_html)
        self.assertIn(
            'restoreDeletedVoicemail(restoreKey, { confirmRestore: false, requireDeletedFolder: false })',
            page_html,
        )
        self.assertIn('currentFolder = "active";', page_html)
        self.assertIn("selectedKey = payload.file_key || null;", page_html)

    def test_portal_playback_speed_label_is_explicit(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("staff", "6001", "", "Staff User", True)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn('<span>Playback Speed</span>', page_html)
        self.assertNotIn('<span>Speed</span>', page_html)

    def test_portal_review_progress_counter_only_renders(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("staff", "6001", "", "Staff User", True)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        for expected in [
            "function reviewProgressHtml()",
            "function refreshReviewProgress()",
            "review-progress",
            "review-progress-fraction",
            "review-progress-current",
            "review-progress-divider",
            "review-progress-total",
            "reviewProgressSlot",
        ]:
            self.assertIn(expected, page_html)

        self.assertIn("${reviewProgressHtml()}", page_html)
        self.assertIn('<strong class="review-progress-current">', page_html)
        self.assertIn('<span class="review-progress-divider" aria-hidden="true"></span>', page_html)
        self.assertIn('<span class="review-progress-total">${items.length}</span>', page_html)
        self.assertNotIn("<span>Review</span>", page_html)
        self.assertNotIn('</strong>/${items.length}', page_html)
        self.assertNotIn("</strong> of ${items.length}", page_html)
        self.assertNotIn("confidence-badge", page_html)
        self.assertNotIn("evidence-callback", page_html)
        self.assertNotIn("fieldConfidenceBadgeHtml", page_html)
        self.assertNotIn("evidenceSpansForItem", page_html)

    def test_portal_keyboard_navigation_scrolls_only_list_container(self):
        import voicemail_portal

        user = voicemail_portal.PortalUser("staff", "6001", "", "Staff User", True)
        page_html = voicemail_portal.portal_page(user, "csrf-token").body.decode("utf-8")

        self.assertIn("function scrollRowIntoListView(row)", page_html)
        self.assertIn("scrollRowIntoListView(row);", page_html)
        self.assertIn('document.getElementById("items")', page_html)
        self.assertIn("desiredBottomComfort", page_html)
        self.assertIn("maxBottomComfort", page_html)
        self.assertIn("bottomComfort", page_html)
        self.assertIn("bottomLimit", page_html)
        self.assertIn("listRect.top + 8", page_html)
        self.assertIn("list.scrollTop", page_html)
        self.assertIn('event.key === "ArrowDown"', page_html)
        self.assertIn('event.key === "ArrowUp"', page_html)
        self.assertIn('event.key.toLowerCase() === "n"', page_html)
        self.assertNotIn("rowRect.bottom - listRect.bottom", page_html)
        self.assertNotIn("row.scrollIntoView", page_html)

    def test_portal_auto_delete_short_seconds_default_and_env_override(self):
        import voicemail_portal

        with self.patched_env({}):
            self.assertEqual(voicemail_portal.Settings.from_env().auto_delete_short_seconds, 5)
        with self.patched_env({"VOICEMAIL_PORTAL_AUTO_DELETE_SHORT_SECONDS": "6"}):
            self.assertEqual(voicemail_portal.Settings.from_env().auto_delete_short_seconds, 6)
        with self.patched_env({"VOICEMAIL_PORTAL_AUTO_DELETE_SHORT_SECONDS": "0"}):
            self.assertEqual(voicemail_portal.Settings.from_env().auto_delete_short_seconds, 0)

    def test_portal_delete_comment_and_retention_settings_parse_from_env(self):
        import voicemail_portal

        with self.patched_env(
            {
                "VOICEMAIL_PORTAL_DELETE_COMMENT_USER_EXTENSIONS": "154, 155\n154",
                "VOICEMAIL_PORTAL_DELETED_RETENTION_DAYS": "60",
            }
        ):
            settings = voicemail_portal.Settings.from_env()

        self.assertEqual(settings.delete_comment_user_extensions, ("154", "155"))
        self.assertEqual(settings.deleted_retention_days, 60)

    def test_portal_ami_mwi_settings_enable_only_with_credentials(self):
        import voicemail_portal

        with self.patched_env({}):
            default_settings = voicemail_portal.Settings.from_env()
        self.assertFalse(default_settings.mwi_refresh_enabled)
        self.assertEqual(default_settings.ami_host, "127.0.0.1")
        self.assertEqual(default_settings.ami_port, 5038)
        self.assertEqual(default_settings.ami_timeout_seconds, 3.0)
        self.assertEqual(default_settings.ami_default_context, "")

        with self.patched_env(
            {
                "VOICEMAIL_PORTAL_AMI_USERNAME": "portal_mwi",
                "VOICEMAIL_PORTAL_AMI_SECRET": "secret",
                "VOICEMAIL_PORTAL_AMI_HOST": "192.0.2.10",
                "VOICEMAIL_PORTAL_AMI_PORT": "5039",
                "VOICEMAIL_PORTAL_AMI_TIMEOUT": "4.5",
                "VOICEMAIL_PORTAL_AMI_CONTEXT": "default",
            }
        ):
            enabled_settings = voicemail_portal.Settings.from_env()
        self.assertTrue(enabled_settings.mwi_refresh_enabled)
        self.assertEqual(enabled_settings.ami_host, "192.0.2.10")
        self.assertEqual(enabled_settings.ami_port, 5039)
        self.assertEqual(enabled_settings.ami_username, "portal_mwi")
        self.assertEqual(enabled_settings.ami_secret, "secret")
        self.assertEqual(enabled_settings.ami_timeout_seconds, 4.5)
        self.assertEqual(enabled_settings.ami_default_context, "default")

    def test_portal_ami_header_values_reject_empty_and_line_breaks(self):
        import voicemail_portal

        self.assertEqual(voicemail_portal._ami_header_value(" mailbox ", "Mailbox"), "mailbox")
        for value in ("", " ", "default\rAction: Ping", "155\nAction: Ping"):
            with self.assertRaises(ValueError):
                voicemail_portal._ami_header_value(value, "AMI")

    def test_portal_mwi_target_derives_context_and_mailbox_from_spool_path(self):
        import voicemail_portal

        with self.patched_env(
            {
                "VOICEMAIL_WATCH_DIR": "/var/spool/asterisk/voicemail",
                "VOICEMAIL_PORTAL_AMI_CONTEXT": "",
            }
        ):
            settings = voicemail_portal.Settings.from_env()

        self.assertEqual(
            voicemail_portal.mailbox_context_from_path(
                "/var/spool/asterisk/voicemail/default/155/INBOX/msg0000.txt",
                settings,
            ),
            ("default", "155"),
        )

    def test_portal_mwi_target_derives_vitalpbx_context_from_test_path(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            path = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX", "msg0111.txt")
            with self.patched_env(
                {
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_AMI_CONTEXT": "",
                }
            ):
                settings = voicemail_portal.Settings.from_env()

            self.assertEqual(
                voicemail_portal.mailbox_context_from_path(path, settings),
                ("vitalpbx-voicemail", "154"),
            )

    def test_portal_delete_requests_targeted_mwi_refresh(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            self.write_portal_message(inbox, "msg0111", duration=20, origtime="1770000111")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_AMI_USERNAME": "portal_mwi",
                    "VOICEMAIL_PORTAL_AMI_SECRET": "secret",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                key = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE msg_name = 'msg0111'"
                ).fetchone()[0]

            user = voicemail_portal.PortalUser("154", "154", "", "154")
            calls = []
            old_get_store = voicemail_portal.get_store
            old_current_user = voicemail_portal.current_user
            old_require_csrf = voicemail_portal.require_csrf
            old_send = voicemail_portal.send_ami_voicemail_refresh
            voicemail_portal.get_store = lambda: store
            voicemail_portal.current_user = lambda _request: user
            voicemail_portal.require_csrf = lambda _request: None
            voicemail_portal.send_ami_voicemail_refresh = lambda context, mailbox, _settings: calls.append(
                (context, mailbox)
            ) or True
            try:
                response = voicemail_portal.delete_voicemail(key, SimpleNamespace())
                self.assertEqual(response.status_code, 200)
            finally:
                voicemail_portal.get_store = old_get_store
                voicemail_portal.current_user = old_current_user
                voicemail_portal.require_csrf = old_require_csrf
                voicemail_portal.send_ami_voicemail_refresh = old_send

            self.assertEqual(calls, [("vitalpbx-voicemail", "154")])

    def test_portal_delete_survives_ami_mwi_refresh_failure(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            txt_path, _wav_path = self.write_portal_message(inbox, "msg0117", duration=20, origtime="1770000117")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_AMI_USERNAME": "portal_mwi",
                    "VOICEMAIL_PORTAL_AMI_SECRET": "secret",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                key = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE msg_name = 'msg0117'"
                ).fetchone()[0]

            user = voicemail_portal.PortalUser("154", "154", "", "154")
            old_get_store = voicemail_portal.get_store
            old_current_user = voicemail_portal.current_user
            old_require_csrf = voicemail_portal.require_csrf
            old_send = voicemail_portal.send_ami_voicemail_refresh
            voicemail_portal.get_store = lambda: store
            voicemail_portal.current_user = lambda _request: user
            voicemail_portal.require_csrf = lambda _request: None
            voicemail_portal.send_ami_voicemail_refresh = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("AMI unavailable")
            )
            try:
                response = voicemail_portal.delete_voicemail(key, SimpleNamespace())
                self.assertEqual(response.status_code, 200)
            finally:
                voicemail_portal.get_store = old_get_store
                voicemail_portal.current_user = old_current_user
                voicemail_portal.require_csrf = old_require_csrf
                voicemail_portal.send_ami_voicemail_refresh = old_send

            with sqlite3.connect(state_db) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT folder, deleted_utc FROM voicemail_transcripts WHERE file_key = ?",
                    (key,),
                ).fetchone()
            self.assertFalse(os.path.exists(txt_path))
            self.assertEqual(row["folder"], "Deleted")
            self.assertIsNotNone(row["deleted_utc"])

    def test_watcher_schema_has_deleted_comment_and_upsert_preserves_saved_value(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db_path = os.path.join(tmp, "state.sqlite3")
            txt_path = os.path.join(tmp, "msg0201.txt")
            wav_path = os.path.join(tmp, "msg0201.wav")
            Path(txt_path).write_text("origmailbox=154\norigtime=1770000201\n", encoding="utf-8")
            Path(wav_path).write_bytes(b"not real wav")
            store = watcher.VoicemailStore(db_path)
            store.upsert_transcript(
                "watcher-comment-key",
                "154",
                txt_path,
                wav_path,
                {"origmailbox": "154", "origtime": "1770000201", "duration": "20"},
                "hello",
                {"name": "Test"},
            )

            with sqlite3.connect(db_path) as conn:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(voicemail_transcripts)")}
                self.assertIn("deleted_comment", columns)
                conn.execute(
                    """
                    UPDATE voicemail_transcripts
                    SET deleted_utc = ?, deleted_by = '154', deleted_comment = 'stale'
                    WHERE file_key = 'watcher-comment-key'
                    """,
                    (watcher.utc_now_iso(),),
                )

            store.upsert_transcript(
                "watcher-comment-key",
                "154",
                txt_path,
                wav_path,
                {"origmailbox": "154", "origtime": "1770000201", "duration": "20"},
                "hello again",
                {"name": "Test"},
            )

            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT deleted_utc, deleted_by, deleted_comment FROM voicemail_transcripts"
                ).fetchone()
            self.assertIsNone(row["deleted_utc"])
            self.assertIsNone(row["deleted_by"])
            self.assertEqual(row["deleted_comment"], "stale")

    def test_portal_protected_user_delete_requires_and_stores_comment(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            txt_path, _wav_path = self.write_portal_message(inbox, "msg0101", duration=20, origtime="1770000101")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_DELETE_COMMENT_USER_EXTENSIONS": "154",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                key = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE msg_name = 'msg0101'"
                ).fetchone()[0]

            user = voicemail_portal.PortalUser("154", "154", "", "154")
            old_get_store = voicemail_portal.get_store
            old_current_user = voicemail_portal.current_user
            old_require_csrf = voicemail_portal.require_csrf
            voicemail_portal.get_store = lambda: store
            voicemail_portal.current_user = lambda _request: user
            voicemail_portal.require_csrf = lambda _request: None
            try:
                with self.assertRaises(Exception) as raised:
                    voicemail_portal.delete_voicemail_with_comment(
                        key,
                        voicemail_portal.DeleteVoicemailRequest(),
                        SimpleNamespace(),
                    )
                self.assertEqual(raised.exception.status_code, 400)
                self.assertTrue(os.path.exists(txt_path))

                response = voicemail_portal.delete_voicemail_with_comment(
                    key,
                    voicemail_portal.DeleteVoicemailRequest(comment="  Duplicate request  "),
                    SimpleNamespace(),
                )
                self.assertEqual(response.status_code, 200)
            finally:
                voicemail_portal.get_store = old_get_store
                voicemail_portal.current_user = old_current_user
                voicemail_portal.require_csrf = old_require_csrf

            with sqlite3.connect(state_db) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """
                    SELECT folder, deleted_by, deleted_comment, txt_path
                    FROM voicemail_transcripts
                    WHERE file_key = ?
                    """,
                    (key,),
                ).fetchone()

            self.assertEqual(row["folder"], "Deleted")
            self.assertEqual(row["deleted_by"], "154")
            self.assertEqual(row["deleted_comment"], "Duplicate request")
            self.assertFalse(os.path.exists(txt_path))
            self.assertIn(os.path.abspath(trash_dir), os.path.abspath(row["txt_path"]))

            deleted = store.list_voicemails(user, "deleted")
            self.assertEqual(deleted[0]["deleted_comment"], "Duplicate request")
            self.assertTrue(deleted[0]["delete_comment_required"])

    def test_portal_admin_delete_requires_comment_for_protected_voicemail_extension(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            txt_path, _wav_path = self.write_portal_message(inbox, "msg0108", duration=20, origtime="1770000108")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_DELETE_COMMENT_USER_EXTENSIONS": "154",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                key = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE msg_name = 'msg0108'"
                ).fetchone()[0]

            admin = voicemail_portal.PortalUser("admin", "*", "", "Admin", True)
            active = store.list_voicemails(admin, "active")
            self.assertTrue(active[0]["delete_comment_required"])

            old_get_store = voicemail_portal.get_store
            old_current_user = voicemail_portal.current_user
            old_require_csrf = voicemail_portal.require_csrf
            voicemail_portal.get_store = lambda: store
            voicemail_portal.current_user = lambda _request: admin
            voicemail_portal.require_csrf = lambda _request: None
            try:
                with self.assertRaises(Exception) as raised:
                    voicemail_portal.delete_voicemail_with_comment(
                        key,
                        voicemail_portal.DeleteVoicemailRequest(),
                        SimpleNamespace(),
                    )
                self.assertEqual(raised.exception.status_code, 400)
                self.assertTrue(os.path.exists(txt_path))

                response = voicemail_portal.delete_voicemail_with_comment(
                    key,
                    voicemail_portal.DeleteVoicemailRequest(comment="admin cleanup"),
                    SimpleNamespace(),
                )
                self.assertEqual(response.status_code, 200)
            finally:
                voicemail_portal.get_store = old_get_store
                voicemail_portal.current_user = old_current_user
                voicemail_portal.require_csrf = old_require_csrf

            with sqlite3.connect(state_db) as conn:
                row = conn.execute(
                    "SELECT deleted_by, deleted_comment FROM voicemail_transcripts WHERE file_key = ?",
                    (key,),
                ).fetchone()
            self.assertEqual(row[0], "admin")
            self.assertEqual(row[1], "admin cleanup")

    def test_portal_unprotected_legacy_delete_still_works_without_comment(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            self.write_portal_message(inbox, "msg0102", duration=20, origtime="1770000102")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_DELETE_COMMENT_USER_EXTENSIONS": "155",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                key = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE msg_name = 'msg0102'"
                ).fetchone()[0]

            user = voicemail_portal.PortalUser("154", "154", "", "154")
            old_get_store = voicemail_portal.get_store
            old_current_user = voicemail_portal.current_user
            old_require_csrf = voicemail_portal.require_csrf
            voicemail_portal.get_store = lambda: store
            voicemail_portal.current_user = lambda _request: user
            voicemail_portal.require_csrf = lambda _request: None
            try:
                response = voicemail_portal.delete_voicemail(key, SimpleNamespace())
                self.assertEqual(response.status_code, 200)
            finally:
                voicemail_portal.get_store = old_get_store
                voicemail_portal.current_user = old_current_user
                voicemail_portal.require_csrf = old_require_csrf

            with sqlite3.connect(state_db) as conn:
                row = conn.execute(
                    "SELECT deleted_comment FROM voicemail_transcripts WHERE file_key = ?",
                    (key,),
                ).fetchone()
            self.assertIsNone(row[0])

    def test_portal_unprotected_delete_stores_optional_comment(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            self.write_portal_message(inbox, "msg0109", duration=20, origtime="1770000109")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_DELETE_COMMENT_USER_EXTENSIONS": "155",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                key = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE msg_name = 'msg0109'"
                ).fetchone()[0]

            user = voicemail_portal.PortalUser("154", "154", "", "154")
            old_get_store = voicemail_portal.get_store
            old_current_user = voicemail_portal.current_user
            old_require_csrf = voicemail_portal.require_csrf
            voicemail_portal.get_store = lambda: store
            voicemail_portal.current_user = lambda _request: user
            voicemail_portal.require_csrf = lambda _request: None
            try:
                response = voicemail_portal.delete_voicemail_with_comment(
                    key,
                    voicemail_portal.DeleteVoicemailRequest(comment="optional note"),
                    SimpleNamespace(),
                )
                self.assertEqual(response.status_code, 200)
            finally:
                voicemail_portal.get_store = old_get_store
                voicemail_portal.current_user = old_current_user
                voicemail_portal.require_csrf = old_require_csrf

            with sqlite3.connect(state_db) as conn:
                row = conn.execute(
                    "SELECT deleted_comment FROM voicemail_transcripts WHERE file_key = ?",
                    (key,),
                ).fetchone()
            self.assertEqual(row[0], "optional note")

    def test_portal_save_comment_persists_without_deleting_voicemail(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            txt_path, _wav_path = self.write_portal_message(inbox, "msg0110", duration=20, origtime="1770000110")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                key = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE msg_name = 'msg0110'"
                ).fetchone()[0]

            user = voicemail_portal.PortalUser("154", "154", "", "154")
            old_get_store = voicemail_portal.get_store
            old_current_user = voicemail_portal.current_user
            old_require_csrf = voicemail_portal.require_csrf
            voicemail_portal.get_store = lambda: store
            voicemail_portal.current_user = lambda _request: user
            voicemail_portal.require_csrf = lambda _request: None
            try:
                response = voicemail_portal.save_voicemail_comment(
                    key,
                    voicemail_portal.SaveVoicemailCommentRequest(comment=" call patient back "),
                    SimpleNamespace(),
                )
                payload = json.loads(response.body.decode("utf-8"))
            finally:
                voicemail_portal.get_store = old_get_store
                voicemail_portal.current_user = old_current_user
                voicemail_portal.require_csrf = old_require_csrf

            self.assertEqual(payload["comment"], "call patient back")
            self.assertTrue(os.path.exists(txt_path))
            with sqlite3.connect(state_db) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT folder, deleted_utc, deleted_comment FROM voicemail_transcripts WHERE file_key = ?",
                    (key,),
                ).fetchone()
            self.assertEqual(row["folder"], "INBOX")
            self.assertIsNone(row["deleted_utc"])
            self.assertEqual(row["deleted_comment"], "call patient back")

            active = store.list_voicemails(user, "active")
            self.assertEqual(active[0]["deleted_comment"], "call patient back")

    def test_portal_bulk_delete_uses_one_shared_comment_for_protected_user(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            self.write_portal_message(inbox, "msg0103", duration=20, origtime="1770000103")
            self.write_portal_message(inbox, "msg0104", duration=20, origtime="1770000104")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_DELETE_COMMENT_USER_EXTENSIONS": "154",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                keys = [
                    row[0]
                    for row in conn.execute(
                        """
                        SELECT file_key
                        FROM voicemail_transcripts
                        WHERE msg_name IN ('msg0103', 'msg0104')
                        ORDER BY msg_name
                        """
                    ).fetchall()
                ]

            user = voicemail_portal.PortalUser("154", "154", "", "154")
            old_get_store = voicemail_portal.get_store
            old_current_user = voicemail_portal.current_user
            old_require_csrf = voicemail_portal.require_csrf
            voicemail_portal.get_store = lambda: store
            voicemail_portal.current_user = lambda _request: user
            voicemail_portal.require_csrf = lambda _request: None
            try:
                response = voicemail_portal.bulk_delete_voicemails(
                    voicemail_portal.BulkDeleteRequest(file_keys=keys, comment="bulk cleanup"),
                    SimpleNamespace(),
                )
                payload = json.loads(response.body.decode("utf-8"))
            finally:
                voicemail_portal.get_store = old_get_store
                voicemail_portal.current_user = old_current_user
                voicemail_portal.require_csrf = old_require_csrf

            self.assertEqual(payload["deleted_count"], 2)
            with sqlite3.connect(state_db) as conn:
                comments = [
                    row[0]
                    for row in conn.execute(
                        "SELECT deleted_comment FROM voicemail_transcripts ORDER BY msg_name"
                    ).fetchall()
                ]
            self.assertEqual(comments, ["bulk cleanup", "bulk cleanup"])

    def test_portal_bulk_delete_dedupes_mwi_refresh_targets(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            self.write_portal_message(inbox, "msg0112", duration=20, origtime="1770000112")
            self.write_portal_message(inbox, "msg0113", duration=20, origtime="1770000113")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_AMI_USERNAME": "portal_mwi",
                    "VOICEMAIL_PORTAL_AMI_SECRET": "secret",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                keys = [
                    row[0]
                    for row in conn.execute(
                        """
                        SELECT file_key
                        FROM voicemail_transcripts
                        WHERE msg_name IN ('msg0112', 'msg0113')
                        ORDER BY msg_name
                        """
                    ).fetchall()
                ]

            user = voicemail_portal.PortalUser("154", "154", "", "154")
            calls = []
            old_get_store = voicemail_portal.get_store
            old_current_user = voicemail_portal.current_user
            old_require_csrf = voicemail_portal.require_csrf
            old_send = voicemail_portal.send_ami_voicemail_refresh
            voicemail_portal.get_store = lambda: store
            voicemail_portal.current_user = lambda _request: user
            voicemail_portal.require_csrf = lambda _request: None
            voicemail_portal.send_ami_voicemail_refresh = lambda context, mailbox, _settings: calls.append(
                (context, mailbox)
            ) or True
            try:
                response = voicemail_portal.bulk_delete_voicemails(
                    voicemail_portal.BulkDeleteRequest(file_keys=keys),
                    SimpleNamespace(),
                )
                payload = json.loads(response.body.decode("utf-8"))
            finally:
                voicemail_portal.get_store = old_get_store
                voicemail_portal.current_user = old_current_user
                voicemail_portal.require_csrf = old_require_csrf
                voicemail_portal.send_ami_voicemail_refresh = old_send

            self.assertEqual(payload["deleted_count"], 2)
            self.assertEqual(calls, [("vitalpbx-voicemail", "154")])

    def test_portal_bulk_delete_does_not_refresh_failed_rows(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            self.write_portal_message(inbox, "msg0114", duration=20, origtime="1770000114")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_AMI_USERNAME": "portal_mwi",
                    "VOICEMAIL_PORTAL_AMI_SECRET": "secret",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                key = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE msg_name = 'msg0114'"
                ).fetchone()[0]

            user = voicemail_portal.PortalUser("154", "154", "", "154")
            calls = []
            old_get_store = voicemail_portal.get_store
            old_current_user = voicemail_portal.current_user
            old_require_csrf = voicemail_portal.require_csrf
            old_move = voicemail_portal.move_message_to_trash
            old_send = voicemail_portal.send_ami_voicemail_refresh
            voicemail_portal.get_store = lambda: store
            voicemail_portal.current_user = lambda _request: user
            voicemail_portal.require_csrf = lambda _request: None
            voicemail_portal.move_message_to_trash = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("move blocked")
            )
            voicemail_portal.send_ami_voicemail_refresh = lambda context, mailbox, _settings: calls.append(
                (context, mailbox)
            ) or True
            try:
                response = voicemail_portal.bulk_delete_voicemails(
                    voicemail_portal.BulkDeleteRequest(file_keys=[key]),
                    SimpleNamespace(),
                )
                payload = json.loads(response.body.decode("utf-8"))
            finally:
                voicemail_portal.get_store = old_get_store
                voicemail_portal.current_user = old_current_user
                voicemail_portal.require_csrf = old_require_csrf
                voicemail_portal.move_message_to_trash = old_move
                voicemail_portal.send_ami_voicemail_refresh = old_send

            self.assertEqual(response.status_code, 400)
            self.assertEqual(payload["deleted_count"], 0)
            self.assertEqual(calls, [])

    def test_portal_restore_clears_deleted_comment(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            txt_path, wav_path = self.write_portal_message(inbox, "msg0105", duration=20, origtime="1770000105")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                key = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE msg_name = 'msg0105'"
                ).fetchone()[0]

            store.mark_deleted(key, "154", [], "wrong mailbox")
            store.mark_restored(key, key, txt_path, wav_path, "msg0105")

            with sqlite3.connect(state_db) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT deleted_utc, deleted_by, deleted_comment FROM voicemail_transcripts WHERE file_key = ?",
                    (key,),
                ).fetchone()
            self.assertIsNone(row["deleted_utc"])
            self.assertIsNone(row["deleted_by"])
            self.assertIsNone(row["deleted_comment"])

    def test_portal_restore_requests_mwi_refresh_for_restored_inbox_target(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            self.write_portal_message(inbox, "msg0115", duration=20, origtime="1770000115")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_AMI_USERNAME": "portal_mwi",
                    "VOICEMAIL_PORTAL_AMI_SECRET": "secret",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()
            user = voicemail_portal.PortalUser("154", "154", "", "154")

            with sqlite3.connect(state_db) as conn:
                key = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE msg_name = 'msg0115'"
                ).fetchone()[0]

            row = store.get_voicemail(key, user)
            moved = voicemail_portal.move_message_to_trash(row, settings)
            store.mark_deleted(key, user.username, moved)

            calls = []
            old_get_store = voicemail_portal.get_store
            old_current_user = voicemail_portal.current_user
            old_require_csrf = voicemail_portal.require_csrf
            old_send = voicemail_portal.send_ami_voicemail_refresh
            voicemail_portal.get_store = lambda: store
            voicemail_portal.current_user = lambda _request: user
            voicemail_portal.require_csrf = lambda _request: None
            voicemail_portal.send_ami_voicemail_refresh = lambda context, mailbox, _settings: calls.append(
                (context, mailbox)
            ) or True
            try:
                response = voicemail_portal.restore_voicemail(key, SimpleNamespace())
                payload = json.loads(response.body.decode("utf-8"))
            finally:
                voicemail_portal.get_store = old_get_store
                voicemail_portal.current_user = old_current_user
                voicemail_portal.require_csrf = old_require_csrf
                voicemail_portal.send_ami_voicemail_refresh = old_send

            self.assertEqual(response.status_code, 200)
            self.assertTrue(payload["ok"])
            self.assertEqual(calls, [("vitalpbx-voicemail", "154")])

    def test_portal_retention_purges_old_deleted_voicemails_only(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            old_txt, _ = self.write_portal_message(inbox, "msg0106", duration=20, origtime="1770000106")
            recent_txt, _ = self.write_portal_message(inbox, "msg0107", duration=20, origtime="1770000107")
            outside_txt = os.path.join(tmp, "outside-msg.txt")
            Path(outside_txt).write_text("keep me", encoding="utf-8")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_DELETED_RETENTION_DAYS": "60",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                conn.row_factory = sqlite3.Row
                old_key = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE msg_name = 'msg0106'"
                ).fetchone()["file_key"]
                recent_key = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE msg_name = 'msg0107'"
                ).fetchone()["file_key"]

            old_row = store.get_voicemail(old_key, voicemail_portal.PortalUser("154", "154", "", "154"))
            recent_row = store.get_voicemail(recent_key, voicemail_portal.PortalUser("154", "154", "", "154"))
            old_moved = voicemail_portal.move_message_to_trash(old_row, settings)
            recent_moved = voicemail_portal.move_message_to_trash(recent_row, settings)
            store.mark_deleted(old_key, "154", old_moved, "old")
            store.mark_deleted(recent_key, "154", recent_moved, "recent")

            old_deleted = (datetime.now(timezone.utc) - timedelta(days=61)).isoformat()
            recent_deleted = (datetime.now(timezone.utc) - timedelta(days=59)).isoformat()
            outside_key = "a" * 32
            now = voicemail_portal.utc_now_iso()
            with sqlite3.connect(state_db) as conn:
                conn.execute(
                    """
                    UPDATE voicemail_transcripts
                    SET deleted_utc = ?
                    WHERE file_key = ?
                    """,
                    (old_deleted, old_key),
                )
                conn.execute(
                    """
                    UPDATE voicemail_transcripts
                    SET deleted_utc = ?
                    WHERE file_key = ?
                    """,
                    (recent_deleted, recent_key),
                )
                conn.execute(
                    """
                    CREATE TABLE voicemail_field_verification (
                        file_key TEXT,
                        field_name TEXT
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO voicemail_field_verification (file_key, field_name) VALUES (?, 'name')",
                    (old_key,),
                )
                conn.execute(
                    """
                    INSERT INTO voicemail_transcripts (
                        file_key, extension, mailbox, folder, msg_name, txt_path, wav_path,
                        callerid, origtime, origdate, duration, transcript, entities_json,
                        created_utc, updated_utc, deleted_utc, deleted_by, deleted_comment
                    )
                    VALUES (?, '154', '154', 'Deleted', 'msg9999', ?, ?, '', 0, '', 20, '',
                            '{}', ?, ?, ?, '154', 'outside')
                    """,
                    (outside_key, outside_txt, outside_txt, now, now, old_deleted),
                )

            self.assertEqual(store.purge_expired_deleted_voicemails(), 2)

            self.assertFalse(any(Path(path).exists() for path in old_moved))
            self.assertTrue(any(Path(path).exists() for path in recent_moved))
            self.assertTrue(os.path.exists(outside_txt))
            with sqlite3.connect(state_db) as conn:
                old_count = conn.execute(
                    "SELECT count(*) FROM voicemail_transcripts WHERE file_key = ?",
                    (old_key,),
                ).fetchone()[0]
                recent_count = conn.execute(
                    "SELECT count(*) FROM voicemail_transcripts WHERE file_key = ?",
                    (recent_key,),
                ).fetchone()[0]
                outside_count = conn.execute(
                    "SELECT count(*) FROM voicemail_transcripts WHERE file_key = ?",
                    (outside_key,),
                ).fetchone()[0]
                aux_count = conn.execute(
                    "SELECT count(*) FROM voicemail_field_verification WHERE file_key = ?",
                    (old_key,),
                ).fetchone()[0]
            self.assertEqual(old_count, 0)
            self.assertEqual(recent_count, 1)
            self.assertEqual(outside_count, 0)
            self.assertEqual(aux_count, 0)

    def test_portal_sync_retains_active_voicemail_files_for_external_delete_recovery(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            txt_path, _wav_path = self.write_portal_message(inbox, "msg0111", duration=20, origtime="1770000111")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            self.assertEqual(store.sync_filesystem(), 1)

            info = voicemail_portal.parse_txt(txt_path)
            key = voicemail_portal.build_file_key("154", info, txt_path)
            retained_dir = voicemail_portal.external_retention_message_dir(key, txt_path, settings)

            self.assertTrue((retained_dir / "msg0111.txt").exists())
            self.assertTrue((retained_dir / "msg0111.wav").exists())
            self.assertTrue((retained_dir / "msg0111.gsm").exists())
            self.assertTrue(Path(txt_path).exists())

    def test_portal_external_delete_uses_retained_audio_in_deleted_tab(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            txt_path, wav_path = self.write_portal_message(inbox, "msg0112", duration=20, origtime="1770000112")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            info = voicemail_portal.parse_txt(txt_path)
            key = voicemail_portal.build_file_key("154", info, txt_path)
            for path in (txt_path, wav_path, os.path.join(inbox, "msg0112.gsm")):
                os.remove(path)

            self.assertEqual(store.sync_filesystem(), 0)

            with sqlite3.connect(state_db) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """
                    SELECT folder, deleted_by, txt_path, wav_path
                    FROM voicemail_transcripts
                    WHERE file_key = ?
                    """,
                    (key,),
                ).fetchone()

            self.assertEqual(row["folder"], "Deleted")
            self.assertEqual(row["deleted_by"], "external")
            self.assertIn(os.path.abspath(trash_dir), os.path.abspath(row["txt_path"]))
            self.assertIn(voicemail_portal.EXTERNAL_RETENTION_DIR_NAME, Path(row["txt_path"]).parts)
            self.assertTrue(os.path.exists(row["wav_path"]))

            user = voicemail_portal.PortalUser("154", "154", "", "154")
            deleted = store.list_voicemails(user, "deleted")
            deleted_item = next(item for item in deleted if item["file_key"] == key)
            self.assertTrue(deleted_item["has_audio"])

    def test_portal_external_delete_can_restore_from_retained_copy_and_cleans_it_up(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            txt_path, wav_path = self.write_portal_message(inbox, "msg0113", duration=20, origtime="1770000113")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            info = voicemail_portal.parse_txt(txt_path)
            key = voicemail_portal.build_file_key("154", info, txt_path)
            retained_key_dir = voicemail_portal.external_retention_key_dir(key, settings)
            for path in (txt_path, wav_path, os.path.join(inbox, "msg0113.gsm")):
                os.remove(path)
            store.sync_filesystem()

            user = voicemail_portal.PortalUser("154", "154", "", "154")
            row = store.get_voicemail(key, user, include_deleted=True)
            restored = voicemail_portal.restore_message_to_inbox(row, settings)
            store.mark_restored(
                key,
                str(restored["file_key"]),
                str(restored["txt_path"]),
                str(restored["wav_path"]),
                str(restored["msg_name"]),
            )

            self.assertTrue(Path(restored["txt_path"]).exists())
            self.assertTrue(Path(restored["wav_path"]).exists())
            self.assertTrue((Path(restored["txt_path"]).parent / "msg0113.gsm").exists())
            self.assertFalse(retained_key_dir.exists())
            active = store.list_voicemails(user, "active")
            restored_item = next(item for item in active if item["file_key"] == restored["file_key"])
            self.assertTrue(restored_item["has_audio"])

    def test_portal_retention_purges_external_retained_copy_after_sixty_days(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            txt_path, wav_path = self.write_portal_message(inbox, "msg0114", duration=20, origtime="1770000114")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_DELETED_RETENTION_DAYS": "60",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            info = voicemail_portal.parse_txt(txt_path)
            key = voicemail_portal.build_file_key("154", info, txt_path)
            retained_key_dir = voicemail_portal.external_retention_key_dir(key, settings)
            for path in (txt_path, wav_path, os.path.join(inbox, "msg0114.gsm")):
                os.remove(path)
            store.sync_filesystem()

            old_deleted = (datetime.now(timezone.utc) - timedelta(days=61)).isoformat()
            with sqlite3.connect(state_db) as conn:
                conn.execute(
                    "UPDATE voicemail_transcripts SET deleted_utc = ? WHERE file_key = ?",
                    (old_deleted, key),
                )

            self.assertTrue(retained_key_dir.exists())
            self.assertEqual(store.purge_expired_deleted_voicemails(), 1)
            self.assertFalse(retained_key_dir.exists())
            with sqlite3.connect(state_db) as conn:
                count = conn.execute(
                    "SELECT count(*) FROM voicemail_transcripts WHERE file_key = ?",
                    (key,),
                ).fetchone()[0]
            self.assertEqual(count, 0)

    def test_portal_sync_auto_deletes_only_short_active_voicemails(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            short_txt, short_wav = self.write_portal_message(inbox, "msg0001", duration=5, origtime="1770000001")
            keep_txt, _ = self.write_portal_message(inbox, "msg0002", duration=6, origtime="1770000002")
            missing_duration_txt, _ = self.write_portal_message(inbox, "msg0003", duration=None, origtime="1770000003")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_AUTO_DELETE_SHORT_SECONDS": "0",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                conn.row_factory = sqlite3.Row
                short_row = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE msg_name = 'msg0001'"
                ).fetchone()
                short_key = short_row["file_key"]
                now = voicemail_portal.utc_now_iso()
                conn.execute(
                    """
                    UPDATE voicemails
                    SET status = 'discovered',
                        txt_path = ?,
                        wav_path = ?,
                        updated_utc = ?
                    WHERE file_key = ?
                    """,
                    (short_txt, short_wav, now, short_key),
                )

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_AUTO_DELETE_SHORT_SECONDS": "6",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            self.assertEqual(store.sync_filesystem(), 3)

            self.assertFalse(os.path.exists(short_txt))
            self.assertFalse(os.path.exists(short_wav))
            self.assertTrue(os.path.exists(keep_txt))
            self.assertTrue(os.path.exists(missing_duration_txt))
            self.assertEqual(len(list(Path(trash_dir).rglob("msg0001.txt"))), 1)
            self.assertEqual(len(list(Path(trash_dir).rglob("msg0001.wav"))), 1)
            self.assertEqual(len(list(Path(trash_dir).rglob("msg0001.gsm"))), 1)

            with sqlite3.connect(state_db) as conn:
                conn.row_factory = sqlite3.Row
                short_row = conn.execute(
                    """
                    SELECT folder, deleted_utc, deleted_by, txt_path, wav_path
                    FROM voicemail_transcripts
                    WHERE msg_name = 'msg0001'
                    """
                ).fetchone()
                keep_row = conn.execute(
                    "SELECT folder, deleted_utc FROM voicemail_transcripts WHERE msg_name = 'msg0002'"
                ).fetchone()
                missing_duration_row = conn.execute(
                    "SELECT folder, deleted_utc FROM voicemail_transcripts WHERE msg_name = 'msg0003'"
                ).fetchone()
                queue_row = conn.execute(
                    "SELECT status, last_error FROM voicemails WHERE txt_path = ?",
                    (short_txt,),
                ).fetchone()

            self.assertEqual(short_row["folder"], "Deleted")
            self.assertIsNotNone(short_row["deleted_utc"])
            self.assertEqual(short_row["deleted_by"], "portal-auto-short")
            self.assertIn(os.path.abspath(trash_dir), os.path.abspath(short_row["txt_path"]))
            self.assertIn(os.path.abspath(trash_dir), os.path.abspath(short_row["wav_path"]))
            self.assertEqual(keep_row["folder"], "INBOX")
            self.assertIsNone(keep_row["deleted_utc"])
            self.assertEqual(missing_duration_row["folder"], "INBOX")
            self.assertIsNone(missing_duration_row["deleted_utc"])
            self.assertEqual(queue_row["status"], "skipped")
            self.assertIn("portal-auto-short", queue_row["last_error"])

            user = voicemail_portal.PortalUser("154", "154", "", "154")
            active_keys = {item["file_key"] for item in store.list_voicemails(user, "active")}
            deleted_keys = {item["file_key"] for item in store.list_voicemails(user, "deleted")}
            self.assertNotIn(short_key, active_keys)
            self.assertEqual(
                sum(1 for item in store.list_voicemails(user, "active") if item["duration"] == 6),
                1,
            )
            self.assertEqual(
                sum(1 for item in store.list_voicemails(user, "active") if item["duration"] is None),
                1,
            )
            self.assertIn(short_key, deleted_keys)

    def test_portal_auto_delete_requests_mwi_refresh(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            self.write_portal_message(inbox, "msg0116", duration=5, origtime="1770000116")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_AUTO_DELETE_SHORT_SECONDS": "0",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_AUTO_DELETE_SHORT_SECONDS": "6",
                    "VOICEMAIL_PORTAL_AMI_USERNAME": "portal_mwi",
                    "VOICEMAIL_PORTAL_AMI_SECRET": "secret",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)

            calls = []
            old_send = voicemail_portal.send_ami_voicemail_refresh
            voicemail_portal.send_ami_voicemail_refresh = lambda context, mailbox, _settings: calls.append(
                (context, mailbox)
            ) or True
            try:
                self.assertEqual(store.auto_delete_short_voicemails(), 1)
            finally:
                voicemail_portal.send_ami_voicemail_refresh = old_send

            self.assertEqual(calls, [("vitalpbx-voicemail", "154")])

    def test_portal_sync_dedupes_same_path_and_preserves_completed_transcript(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            txt_path, wav_path = self.write_portal_message(inbox, "msg0005", duration=20, origtime="1770000005")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            info = voicemail_portal.parse_txt(txt_path)
            current_key = voicemail_portal.build_file_key("154", info, txt_path)
            old_key = "old-completed-key"
            now = voicemail_portal.utc_now_iso()
            with sqlite3.connect(state_db) as conn:
                conn.execute(
                    """
                    INSERT INTO voicemail_transcripts (
                        file_key, extension, mailbox, folder, msg_name, txt_path, wav_path,
                        callerid, origtime, origdate, duration, transcript, entities_json,
                        created_utc, updated_utc, deleted_utc, deleted_by
                    )
                    VALUES (?, '154', '154', 'INBOX', 'msg0005', ?, ?, ?, ?, ?, 20,
                            'already transcribed', '{"callback_number":"217-555-0100"}',
                            ?, ?, NULL, NULL)
                    """,
                    (
                        old_key,
                        txt_path,
                        wav_path,
                        info["callerid"],
                        int(info["origtime"]),
                        info["origdate"],
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO voicemails (
                        file_key, status, extension, txt_path, wav_path,
                        attempts, first_seen_utc, updated_utc, emailed_utc, transcript_chars
                    )
                    VALUES (?, 'completed', '154', ?, ?, 1, ?, ?, ?, 19)
                    """,
                    (old_key, txt_path, wav_path, now, now, now),
                )

            store.sync_filesystem()

            with sqlite3.connect(state_db) as conn:
                conn.row_factory = sqlite3.Row
                active_rows = conn.execute(
                    """
                    SELECT file_key, transcript, entities_json
                    FROM voicemail_transcripts
                    WHERE txt_path = ?
                      AND deleted_utc IS NULL
                      AND folder = 'INBOX'
                    """,
                    (txt_path,),
                ).fetchall()
                old_row = conn.execute(
                    "SELECT deleted_utc, deleted_by FROM voicemail_transcripts WHERE file_key = ?",
                    (old_key,),
                ).fetchone()
                queue_row = conn.execute(
                    "SELECT status, transcript_chars FROM voicemails WHERE file_key = ?",
                    (current_key,),
                ).fetchone()

            self.assertEqual(len(active_rows), 1)
            self.assertEqual(active_rows[0]["file_key"], current_key)
            self.assertEqual(active_rows[0]["transcript"], "already transcribed")
            self.assertIn("217-555-0100", active_rows[0]["entities_json"])
            self.assertIsNotNone(old_row["deleted_utc"])
            self.assertEqual(old_row["deleted_by"], "deduped_by_current_inbox_path")
            self.assertEqual(queue_row["status"], "completed")
            self.assertEqual(queue_row["transcript_chars"], 19)

    def test_portal_sync_queues_current_untranscribed_rows_missing_watcher_row(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            txt_path, _ = self.write_portal_message(inbox, "msg0006", duration=20, origtime="1770000006")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            info = voicemail_portal.parse_txt(txt_path)
            current_key = voicemail_portal.build_file_key("154", info, txt_path)
            with sqlite3.connect(state_db) as conn:
                conn.row_factory = sqlite3.Row
                queue_row = conn.execute(
                    "SELECT status, attempts FROM voicemails WHERE file_key = ?",
                    (current_key,),
                ).fetchone()

            self.assertIsNotNone(queue_row)
            self.assertEqual(queue_row["status"], "discovered")
            self.assertEqual(queue_row["attempts"], 0)

    def test_portal_auto_delete_move_failure_leaves_voicemail_active(self):
        import voicemail_portal

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            watch_dir = os.path.join(tmp, "spool")
            trash_dir = os.path.join(tmp, "trash")
            inbox = os.path.join(watch_dir, "vitalpbx-voicemail", "154", "INBOX")
            state_db = os.path.join(tmp, "state.sqlite3")
            short_txt, _ = self.write_portal_message(inbox, "msg0004", duration=5, origtime="1770000004")

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_AUTO_DELETE_SHORT_SECONDS": "0",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            store.sync_filesystem()

            with self.patched_env(
                {
                    "VOICEMAIL_STATE_DB": state_db,
                    "VOICEMAIL_WATCH_DIR": watch_dir,
                    "VOICEMAIL_PORTAL_TRASH_DIR": trash_dir,
                    "VOICEMAIL_PORTAL_AUTO_DELETE_SHORT_SECONDS": "6",
                    "VOICEMAIL_PORTAL_AMI_USERNAME": "portal_mwi",
                    "VOICEMAIL_PORTAL_AMI_SECRET": "secret",
                    "VOICEMAIL_PORTAL_SYNC_INTERVAL": "60",
                }
            ):
                settings = voicemail_portal.Settings.from_env()
            store = voicemail_portal.PortalStore(settings)
            calls = []
            original_move = voicemail_portal.move_message_to_trash
            old_send = voicemail_portal.send_ami_voicemail_refresh
            voicemail_portal.move_message_to_trash = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("move blocked")
            )
            voicemail_portal.send_ami_voicemail_refresh = lambda context, mailbox, _settings: calls.append(
                (context, mailbox)
            ) or True
            try:
                self.assertEqual(store.auto_delete_short_voicemails(), 0)
            finally:
                voicemail_portal.move_message_to_trash = original_move
                voicemail_portal.send_ami_voicemail_refresh = old_send

            with sqlite3.connect(state_db) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT folder, deleted_utc FROM voicemail_transcripts WHERE msg_name = 'msg0004'"
                ).fetchone()

            self.assertTrue(os.path.exists(short_txt))
            self.assertEqual(row["folder"], "INBOX")
            self.assertIsNone(row["deleted_utc"])
            self.assertEqual(calls, [])

    def test_optional_service_api_auth_helpers(self):
        import litert_chat_web
        import parakeet_server

        old_litert_required = litert_chat_web.LITERT_REQUIRE_AUTH
        old_litert_key = litert_chat_web.LITERT_API_KEY
        old_parakeet_required = parakeet_server.PARAKEET_REQUIRE_AUTH
        old_parakeet_key = parakeet_server.PARAKEET_API_KEY
        try:
            litert_chat_web.LITERT_REQUIRE_AUTH = True
            litert_chat_web.LITERT_API_KEY = "secret"
            with self.assertRaises(Exception):
                litert_chat_web.authorize_request("Bearer wrong", None)
            litert_chat_web.authorize_request("Bearer secret", None)

            parakeet_server.PARAKEET_REQUIRE_AUTH = True
            parakeet_server.PARAKEET_API_KEY = "secret"
            with self.assertRaises(Exception):
                parakeet_server.authorize_request(None, "wrong")
            parakeet_server.authorize_request(None, "secret")
        finally:
            litert_chat_web.LITERT_REQUIRE_AUTH = old_litert_required
            litert_chat_web.LITERT_API_KEY = old_litert_key
            parakeet_server.PARAKEET_REQUIRE_AUTH = old_parakeet_required
            parakeet_server.PARAKEET_API_KEY = old_parakeet_key

    def test_parakeet_endpoint_uses_threaded_locked_inference(self):
        import parakeet_server

        self.assertTrue(hasattr(parakeet_server, "MODEL_LOCK"))
        self.assertTrue(hasattr(parakeet_server, "transcribe_path_locked"))
        endpoint_source = inspect.getsource(parakeet_server.transcribe)
        self.assertIn("asyncio.to_thread", endpoint_source)
        self.assertIn("transcribe_path_locked", endpoint_source)


class MailboxSpellingRulesTests(unittest.TestCase):
    def test_applies_whole_word_transcript_rule_for_matching_mailbox(self):
        from voicemail_watcher.mailbox_spelling_rules import apply_mailbox_spelling_rules

        transcript, entities, count = apply_mailbox_spelling_rules(
            "103",
            "Kev called today. Kevin called yesterday.",
            {"name": "Kev Example"},
            {"103": [{"from": "Kev", "to": "Kiev"}]},
        )

        self.assertEqual(transcript, "Kiev called today. Kevin called yesterday.")
        self.assertEqual(entities["name"], "Kiev Example")
        self.assertEqual(count, 2)

    def test_wrong_mailbox_does_not_apply_rules(self):
        from voicemail_watcher.mailbox_spelling_rules import apply_mailbox_spelling_rules

        transcript, entities, count = apply_mailbox_spelling_rules(
            "581",
            "Kev called today.",
            {"name": "Kev Example"},
            {"103": [{"from": "Kev", "to": "Kiev"}]},
        )

        self.assertEqual(transcript, "Kev called today.")
        self.assertEqual(entities["name"], "Kev Example")
        self.assertEqual(count, 0)

    def test_disabled_rules_leave_values_unchanged(self):
        from voicemail_watcher.mailbox_spelling_rules import apply_mailbox_spelling_rules

        transcript, entities, count = apply_mailbox_spelling_rules(
            "581",
            "Casey called today.",
            {"name": "Casey Example"},
            {"581": [{"from": "Casey", "to": "Amie"}]},
            enabled=False,
        )

        self.assertEqual(transcript, "Casey called today.")
        self.assertEqual(entities["name"], "Casey Example")
        self.assertEqual(count, 0)

    def test_watcher_final_output_hook_uses_loaded_mailbox_rules(self):
        settings = SimpleNamespace(
            mailbox_spelling_rules_enabled=True,
            mailbox_spelling_rules_path="/unused/mailbox_spelling_rules.json",
        )
        old_loader = watcher.load_mailbox_spelling_rules
        old_disabled = watcher.logger.disabled
        watcher.load_mailbox_spelling_rules = lambda _path: {"103": [{"from": "Kev", "to": "Kiev"}]}
        watcher.logger.disabled = True
        try:
            transcript, entities = watcher.apply_final_mailbox_spelling_rules(
                "103",
                "Kev called today.",
                {"name": "Kev Example", "_word_timestamps": [{"word": "Kev"}]},
                settings,
                file_key="synthetic-key",
            )
        finally:
            watcher.load_mailbox_spelling_rules = old_loader
            watcher.logger.disabled = old_disabled

        self.assertEqual(transcript, "Kiev called today.")
        self.assertEqual(entities["name"], "Kiev Example")
        self.assertEqual(entities["_word_timestamps"], [{"word": "Kev"}])

    def test_load_rules_fails_safe_for_missing_or_invalid_file(self):
        from voicemail_watcher.mailbox_spelling_rules import load_mailbox_spelling_rules

        missing = os.path.join(os.getcwd(), "__missing_mailbox_rules__.json")
        invalid = os.getcwd()

        old_disabled = watcher.logger.disabled
        watcher.logger.disabled = True
        try:
            self.assertEqual(load_mailbox_spelling_rules(missing), {})
            self.assertEqual(load_mailbox_spelling_rules(invalid), {})
        finally:
            watcher.logger.disabled = old_disabled

    def test_load_rules_fails_safe_when_optional_path_is_inaccessible(self):
        from voicemail_watcher.mailbox_spelling_rules import load_mailbox_spelling_rules

        load_mailbox_spelling_rules.cache_clear()
        with patch(
            "voicemail_watcher.mailbox_spelling_rules.Path.exists",
            side_effect=PermissionError(13, "Permission denied"),
        ):
            self.assertEqual(load_mailbox_spelling_rules("/etc/lvt/rules.json"), {})


if __name__ == "__main__":
    unittest.main()

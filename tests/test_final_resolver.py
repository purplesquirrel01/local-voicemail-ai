import unittest
from pathlib import Path

from final_resolver import FINAL_SCHEMA_KEYS, merge_agent_outputs, merge_split_agent_outputs, validate_final_json


def base_candidates():
    return {
        "number_candidates": [
            {
                "id": "number:0",
                "raw": "202-555-0108",
                "normalized": "2025550108",
                "formatted": "(202) 555-0108",
                "evidence_text": "call me back at 202-555-0108",
                "nearby_cues": ["callback"],
                "cue_phrases": ["call me back"],
                "source": "numeric_phone",
            },
            {
                "id": "number:1",
                "raw": "202-555-0114",
                "normalized": "2025550114",
                "formatted": "(202) 555-0114",
                "evidence_text": "fax this to 202-555-0114",
                "nearby_cues": ["fax"],
                "cue_phrases": ["fax"],
                "source": "numeric_phone",
            },
        ],
        "dob_candidates": [
            {
                "id": "dob:0",
                "raw": "1/4/72",
                "normalized": "01/04/1972",
                "evidence_text": "date of birth is 1/4/72",
                "nearby_cues": ["dob"],
                "source": "date_numeric",
            }
        ],
        "name_candidates": [
            {
                "id": "name:0",
                "raw": "Jordan Example",
                "value": "Jordan Example",
                "evidence_text": "about my father Jordan Example",
                "source": "relationship_subject",
                "caller_id_used": "",
                "nearby_cues": ["relationship_subject"],
            }
        ],
        "name_correction_candidates": [
            {
                "id": "name_correction:0",
                "raw": "Bailey Sampel",
                "suggested_value": "Bailey Example",
                "evidence_text": "this is Bailey Sampel",
                "caller_id_used": "EXAMPLE,BAILEY",
                "reason": "last_name_phonetic_match",
            }
        ],
        "spelled_sequences": [],
    }


class FinalResolverTests(unittest.TestCase):
    def test_final_resolver_has_no_watcher_verification_dependency(self):
        source = Path("final_resolver.py").read_text(encoding="utf-8")

        self.assertNotIn("from verification import", source)

    def test_compact_agent_ids_resolve_to_full_final_objects(self):
        merged = merge_agent_outputs(
            {
                "callback_ids": ["number:0"],
                "fax_ids": ["number:1"],
                "uncertain_ids": [],
                "errors": [],
            },
            {
                "name_ids": ["name:0"],
                "dob_ids": ["dob:0"],
                "errors": [],
            },
        )

        final = validate_final_json(
            merged,
            base_candidates(),
            "This is Avery calling about my father Jordan Example. His date of birth is 1/4/72. "
            "Please call me back at 202-555-0108 and fax this to 202-555-0114.",
        )

        self.assertEqual(final["callback_numbers"][0]["normalized"], "2025550108")
        self.assertEqual(final["callback_numbers"][0]["formatted"], "(202) 555-0108")
        self.assertEqual(final["callback_numbers"][0]["evidence_text"], "call me back at 202-555-0108")
        self.assertEqual(final["fax_numbers"][0]["normalized"], "2025550114")
        self.assertEqual(final["patient_names"][0]["value"], "Jordan Example")
        self.assertEqual(final["patient_names"][0]["source"], "relationship_subject")
        self.assertEqual(final["dob_candidates"][0]["normalized"], "01/04/1972")
        self.assertEqual(final["possible_errors"], [])
        for field in ("callback_numbers", "fax_numbers", "patient_names", "dob_candidates"):
            for item in final[field]:
                self.assertNotIn("confidence", item)

    def test_split_name_and_dob_agent_ids_resolve_to_full_final_objects(self):
        merged = merge_split_agent_outputs(
            {
                "callback_ids": ["number:0"],
                "fax_ids": ["number:1"],
                "uncertain_ids": [],
                "errors": [],
            },
            {
                "name_ids": ["name:0"],
                "errors": [],
            },
            {
                "dob_ids": ["dob:0"],
                "errors": [],
            },
        )

        final = validate_final_json(
            merged,
            base_candidates(),
            "This is Avery calling about my father Jordan Example. His date of birth is 1/4/72. "
            "Please call me back at 202-555-0108 and fax this to 202-555-0114.",
        )

        self.assertEqual(list(final.keys()), FINAL_SCHEMA_KEYS)
        self.assertEqual(final["patient_names"][0]["value"], "Jordan Example")
        self.assertEqual(final["dob_candidates"][0]["normalized"], "01/04/1972")
        self.assertEqual(final["callback_numbers"][0]["formatted"], "(202) 555-0108")
        self.assertEqual(final["fax_numbers"][0]["formatted"], "(202) 555-0114")
        self.assertEqual(final["possible_errors"], [])

    def test_malformed_fax_candidate_is_rejected_and_schema_stays_compact(self):
        candidates = base_candidates()
        candidates["number_candidates"] = [
            {
                "id": "number:0",
                "raw": "800-1230-4567",
                "normalized": "80012304567",
                "formatted": "",
                "evidence_text": "fax records to 800-1230-4567",
                "nearby_cues": ["fax"],
                "source": "numeric_phone",
            },
            {
                "id": "number:1",
                "raw": "202-555-0103",
                "normalized": "2025550103",
                "formatted": "(202) 555-0103",
                "evidence_text": "fax number is 202-555-0103",
                "nearby_cues": ["fax"],
                "source": "numeric_phone",
            },
        ]

        final = validate_final_json(
            {
                "patient_names": [],
                "name_correction_candidates": [],
                "dob_candidates": [],
                "callback_numbers": [],
                "fax_numbers": [{"candidate_id": "number:0"}, {"candidate_id": "number:1"}],
                "uncertain_numbers": [],
                "possible_errors": [],
            },
            candidates,
            "Please fax records to 800-1230-4567. The fax number is 202-555-0100.",
        )

        self.assertEqual(final["fax_numbers"], [
            {
                "raw": "202-555-0103",
                "normalized": "2025550103",
                "formatted": "(202) 555-0103",
                "label_cue": "fax",
                "evidence_text": "fax number is 202-555-0103",
            }
        ])
        self.assertTrue(any("invalid_phone" in str(item) for item in final["possible_errors"]))
        self.assertEqual(list(final.keys()), FINAL_SCHEMA_KEYS)

    def test_compact_name_correction_ids_resolve_to_audit_objects(self):
        merged = merge_split_agent_outputs(
            {"callback_ids": [], "fax_ids": [], "uncertain_ids": [], "errors": []},
            {"name_ids": ["name:0"], "name_correction_ids": ["name_correction:0"], "errors": []},
            {"dob_ids": [], "errors": []},
        )

        final = validate_final_json(
            merged,
            base_candidates(),
            "This is Bailey Sampel. Please call me back at 202-555-0108.",
        )

        self.assertEqual(final["patient_names"][0]["value"], "Jordan Example")
        self.assertEqual(
            final["name_correction_candidates"],
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

    def test_unknown_name_correction_candidate_id_is_rejected(self):
        final = validate_final_json(
            {
                "patient_names": [],
                "name_correction_candidates": [{"candidate_id": "name_correction:missing"}],
                "dob_candidates": [],
                "callback_numbers": [],
                "fax_numbers": [],
                "uncertain_numbers": [],
                "possible_errors": [],
            },
            base_candidates(),
            "This is Bailey Sampel.",
        )

        self.assertEqual(final["name_correction_candidates"], [])
        self.assertTrue(any("unknown_candidate_id" in str(item) for item in final["possible_errors"]))

    def test_split_name_correction_output_precedes_common_name_output(self):
        candidates = base_candidates()
        candidates["name_candidates"] = [
            {
                "id": "name:0",
                "raw": "Bailey Sampel",
                "value": "Bailey Sampel",
                "evidence_text": "this is Bailey Sampel",
                "source": "self_identification",
                "caller_id_used": "",
            },
            {
                "id": "name:1",
                "raw": "Bailey Sampel",
                "value": "Bailey Sample",
                "evidence_text": "Bailey Sampel from Example Clinic. S-A-M-P-L-E",
                "source": "transcript_spelling_corrected",
                "caller_id_used": "",
            },
        ]

        merged = merge_split_agent_outputs(
            {"callback_ids": [], "fax_ids": [], "uncertain_ids": [], "errors": []},
            {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
            {"dob_ids": [], "errors": []},
            {"name_ids": ["name:1"], "name_correction_ids": [], "errors": []},
        )
        final = validate_final_json(
            merged,
            candidates,
            "Yeah, this is Bailey Sampel from Example Clinic. S-A-M-P-L-E.",
        )

        self.assertEqual(final["patient_names"], [
            {
                "raw": "Bailey Sampel",
                "value": "Bailey Sample",
                "evidence_text": "Bailey Sampel from Example Clinic. S-A-M-P-L-E",
                "source": "transcript_spelling_corrected",
                "caller_id_used": "",
            }
        ])

    def test_split_dual_correction_outputs_precede_common_name_output(self):
        candidates = base_candidates()
        candidates["name_candidates"] = [
            {
                "id": "name:0",
                "raw": "Bailey Sampel",
                "value": "Bailey Sampel",
                "evidence_text": "this is Bailey Sampel",
                "source": "self_identification",
                "caller_id_used": "",
            },
            {
                "id": "name:1",
                "raw": "Bailey Sampel",
                "value": "Bailey Sample",
                "evidence_text": "Bailey Sampel from Example Clinic. S-A-M-P-L-E",
                "source": "transcript_spelling_corrected",
                "caller_id_used": "",
            },
            {
                "id": "name:2",
                "raw": "Bailey Sampel",
                "value": "Bailey Exampel",
                "evidence_text": "this is Bailey Sampel",
                "source": "caller_id_corrected",
                "caller_id_used": "EXAMPEL BAILEY",
            },
        ]

        merged = merge_split_agent_outputs(
            {"callback_ids": [], "fax_ids": [], "uncertain_ids": [], "errors": []},
            {"name_ids": ["name:0"], "name_correction_ids": [], "errors": []},
            {"dob_ids": [], "errors": []},
            {"name_ids": ["name:1"], "name_correction_ids": [], "errors": []},
            {"name_ids": ["name:2"], "name_correction_ids": [], "errors": []},
        )
        final = validate_final_json(
            merged,
            candidates,
            "Yeah, this is Bailey Sampel from Example Clinic. S-A-M-P-L-E.",
        )

        self.assertEqual(len(final["patient_names"]), 1)
        self.assertEqual(final["patient_names"][0]["value"], "Bailey Sample")
        self.assertEqual(final["patient_names"][0]["source"], "transcript_spelling_corrected")

    def test_name_evidence_must_support_raw_name(self):
        candidates = base_candidates()
        candidates["name_candidates"][0]["evidence_text"] = "this is"

        final = validate_final_json(
            {
                "patient_names": [{"candidate_id": "name:0"}],
                "name_correction_candidates": [],
                "dob_candidates": [],
                "callback_numbers": [],
                "fax_numbers": [],
                "uncertain_numbers": [],
                "possible_errors": [],
            },
            candidates,
            "This is Jordan Example.",
        )

        self.assertEqual(final["patient_names"], [])
        self.assertTrue(any("unsupported_name_evidence" in str(item) for item in final["possible_errors"]))

    def test_unknown_compact_candidate_id_is_rejected(self):
        merged = merge_agent_outputs(
            {"callback_ids": ["number:missing"], "fax_ids": [], "uncertain_ids": [], "errors": []},
            {"name_ids": [], "dob_ids": [], "errors": []},
        )

        final = validate_final_json(merged, base_candidates(), "Please call me back at 202-555-0108.")

        self.assertEqual(final["callback_numbers"], [])
        self.assertTrue(any("unknown_candidate_id" in str(item) for item in final["possible_errors"]))

    def test_agents_cannot_populate_fields_they_do_not_own(self):
        merged = merge_agent_outputs(
            {
                "callback_numbers": [{"candidate_id": "number:0", "evidence_text": "call me back at 202-555-0108"}],
                "patient_names": [{"candidate_id": "name:0", "evidence_text": "about my father Jordan Example"}],
                "possible_errors": [],
            },
            {
                "patient_names": [{"candidate_id": "name:0", "evidence_text": "about my father Jordan Example"}],
                "fax_numbers": [{"candidate_id": "number:1", "evidence_text": "fax this to 202-555-0114"}],
                "possible_errors": [],
            },
        )

        self.assertEqual(merged["fax_numbers"], [])
        self.assertEqual(len(merged["callback_numbers"]), 1)
        self.assertEqual(len(merged["patient_names"]), 1)

    def test_unknown_number_candidate_id_is_rejected(self):
        final = validate_final_json(
            {
                "callback_numbers": [{"candidate_id": "number:missing", "evidence_text": "call me back at 202-555-0108"}],
                "fax_numbers": [],
                "uncertain_numbers": [],
                "patient_names": [],
                "dob_candidates": [],
                "possible_errors": [],
            },
            base_candidates(),
            "Please call me back at 202-555-0108.",
        )

        self.assertEqual(final["callback_numbers"], [])
        self.assertTrue(any("unknown_candidate_id" in str(item) for item in final["possible_errors"]))

    def test_invalid_dob_is_rejected(self):
        candidates = base_candidates()
        candidates["dob_candidates"][0]["normalized"] = "02/30/1972"

        final = validate_final_json(
            {
                "patient_names": [],
                "dob_candidates": [{"candidate_id": "dob:0", "evidence_text": "date of birth is 2/30/72"}],
                "callback_numbers": [],
                "fax_numbers": [],
                "uncertain_numbers": [],
                "possible_errors": [],
            },
            candidates,
            "His date of birth is 2/30/72.",
        )

        self.assertEqual(final["dob_candidates"], [])
        self.assertTrue(any("invalid_dob" in str(item) for item in final["possible_errors"]))

    def test_missing_evidence_text_is_rejected(self):
        candidates = base_candidates()
        candidates["number_candidates"][0]["evidence_text"] = ""

        final = validate_final_json(
            {
                "callback_numbers": [{"candidate_id": "number:0"}],
                "fax_numbers": [],
                "uncertain_numbers": [],
                "patient_names": [],
                "dob_candidates": [],
                "possible_errors": [],
            },
            candidates,
            "Please call me back at 202-555-0108.",
        )

        self.assertEqual(final["callback_numbers"], [])
        self.assertTrue(any("missing_evidence_text" in str(item) for item in final["possible_errors"]))

    def test_final_schema_always_contains_required_top_level_fields(self):
        final = validate_final_json({}, base_candidates(), "")

        self.assertEqual(list(final.keys()), FINAL_SCHEMA_KEYS)
        self.assertTrue(all(isinstance(final[key], list) for key in FINAL_SCHEMA_KEYS))


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from agent_constraints import build_agent_constraint_schema


class AgentConstraintSchemaTests(unittest.TestCase):
    def test_numbers_schema_restricts_all_number_id_fields_to_candidate_ids(self):
        schema = build_agent_constraint_schema(
            "numbers",
            {
                "number_candidates": [
                    {"id": "number:0", "raw": "217-555-0100", "evidence_text": "call back"},
                    {"id": "number:1", "raw": "217-555-0101", "evidence_text": "fax"},
                ]
            },
        )

        self.assertEqual(schema["required"], ["callback_ids", "fax_ids", "uncertain_ids", "errors"])
        self.assertFalse(schema["additionalProperties"])
        for field in ("callback_ids", "fax_ids", "uncertain_ids"):
            self.assertEqual(schema["properties"][field]["items"]["enum"], ["number:0", "number:1"])
            self.assertTrue(schema["properties"][field]["uniqueItems"])

    def test_name_schema_uses_name_and_name_correction_id_buckets(self):
        schema = build_agent_constraint_schema(
            "spelling_correction",
            {
                "name_candidates": [{"id": "name:0"}, {"id": "name:1"}],
                "name_correction_candidates": [{"id": "name_correction:0"}],
            },
        )

        self.assertEqual(schema["required"], ["name_ids", "name_correction_ids", "errors"])
        self.assertEqual(schema["properties"]["name_ids"]["items"]["enum"], ["name:0", "name:1"])
        self.assertEqual(
            schema["properties"]["name_correction_ids"]["items"]["enum"],
            ["name_correction:0"],
        )

    def test_identity_schema_includes_name_correction_and_dob_ids(self):
        schema = build_agent_constraint_schema(
            "identity",
            {
                "name_candidates": [{"id": "name:0"}],
                "name_correction_candidates": [{"id": "name_correction:0"}],
                "dob_candidates": [{"id": "dob:0"}],
            },
        )

        self.assertEqual(schema["required"], ["name_ids", "name_correction_ids", "dob_ids", "errors"])
        self.assertEqual(schema["properties"]["name_ids"]["items"]["enum"], ["name:0"])
        self.assertEqual(schema["properties"]["name_correction_ids"]["items"]["enum"], ["name_correction:0"])
        self.assertEqual(schema["properties"]["dob_ids"]["items"]["enum"], ["dob:0"])

    def test_empty_candidate_bucket_allows_only_empty_id_array(self):
        schema = build_agent_constraint_schema("dob", {"dob_candidates": []})

        self.assertEqual(schema["required"], ["dob_ids", "errors"])
        self.assertEqual(schema["properties"]["dob_ids"]["maxItems"], 0)
        self.assertNotIn("enum", schema["properties"]["dob_ids"]["items"])

    def test_schema_does_not_embed_phi_candidate_values_or_evidence(self):
        schema = build_agent_constraint_schema(
            "subject_name",
            {
                "name_candidates": [
                    {
                        "id": "name:0",
                        "raw": "Alice Sample",
                        "value": "Alice Sample",
                        "evidence_text": "Hello this is Alice Sample",
                    }
                ],
                "transcript": "Hello this is Alice Sample. Please call 217-555-0100.",
            },
        )

        serialized = json.dumps(schema, sort_keys=True)
        self.assertIn("name:0", serialized)
        self.assertNotIn("Alice", serialized)
        self.assertNotIn("Sample", serialized)
        self.assertNotIn("217-555-0100", serialized)
        self.assertNotIn("Hello this is", serialized)


if __name__ == "__main__":
    unittest.main()

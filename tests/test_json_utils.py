import unittest

from json_utils import AgentOutputError, parse_json_strict_or_repair


class JsonUtilsTests(unittest.TestCase):
    def test_strips_markdown_and_coerces_missing_arrays(self):
        parsed = parse_json_strict_or_repair(
            '```json\n{"callback_numbers": [{"candidate_id": "number:0"}], "extra": true}\n```',
            ["callback_numbers", "fax_numbers"],
        )

        self.assertEqual(parsed, {"callback_numbers": [{"candidate_id": "number:0"}], "fax_numbers": []})

    def test_extracts_first_json_object_from_extra_text(self):
        parsed = parse_json_strict_or_repair(
            'Here is JSON: {"patient_names": [], "dob_candidates": []} trailing',
            ["patient_names", "dob_candidates"],
        )

        self.assertEqual(parsed, {"patient_names": [], "dob_candidates": []})

    def test_raises_clear_error_for_malformed_output(self):
        with self.assertRaises(AgentOutputError):
            parse_json_strict_or_repair("{not valid", ["callback_numbers"])


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path


PROMPT_DIR = Path("prompts")


def read_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


class CandidateAgentPromptTests(unittest.TestCase):

    def test_number_rules_are_only_in_numbers_prompt(self):
        numbers = read_prompt("numbers_agent.md")
        name = read_prompt("name_agent.md")
        dob = read_prompt("dob_agent.md")

        for phrase in (
            "Do not use caller ID phone numbers as callback or fax candidates.",
            "Put uncertain or incomplete numbers in uncertain_numbers, not callback_numbers or fax_numbers.",
            "Select only valid 10-digit US callback/fax candidates.",
            "Ignore malformed 11-digit lookalikes",
            'explicit "fax number"',
            "Deduplicate by normalized 10-digit number.",
            "Return each callback, fax, or uncertain number at most once.",
            "Use the strongest evidence_text only.",
            "Begin output with { and end after the final }.",
            '"callback_ids":[]',
            '"fax_ids":[]',
            '"uncertain_ids":[]',
        ):
            self.assertIn(phrase, numbers)
            self.assertNotIn(phrase, name)
            self.assertNotIn(phrase, dob)

    def test_name_rules_are_only_in_name_prompt(self):
        numbers = read_prompt("numbers_agent.md")
        name = read_prompt("name_agent.md")
        correction = read_prompt("name_correction_agent.md")
        spelling_correction = read_prompt("spelling_correction_agent.md")
        caller_id_correction = read_prompt("caller_id_correction_agent.md")
        spelling = read_prompt("name_agent_spelling.md")
        caller_id = read_prompt("name_agent_caller_id.md")
        dob = read_prompt("dob_agent.md")

        for phrase in (
            "Name priority: first find the patient/subject of the voicemail.",
            '"this is for"',
            "Facility or clinical staff caller names do not beat a separate patient/subject candidate.",
            '"from Facility for Name, date of birth ..."',
            '"message about him, Name, being..."',
            '"patient of Dr. X, Name"',
            '"This is Bailey Example calling for my sibling Jordan Sample."',
            '"name_ids":[]',
            '"name_correction_ids":[]',
        ):
            self.assertIn(phrase, name)
            self.assertNotIn(phrase, numbers)
            self.assertNotIn(phrase, dob)
        for phrase in ('"name_ids":[]', '"name_correction_ids":[]'):
            self.assertIn(phrase, correction)
        self.assertIn("name_correction_ids must always be [] in this common name worker.", name)
        self.assertIn("Select name_ids only from provided name_candidates whose source is", correction)
        self.assertIn("name_correction_candidates are audit/review hints only.", correction)
        self.assertIn("Select name_ids only from transcript_spelling_corrected candidates.", spelling_correction)
        self.assertIn("Select name_ids only from caller_id_corrected candidates.", caller_id_correction)
        self.assertIn("Select name_correction_ids only from provided name_correction_candidates.", caller_id_correction)
        for phrase in (
            "Explicit spelling immediately after a spoken patient/caller name may correct the name.",
            "Bailey Sampel, spelled S-A-M-P-L-E",
            "Avery Exampel, spelled E-X-A-M-P-L-E",
        ):
            self.assertIn(phrase, spelling)
            self.assertNotIn(phrase, name)
            self.assertNotIn(phrase, correction)
            self.assertNotIn(phrase, spelling_correction)
            self.assertNotIn(phrase, caller_id_correction)
            self.assertNotIn(phrase, numbers)
            self.assertNotIn(phrase, dob)
        for phrase in (
            "Caller ID may correct a clearly self-identified spoken name only when it appears to be the same person",
            "name_correction_candidates must not replace patient_names.",
            "phonetic_last_first_match",
            "last_name_phonetic_match",
            "Example Caller ID corrections:",
            "Example close LAST FIRST Caller ID correction:",
            "Example truncated Caller ID not used:",
        ):
            self.assertIn(phrase, caller_id)
            self.assertNotIn(phrase, name)
            self.assertNotIn(phrase, correction)
            self.assertNotIn(phrase, spelling_correction)
            self.assertNotIn(phrase, caller_id_correction)
            self.assertNotIn(phrase, numbers)
            self.assertNotIn(phrase, dob)
        for phrase in ("same first name", "middle initial", "FIRST LAST or LAST FIRST", "last-name-only", "CANDIDATE_AGENT_CALLER_ID_LAST_NAME_ONLY_CORRECTION"):
            self.assertIn(phrase, caller_id)
            self.assertIn(phrase, caller_id_correction)

    def test_subject_and_fallback_name_prompts_are_separate(self):
        subject = read_prompt("subject_name_agent.md")
        fallback = read_prompt("caller_name_fallback_agent.md")
        numbers = read_prompt("numbers_agent.md")
        spelling_correction = read_prompt("spelling_correction_agent.md")
        caller_id_correction = read_prompt("caller_id_correction_agent.md")

        for phrase in (
            "You own only patient/subject name selection.",
            "Select only patient/subject candidates.",
            '"calling for"',
            '"from Facility for Name, date of birth ..."',
            '"message about him, Name, being..."',
            '"for patient Name"',
            '"calling about patient Name"',
            '"mutual client Ms. Name"',
            '"patient here ... it is Name"',
            '"office of Caller regarding Name"',
            '"request ... for Name"',
            '"request ... received on Name"',
            '"patient of Dr. X" means the speaker is the patient',
        ):
            self.assertIn(phrase, subject)
            self.assertNotIn(phrase, fallback)
            self.assertNotIn(phrase, numbers)
            self.assertNotIn(phrase, spelling_correction)
            self.assertNotIn(phrase, caller_id_correction)
        self.assertIn("relationship_subject", subject)
        self.assertIn("explicit_patient", subject)
        self.assertIn("broad_name_recall", subject)

        for phrase in (
            "You own only caller/speaker fallback name selection.",
            "Use this worker only when no patient/subject name was selected.",
            "self_identification",
            "spelled_sequence_context",
            "broad_name_recall",
            "Do not select relationship_subject or explicit_patient candidates.",
        ):
            self.assertIn(phrase, fallback)
            if phrase != "broad_name_recall":
                self.assertNotIn(phrase, subject)
            self.assertNotIn(phrase, numbers)
            self.assertNotIn(phrase, spelling_correction)
            self.assertNotIn(phrase, caller_id_correction)

        for phrase in (
            "Explicit spelling immediately after a spoken patient/caller name may correct the name.",
            "Caller ID may correct a clearly self-identified spoken name",
        ):
            self.assertNotIn(phrase, subject)
            self.assertNotIn(phrase, fallback)

    def test_candidate_scout_prompt_is_recall_focused_not_final_selection(self):
        prompt = read_prompt("candidate_scout_agent.md")

        self.assertIn("semantic candidate scout", prompt)
        self.assertIn("Do not choose final fields", prompt)
        self.assertIn("exact evidence_text", prompt)
        self.assertIn("name_candidates", prompt)
        self.assertIn("Names only", prompt)
        self.assertIn("mutual client Savannah Example", prompt)
        self.assertIn("Most voicemail messages contain at least one person name.", prompt)
        self.assertIn("Return empty arrays only when there is no supported person name span.", prompt)
        self.assertIn("When a cue is immediately followed by a person name, return that name.", prompt)
        self.assertIn("It is better to return an extra supported name candidate", prompt)
        self.assertNotIn("dob_candidates", prompt)
        self.assertNotIn("number_candidates", prompt)
        self.assertNotIn("spelled_sequences", prompt)

    def test_dob_rules_are_only_in_dob_prompt(self):
        numbers = read_prompt("numbers_agent.md")
        name = read_prompt("name_agent.md")
        dob = read_prompt("dob_agent.md")

        for phrase in (
            "DOB must be clearly identified as date of birth, birth date, or DOB.",
            "Compact DOB fragments after DOB cues are complete DOB candidates",
            '"Jane Example, 625-54"',
            '"Jane Example 12-16-62"',
            "date_numeric_adjacent_patient",
            '"424 of 60"',
            '"dob_ids":[]',
        ):
            self.assertIn(phrase, dob)
            self.assertNotIn(phrase, name)
            self.assertNotIn(phrase, numbers)

    def test_shared_deterministic_rules_are_in_both_prompts(self):
        numbers = read_prompt("numbers_agent.md")
        name = read_prompt("name_agent.md")
        dob = read_prompt("dob_agent.md")
        subject = read_prompt("subject_name_agent.md")
        fallback = read_prompt("caller_name_fallback_agent.md")

        for phrase in (
            "Return exactly one minified JSON object. No markdown. No explanation.",
            "Extract only values directly supported by transcript evidence.",
            "possible_errors should usually be [].",
        ):
            self.assertIn(phrase, numbers)
            self.assertIn(phrase, name)
            self.assertIn(phrase, dob)
            self.assertIn(phrase, subject)
            self.assertIn(phrase, fallback)


if __name__ == "__main__":
    unittest.main()

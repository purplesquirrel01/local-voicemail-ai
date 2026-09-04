import os
import unittest

from candidate_extractor import extract_candidates


def normalized_numbers(result):
    return {item.get("normalized") for item in result["number_candidates"]}


def candidate_by_number(result, digits):
    for item in result["number_candidates"]:
        if item.get("normalized") == digits:
            return item
    raise AssertionError(f"number candidate {digits} not found in {result['number_candidates']!r}")


def name_values(result, source=None):
    return [
        item.get("value")
        for item in result["name_candidates"]
        if source is None or item.get("source") == source
    ]


def name_candidates(result, source=None):
    return [
        item
        for item in result["name_candidates"]
        if source is None or item.get("source") == source
    ]


FULL_SYNTHETIC_TRANSCRIPT = (
    "This is Avery Exampel. I have an appointment on June the 1st. "
    "I would like to ask about moving the appointment to July. "
    "I am going to be available then. Please check the available dates. "
    "For the return call this is Avery A-V-E-R-Y Exampel E-X-A-M-P-L-E "
    "and my number is 202-555-0123 thank you"
)


class CandidateExtractorTests(unittest.TestCase):
    def test_numeric_callback_has_evidence_and_callback_cue(self):
        result = extract_candidates("Please call me back at 202-555-0108.", transcript_id="msg-001")

        item = candidate_by_number(result, "2025550108")
        self.assertEqual(item["formatted"], "(202) 555-0108")
        self.assertEqual(item["span"], [23, 35])
        self.assertIn("call me back", item["window"].lower())
        self.assertIn("callback", item["nearby_cues"])
        self.assertTrue(item["evidence_text"])
        self.assertEqual(result["transcript_id"], "msg-001")

    def test_explicit_fax_has_fax_cue(self):
        result = extract_candidates("Please fax this to 202-555-0114.")

        item = candidate_by_number(result, "2025550114")
        self.assertIn("fax", item["nearby_cues"])
        self.assertIn("fax this to", " ".join(item["cue_phrases"]).lower())

    def test_implied_medical_fax_has_referral_send_cues(self):
        result = extract_candidates("Please send the referral to 202-555-0114.")

        item = candidate_by_number(result, "2025550114")
        self.assertIn("medical_send", item["nearby_cues"])
        self.assertIn("referral", item["window"].lower())

    def test_two_numbers_are_deduplicated_and_cued_separately(self):
        result = extract_candidates("My number is 202-555-0101 and the fax is 202-555-0105.")

        self.assertEqual(normalized_numbers(result), {"2025550101", "2025550105"})
        self.assertIn("callback", candidate_by_number(result, "2025550101")["nearby_cues"])
        self.assertIn("fax", candidate_by_number(result, "2025550105")["nearby_cues"])

    def test_dob_cue_extracts_valid_date_candidate(self):
        result = extract_candidates("His date of birth is 1/4/72.")

        self.assertEqual(len(result["dob_candidates"]), 1)
        item = result["dob_candidates"][0]
        self.assertEqual(item["raw"], "1/4/72")
        self.assertEqual(item["normalized"], "01/04/1972")
        self.assertIn("date of birth", item["window"].lower())
        self.assertIn("dob", item["nearby_cues"])

    def test_appointment_date_is_not_a_dob_candidate(self):
        result = extract_candidates("My appointment is 1/4/72.")

        self.assertEqual(result["dob_candidates"], [])
        self.assertTrue(
            any(event.get("event_type") == "appointment_date" for event in result["semantic_events"]),
            result["semantic_events"],
        )

    def test_leading_patient_name_and_numeric_date_with_surgery_context(self):
        result = extract_candidates(
            "Avery Example 12-16-62 wondering if the hospital sent the test "
            "results. Please call back because I would like to schedule surgery."
        )

        self.assertIn("Avery Example", name_values(result, "relationship_subject"))
        self.assertTrue(
            any(
                item.get("normalized") == "12/16/1962"
                and item.get("source") == "date_numeric_adjacent_patient"
                for item in result["dob_candidates"]
            ),
            result["dob_candidates"],
        )

    def test_spelled_sequence_is_extracted_with_name_context(self):
        result = extract_candidates("My name is Example, E-X-A-M-P-L-E.")

        self.assertEqual(len(result["spelled_sequences"]), 1)
        spelled = result["spelled_sequences"][0]
        self.assertEqual(spelled["letters"], "EXAMPLE")
        self.assertIn("Example", spelled["window"])
        self.assertTrue(any(item.get("raw") == "Example" for item in result["name_candidates"]))

    def test_test_voicemail_intro_does_not_hide_self_identified_name(self):
        result = extract_candidates(
            "Testing, testing, one, two, three. This is a test voicemail. "
            "This is Avery Sample. Thank you."
        )

        self.assertIn("Avery Sample", name_values(result, "self_identification"))
        self.assertNotIn("Test Voicemail", name_values(result))

    def test_test_voicemail_intro_preserves_name_dob_and_numbers(self):
        result = extract_candidates(
            "Testing, testing, one, two, three. This is a test voicemail. "
            "This is Avery Sample, date of birth, 05/21/2001. "
            "Good callback number is 217-555-0184. "
            "And good fax number is 202-555-0113."
        )

        self.assertIn("Avery Sample", name_values(result, "self_identification"))
        self.assertNotIn("Test Voicemail", name_values(result))
        self.assertTrue(any(item.get("normalized") == "05/21/2001" for item in result["dob_candidates"]))
        self.assertIn("2175550184", normalized_numbers(result))
        self.assertIn("2025550113", normalized_numbers(result))

    def test_adjacent_spelling_corrects_name_candidate_value(self):
        result = extract_candidates("Yeah, this is Bailey Sampel from Example Clinic. S-A-M-P-L-E.")

        raw = [
            item for item in result["name_candidates"]
            if item.get("source") == "self_identification"
        ]
        corrected = [
            item for item in result["name_candidates"]
            if item.get("source") == "transcript_spelling_corrected"
        ]
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0]["raw"], "Bailey Sampel")
        self.assertEqual(raw[0]["value"], "Bailey Sampel")
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["raw"], "Bailey Sampel")
        self.assertEqual(corrected[0]["value"], "Bailey Sample")
        self.assertEqual(corrected[0]["caller_id_used"], "")
        self.assertIn("S-A-M-P-L-E", corrected[0]["evidence_text"])

    def test_contiguous_full_name_spelling_preserves_name_spacing(self):
        result = extract_candidates(
            "Hello. My name is Bailey Example, B-A-I-L-E-Y-E-X-A-M-P-L-E. "
            "I was wanting to speak to Dr. Example."
        )

        corrected = [
            item for item in result["name_candidates"]
            if item.get("source") == "transcript_spelling_corrected"
        ]

        self.assertTrue(
            any(item.get("raw") == "Bailey Example" and item.get("value") == "Bailey Example" for item in corrected),
            corrected,
        )
        self.assertFalse(
            any(item.get("value") == "Bailey BaileyExample" for item in result["name_candidates"]),
            result["name_candidates"],
        )

    def test_interleaved_first_and_last_spelling_ignores_spelling_tokens_as_name_words(self):
        result = extract_candidates(
            "call me back this is Avery A-V-E-R-Y Exampel E-X-A-M-P-L-E and my number is 202-555-0123 thank you"
        )

        names = {item.get("raw") for item in result["name_candidates"]}
        self.assertIn("Avery Exampel", names)
        self.assertNotIn("Is Avery", names)
        self.assertNotIn("A-V-E-R-Y Exampel", names)
        corrected = [
            item for item in result["name_candidates"]
            if item.get("source") == "transcript_spelling_corrected"
            and item.get("value") == "Avery Example"
        ]
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["raw"], "Avery Exampel")
        self.assertIn("E-X-A-M-P-L-E", corrected[0]["evidence_text"])
        self_id = next(item for item in result["name_candidates"] if item.get("source") == "self_identification")
        self.assertIn("Avery A-V-E-R-Y Exampel", self_id["evidence_text"])

    def test_full_transcript_interleaved_spelling_corrects_repeated_name(self):
        result = extract_candidates(FULL_SYNTHETIC_TRANSCRIPT)

        names = {item.get("raw") for item in result["name_candidates"]}
        self.assertIn("Avery Exampel", names)
        self.assertNotIn("Going To Be", names)
        self.assertNotIn("Surgery", names)
        corrected = [
            item for item in result["name_candidates"]
            if item.get("source") == "transcript_spelling_corrected"
            and item.get("raw") == "Avery Exampel"
            and item.get("value") == "Avery Example"
        ]
        self.assertEqual(len(corrected), 1)
        self.assertIn("Avery A-V-E-R-Y Exampel E-X-A-M-P-L-E", corrected[0]["evidence_text"])
        self.assertNotIn("going to be", corrected[0]["evidence_text"].lower())
        item = candidate_by_number(result, "2025550123")
        self.assertEqual(item["formatted"], "(202) 555-0123")

    def test_patient_name_alternate_phrase_uses_spelled_first_and_last_name(self):
        result = extract_candidates(
            "Good morning. My name is Avery. I am calling from a payer unit. "
            "The patient's name, Casey, or Casey, Example, or Example, E-X-A-M-P-L-E. "
            "Again, Casey, or Casey, C-A-S-E-Y, Example, or Example, E-X-A-M-P-L-E, "
            "date of birth 07/30/1948."
        )

        subject_names = name_candidates(result, "relationship_subject")
        self.assertTrue(
            any(item.get("value") == "Casey Example" for item in subject_names),
            result["name_candidates"],
        )
        self.assertNotIn("Or Example", {item.get("raw") for item in result["name_candidates"]})
        self.assertNotIn("Or Casey", {item.get("raw") for item in result["name_candidates"]})

    def test_name_regression_examples_extract_clean_candidates(self):
        examples = [
            ("Hi, this is Avery Example.", "EXAMPLE,AVERY", "Avery Example", []),
            (
                "Hi, this is Bailey Sample. I am following up about an appointment. Please return the call.",
                "WIRELESS CALLER", "Bailey Sample", ["Following Up"],
            ),
            (
                "Hi, this is Casey Example. Could you check the appointment time? Call me at 202-555-0101.",
                "WIRELESS CALLER", "Casey Example", [],
            ),
            (
                "Hi, Avery. I am calling for Jordan Sample. Dr. Example suggested a routine appointment.",
                "EXAMPLE AVERY", "Jordan Sample", [],
            ),
            (
                "Hello, my name is Bailey. I am calling from Example Clinic about a client. "
                "His name is Taylor Example. Please return the call at 202-555-0102.",
                "EXAMPLE CLINIC", "Taylor Example", ["Bailey I Am"],
            ),
            (
                "Hi, this is Morgan Sample. That's S-A-M-P-L-E. I have an appointment question. "
                "My number is 202-555-0103. Thank you.",
                "SAMPLE MORGAN", "Morgan Sample", ["Sample That"],
            ),
        ]
        for transcript, caller_id, expected, rejected in examples:
            with self.subTest(expected=expected):
                result = extract_candidates(transcript, caller_id=caller_id)
                names = {item.get("raw") for item in result["name_candidates"]}
                self.assertIn(expected, names)
                for raw in rejected:
                    self.assertNotIn(raw, names)

    def test_for_patient_name_trims_patient_descriptor(self):
        result = extract_candidates("This is caller with insurer, and this is for patient Bailey Example.")

        self.assertIn("Bailey Example", name_values(result, "explicit_patient"))
        self.assertNotIn("Patient Bailey Example", name_values(result))

    def test_calling_about_patient_name_is_subject(self):
        result = extract_candidates("Hello, I am calling about patient Casey Example.")

        self.assertIn("Casey Example", name_values(result, "relationship_subject"))

    def test_request_for_subject_ignores_generic_request_descriptor(self):
        result = extract_candidates(
            "Hello, I am with Example Clinic calling regarding your request for Gelson 3 for Jordan Example."
        )

        self.assertIn("Jordan Example", name_values(result, "relationship_subject"))
        self.assertNotIn("Your Request", name_values(result))

    def test_availability_request_received_on_subject_ignores_request_descriptor(self):
        result = extract_candidates(
            "Hi, this is Bailey with Example Clinic prior authorizations. "
            "I'm talking about an availability request that we received on Taylor Example."
        )

        self.assertIn("Taylor Example", name_values(result, "relationship_subject"))
        self.assertNotIn("An Availability Request", name_values(result))
        self.assertNotIn("Talking", name_values(result))

    def test_one_of_provider_patients_name_is_subject(self):
        result = extract_candidates(
            "I am calling in regards to one of Dr. Sample's patients, Quinn Example."
        )

        self.assertIn("Quinn Example", name_values(result, "relationship_subject"))

    def test_regarding_subject_after_office_of_caller(self):
        result = extract_candidates(
            "This message is from the office of Robin Example regarding Morgan Example."
        )

        self.assertIn("Morgan Example", name_values(result, "relationship_subject"))
        self.assertNotIn("Robin Example", name_values(result, "relationship_subject"))

    def test_calling_on_behalf_of_subject(self):
        result = extract_candidates("I am calling on behalf of Avery Example.")

        self.assertIn("Avery Example", name_values(result, "relationship_subject"))

    def test_mutual_client_with_honorific_subject(self):
        result = extract_candidates("It is pertaining to our mutual client Ms. Bailey Example.")

        self.assertIn("Bailey Example", name_values(result, "relationship_subject"))

    def test_patient_here_followed_by_it_is_name(self):
        result = extract_candidates("I have a patient here of Dr. Sample. It is Casey Example.")

        self.assertIn("Casey Example", name_values(result, "relationship_subject"))

    def test_relationship_subject_single_name(self):
        result = extract_candidates("This is Taylor. I am calling for my husband, Quinn.")

        self.assertIn("Quinn", name_values(result, "relationship_subject"))

    def test_self_identification_stops_before_location_text(self):
        result = extract_candidates("Yeah, this is Jordan Example over in Example Town.")

        self.assertIn("Jordan Example", name_values(result, "self_identification"))
        self.assertNotIn("Jordan Example Over", name_values(result))

    def test_regarding_name_with_first_and_last_spelling_override(self):
        result = extract_candidates(
            "We are calling regarding Casey Sampel, C-A-S-E-Y, Sample, S-A-M-P-L-E."
        )

        corrected = name_candidates(result, "transcript_spelling_corrected")
        self.assertTrue(
            any(item.get("raw") == "Casey Sampel" and item.get("value") == "Casey Sample" for item in corrected),
            corrected,
        )

    def test_hyphenated_last_name_preserves_component_capitalization(self):
        result = extract_candidates("Hi, this is Quinn Example.")

        self.assertIn("Quinn Example", name_values(result, "self_identification"))

    def test_honorific_and_middle_initial_name_cleanup(self):
        result = extract_candidates("Yes, this is Mrs. Robin R. Sample.")

        self.assertIn("Robin R. Sample", name_values(result, "self_identification"))

    def test_same_first_caller_id_corrects_close_surname_error(self):
        result = extract_candidates(
            "Hey, it is Jordan Exampel again.",
            caller_id='"JORDAN EXAMPLE" (202) 555-0102',
        )

        self.assertIn("Jordan Example", name_values(result, "caller_id_corrected"))

    def test_im_sorry_before_it_is_name_does_not_block_caller_id_correction(self):
        result = extract_candidates(
            "Hey, Bailey, I'm sorry. It's Jordan Exampel again. Can you give me a call?",
            caller_id='"JORDAN EXAMPLE" (202) 555-0102',
        )

        self.assertNotIn("Sorry", name_values(result))
        self.assertIn("Jordan Exampel", name_values(result, "self_identification"))
        self.assertIn("Jordan Example", name_values(result, "caller_id_corrected"))

    def test_malformed_long_fax_like_number_is_not_recalled_when_valid_fax_exists(self):
        # An extra digit makes the first synthetic number malformed.
        malformed = "202" + "5550" + "0142"
        result = extract_candidates(
            "Please fax records to 202-5550-0142. The correct fax number is 202-555-0143."
        )
        normalized = normalized_numbers(result)
        self.assertIn("2025550143", normalized)
        self.assertNotIn(malformed, normalized)

    def test_this_is_for_subject_name_is_collected(self):
        result = extract_candidates("Hi, this is for Robin Example again.", caller_id="EXAMPLE ROBIN")

        subject = [
            item for item in result["name_candidates"]
            if item.get("raw") == "Robin Example"
            and item.get("source") == "relationship_subject"
        ]
        self.assertEqual(len(subject), 1)
        self.assertIn("this is for Robin Example", subject[0]["evidence_text"])

    def test_facility_caller_patient_statement_collects_patient_subject(self):
        result = extract_candidates(
            "Morgan with Example Clinic and Example Clinic. Morgan Example, 424 of 60, had a knee revision today. "
            "Again, this is Morgan from Example Clinic and Example Clinic.",
            caller_id="EXAMPLE MORGAN",
        )

        names = {item.get("raw") for item in result["name_candidates"]}
        self.assertIn("Morgan", names)
        self.assertIn("Morgan Example", names)
        subject = next(item for item in result["name_candidates"] if item.get("raw") == "Morgan Example")
        self.assertEqual(subject["source"], "relationship_subject")
        self.assertIn("424 of 60", subject["evidence_text"])
        self.assertIn("relationship_subject", subject["nearby_cues"])

    def test_patient_of_provider_appositive_collects_patient_subject(self):
        result = extract_candidates(
            "Hi, this is Avery calling from Example Home Care. "
            "I'm a physical therapist for a patient of Dr. Sample, Robin Example.",
            caller_id="EXAMPLE HOME CARE",
        )

        names = {item.get("raw") for item in result["name_candidates"]}
        self.assertIn("Avery", names)
        self.assertIn("Robin Example", names)
        subject = next(item for item in result["name_candidates"] if item.get("raw") == "Robin Example")
        self.assertEqual(subject["source"], "relationship_subject")
        self.assertIn("patient of Dr. Sample, Robin Example", subject["evidence_text"])

    def test_self_identified_patient_of_provider_promotes_self_not_provider(self):
        result = extract_candidates("Hi, yes, this is Avery Example. I'm a patient of Dr. Sample.")

        self.assertIn("Avery Example", name_values(result, "self_identification"))
        self.assertIn("Avery Example", name_values(result, "relationship_subject"))
        self.assertNotIn("Dr Sample", name_values(result))
        self.assertNotIn("Sample", name_values(result))

    def test_possessive_relationship_subject_is_collected(self):
        result = extract_candidates(
            "This is Casey Example. I am calling from my husband Morgan Example. "
            "Morgan has a concern about his left knee.",
            caller_id="Unavailable",
        )

        names = {item.get("raw") for item in result["name_candidates"]}
        self.assertIn("Casey Example", names)
        self.assertIn("Morgan Example", names)
        subject = next(item for item in result["name_candidates"] if item.get("raw") == "Morgan Example")
        self.assertEqual(subject["source"], "relationship_subject")
        self.assertIn("my husband Morgan Example", subject["evidence_text"])

    def test_reverse_possessive_relationship_subject_is_collected(self):
        result = extract_candidates(
            "Yes, this is Bailey Example, Casey Sample's daughter. "
            "He is needing a follow-up appointment from a hospital stay.",
            caller_id="Unavailable",
        )

        names = {item.get("raw") for item in result["name_candidates"]}
        self.assertIn("Bailey Example", names)
        self.assertIn("Casey Sample", names)
        subject = next(item for item in result["name_candidates"] if item.get("raw") == "Casey Sample")
        self.assertEqual(subject["source"], "relationship_subject")
        self.assertIn("Casey Sample's daughter", subject["evidence_text"])

    def test_facility_for_dob_subject_is_collected_with_dob_and_callback(self):
        result = extract_candidates(
            "Hi, this is Casey. I am calling from Example Clinic for Robin Example, "
            "date of birth 10/11/1981. You can give me a call at Example Clinic. "
            "The number is 618-555-0100.",
            caller_id="Unavailable",
        )

        names = {item.get("raw") for item in result["name_candidates"]}
        self.assertIn("Casey", names)
        self.assertIn("Robin Example", names)
        subject = next(item for item in result["name_candidates"] if item.get("raw") == "Robin Example")
        self.assertEqual(subject["source"], "relationship_subject")
        self.assertIn("for Robin Example, date of birth 10/11/1981", subject["evidence_text"])
        self.assertEqual(result["dob_candidates"][0]["normalized"], "10/11/1981")
        self.assertEqual(candidate_by_number(result, "6185550100")["formatted"], "(618) 555-0100")

    def test_pronoun_appositive_subject_is_collected(self):
        result = extract_candidates(
            "Hello, this is Morgan from Example Clinic and Example Clinic. "
            "I did get your message about him, Robin Example, being there at 8 a.m.",
            caller_id="Unavailable",
        )

        names = {item.get("raw") for item in result["name_candidates"]}
        self.assertIn("Morgan", names)
        self.assertIn("Robin Example", names)
        subject = next(item for item in result["name_candidates"] if item.get("raw") == "Robin Example")
        self.assertEqual(subject["source"], "relationship_subject")
        self.assertIn("message about him, Robin Example", subject["evidence_text"])

    def test_that_spelling_confirmation_keeps_full_spoken_name(self):
        result = extract_candidates(
            "Hi, this is Jordan Example. That's E-X-A-M-P-L-E. "
            "My number is 202-555-0104.",
            caller_id="EXAMPLE JORDAN",
        )

        corrected = [
            item for item in result["name_candidates"]
            if item.get("source") == "transcript_spelling_corrected"
            and item.get("raw") == "Jordan Example"
            and item.get("value") == "Jordan Example"
        ]
        self.assertEqual(len(corrected), 1)
        self.assertIn("That's E-X-A-M-P-L-E", corrected[0]["evidence_text"])

    def test_caller_and_patient_name_candidates_are_both_collected(self):
        result = extract_candidates("This is Bailey calling about my father Jordan Example.")

        names = {item.get("raw") for item in result["name_candidates"]}
        self.assertIn("Bailey", names)
        self.assertIn("Jordan Example", names)
        jordan = next(item for item in result["name_candidates"] if item.get("raw") == "Jordan Example")
        self.assertIn("relationship_subject", jordan["nearby_cues"])

    def test_self_identification_stops_before_dob_sentence(self):
        result = extract_candidates(
            "Hi, this is Avery Sample. Date of birth is 09/22/1975. "
            "Please call me back at 202-555-0106."
        )

        names = {item.get("raw") for item in result["name_candidates"]}
        self.assertIn("Avery Sample", names)
        self.assertNotIn("Avery Sample Date", names)

    def test_callback_request_without_number_is_a_semantic_event(self):
        result = extract_candidates("Please call me back at the number on file.")

        self.assertEqual(result["number_candidates"], [])
        self.assertTrue(
            any(event.get("event_type") == "callback_request_no_number" for event in result["semantic_events"]),
            result["semantic_events"],
        )

    def test_spoken_digits_extract_conservative_phone_candidate(self):
        result = extract_candidates("Call me back at two zero two five five five zero one zero eight.")

        item = candidate_by_number(result, "2025550108")
        self.assertEqual(item["source"], "spoken_digits")
        self.assertIn("callback", item["nearby_cues"])

    def test_strong_caller_id_match_adds_corrected_name_candidate(self):
        result = extract_candidates("This is Robin Sampel.", caller_id="ROBIN SAMPLE")

        corrected = [
            item for item in result["name_candidates"] if item.get("source") == "caller_id_corrected"
        ]
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["raw"], "Robin Sampel")
        self.assertEqual(corrected[0]["value"], "Robin Sample")
        self.assertEqual(corrected[0]["caller_id_used"], "ROBIN SAMPLE")
        self.assertEqual(result["name_correction_candidates"], [])

    def test_middle_initial_caller_id_adds_same_first_last_name_corrected_candidate(self):
        result = extract_candidates("This is Quinn Exampel.", caller_id="QUINN L EXAMPLE")

        corrected = [
            item for item in result["name_candidates"] if item.get("source") == "caller_id_corrected"
        ]
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["raw"], "Quinn Exampel")
        self.assertEqual(corrected[0]["value"], "Quinn Example")
        self.assertEqual(corrected[0]["caller_id_used"], "QUINN L EXAMPLE")

    def test_parenthesized_phone_caller_id_preserves_display_name_only(self):
        result = extract_candidates(
            "This is Quinn Exampel.",
            caller_id='"QUINN L EXAMPLE" (202) 555-0104',
        )

        corrected = [
            item for item in result["name_candidates"] if item.get("source") == "caller_id_corrected"
        ]
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["raw"], "Quinn Exampel")
        self.assertEqual(corrected[0]["value"], "Quinn Example")
        self.assertEqual(corrected[0]["caller_id_used"], "QUINN L EXAMPLE")

    def test_first_last_parenthesized_caller_id_adds_corrected_candidate(self):
        result = extract_candidates(
            "This is Taylor Exampel.",
            caller_id='"TAYLOR EXAMPLE" (202) 555-0105',
        )

        corrected = [
            item for item in result["name_candidates"] if item.get("source") == "caller_id_corrected"
        ]
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["raw"], "Taylor Exampel")
        self.assertEqual(corrected[0]["value"], "Taylor Example")
        self.assertEqual(corrected[0]["caller_id_used"], "TAYLOR EXAMPLE")

    def test_last_first_parenthesized_caller_id_adds_corrected_candidate(self):
        result = extract_candidates(
            "This is Taylor Exampel.",
            caller_id='"EXAMPLE TAYLOR" (202) 555-0106',
        )

        corrected = [
            item for item in result["name_candidates"] if item.get("source") == "caller_id_corrected"
        ]
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["raw"], "Taylor Exampel")
        self.assertEqual(corrected[0]["value"], "Taylor Example")
        self.assertEqual(corrected[0]["caller_id_used"], "EXAMPLE TAYLOR")

    def test_last_first_trailing_initial_caller_id_adds_corrected_candidate(self):
        result = extract_candidates(
            "This is Taylor Exampel.",
            caller_id='"EXAMPLE TAYLOR P" (202) 555-0107',
        )

        corrected = [
            item for item in result["name_candidates"] if item.get("source") == "caller_id_corrected"
        ]
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["raw"], "Taylor Exampel")
        self.assertEqual(corrected[0]["value"], "Taylor Example")
        self.assertEqual(corrected[0]["caller_id_used"], "EXAMPLE TAYLOR P")

    def test_truncated_first_token_caller_id_can_still_correct_last_name(self):
        result = extract_candidates("This is Taylor Exampel.", caller_id='"EXAMPLE TAY" (202) 555-0108')
        corrected = [item for item in result["name_candidates"] if item.get("source") == "caller_id_corrected"]
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["raw"], "Taylor Exampel")
        self.assertEqual(corrected[0]["value"], "Taylor Example")
        self.assertEqual(corrected[0]["caller_id_used"], "EXAMPLE TAY")

    def test_truncated_last_token_caller_id_is_not_used_as_replacement(self):
        result = extract_candidates(
            "This is Bailey Sample.",
            caller_id='"BAILEY SAM" (202) 555-0109',
        )
        corrected = [
            item for item in result["name_candidates"] if item.get("source") == "caller_id_corrected"
        ]
        self.assertEqual(corrected, [])
        self.assertNotEqual(
            [item.get("value") for item in result["name_candidates"]],
            ["Bailey Sam"],
        )

    def test_last_name_only_caller_id_correction_adds_corrected_candidate_by_default(self):
        result = extract_candidates("This is Taylor Exampel.", caller_id="QUINN L EXAMPLE")

        corrected = [
            item for item in result["name_candidates"] if item.get("source") == "caller_id_corrected"
        ]
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["raw"], "Taylor Exampel")
        self.assertEqual(corrected[0]["value"], "Taylor Example")
        self.assertEqual(corrected[0]["caller_id_used"], "QUINN L EXAMPLE")

    def test_last_name_only_caller_id_correction_accepts_sip_display_name(self):
        result = extract_candidates("This is Taylor Exampel.", caller_id='"QUINN L EXAMPLE" <2025550124>')

        corrected = [
            item for item in result["name_candidates"] if item.get("source") == "caller_id_corrected"
        ]
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["raw"], "Taylor Exampel")
        self.assertEqual(corrected[0]["value"], "Taylor Example")
        self.assertEqual(corrected[0]["caller_id_used"], "QUINN L EXAMPLE")

    def test_last_name_only_caller_id_correction_can_be_disabled(self):
        old_value = os.environ.get("CANDIDATE_AGENT_CALLER_ID_LAST_NAME_ONLY_CORRECTION")
        os.environ["CANDIDATE_AGENT_CALLER_ID_LAST_NAME_ONLY_CORRECTION"] = "false"
        try:
            result = extract_candidates("This is Taylor Exampel.", caller_id="QUINN L EXAMPLE")
        finally:
            if old_value is None:
                os.environ.pop("CANDIDATE_AGENT_CALLER_ID_LAST_NAME_ONLY_CORRECTION", None)
            else:
                os.environ["CANDIDATE_AGENT_CALLER_ID_LAST_NAME_ONLY_CORRECTION"] = old_value

        self.assertFalse(
            any(item.get("source") == "caller_id_corrected" for item in result["name_candidates"])
        )
        self.assertEqual(result["name_correction_candidates"][0]["raw"], "Taylor Exampel")
        self.assertEqual(result["name_correction_candidates"][0]["suggested_value"], "Taylor Example")
        self.assertEqual(result["name_correction_candidates"][0]["reason"], "last_name_phonetic_match")

    def test_close_last_first_caller_id_adds_corrected_name_candidate(self):
        result = extract_candidates("This is Morgan Sampel.", caller_id="SAMPLE MORGAN")
        corrected = [item for item in result["name_candidates"] if item.get("source") == "caller_id_corrected"]
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["raw"], "Morgan Sampel")
        self.assertEqual(corrected[0]["value"], "Morgan Sample")
        self.assertEqual(corrected[0]["caller_id_used"], "SAMPLE MORGAN")
        self.assertEqual(result["name_correction_candidates"], [])

    def test_close_first_last_caller_id_adds_corrected_name_candidate(self):
        result = extract_candidates("This is Morgan Sampel.", caller_id="MORGAN SAMPLE")
        corrected = [item for item in result["name_candidates"] if item.get("source") == "caller_id_corrected"]
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["raw"], "Morgan Sampel")
        self.assertEqual(corrected[0]["value"], "Morgan Sample")
        self.assertEqual(corrected[0]["caller_id_used"], "MORGAN SAMPLE")
        self.assertEqual(result["name_correction_candidates"], [])

    def test_caller_id_does_not_expand_complete_spoken_first_name(self):
        result = extract_candidates("This is Casey Example.", caller_id='"EXAMPLE CASEYAN" (202) 555-0110')

        self.assertIn("Casey Example", name_values(result, "self_identification"))
        self.assertNotIn("Caseyan Example", name_values(result, "caller_id_corrected"))
        self.assertFalse(
            any(
                item.get("source") == "caller_id_corrected"
                and item.get("raw") == "Casey Example"
                and item.get("value") == "Caseyan Example"
                for item in result["name_candidates"]
            )
        )

    def test_caller_id_does_not_expand_first_name_only_candidate(self):
        result = extract_candidates("This is Casey.", caller_id='"EXAMPLE CASEYAN" (202) 555-0110')

        self.assertIn("Casey", name_values(result, "self_identification"))
        self.assertNotIn("Caseyan", name_values(result, "caller_id_corrected"))

    def test_uncertain_last_name_only_caller_id_adds_name_correction_candidate(self):
        result = extract_candidates("This is Bailey Sampel.", caller_id="EXAMPLE,AVERYS")

        self.assertEqual(result["name_correction_candidates"][0]["raw"], "Bailey Sampel")
        self.assertEqual(result["name_correction_candidates"][0]["suggested_value"], "Bailey Example")
        self.assertEqual(result["name_correction_candidates"][0]["reason"], "last_name_phonetic_match")

    def test_generic_and_organization_caller_ids_do_not_add_name_corrections(self):
        for caller_id in ("Wireless Caller", "Example Clinic", "202-555-0108", "UNKNOWN"):
            result = extract_candidates("This is Bailey Sampel.", caller_id=caller_id)
            self.assertEqual(result["name_correction_candidates"], [], caller_id)
            self.assertFalse(
                any(item.get("source") == "caller_id_corrected" for item in result["name_candidates"]),
                caller_id,
            )

    def test_broad_name_recall_adds_supported_subject_name_by_default(self):
        result = extract_candidates(
            "Hello, this is the office following up. "
            "The message concerns Casey Sample and the appointment."
        )

        broad = name_candidates(result, "broad_name_recall")
        self.assertTrue(any(item.get("value") == "Casey Sample" for item in broad), result["name_candidates"])
        casey = next(item for item in broad if item.get("value") == "Casey Sample")
        self.assertEqual(casey["raw"], "Casey Sample")
        self.assertIn("message concerns Casey Sample", casey["evidence_text"])
        self.assertIn("name_recall", casey["nearby_cues"])

    def test_broad_name_recall_can_be_disabled(self):
        old_value = os.environ.get("CANDIDATE_AGENT_BROAD_NAME_RECALL")
        os.environ["CANDIDATE_AGENT_BROAD_NAME_RECALL"] = "false"
        try:
            result = extract_candidates(
                "Hello, this is the office following up. "
                "The message concerns Casey Sample and the appointment."
            )
        finally:
            if old_value is None:
                os.environ.pop("CANDIDATE_AGENT_BROAD_NAME_RECALL", None)
            else:
                os.environ["CANDIDATE_AGENT_BROAD_NAME_RECALL"] = old_value

        self.assertFalse(
            any(item.get("source") == "broad_name_recall" for item in result["name_candidates"]),
            result["name_candidates"],
        )

    def test_broad_name_recall_rejects_obvious_descriptors(self):
        result = extract_candidates(
            "Hello, I am calling about your request. "
            "The message concerns the authorization request."
        )

        values = {item.get("value") for item in result["name_candidates"]}
        self.assertNotIn("Your Request", values)
        self.assertNotIn("The Authorization Request", values)


if __name__ == "__main__":
    unittest.main()

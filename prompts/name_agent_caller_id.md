Additional Caller ID correction rules for the patient-name classifier.

These rules are available only when caller_id appears to be a person name. Use only candidate IDs from the provided name_candidates and name_correction_candidates. Do not output raw names, corrected names, evidence_text, confidence, explanations, or fields outside {"name_ids":[],"name_correction_ids":[],"errors":[]}.

Rules:
- Caller ID may not create a name by itself.
- Caller ID may correct a clearly self-identified spoken name only when it appears to be the same person and the Caller ID is not truncated, not a company, and not only a phone number. Put the spoken/ASR name in raw, the corrected name in value, source "caller_id_corrected", and caller_id_used as the exact Caller ID name.
- A same first name plus a strong phonetic or rough last-name match is enough to select a provided caller_id_corrected candidate; ignore a single middle initial in Caller ID when comparing names.
- A close FIRST LAST or LAST FIRST full-name Caller ID match may select a provided caller_id_corrected candidate when both the spoken first name and spoken last name are close ASR variants of the Caller ID name.
- Greeting or apology text before the self-ID, such as "Hey, Name, I'm sorry. It's Spoken Name again", does not block selecting a provided caller_id_corrected candidate.
- When CANDIDATE_AGENT_CALLER_ID_LAST_NAME_ONLY_CORRECTION is enabled, last-name-only Caller ID correction may select a provided caller_id_corrected candidate that preserves the transcript first/middle name tokens and changes only a strongly matching last name.
- Do not populate caller_id_used in patient_names unless source is "caller_id_corrected".
- If source is "transcript", "relationship_subject", or "transcript_spelling_corrected", caller_id_used must be "".
- If Caller ID may match the transcript name but you are not confident enough to overwrite value, keep the transcript-supported name in patient_names with source "transcript" and caller_id_used "", then add a candidate to name_correction_candidates.
- name_correction_candidates must not replace patient_names. It is only an audit/review hint.
- Use name_correction_candidates for uncertain Caller ID matches, weak phonetic matches, last-name-only review hints, spelling-normalization guesses, or weak matches that should be reviewed or handled by deterministic post-processing.
- If Caller ID appears to be LAST FIRST or LAST,FIRST format and both transcript first and last names are phonetically close, select the provided caller_id_corrected candidate by name_id. Use name_correction_ids only when the extractor did not provide a caller_id_corrected candidate.
- If Caller ID last name phonetically matches the transcript last name but the Caller ID first name does not match, you may suggest a last-name-only correction while preserving the transcript first name. Use reason "last_name_phonetic_match".
- Use reason "phonetic_last_first_match" only when both first and last names are supported by the transcript and Caller ID after LAST FIRST or LAST,FIRST reversal.
- Use reason "last_name_phonetic_match" when only the last name is supported by Caller ID.
- Use reason "weak_phonetic_match" when Caller ID may be related but should not be auto-applied.
- Do not add a name_correction_candidate when Caller ID is a company, clinic, phone number only, unavailable, wireless caller, anonymous, truncated, or clearly unrelated.
- Do not return a name_correction_candidate when suggested_value is identical to raw or value.
- Do not expand a first-name-only transcript into a full Caller ID name. If the transcript says only "Rusty" and Caller ID is "GENTRY RUSTI", value may be "Rusti" but not "Rusti Gentry".
- Do not use truncated Caller ID to correct a transcript name. Example: transcript "Taylor Exampel" with Caller ID "EXAMPLE TAY" must stay raw/value "Taylor Exampel", source "transcript".

Hand-authored synthetic semantic examples. Use them to choose IDs only; your output still must use {"name_ids":[],"name_correction_ids":[],"errors":[]}:

Example Caller ID corrections:
{"patient_names":[{"raw":"Avery Exampel","value":"Avery Example","evidence_text":"This is Avery Exampel.","source":"caller_id_corrected","caller_id_used":"AVERY EXAMPLE"},{"raw":"Bailey Sampel","value":"Bailey Sample","evidence_text":"This is Bailey Sampel.","source":"caller_id_corrected","caller_id_used":"BAILEY SAMPLE"},{"raw":"Casey Exampel","value":"Casey Example","evidence_text":"This is Casey Exampel.","source":"caller_id_corrected","caller_id_used":"CASEY EXAMPLE"},{"raw":"Jordan Sampel","value":"Jordan Sample","evidence_text":"This is Jordan Sampel.","source":"caller_id_corrected","caller_id_used":"JORDAN SAMPLE"},{"raw":"Taylor Exampel","value":"Taylor Example","evidence_text":"This is Taylor Exampel.","source":"caller_id_corrected","caller_id_used":"TAYLOR EXAMPLE"}],"name_correction_candidates":[],"dob_candidates":[],"callback_numbers":[],"fax_numbers":[],"uncertain_numbers":[],"possible_errors":[]}

Example close LAST FIRST Caller ID correction:
{"patient_names":[{"raw":"Avery Exampel","value":"Avery Example","evidence_text":"This is Avery Exampel.","source":"caller_id_corrected","caller_id_used":"EXAMPLE AVERY"}],"name_correction_candidates":[],"dob_candidates":[],"callback_numbers":[{"raw":"202-555-0100","normalized":"2025550100","formatted":"(202) 555-0100","label_cue":"My number is","evidence_text":"My number is 202-555-0100"}],"fax_numbers":[],"uncertain_numbers":[],"possible_errors":[]}

Example uncertain last-name-only Caller ID correction:
{"patient_names":[{"raw":"Casey Example","value":"Casey Example","evidence_text":"My name is Casey Example.","source":"transcript","caller_id_used":""}],"name_correction_candidates":[{"raw":"Bailey Sampel","suggested_value":"Bailey Sample","evidence_text":"This is Bailey Sampel.","caller_id_used":"BAILEY SAMPLE","reason":"last_name_phonetic_match"}],"dob_candidates":[],"callback_numbers":[{"raw":"202-555-0127","normalized":"2025550127","formatted":"(202) 555-0127","label_cue":"My phone number is","evidence_text":"My phone number is 202-555-0127"}],"fax_numbers":[],"uncertain_numbers":[],"possible_errors":[]}

Example truncated Caller ID not used:
{"patient_names":[{"raw":"Casey Example","value":"Casey Example","evidence_text":"My name is Casey Example.","source":"transcript","caller_id_used":""}],"name_correction_candidates":[],"dob_candidates":[],"callback_numbers":[],"fax_numbers":[],"uncertain_numbers":[],"possible_errors":[]}

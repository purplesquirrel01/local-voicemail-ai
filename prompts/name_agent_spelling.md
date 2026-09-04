Additional spelling rules for the patient-name classifier.

Use only candidate IDs from the provided name_candidates and spelled_sequences. Do not output raw names, corrected names, evidence_text, confidence, explanations, or fields outside {"name_ids":[],"name_correction_ids":[],"errors":[]}.

Rules:
- Select from provided name_candidates and spelled_sequences.
- Explicit spelling immediately after a spoken patient/caller name may correct the name. Prefer spelling-associated candidates when the supplied candidate evidence supports the spelling.
- If source is "transcript_spelling_corrected", value must reflect the spelled letters from evidence_text. Do not copy raw into value unless the spelled letters exactly reconstruct the same name token. When ASR hears a plausible but different name, the spelled letters win.
- Do not label a candidate as source "transcript_spelling_corrected" unless value has been corrected from explicit transcript spelling, or unless the spelled letters exactly match the raw name.
- Synthetic examples: "Avery Exampel, spelled E-X-A-M-P-L-E" -> raw Avery Exampel, value Avery Example. "Bailey Sampel, spelled S-A-M-P-L-E" -> raw Bailey Sampel, value Bailey Sample.
- If first and last name are spelled separately, return the full name, not only the last spelled word. Synthetic example: "Avery, A-V-E-R-Y, Example, E-X-A-M-P-L-E" -> raw Avery Example, value Avery Example, source "transcript_spelling".
- If a name is corrected by spelling, use the spelled letters even when ASR heard a similar name. Synthetic example: "Bailey Sampel, B-A-I-L-E-Y, S-A-M-P-L-E" -> raw Bailey Sampel, value Bailey Sample, source "transcript_spelling".
- Spelled letters may include clarification words like "B as in boy". Treat "B as in boy" as the letter B and combine it with nearby spelled letters for the same name token.

Hand-authored synthetic semantic examples. Use them to choose IDs only; your output still must use {"name_ids":[],"name_correction_ids":[],"errors":[]}:

Example spelled names:
{"patient_names":[{"raw":"Avery Exampel","value":"Avery Example","evidence_text":"Avery Exampel, spelled A-V-E-R-Y E-X-A-M-P-L-E.","source":"transcript_spelling_corrected","caller_id_used":""},{"raw":"Bailey Sampel","value":"Bailey Sample","evidence_text":"Bailey Sampel, spelled B-A-I-L-E-Y S-A-M-P-L-E.","source":"transcript_spelling_corrected","caller_id_used":""}],"name_correction_candidates":[],"dob_candidates":[],"callback_numbers":[],"fax_numbers":[],"uncertain_numbers":[],"possible_errors":[]}

Example spelling override:
{"patient_names":[{"raw":"Avery Exampel","value":"Avery Example","evidence_text":"Avery Exampel, spelled A-V-E-R-Y E-X-A-M-P-L-E.","source":"transcript_spelling_corrected","caller_id_used":""}],"name_correction_candidates":[],"dob_candidates":[],"callback_numbers":[],"fax_numbers":[],"uncertain_numbers":[],"possible_errors":[]}

Example separately spelled full name:
{"patient_names":[{"raw":"Avery Example","value":"Avery Example","evidence_text":"My name is Avery, A-V-E-R-Y, Example, E-X-A-M-P-L-E.","source":"transcript_spelling_corrected","caller_id_used":""}],"name_correction_candidates":[],"dob_candidates":[],"callback_numbers":[],"fax_numbers":[],"uncertain_numbers":[],"possible_errors":[]}

Example spelling-corrected subject name:
{"patient_names":[{"raw":"Avery Exampel","value":"Avery Example","evidence_text":"I am calling for Avery Exampel, spelled E-X-A-M-P-L-E.","source":"transcript_spelling_corrected","caller_id_used":""}],"name_correction_candidates":[],"dob_candidates":[],"callback_numbers":[],"fax_numbers":[],"uncertain_numbers":[],"possible_errors":[]}

Example spelling clarification words:
{"patient_names":[{"raw":"Bailey Sampel","value":"Bailey Sample","evidence_text":"Bailey Sampel, B as in boy, A-I-L-E-Y, S-A-M-P-L-E.","source":"transcript_spelling_corrected","caller_id_used":""}],"name_correction_candidates":[],"dob_candidates":[],"callback_numbers":[],"fax_numbers":[],"uncertain_numbers":[],"possible_errors":[]}

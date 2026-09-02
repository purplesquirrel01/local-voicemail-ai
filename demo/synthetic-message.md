# Synthetic Voicemail Walkthrough

This example is entirely synthetic. It contains no real patient, employee,
caller, organization, or production information.

## Example Voicemail

> Hi, this is Maria Sample calling for John Example. His date of birth is
> January 15, 1980. Please call me back at 217-555-0142 about rescheduling
> an appointment. Thank you.

## Primary Transcription

**Model:** Whisper large-v3 via faster-whisper

**Transcript:**

Hi, this is Maria Sample calling for John Example. His date of birth is
January 15, 1980. Please call me back at 217-555-0142 about rescheduling
an appointment. Thank you.

## Structured Candidates

| Field | Candidate |
|---|---|
| Caller | Maria Sample |
| Patient/subject | John Example |
| Date of birth | 01/15/1980 |
| Callback number | 217-555-0142 |

## Callback Verification

1. Gemma identifies the callback-number candidate and its supporting text.
2. The evidence is mapped to Whisper word-level timestamps.
3. FFmpeg extracts the corresponding audio window.
4. Parakeet independently transcribes the selected audio clip.
5. Deterministic Python compares the normalized results.

| Source | Normalized value |
|---|---|
| Whisper evidence | 2175550142 |
| Gemma candidate | 2175550142 |
| Parakeet verification | 2175550142 |

**Resolver decision:** Verified  
**Applied:** Yes  

## Safety Behavior

If the models disagreed, the evidence could not be mapped to audio, or the
clip contained multiple numbers, the callback number would be marked
**Needs Review** instead of being silently accepted.

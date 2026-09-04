# Synthetic voicemail walkthrough

This newly written example matches the recreated portal image. Every identity,
date, number, and mailbox is fictional. The displayed verification and review
states are illustrative fixtures, not measured model results.

> This is Bailey Sample calling for Jordan Sample. Their date of birth is
> February 3, 1970. Please call 202-555-0142 to arrange an appointment. Thank you.

| Field | Synthetic value |
|---|---|
| Reviewer | Avery Example |
| Mailbox | 9901 |
| Caller | Bailey Sample |
| Subject | Jordan Sample |
| Date of birth | 02/03/1970 |
| Callback | 202-555-0142 |

The normal processing path uses Whisper for primary transcription, Gemma for
constrained extraction, and deterministic Python for normalization and evidence
checks. Where configured, Parakeet verifies the callback against its audio clip.
The reviewer sees disagreement and ambiguity instead of an unconditional approval.

The screenshot deliberately illustrates a verified callback and a date of birth
requiring review. Its audio control plays generated silence. No model was run to
produce these demonstration states, and no production record was used.

See [image provenance](../docs/images/README.md) and the
[architecture](../docs/ARCHITECTURE.md) for the implementation context.

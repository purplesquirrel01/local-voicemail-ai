# On-Premises AI Voicemail Processing Platform

A production-oriented voicemail transcription and review system that
processed approximately 150 healthcare voicemails per day across a
five-site orthopedic practice while keeping audio, transcripts, and
model inference inside the local environment.

The system combines Whisper large-v3 transcription, structured extraction
with Gemma 4 E4B, independent Parakeet TDT verification for high-risk audio
spans, deterministic Python resolution, field-level audit records, and a
FastAPI staff review portal.

No patient audio or PHI was sent to external transcription services or
LLM APIs.

## What It Does

- Monitors an Asterisk/VitalPBX voicemail queue for new messages.
- Transcribes voicemail locally with faster-whisper and Whisper large-v3.
- Extracts structured fields such as names, dates of birth, callback
  numbers, fax numbers, and message context.
- Maps high-risk fields back to word-level audio timestamps.
- Re-transcribes selected audio clips with Parakeet TDT.
- Uses deterministic Python rules to accept, reject, or flag values.
- Records field-level verification and review state in SQLite.
- Presents transcripts, fields, verification status, search, and audio
  playback through a FastAPI portal.
- Preserves the original PBX voicemail workflow as a fallback.

## Architecture

1. A watcher detects new voicemail audio and metadata.
2. Whisper produces the primary transcript and word-level timestamps.
3. Gemma proposes bounded structured-field candidates.
4. Evidence is mapped from the transcript back to the original audio.
5. Parakeet independently re-transcribes clips containing high-risk fields.
6. Deterministic Python decides whether each field can be accepted.
7. SQLite stores transcripts, fields, audit records, and review state.
8. Staff review the result through the FastAPI portal.

## Verification Design

Phone and fax numbers can look plausible even when a transcription model
gets one digit wrong. This system does not allow a single model to become
the final authority.

Whisper produces the primary transcript and word timings. Gemma proposes
structured candidates with supporting evidence. Parakeet independently
re-transcribes timestamp-grounded audio clips containing high-risk fields.
Deterministic Python then decides whether a value can be accepted or should
be sent for human review.

> Models propose, evidence grounds, deterministic code decides, and humans
> retain control.

## Technology

- Python
- FastAPI
- SQLite
- faster-whisper / Whisper large-v3
- Gemma 4 E4B / LiteRT
- NVIDIA Parakeet TDT 0.6B v2
- FFmpeg
- Asterisk / VitalPBX
- Nginx

## Privacy

This public repository contains no real voicemail recordings, transcripts,
patient information, employee information, credentials, internal network
addresses, or organization-identifying data.

All public examples, screenshots, names, phone numbers, dates, and audio
files are synthetic.

The system assists administrative review. It does not diagnose, treat,
make clinical decisions, or replace human judgment.

## Public Repository Scope

This repository documents a system originally developed for an internal
healthcare workflow. Organization-specific configuration, production data,
credentials, deployment details, and protected information are excluded
from the public version.

# Model and dependency terms

Apache-2.0 applies to this project's source. It does not replace the terms of
third-party dependencies, model weights, or vendor services. No weights, runtime
binaries, or copied dependency source are bundled.

The integrations target Whisper Large-v3 through faster-whisper, NVIDIA Parakeet
TDT, and a local Gemma model through LiteRT. Before provisioning a model, consult
the exact artifact's upstream model card and license, including any attribution
or use conditions:

- [Whisper/faster-whisper model card](https://huggingface.co/Systran/faster-whisper-large-v3)
- [Parakeet TDT model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [Gemma terms](https://ai.google.dev/gemma/terms)
- [LiteRT LM](https://github.com/google-ai-edge/LiteRT-LM)

The optional dependency versions in `pyproject.toml` describe source integrations,
not a bundled or validated model distribution. Model download and provisioning
are outside this repository's development/test setup.

# VoiceSep test audio

Upload a WAV from the numbered folders into VoiceSep. Each file is a four-second synthetic mixture of recorded voices, at 8 kHz mono PCM16. Start with automatic speaker counting, then compare using the manual count shown in the folder name.

| File | Voices | Language |
|---|---:|---|
| [2-speakers/en-2-speakers.wav](2-speakers/en-2-speakers.wav) | 2 | en |
| [2-speakers/hi-2-speakers.wav](2-speakers/hi-2-speakers.wav) | 2 | hi |
| [2-speakers/mixed-2-speakers.wav](2-speakers/mixed-2-speakers.wav) | 2 | mixed |
| [3-speakers/en-3-speakers.wav](3-speakers/en-3-speakers.wav) | 3 | en |
| [3-speakers/hi-3-speakers.wav](3-speakers/hi-3-speakers.wav) | 3 | hi |
| [3-speakers/mixed-3-speakers.wav](3-speakers/mixed-3-speakers.wav) | 3 | mixed |
| [4-speakers/en-4-speakers.wav](4-speakers/en-4-speakers.wav) | 4 | en |
| [4-speakers/hi-4-speakers.wav](4-speakers/hi-4-speakers.wav) | 4 | hi |
| [4-speakers/mixed-4-speakers.wav](4-speakers/mixed-4-speakers.wav) | 4 | mixed |
| [5-speakers/en-5-speakers.wav](5-speakers/en-5-speakers.wav) | 5 | en |
| [5-speakers/hi-5-speakers.wav](5-speakers/hi-5-speakers.wav) | 5 | hi |
| [5-speakers/mixed-5-speakers.wav](5-speakers/mixed-5-speakers.wav) | 5 | mixed |

The `references` folder contains the clean component tracks for listening comparisons. These mixtures reuse the held-out five-speaker example references and serve as local test fixtures. The original benchmark report remains in the research bundle.

English recordings come from LibriSpeech. Hindi recordings come from IndicVoices-R under the access and usage terms accepted for this project. Keep these local fixtures and source audio out of Git. See `docs/DATA.md` for attribution and corpus details.

Rebuild with `python scripts/make_test_examples.py --archive PATH_TO_LISTENING_EXAMPLES_ZIP --output examples/test-audio`.

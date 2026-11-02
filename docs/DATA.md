# Data preparation

Use [LibriSpeech](https://www.openslr.org/12) and the Hindi portion of [IndicVoices-R](https://github.com/AI4Bharat/IndicVoices-R). Preserve original speaker identifiers and dataset attribution. Download only the required portions on the free GPU host; do not commit audio.

LibriSpeech: index a training pool such as train-clean-100. For an exact published benchmark reproduction, use its prescribed pools and frozen metadata instead; do not compare custom mixtures as though they were the official Libri5Mix test set.

IV-R: use the provided JSONL manifest. Each row must contain `filename`, `speaker_id`, `lang`, and preferably `text`, `duration`, and `snr`. `voicesep index-sources` accepts Hindi codes hi/hin/Hindi and filters supplied SNR values below 10 dB. Missing SNR values are retained for manual QA. IndicVoices-R contains enhanced speech, so inspect enhancement artifacts before treating it as a clean reference.

Normalized source JSONL:

```json
{"path":"/absolute/path/to/clip.wav","speaker_id":"original-id","language":"hi","corpus":"indicvoices-r","text":"Original transcript"}
```

Relative paths are resolved from the source manifest's directory. Speaker splits are deterministic corpus-qualified hashes, 80/10/10; each split needs at least five speakers per language. Do not split already generated mixtures randomly. Check for duplicate recordings or identity aliases during source QA.

Dynamic recipes record source paths/identities, crop offsets, gains, augmentation seeds, language condition, sample rate, and source-manifest SHA. Frozen validation and test recipes are generated from disjoint source manifests. Keep validation/test seeds separate. Full within-speaker code-switching requires appropriately transcribed code-switched sources; mixing one English speaker with a Hindi speaker does not establish that capability.

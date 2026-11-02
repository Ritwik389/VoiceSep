# Data preparation

Use [LibriSpeech](https://www.openslr.org/12) and the Hindi portion of [IndicVoices-R](https://github.com/AI4Bharat/IndicVoices-R). Preserve original speaker identifiers and dataset attribution. Prepare data on a CPU host before allocating the free GPU; do not commit audio.

The automated Kaggle path is described in [KAGGLE.md](KAGGLE.md). As checked on 2 November 2026, the old direct Hindi archive returns 403; the publisher's [Hugging Face dataset](https://huggingface.co/datasets/ai4bharat/indicvoices_r) requires approved access. Configure a read-only `HF_TOKEN` in Kaggle Secrets after approval. The bootstrap preflights access before downloading English and never bypasses the publisher gate.

`pip install -e '.[data]'` enables `voicesep bootstrap-data --destination data/corpus`. Defaults cap selection at 30 English hours and 20 Hindi hours, with 40 clips per original speaker. Retained clips are 4–30 seconds, finite and non-silent, converted to mono 8 kHz FLAC. Actual retained hours can be lower and are recorded in `preparation.json`. Hindi loading pins the dataset revision, preserves original IDs/transcripts, and filters supplied SNR below 10 dB. The gated dataset schema and the complete download still need verification in an authorized session. Preparation stops if any split lacks five speakers per language; increase the corpus budget in a fresh destination rather than changing the split after seeing test results.

LibriSpeech: index a training pool such as train-clean-100. For an exact published benchmark reproduction, use its prescribed pools and frozen metadata instead; do not compare custom mixtures as though they were the official Libri5Mix test set.

IV-R: use the provided JSONL manifest. Each row must contain `filename`, `speaker_id`, `lang`, and preferably `text`, `duration`, and `snr`. `voicesep index-sources` accepts Hindi codes hi/hin/Hindi and filters supplied SNR values below 10 dB. Missing SNR values are retained for manual QA. IndicVoices-R contains enhanced speech, so inspect enhancement artifacts before treating it as a clean reference.

Normalized source JSONL:

```json
{"path":"/absolute/path/to/clip.wav","speaker_id":"original-id","language":"hi","corpus":"indicvoices-r","text":"Original transcript"}
```

Relative paths are resolved from the source manifest's directory. Speaker splits are deterministic corpus-qualified hashes, 80/10/10; each split needs at least five speakers per language. Do not split already generated mixtures randomly. Check for duplicate recordings or identity aliases during source QA.

Dynamic recipes record source paths/identities, crop offsets, gains, augmentation seeds, language condition, sample rate, and source-manifest SHA. Frozen validation and test recipes are generated from disjoint source manifests. Bootstrap manifests and recipes use relative paths so the exact manifest bytes can move from CPU outputs to GPU input mounts without changing their hashes. Keep validation/test seeds separate. Full within-speaker code-switching requires appropriately transcribed code-switched sources; mixing one English speaker with a Hindi speaker does not establish that capability.

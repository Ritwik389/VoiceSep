# VoiceSep model card

**Current five-speaker training/evaluation status: not run.**

## Intended task

Offline, single-channel separation of English/Hindi mixtures with one to five distinct speakers per recording. Five simultaneous voices are the target; recordings containing more than five distinct speakers exceed the initial capacity.

## Architecture and initialization

TF-Locoformer, six blocks, hidden dimension 128, two SwiGLU convolution FFNs of dimension 384, four attention heads. STFT/iSTFT use periodic Hann windows, FFT 128, hop 64 at 8 kHz. Shared weights are loaded strictly from the verified MERL Libri2Mix checkpoint. The five-slot decoder and shared activity head require training. Shared-weight mismatches fail rather than loading partially without notice.

Objective: Hungarian-matched SI-SDR plus amplitude error on active sources, normalized energy cost on inactive slots, matched speech-activity BCE, and mixture reconstruction. Counts come from calibrated activity/relative energy with duplicate suppression. No reference count is used in automatic inference.

## Known limitations

- No validated five-speaker, Hindi, code-switching, noisy-room, or live capability until measured.
- 8 kHz outputs contain no original content above 4 kHz.
- Silence detection is an amplitude guard, not a certified speech detector.
- Default tracking aligns overlapping audio. Optional ECAPA embeddings can improve tracking after pauses, but require separate evaluation.
- Similar voices, very quiet sources, and non-overlapping turns can cause merges, omissions, duplicates, and identity swaps.
- Filling all five output slots does not prove that only five speakers are present.
- Uploaded audio has no clean references: reference-dependent separation scores are never shown in the UI.
- Transcripts are generated from separated audio and can inherit separation errors. They are not ground truth.

## Release evidence

Record checkpoint and benchmark hashes, sampling rate, training-data size, automatic vs oracle count, each language/count condition, actual compute, numeric gates, and a signed listening review. A checkpoint having optimizer steps does not establish quality.

## Later analysis

Age range, perceived vocal presentation, and emotion are later experimental features. Native 16 kHz training, bilingual calibration, uncertainty/abstention, and comparison of original versus separated speech are prerequisites. They are not implemented as fabricated V1 predictions.

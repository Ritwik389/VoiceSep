# VoiceSep model card

**Status: evaluated research checkpoint. The numeric release gate failed.**

Training ended at step 16,645 under the free GPU budget. Step 16,000 had the best validation selection score of 11.81 and was calibrated on validation data before held-out evaluation. See [the compute budget](COMPUTE_BUDGET.md) for the stop conditions.

## Intended task

Offline, single-channel separation of English/Hindi mixtures with one to five distinct speakers per recording. Five simultaneous voices are the target; recordings containing more than five distinct speakers exceed the initial capacity.

## Architecture and initialization

TF-Locoformer, six blocks, hidden dimension 128, two SwiGLU convolution FFNs of dimension 384, four attention heads. STFT/iSTFT use periodic Hann windows, FFT 128, hop 64 at 8 kHz. Shared weights are loaded strictly from the verified MERL Libri2Mix checkpoint. The five-slot decoder and shared activity head require training. Shared-weight mismatches fail rather than loading partially without notice.

Objective: Hungarian-matched SI-SDR plus amplitude error on active sources, normalized energy cost on inactive slots, matched speech-activity BCE, and mixture reconstruction. Counts come from calibrated activity/relative energy with duplicate suppression. No reference count is used in automatic inference.

## Known limitations

- Held-out English, Hindi, and mixed-language mixtures measure scripted overlap at 8 kHz. Noisy rooms, spontaneous conversations, code-switching within an utterance, and live use remain unvalidated.
- 8 kHz outputs contain no original content above 4 kHz.
- Silence detection is an amplitude guard, not a certified speech detector.
- Default tracking aligns overlapping audio. Optional ECAPA embeddings can improve tracking after pauses, but require separate evaluation.
- Similar voices, very quiet sources, and non-overlapping turns can cause merges, omissions, duplicates, and identity swaps.
- Filling all five output slots does not prove that only five speakers are present.
- Uploaded audio has no clean references: reference-dependent separation scores are never shown in the UI.
- Transcripts are generated from separated audio and can inherit separation errors. They are not ground truth.

## Release evidence

The automatic-count evaluation covered 1,200 held-out mixtures. The gate requires at least 10 dB mean SI-SDR improvement and 90% exact count accuracy in each four- and five-speaker condition.

| Condition | 4-speaker SI-SDRi / count | 5-speaker SI-SDRi / count |
|---|---:|---:|
| English | 12.05 dB / 89% | 9.17 dB / 88% |
| Hindi | 9.96 dB / 71% | 8.84 dB / 89% |
| Mixed | 12.41 dB / 89% | 9.51 dB / 95% |

The research bundle contains the detailed automatic-count evaluation. The Kaggle output also contains the known-count diagnostic and listening examples. Human listening review and reference-aligned ASR scoring remain pending.

## Later analysis

Age range, vocal gender presentation and temporal emotion now have a local pretrained research preview. Results carry source-bandwidth and validation warnings, with abstention for insufficient speech. Native 16 kHz separation, bilingual calibration and clean-versus-separated comparisons are the next validation steps. See [the implementation plan](SPEAKER_ANALYSIS_PLAN.md).

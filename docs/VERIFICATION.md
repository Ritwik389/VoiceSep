# Local software verification — 2 November 2026

## Verified

- Official Libri2Mix checkpoint SHA-256 matched the published Git LFS identifier.
- Strict shared-backbone loading succeeded: 172 tensors; the decoder/activity extension has 14,982,731 parameters in total.
- Real forward/backward execution passed for short waveforms and synthetic-tone smoke inputs. These are execution checks, not speech-quality measurements.
- The interrupted-training test reproduces the uninterrupted optimizer/model state exactly, preserving mixture index and early-stopping state.
- Python checks cover silent targets, permutation invariance, degenerate estimates, brief voices in long files, duplicate suppression, overlap alignment, deterministic mixtures, speaker-disjoint splits, automatic-count penalties, import gating, cancellation, ZIP exports, and remixing.
- Shared-clock tests verify common source start times, seek offsets, solo/gain changes without restarting, and cancellation of pending playback.
- Frontend TypeScript compilation and Vite production build pass.
- Private GitHub source push succeeded; its initial Software checks workflow passed. No GitHub release has been published.
- Kaggle stage generation produces private CPU/GPU notebooks without credentials. Portable mixture recipes reproduce identical arrays after moving the corpus to a new input mount. Actual Kaggle execution is pending authentication and Hindi access.
- Browser inspection at 1440, 768, and 390 px: no page-level horizontal overflow observed. A clearly named synthetic timing fixture was imported and played. The model-settings dialog rejected raw two-speaker weights. Separation remained disabled without a trained checkpoint.

The browser was inspected through the supported desktop browser tools; the design skill's separate inspector executable was not run. No screenshot is evidence of separation accuracy. The generated design image in docs/design is a design reference, not a processed recording.

## Still required

- Actual free-GPU English/Hindi fine-tuning, calibration, and held-out speech evaluation.
- Real four/five-speaker audio examples, numeric release gates, and human listening review.
- Controlled Conv-TasNet training/comparison and ECAPA tracking evaluation.
- Multilingual ASR accuracy, WER/CER and code-switching tests, real-room tests, and later age/affect features.
- Actual Kaggle submission, GPU memory/throughput results, and resumable training across saved notebook outputs. No model-quality achievement is claimed.

The locally available execution device has neither CUDA nor MPS. A SciPy 1.15 native-library loader failure on this Mac was resolved by using project-local SciPy 1.13.1; the conditional dependency preserves that known-compatible version for this Python/macOS combination. Test warnings from the pinned upstream rotary/attention APIs are deprecations rather than test failures.

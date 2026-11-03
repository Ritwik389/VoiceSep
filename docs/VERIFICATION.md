# Software verification — updated 3 November 2026

## Verified

- Official Libri2Mix checkpoint SHA-256 matched the published Git LFS identifier.
- Strict shared-backbone loading succeeded: 172 tensors; the decoder/activity extension has 14,982,731 parameters in total.
- Real forward/backward execution passed for short waveforms and synthetic-tone smoke inputs. These are execution checks, not speech-quality measurements.
- The interrupted-training test reproduces the uninterrupted optimizer/model state exactly, preserving mixture index and early-stopping state.
- Python checks cover silent targets, permutation invariance, degenerate estimates, brief voices in long files, duplicate suppression, overlap alignment, deterministic mixtures, speaker-disjoint splits, automatic-count penalties, import gating, cancellation, ZIP exports, and remixing.
- Shared-clock tests verify common source start times, seek offsets, solo/gain changes without restarting, and cancellation of pending playback.
- Frontend TypeScript compilation and Vite production build pass.
- Private GitHub source push succeeded; its initial Software checks workflow passed. No GitHub release has been published.
- Kaggle login, free quota lookup, private source upload, and private CPU-only draft metadata were verified through the official CLI. Kaggle expands ZIP uploads into a nested archive directory; the corrected loader verifies a pinned file-checksum manifest and supports flat expanded or intact archives. These layouts and tamper rejection pass a regression check. The UI-started CPU run (version 4) retrieved `HF_TOKEN`, passed IndicVoices-R access preflight, and retained 2,500 English clips (9.24 hours), then failed on a socket read timeout while streaming the archive. This run did not reach Hindi materialization; the later version-6 recovery completed both languages. Portable mixture recipes reproduce identical arrays after moving the corpus to a new input mount.
- The replacement downloader resumes saved bytes after timeouts or truncated transfers, handles ignored/malformed ranges, rejects checksum failures, and preserves a partial cache after exhausted retries. Seven regression cases pass, including local extraction with transcript preservation. These same tests passed in a private Kaggle CPU notebook on Python 3.13, along with a live HTTP 206 request for exactly 1 KB at the requested offset. Recovery preparation version 6 subsequently completed the bilingual corpus; its manifests and frozen recipes have passed validation.
- The bounded controller stopped on the failed CPU run before allocating a GPU. Its tests cover corpus integrity, failed benchmarks, insufficient free quota at both submission boundaries, stop-file handling, pagination, pinned parent versions, recovery waiting for a new UI run, and prevention of repeated training sessions. These are orchestration tests, not measured GPU results. All 36 Python tests pass.
- Preparation version 6 completed: 30.0017 English hours and 20.0014 Hindi hours. Downloaded manifests pass speaker-disjoint and frozen-recipe hash checks; an exported sample decodes as 8 kHz audio. A long-lived client received a permission error while polling; controller requests now reload cached credentials before use, with a regression case for credentials becoming stale between polls.
- GPU benchmark version 1 reached a Tesla T4 but aborted on non-finite gradient norms before GradScaler could skip the overflowing update. AMP now follows [PyTorch's unscale/clip/step/update sequence](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html#gradient-clipping), counts only completed optimizer updates, and saves the actual consumed-mixture index. CPU GradScaler tests confirm overflow leaves parameters unchanged and reduces the scale; full-precision non-finite gradients still fail. Resume after a skipped update matches uninterrupted training exactly. Corrected benchmark version 2 has completed its first finite optimizer update on a Tesla T4; the complete 100-update result remains pending.
- Browser inspection at 1440, 768, and 390 px: no page-level horizontal overflow observed. A clearly named synthetic timing fixture was imported and played. The model-settings dialog rejected raw two-speaker weights. Separation remained disabled without a trained checkpoint.

The browser was inspected through the supported desktop browser tools; the design skill's separate inspector executable was not run. No screenshot is evidence of separation accuracy. The generated design image in docs/design is a design reference, not a processed recording.

## Still required

- Actual free-GPU English/Hindi fine-tuning, calibration, and held-out speech evaluation.
- Real four/five-speaker audio examples, numeric release gates, and human listening review.
- Controlled Conv-TasNet training/comparison and ECAPA tracking evaluation.
- Multilingual ASR accuracy, WER/CER and code-switching tests, real-room tests, and later age/affect features.
- Completed CPU corpus preparation, GPU memory/throughput results, and resumable training across saved notebook outputs. No model-quality achievement is claimed.

The locally available execution device has neither CUDA nor MPS. A SciPy 1.15 native-library loader failure on this Mac was resolved by using project-local SciPy 1.13.1; the conditional dependency preserves that known-compatible version for this Python/macOS combination. Test warnings from the pinned upstream rotary/attention APIs are deprecations rather than test failures.

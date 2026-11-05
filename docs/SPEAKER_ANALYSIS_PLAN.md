# VoiceSep speaker analysis and next phases

## Current delivery

The requested local test folder is `examples/test-audio`. It contains 12 WAV mixtures covering two, three, four and five speakers in English, Hindi and mixed English/Hindi. Each mixture is four seconds at 8 kHz mono. Clean component tracks and a manifest support listening comparisons. They derive from the held-out example references. The original 1,200-mixture evaluation remains the release benchmark.

The speaker-attribute preview runs on completed separation sessions through **Analyze all speakers**. It also accepts an isolated voice through `voicesep analyze`. It uses local CPU inference, so the existing one-hour cloud reserves remain available. This follows the user's priority to add features after the usable separation preview. The separator's release thresholds still require improvement.

## Pretrained features

| Feature | Model and revision | Product behavior |
|---|---|---|
| Age range and vocal gender presentation | `audeering/wav2vec2-large-robust-6-ft-age-gender` at `a681b720dafd12b9dd7b6d13fb437c7b6b197fd3` | Run on up to three speech segments per voice. Aggregate age estimates into a broad band. Return uncertain when predictions disagree. Present acoustic labels as feminine-sounding, masculine-sounding or child-like. |
| Emotion over time | `emotion2vec/emotion2vec_plus_base` at `b318240bfe67db81a8c572ecb37ce9c3759b81c9` | Analyze up to 12 speech windows per voice, retain timeline offsets, and seek on click. Preserve uncertain results when class scores are weak or the model returns other/unknown. |

The audEERING weights use CC-BY-NC-SA-4.0. The emotion model points to the publisher's model-license and FunASR agreement. Keep the pretrained model identity, revision and license in exported results. This application is a noncommercial research preview. [audEERING model card](https://huggingface.co/audeering/wav2vec2-large-robust-6-ft-age-gender), [emotion2vec model card](https://huggingface.co/emotion2vec/emotion2vec_plus_base), [FunASR model agreement](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)

Both models receive 16 kHz input. Current separated tracks are native 8 kHz, so preprocessing resamples them and records the bandwidth limit. The age band's margin and class scores are uncalibrated. English and Hindi reliability on separated voices remains an evaluation task.

## Implementation and acceptance

- Analyze existing completed sessions through `POST /api/jobs/{id}/analysis`. Use the existing local worker, cancellation and event stream. Keep voice playback and exports available during analysis.
- Store results per speaker in the session JSON and SQLite payload. Add the analysis JSON to the ZIP export. Older sessions can acquire attributes without repeating separation.
- Select speech windows using existing activity regions plus duration, energy and clipping checks. Require at least two seconds for emotion and three for age. Load age and emotion models sequentially after releasing the separator.
- Preserve usable separation outputs when model loading or analysis fails. Show missing dependencies, retryable errors and uncertain attributes in the inspector. A restart marks interrupted analysis as failed while retaining the completed voice tracks.
- Verify pinned pretrained weights with actual CPU inference. Test abstention, original timestamp alignment, model failure, cancellation, restart, duplicate requests and exported metadata. Check the inspector at desktop and mobile sizes.

## Ordered next work

1. **Quality-aware attribute evaluation.** Run the same pretrained models on clean references and matched separated tracks. Report age-band shifts, vocal-class agreement, emotion distribution drift and abstention coverage separately by language and speaker count. These measure stability. Use recordings with consented age-band and emotion annotations to measure age error, band accuracy and emotion macro-F1. Compare all speech against quality-selected windows and uncertainty gating in controlled ablations.
2. **Native 16 kHz separation.** Prepare native 16 kHz clean sources and frozen speaker-disjoint mixtures before training. Retain the five-slot design, use a 256-point FFT and 128-sample hop to preserve the current time resolution, and benchmark memory/throughput first. Save this as a separate checkpoint/config family. Drive decoding, trimming, session metadata, checkpoint checks and UI bandwidth labels from the checkpoint sample rate so native 16 kHz audio stays at 16 kHz throughout processing. Re-evaluate four/five-speaker quality and attribute stability against the 8 kHz baseline. Allocate a new measured free-GPU budget before launching this training.
3. **V2 robustness.** Freeze separate noisy/reverberant validation and test recipes using SNR levels of 20, 10 and 5 dB, room decay targets of 0.2, 0.4 and 0.6 seconds, and speaker level differences up to 15 dB. Include pauses, partial overlap and similar timbres. Evaluate automatic counting, SI-SDRi, weakest-speaker quality, leakage, long-recording identity swaps, runtime and attribute coverage. Add consented real conversations after the synthetic study.
4. **Scaling and live use.** Compare recursive extraction at six, eight and ten speakers and publish quality-versus-count curves. Streaming follows with a causal/distilled separator and measured latency, buffering and persistent identity tracking.

Native 16 kHz training and robustness validation remain separate milestones. The current pretrained feature preview can be used while those experiments are prepared.

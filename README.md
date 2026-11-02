# VoiceSep

A local audio workspace and reproducible English/Hindi speech-separation research pipeline.

**Status: implementation under validation. The five-speaker model has not been trained or evaluated.**
The verified upstream checkpoint is a **two-speaker** backbone. Expanding its decoder does not confer five-speaker capability. The app refuses untrained checkpoints; it does not substitute filters, sample outputs, or duplicated audio for separation.

## Run locally

Python 3.10+ and FFmpeg are required. Use a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cd frontend
npm ci
npm run build
cd ..
voicesep serve
```

Open http://127.0.0.1:8000. Without a trained checkpoint you can import, preview, trim the selected range, manage recordings, and download the training notebook. Separation unlocks when a fine-tuned checkpoint is loaded in Model settings. The native model bandwidth is 8 kHz; resampling will not restore frequencies above 4 kHz.

For frontend development, run `npm run dev` inside `frontend` and `voicesep serve` in a second terminal. Vite proxies `/api` to port 8000.

Optional packages:

```sh
pip install -e '.[transcription]'  # local multilingual Whisper, no paid API
pip install -e '.[baseline]'       # controlled Conv-TasNet comparison
pip install -e '.[tracking]'       # optional ECAPA embeddings for long pauses
```

Transcription and embedding models are downloaded on first explicitly enabled use. Separation runs first; ASR runs on CPU afterward. Recordings stay in `data/sessions/` unless you export them. Bind to loopback for the intended single-user local deployment.

## Train on a free GPU

Open [the Colab notebook](notebooks/01_train_colab.ipynb) or [the Kaggle notebook](notebooks/02_train_kaggle.ipynb). Both detect CUDA, persist checkpoints, benchmark memory/throughput, and use the same CLI as local execution. They do not purchase compute. Free GPU quotas and actual training time depend on the provider.

1. Download LibriSpeech training audio and the Hindi portion of IndicVoices-R, including its original speaker metadata. See [data preparation](docs/DATA.md).
2. Index sources and split **speakers** before generating mixtures:

```sh
voicesep index-sources /path/to/LibriSpeech /path/to/Hindi/manifest.jsonl /path/to/Hindi
voicesep split data/sources.jsonl --output-dir data
voicesep freeze data/validation.jsonl data/validation-mixtures.jsonl --per-condition 10
voicesep freeze data/test.jsonl data/test-mixtures.jsonl --per-condition 100 --seed 98765
voicesep download-backbone
voicesep train --pretrained models/tflocoformer-libri2mix.pth --benchmark --output runs/benchmark
voicesep train --pretrained models/tflocoformer-libri2mix.pth
# After interruption; retain the original configuration:
voicesep train --resume runs/tflocoformer/last.pt
```

`configs/train.yaml` defaults to five slots, four-second crops, batch one, accumulation eight, 500 head warmup steps, 20,000 total optimizer steps, mixed precision on CUDA, and checkpointing every 500 steps. Use `--segment-seconds 2` for memory benchmarking; set the same value in the training configuration before the real run. Benchmark output is not a speech-quality evaluation and its checkpoint should not be promoted to `best.pt`.

3. Calibrate on validation data, then evaluate once on held-out test mixtures:

```sh
voicesep calibrate runs/tflocoformer/best.pt data/validation.jsonl data/validation-mixtures.jsonl --device cuda
voicesep evaluate models/best-calibrated.pt data/test.jsonl data/test-mixtures.jsonl --device cuda
voicesep separate conversation.wav --checkpoint models/best-calibrated.pt --output runs/conversation
```

Calibration and test recipes record the source-manifest hash. Reuse the exact source manifests rather than modifying them after freezing a benchmark. Copy `best-calibrated.pt` from the notebook to this computer, then set its path in the app. The service reads `VOICESEP_CHECKPOINT` and `VOICESEP_DATA_DIR`; defaults are `models/best.pt` and `data/sessions`.

## Evaluation and release gates

Primary evaluation uses automatic counting. Missing references and excess outputs are penalized rather than removed; oracle-count evaluation is diagnostic only. Report SI-SDR/i, STOI, weakest-speaker improvement, leakage, exact-count confusion, runtime, and memory by English/Hindi/mixed and source count.

The numeric gates are ≥10 dB mean SI-SDR improvement and ≥90% exact-count accuracy for **each** 4/5-speaker English, Hindi, and mixed condition, with at least 100 mixtures per condition. A documented human listening review is additionally required. WER/CER and real-room quality are not implied by these metrics.

Run software verification independently:

```sh
voicesep smoke-test
pytest -q
cd frontend
npm test
npm run build
```

The smoke test uses tones to check forward/backward execution. It never counts as speech evaluation. See [model card](docs/MODEL_CARD.md), [research notes](docs/RESEARCH.md), and [roadmap](docs/ROADMAP.md).

## API

`POST /api/jobs` accepts audio and processing options; `process=false` imports without a checkpoint. `GET /api/jobs/{id}` reads status; `/events` streams progress. `/start`, `/cancel`, and `DELETE /api/jobs/{id}` control jobs. Audio, ZIP/transcript exports, and gain-controlled remix endpoints are available. Interactive API documentation is at `/docs`.

## Attribution

The vendored TF-Locoformer is by Mitsubishi Electric Research Laboratories (MERL), pinned to `7a615460d347ff7334a13dbb831d16280da72cdc`; original headers are retained. Source and official weights are Apache-2.0, with upstream weight declarations preserved in `docs/upstream-license.txt`. Conv-TasNet comparison weights have their own share-alike notice. Dataset and optional-model terms remain separate. See [NOTICE](NOTICE).

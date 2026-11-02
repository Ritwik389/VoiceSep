# Private, free-GPU Kaggle workflow

The private CPU preparation notebook is available at [ritwik8307/voicesep-data-preparation](https://www.kaggle.com/code/ritwik8307/voicesep-data-preparation). Kaggle authentication and the private source upload are verified. Data preparation and GPU results remain separate execution stages. No five-speaker checkpoint is available yet.

## Account setup

Use your existing free Kaggle account. Select the free T4 accelerator; never enable a paid fallback. Kaggle account verification, available quota, and hardware availability may prevent a run.

The official Kaggle CLI needs Python 3.11+. Keep it separate from the app environment:

```sh
python3.11 -m venv .tools/kaggle
.tools/kaggle/bin/pip install kaggle==2.2.5 six==1.17.0
.tools/kaggle/bin/kaggle auth login
```

The official login opens a browser and requests Kaggle API access. Complete it yourself; do not paste credentials into chat, source, notebooks, or GitHub. `.tools/` is ignored by Git. Your Kaggle username can differ from your GitHub username.

Request publisher access to [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r). After approval, create a read-only Hugging Face token and add it as `HF_TOKEN` through **Kaggle → Add-ons → Secrets**. Grant this secret only to the CPU preparation notebook. The old direct Hindi download is currently denied. The jobs do not bypass access requirements or silently replace the Hindi corpus.

## Generate and upload private source

From the repository root, replace `YOUR_KAGGLE_USERNAME` with your actual account name:

```sh
python scripts/kaggle_jobs.py --owner YOUR_KAGGLE_USERNAME
.tools/kaggle/bin/kaggle datasets create -p dist/kaggle/source
.tools/kaggle/bin/kaggle kernels push -p dist/kaggle/prepare --no-run
```

The generated source dataset and all notebooks are private. The archive contains code, configuration and documentation; it excludes recordings, weights and credentials. Kaggle expands uploaded ZIPs, so generation adds `voicesep-source-manifest.json` alongside the ZIP. Each job verifies the pinned manifest and every expanded file; intact ZIP inputs are also supported. Upload the entire generated source directory. Generation alone does not submit a job. If the private source dataset already exists, use `kaggle datasets version -p dist/kaggle/source -m "Update VoiceSep source"` instead of `datasets create`, then regenerate all pending stages against the new archive. Keep one source version consistent throughout a training sequence.

Open `https://www.kaggle.com/code/YOUR_KAGGLE_USERNAME/voicesep-data-preparation`, enable the `HF_TOKEN` secret, and run/save the notebook with outputs. It is CPU only. Alternatively, after enabling the secret, submit it using:

```sh
.tools/kaggle/bin/kaggle kernels push -p dist/kaggle/prepare
.tools/kaggle/bin/kaggle kernels status YOUR_KAGGLE_USERNAME/voicesep-data-preparation
```

The CPU job selects a bounded corpus, resamples to 8 kHz, splits original speakers, and freezes validation/test recipes. Defaults target up to 30 English and 20 Hindi hours; `preparation.json` records actual retained hours. Access failure or insufficient speakers stops preparation. Do not run a GPU stage before preparation succeeds and its output is saved. Do not make outputs containing publisher-controlled data public.

## Benchmark, then train

```sh
.tools/kaggle/bin/kaggle kernels push -p dist/kaggle/benchmark --accelerator NvidiaTeslaT4
.tools/kaggle/bin/kaggle kernels status YOUR_KAGGLE_USERNAME/voicesep-gpu-benchmark
```

This job attaches the completed CPU output. It benchmarks 100 full optimizer steps with the backbone unfrozen, trying four-second crops first and two seconds after a failed run. Inspect failures before continuing; a non-memory error also needs attention. `benchmark-selection.json` records the chosen crop size, throughput, and peak allocated CUDA memory. The model uses one T4; a second T4 does not increase that device's available memory. Benchmark weights are not a release model.

Only after the benchmark completes:

```sh
.tools/kaggle/bin/kaggle kernels push -p dist/kaggle/train --accelerator NvidiaTeslaT4
.tools/kaggle/bin/kaggle kernels status YOUR_KAGGLE_USERNAME/voicesep-gpu-training-01
```

Training stops after approximately six hours at an optimizer-step boundary and writes `voicesep-checkpoints/`. A final validation pass or checkpoint save can extend that time. Checkpoints also save every 500 steps; save notebook outputs before ending a session. Abrupt provider termination can lose work since the last saved output. Check actual remaining quota before submitting another GPU job.

## Resume and retrieve

Read `run-summary.json`. If `resume_required` is true, generate a new notebook with the previous completed output attached:

```sh
python scripts/kaggle_jobs.py --owner YOUR_KAGGLE_USERNAME \
  --run-number 2 --resume-kernel YOUR_KAGGLE_USERNAME/voicesep-gpu-training-01
.tools/kaggle/bin/kaggle kernels push -p dist/kaggle/train --accelerator NvidiaTeslaT4
```

Use a new run number each time and the immediately preceding training output. Keep source code, corpus, benchmark and training configuration unchanged. The resume job restores the model, optimizer, scheduler, random states and mixture index; it rejects changed configurations or manifests. Copying source into the private dataset again is unnecessary when its contents have not changed.

Calibration and held-out evaluation run only when training reaches its configured step limit or validation early stopping. If the provider stops before evaluation finishes, retain the trained checkpoint and run those commands in a separate session using the same frozen corpus. Evaluation reports numeric targets honestly; release approval also requires human listening, and WER/CER remains separate work.

Download the saved output:

```sh
.tools/kaggle/bin/kaggle kernels output YOUR_KAGGLE_USERNAME/voicesep-gpu-training-02 -p runs/kaggle-02
```

Inspect the evaluation before installing `voicesep-checkpoints/best-calibrated.pt` locally. The presence of a checkpoint alone does not establish the four/five-speaker quality gates. Keep training audio and checkpoints out of Git; publish permitted model assets separately only after evaluation and license review.

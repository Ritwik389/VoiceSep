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

The generated source dataset and all notebooks are private. The archive contains code, configuration and documentation; it excludes recordings, weights and credentials. Kaggle expands uploaded ZIPs into a nested archive folder, so generation adds `voicesep-source-manifest.json` alongside the ZIP. Each job finds that source root and verifies the pinned manifest and required source files; intact ZIP inputs are also supported. Repository dotfiles are not required by the runtime checks. Upload the entire generated source directory. Generation alone does not submit a job. If the private source dataset already exists, use `kaggle datasets version -p dist/kaggle/source -m "Update VoiceSep source"` instead of `datasets create`, then regenerate all pending stages against the new archive. Keep one source version consistent throughout a training sequence.

Open `https://www.kaggle.com/code/YOUR_KAGGLE_USERNAME/voicesep-data-preparation`, choose Edit on the latest version, enable the checkbox for `HF_TOKEN` under Add-ons → Secrets, and use **Save Version → Save & Run All**, retaining CPU and Internet settings. Save outputs. It is CPU only. The tested CLI-started session could not retrieve the configured secret; start this secret-dependent stage through the UI. The [official CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#using-secrets-in-kernels) confirms that secret setup is UI-only. Do not put a token in notebook code or a private source dataset as a workaround. Monitor the UI-started run with:

```sh
.tools/kaggle/bin/kaggle kernels status YOUR_KAGGLE_USERNAME/voicesep-data-preparation
```

The CPU job selects a bounded corpus, resamples to 8 kHz, splits original speakers, and freezes validation/test recipes. Defaults target up to 30 English and 20 Hindi hours; `preparation.json` records actual retained hours. Access failure or insufficient speakers stops preparation. Do not run a GPU stage before preparation succeeds and its output is saved. Do not make outputs containing publisher-controlled data public.

The original version-4 run stopped on a network timeout while streaming LibriSpeech directly into the archive reader. The replacement first downloads a resumable, checksum-verified archive, then extracts it locally. If retries are exhausted, retain `voicesep-corpus/downloads/*.part` along with any indexed corpus files. Copying that saved output back to the same writable corpus directory enables a later session to resume. Version 4 did not cache compressed bytes, so its partial FLAC output cannot resume that old network stream; the archived Kaggle output remains available.

## Continue through one training session automatically

After starting preparation through the UI, the local controller can wait for its saved output, validate the corpus, submit the T4 benchmark, and then submit **one** training session. It uses the existing generated jobs; do not regenerate or replace that source snapshot while the controller runs.

```sh
.tools/kaggle/bin/python scripts/run_kaggle_workflow.py --owner YOUR_KAGGLE_USERNAME
```

The controller checks disjoint speakers and frozen-manifest hashes before allocating a GPU, requires a successful 100-step benchmark within 16 GB, and checks free quota before each submission. It pins the exact completed notebook versions as inputs. Benchmark execution is capped at one hour; the training loop has its existing six-hour budget within an eight-hour session cap. Quota reservations allow conservatively for the two allocated T4s. It never enables billing, repeats a submission after an ambiguous response, or automatically launches a second training session.

Progress is written atomically to `runs/kaggle-workflow/status.json`. Selected reports and checkpoints download to that directory when stages complete; audio stays on Kaggle. The CPU preparation session must complete before its output can be validated. A failed or canceled stage stops the controller. Restart the same command with the same jobs/output directory after resolving a recoverable issue; a lock prevents duplicate local controllers. A completed run does not resubmit itself.

The controller runs on this computer, so it needs the computer running and connected to advance stages. Already submitted Kaggle jobs run remotely. To prevent subsequent submissions, create `runs/kaggle-workflow/STOP`; cancel a currently running remote job through Kaggle's UI. Remove the stop file before intentionally restarting the controller. The final `resume_required` field distinguishes a saved training session from a finished model; quality gates and listening review still apply.

When replacing a failed preparation notebook with corrected source, preserve the failed controller output and start a fresh controller against the new generated jobs. `--after-preparation-version N` waits for a UI-saved run newer than the corrected no-run draft N; it does not pin the failed run or execute the draft without its secret. Use the actual draft version returned by Kaggle:

```sh
.tools/kaggle/bin/python scripts/run_kaggle_workflow.py --owner YOUR_KAGGLE_USERNAME \
  --jobs dist/kaggle-retry --output runs/kaggle-workflow --after-preparation-version N
```

## Benchmark, then train

These manual commands are an alternative to the controller; do not use both for the same stages.

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

"""Generate readable, credential-free notebooks from the shared CLI workflow."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def write(name, setup, storage):
    cells = [markdown("""# VoiceSep · free GPU training

Train a five-slot English/Hindi separator from a verified two-speaker TF-Locoformer backbone. **No five-speaker quality is claimed until held-out evaluation and listening review pass.**

Select a **free GPU** runtime. This notebook does not purchase compute. Interrupted sessions resume from persistent checkpoints. Dataset downloads need disk space; clean-speech datasets are not included in the source archive.
"""), code(setup), code(storage), code("""import os, subprocess, sys
os.chdir(PROJECT)
def cli(*args):
    subprocess.run([sys.executable, '-m', 'voicesep.cli', *map(str,args)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[dev]'], check=True)
import torch
assert torch.cuda.is_available(), 'Choose a free GPU runtime before training; CPU is for software smoke tests only.'
print(torch.cuda.get_device_name(0))
cli('download-backbone')
"""), markdown("""## Prepare English and Hindi sources

Download LibriSpeech train-clean-100 from https://www.openslr.org/12 and the Hindi IV-R audio plus JSONL speaker metadata from https://github.com/AI4Bharat/IndicVoices-R . Preserve attribution. Use a persistent copy or extract to notebook scratch storage, then keep all source paths stable between resumed sessions. Set the three real paths below.

At least five distinct speakers per language are required in each split. Never mix sources before speaker-disjoint splitting. Exact published benchmark comparisons require the original manifests; the generated bilingual benchmark is custom.
"""), code("""from pathlib import Path
ENGLISH_ROOT = Path('/path/to/LibriSpeech/train-clean-100')
HINDI_ROOT = Path('/path/to/Hindi')
HINDI_MANIFEST = HINDI_ROOT / 'manifest.jsonl'
assert ENGLISH_ROOT.is_dir() and HINDI_MANIFEST.is_file(), 'Set the actual dataset paths above.'
cli('index-sources', ENGLISH_ROOT, HINDI_MANIFEST, HINDI_ROOT, '--output', 'data/sources.jsonl')
cli('split', 'data/sources.jsonl', '--output-dir', 'data')
cli('freeze', 'data/validation.jsonl', 'data/validation-mixtures.jsonl', '--per-condition', 10)
cli('freeze', 'data/test.jsonl', 'data/test-mixtures.jsonl', '--per-condition', 100, '--seed', 98765)
"""), markdown("""## Benchmark memory and throughput first

Runs 100 complete optimizer steps. If four-second crops exhaust memory, rerun this cell with `SEGMENT_SECONDS = 2`, then keep that value for the actual training configuration. A benchmark checkpoint is not a trained-quality release.
"""), code("""SEGMENT_SECONDS = 4
cli('train', '--pretrained', 'models/tflocoformer-libri2mix.pth', '--benchmark',
    '--output', str(PERSIST / 'benchmark'), '--device', 'cuda', '--segment-seconds', SEGMENT_SECONDS)
"""), markdown("""## Train or resume

One checkpoint is saved every 500 completed optimizer steps, plus interruption/final saves. Keep configuration, manifests, and source files unchanged when resuming. The validation score weights language/count groups equally and penalizes counting errors. Free GPU quotas can require multiple sessions.
"""), code("""import yaml
config = yaml.safe_load(Path('configs/train.yaml').read_text())
config['data']['segment_seconds'] = SEGMENT_SECONDS
CONFIG = PERSIST / 'train.yaml'
text = yaml.safe_dump(config, sort_keys=False)
if CONFIG.exists():
    assert CONFIG.read_text() == text, 'Do not change the saved configuration when resuming.'
else:
    CONFIG.write_text(text)
RUN = PERSIST / 'tflocoformer'
resume = RUN / 'last.pt'
args = ['--resume', resume] if resume.exists() else ['--pretrained', 'models/tflocoformer-libri2mix.pth']
cli('train', '--config', CONFIG, '--output', RUN, '--device', 'cuda', *args)
assert (RUN / 'best.pt').is_file(), 'Training must reach a validation step before a best checkpoint exists.'
"""), markdown("""## Calibrate, evaluate, and retrieve

Calibrate on validation mixtures only, then evaluate the held-out test benchmark. Automatic counting is the primary result. Numeric gates require ≥10 dB SI-SDRi and ≥90% count accuracy in every 4/5-speaker English/Hindi/mixed condition, each with ≥100 mixtures. A separate documented listening review is still required. Do not use these test results to tune thresholds.
"""), code("""CALIBRATED = PERSIST / 'best-calibrated.pt'
cli('calibrate', RUN / 'best.pt', 'data/validation.jsonl', 'data/validation-mixtures.jsonl',
    '--destination', CALIBRATED, '--device', 'cuda')
cli('evaluate', CALIBRATED, 'data/test.jsonl', 'data/test-mixtures.jsonl',
    '--output', PERSIST / 'evaluation', '--device', 'cuda')
print('Copy to the local app:', CALIBRATED)
print('Evaluation and listening-review evidence:', PERSIST / 'evaluation')
"""), markdown("""## Controlled Conv-TasNet comparison (optional)

Install `pip install -e '.[baseline]'`; copy the saved training config with `model.architecture: convtasnet`, preserve data/steps/validation and use a different run directory. The code downloads the three-speaker comparison weights and replaces only the source-dependent mask layer. Keep its separate share-alike weight notice; do not relabel it Apache-2.0.

After model quality is established, run the local application. Next research priority is native 16 kHz separation and quality-aware age, vocal-characteristic, and emotion analysis.
""")]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    for index, cell in enumerate(cells):
        cell["id"] = f"voicesep-{index:02}"
    (ROOT / 'notebooks' / name).write_text(json.dumps(notebook, indent=2) + '\n')


write('01_train_colab.ipynb', """from google.colab import files, drive
from pathlib import Path
import zipfile
# First run scripts/package_source.py locally. Upload voicesep-source.zip here.
uploaded = files.upload()
archive = next(Path(name) for name in uploaded if name.endswith('.zip'))
PROJECT = Path('/content/voicesep')
PROJECT.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as source:
    assert all((PROJECT / name).resolve().is_relative_to(PROJECT.resolve()) for name in source.namelist())
    source.extractall(PROJECT)
drive.mount('/content/drive')
""", """PERSIST = Path('/content/drive/MyDrive/VoiceSep')
PERSIST.mkdir(parents=True, exist_ok=True)
(PROJECT / 'data').mkdir(exist_ok=True)
# Keep source audio and manifests at stable paths when restarting runtimes.
print('Checkpoints persist in:', PERSIST)
""")
write('02_train_kaggle.ipynb', """from pathlib import Path
import zipfile
# Add voicesep-source.zip as a private Kaggle notebook input. Enable GPU + Internet.
archive = next(Path('/kaggle/input').rglob('voicesep-source.zip'))
PROJECT = Path('/kaggle/working/voicesep')
PROJECT.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as source:
    assert all((PROJECT / name).resolve().is_relative_to(PROJECT.resolve()) for name in source.namelist())
    source.extractall(PROJECT)
""", """PERSIST = Path('/kaggle/working/voicesep-checkpoints')
PERSIST.mkdir(parents=True, exist_ok=True)
# Kaggle working files are NOT durable until you save a notebook version with outputs.
# Before ending a session, save/download outputs. To resume, attach the saved outputs as input.
RESTORE = None  # Set to the attached prior voicesep-checkpoints directory when resuming.
if RESTORE is not None:
    import shutil
    shutil.copytree(RESTORE, PERSIST, dirs_exist_ok=True)
print('Save notebook outputs before leaving:', PERSIST)
""")

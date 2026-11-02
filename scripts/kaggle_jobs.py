"""Build private Kaggle stages without embedding credentials or publishing model claims.

Generation is offline. Submit the generated directories using the official Kaggle CLI.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def notebook(cells):
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"}},
            "cells": [{"cell_type": "markdown" if kind == "markdown" else "code", "metadata": {},
                       "id": f"voicesep-{index}", "source": text.splitlines(keepends=True),
                       **({"execution_count": None, "outputs": []} if kind == "code" else {})}
                      for index, (kind, text) in enumerate(cells)]}


def build(owner, output=None, resume_kernel=None, run_number=1):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", owner):
        raise ValueError("Supply the Kaggle username, not a URL or API token")
    if run_number < 1:
        raise ValueError("Run number must be positive")
    if resume_kernel and not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+(?:/[0-9]+)?", resume_kernel):
        raise ValueError("Resume kernel must be an owner/notebook-slug reference")
    output = Path(output or ROOT / "dist" / "kaggle")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "package_source.py")], check=True)
    source = ROOT / "dist" / "voicesep-source.zip"
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    bundle_dir = output / "source"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, bundle_dir / source.name)
    # Kaggle expands uploaded ZIPs. Verify extracted files against a pinned sidecar manifest.
    with zipfile.ZipFile(source) as archive:
        file_hashes = {name: hashlib.sha256(archive.read(name)).hexdigest()
                       for name in archive.namelist() if not name.endswith("/")}
    manifest = json.dumps({"archive_sha256": sha, "files": file_hashes}, sort_keys=True)
    (bundle_dir / "voicesep-source-manifest.json").write_text(manifest)
    manifest_sha = hashlib.sha256(manifest.encode()).hexdigest()
    source_ref = f"{owner}/voicesep-research-source"
    (bundle_dir / "dataset-metadata.json").write_text(json.dumps({"id": source_ref,
        "title": "VoiceSep research source", "licenses": [{"name": "other"}],
        "description": "Private VoiceSep source bundle. Apache-2.0; MERL attribution retained. No audio or model binaries. SHA256 " + sha}, indent=2))
    setup = f'''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, zipfile
manifests = list(Path('/kaggle/input').rglob('voicesep-source-manifest.json'))
assert len(manifests) == 1, 'Attach the intended private VoiceSep source dataset.'
manifest_file = manifests[0]
assert hashlib.sha256(manifest_file.read_bytes()).hexdigest() == {manifest_sha!r}, 'Source manifest changed; regenerate job metadata.'
source_manifest = json.loads(manifest_file.read_text())
PROJECT = Path('/kaggle/working/voicesep')
PROJECT.mkdir(exist_ok=True)
archive = manifest_file.parent / 'voicesep-source.zip'
if archive.is_file():
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == {sha!r}, 'Source archive changed.'
    with zipfile.ZipFile(archive) as bundle:
        assert all((PROJECT / name).resolve().is_relative_to(PROJECT.resolve()) for name in bundle.namelist())
        bundle.extractall(PROJECT)
else:
    for name, digest in source_manifest['files'].items():
        original = manifest_file.parent / name
        destination = PROJECT / name
        assert original.resolve().is_relative_to(manifest_file.parent.resolve())
        assert destination.resolve().is_relative_to(PROJECT.resolve())
        assert hashlib.sha256(original.read_bytes()).hexdigest() == digest, f'Source file changed: {{name}}'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[data]'], check=True)
def cli(*args):
    subprocess.run([sys.executable, '-m', 'voicesep.cli', *map(str, args)], check=True)
'''
    mount_data = '''import json
prepared = sorted(Path('/kaggle/input').rglob('preparation.json'))
assert len(prepared) == 1, 'Attach only the completed VoiceSep preparation output as the corpus input.'
CORPUS = prepared[0].parent
assert (CORPUS / 'test-mixtures.jsonl').is_file()
link = PROJECT / 'data'
if not link.exists():
    link.symlink_to(CORPUS, target_is_directory=True)
import torch
assert torch.cuda.is_available(), 'A free Kaggle GPU is required; no CPU/paid fallback is enabled.'
print({'gpu': torch.cuda.get_device_name(0), 'vram_bytes': torch.cuda.get_device_properties(0).total_memory})
cli('download-backbone')
'''
    preparation = '''from kaggle_secrets import UserSecretsClient
try:
    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
except Exception:
    raise RuntimeError('Approve the IndicVoices-R access request, add a read-only HF_TOKEN to Kaggle Secrets, and enable it for this notebook. Never paste a token into source or output.') from None
cli('bootstrap-data', '--destination', '/kaggle/working/voicesep-corpus', '--english-hours', 30, '--hindi-hours', 20)
print('CPU preparation complete; no model training or quality claim has been made.')
'''
    benchmark = '''PERSIST = Path('/kaggle/working/voicesep-benchmark')
PERSIST.mkdir(exist_ok=True)
selected = None
for seconds in (4, 2):
    destination = PERSIST / f'{seconds}s'
    try:
        cli('train', '--pretrained', 'models/tflocoformer-libri2mix.pth', '--output', destination,
            '--device', 'cuda', '--benchmark', '--segment-seconds', seconds)
        report = json.loads((destination / 'throughput.json').read_text())
        selected = {'segment_seconds': seconds, 'benchmark': report}
        break
    except subprocess.CalledProcessError:
        # Each CLI run is a separate process, releasing failed CUDA allocations before retry.
        if seconds == 2:
            raise
assert selected is not None
(PERSIST / 'benchmark-selection.json').write_text(json.dumps(selected, indent=2))
print(selected)
print('Benchmark only: a separate fine-tuning job is required.')
'''
    training = '''import shutil, yaml
PERSIST = Path('/kaggle/working/voicesep-checkpoints')
PERSIST.mkdir(exist_ok=True)
benchmark_files = sorted(Path('/kaggle/input').rglob('benchmark-selection.json'))
assert len(benchmark_files) == 1, 'Attach the successful VoiceSep benchmark output.'
choice = json.loads(benchmark_files[0].read_text())
config = yaml.safe_load(Path('configs/train.yaml').read_text())
config['data']['segment_seconds'] = choice['segment_seconds']
CONFIG = PERSIST / 'train.yaml'
text = yaml.safe_dump(config, sort_keys=False)
previous = sorted(Path('/kaggle/input').rglob('last.pt'))
# Ignore benchmark weights; only resume checkpoints accompanied by the saved training config.
previous = [path for path in previous if (path.parent.parent / 'train.yaml').is_file()]
assert len(previous) <= 1, 'Attach only one previous VoiceSep training output.'
if previous:
    saved_config = previous[0].parent.parent / 'train.yaml'
    assert saved_config.read_text() == text, 'Resume config differs from this job.'
    shutil.copytree(previous[0].parent, PERSIST / 'tflocoformer', dirs_exist_ok=True)
CONFIG.write_text(text)
RUN = PERSIST / 'tflocoformer'
checkpoint = RUN / 'last.pt'
args = ['--resume', checkpoint] if checkpoint.exists() else ['--pretrained', 'models/tflocoformer-libri2mix.pth']
cli('train', '--config', CONFIG, '--output', RUN, '--device', 'cuda', '--max-seconds', 21600, *args)
state = torch.load(RUN / 'last.pt', map_location='cpu', weights_only=True)
finished = state['step'] >= config['training']['steps'] or state.get('stale_evaluations', 0) >= config['training']['early_stopping_evaluations']
summary = {'step': state['step'], 'resume_required': not finished, 'release_ready': False}
(PERSIST / 'run-summary.json').write_text(json.dumps(summary, indent=2))
print(summary)
if finished:
    assert (RUN / 'best.pt').is_file()
    cli('calibrate', RUN / 'best.pt', 'data/validation.jsonl', 'data/validation-mixtures.jsonl',
        '--destination', PERSIST / 'best-calibrated.pt', '--device', 'cuda')
    cli('evaluate', PERSIST / 'best-calibrated.pt', 'data/test.jsonl', 'data/test-mixtures.jsonl',
        '--output', PERSIST / 'evaluation', '--device', 'cuda')
else:
    print('Save this output and attach it to the next private training job to resume. Held-out test evaluation has not run.')
'''
    stages = {
        "prepare": ("VoiceSep data preparation", False, [], [setup, preparation]),
        "benchmark": ("VoiceSep GPU benchmark", True, [f"{owner}/voicesep-data-preparation"], [setup, mount_data, benchmark]),
        "train": (f"VoiceSep GPU training {run_number:02}", True,
                  [f"{owner}/voicesep-data-preparation", f"{owner}/voicesep-gpu-benchmark"] + ([resume_kernel] if resume_kernel else []),
                  [setup, mount_data, training]),
    }
    for stage, (title, gpu, inputs, cells) in stages.items():
        directory = output / stage
        directory.mkdir(parents=True, exist_ok=True)
        slug = title.lower().replace(" ", "-")
        note = "Private research job. No verified five-speaker result is claimed. "
        if stage == "prepare":
            note += "CPU only. Configure a user-approved read-only HF_TOKEN through Kaggle Secrets before running."
        else:
            note += "Uses the free T4 pool. One T4 is used by the model; the two devices do not provide one shared memory pool."
        nb = notebook([("markdown", f"# {title}\n\n{note}\n")] + [("code", cell) for cell in cells])
        (directory / "job.ipynb").write_text(json.dumps(nb, indent=2))
        metadata = {"id": f"{owner}/{slug}", "title": title, "code_file": "job.ipynb", "language": "python",
                    "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
                    "dataset_sources": [source_ref], "kernel_sources": inputs, "competition_sources": [], "model_sources": []}
        if gpu:
            metadata["machine_shape"] = "NvidiaTeslaT4"
        (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"source_dataset": source_ref, "source_sha256": sha, "directory": str(output),
            "stages": [str(output / name) for name in stages], "submitted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True, help="Your Kaggle username (can differ from GitHub)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-kernel", help="Previous completed private training kernel ref")
    parser.add_argument("--run-number", type=int, default=1, help="Use a new run number when resuming to avoid self-dependencies")
    args = parser.parse_args()
    print(json.dumps(build(args.owner, args.output, args.resume_kernel, args.run_number), indent=2))

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pytest

from voicesep.research.bootstrap import _write_source
from voicesep.research.data import MixtureFactory, write_jsonl


def test_frozen_mixtures_survive_moving_to_a_kaggle_input_mount(tmp_path):
    original = tmp_path / "cpu-output"
    entries = []
    time = np.arange(36000) / 8000
    for language in ("en", "hi"):
        for index in range(5):
            entry = _write_source(original, language, "fixture", f"{language}-{index}", index,
                                  np.sin(time * 2 * np.pi * (110 + index * 150)) * .05, 8000)
            entries.append(entry)
    write_jsonl(original / "sources.jsonl", entries)
    factory = MixtureFactory(original / "sources.jsonl", segment_seconds=.1)
    wave, refs, recipe = factory.draw(2, "mixed", 5)
    assert all(not Path(source["path"]).is_absolute() for source in recipe["sources"])
    destination = tmp_path / "gpu-input"
    shutil.copytree(original, destination)
    shutil.rmtree(original)
    migrated = MixtureFactory(destination / "sources.jsonl", segment_seconds=.1)
    other_wave, other_refs = migrated.render(recipe)
    assert np.array_equal(wave, other_wave) and np.array_equal(refs, other_refs)


def test_bootstrap_rejects_silent_and_nonfinite_sources(tmp_path):
    assert _write_source(tmp_path, "hi", "fixture", "a", "silence", np.zeros(36000), 8000) is None
    assert _write_source(tmp_path, "hi", "fixture", "a", "nan", np.full(36000, np.nan), 8000) is None


def test_generated_kaggle_jobs_are_private_compilable_and_cpu_first(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_jobs.py"
    spec = importlib.util.spec_from_file_location("kaggle_jobs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / "dist" / "voicesep-source.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('pyproject.toml', '# software fixture, not a training source\n')
        bundle.writestr('src/voicesep/cli.py', '# source integrity fixture\n')
        bundle.writestr('.github/workflows/checks.yml', '# hidden repository file\n')
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    result = module.build("researcher", tmp_path / "jobs", "researcher/voicesep-gpu-training-01", run_number=2)
    assert result["submitted"] is False
    for stage in ("prepare", "benchmark", "train"):
        directory = tmp_path / "jobs" / stage
        metadata = json.loads((directory / "kernel-metadata.json").read_text())
        assert metadata["is_private"] is True
        assert metadata["enable_gpu"] is (stage != "prepare")
        assert "HF_TOKEN" not in metadata
        notebook = json.loads((directory / "job.ipynb").read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
    training = json.loads((tmp_path / "jobs/train/kernel-metadata.json").read_text())
    assert training["id"] == "researcher/voicesep-gpu-training-02"
    assert "researcher/voicesep-gpu-training-01" in training["kernel_sources"]
    training_code=''.join(''.join(cell['source']) for cell in
        json.loads((tmp_path/'jobs/train/job.ipynb').read_text())['cells'] if cell['cell_type']=='code')
    assert 'assert len(previous) == 1' in training_code
    with pytest.raises(ValueError):
        module.build("not a username", tmp_path / "invalid")

    # Execute the actual generated loader against both Kaggle's expanded layout and an intact ZIP.
    nb=json.loads((tmp_path/'jobs/prepare/job.ipynb').read_text())
    setup=''.join(nb['cells'][1]['source'])
    input_dir=tmp_path/'input'
    input_dir.mkdir()
    sidecar=tmp_path/'jobs/source/voicesep-source-manifest.json'
    shutil.copyfile(sidecar,input_dir/sidecar.name)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(input_dir)
    # GitHub workflow files are not required by the notebook runtime.
    shutil.rmtree(input_dir/'.github')
    cwd=Path.cwd()
    try:
        for layout in ('expanded','wrapped','zip'):
            destination=tmp_path/f'workspace-{layout}'
            if layout=='wrapped':
                wrapper=input_dir/'voicesep-source'
                wrapper.mkdir()
                for name in ('pyproject.toml','src'):
                    shutil.move(str(input_dir/name),wrapper/name)
            if layout=='zip':
                shutil.copyfile(archive,input_dir/archive.name)
            cell=setup.replace("'/kaggle/input'",repr(str(input_dir)))
            cell=cell.replace("'/kaggle/working/voicesep'",repr(str(destination)))
            exec(compile(cell,'generated-kaggle-setup','exec'),{})
            assert (destination/'src/voicesep/cli.py').read_text()=='# source integrity fixture\n'
        (input_dir/archive.name).unlink()
        next(input_dir.rglob('cli.py')).write_text('# changed file\n')
        with pytest.raises(AssertionError,match='Source file changed'):
            exec(compile(cell.replace(str(destination),str(tmp_path/'tampered')),'changed-source','exec'),{})
    finally:
        import os
        os.chdir(cwd)

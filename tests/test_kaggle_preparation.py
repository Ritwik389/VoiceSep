import ast
import importlib.util
import json
from pathlib import Path
import shutil

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
    archive.write_bytes(b"software fixture, not a training source")
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
    with pytest.raises(ValueError):
        module.build("not a username", tmp_path / "invalid")

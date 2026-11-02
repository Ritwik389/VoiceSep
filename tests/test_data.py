from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from voicesep.research.data import MixtureFactory, write_jsonl, split_sources, read_jsonl, assert_disjoint


def make_sources(tmp_path):
    entries = []
    time = np.arange(1600) / 8000
    for language in ("en", "hi"):
        for index in range(6):
            path = tmp_path / f"{language}-{index}.wav"
            sf.write(path, np.sin(time * 2 * np.pi * (100 + index * 130)) * .1, 8000)
            entries.append({"path": str(path), "speaker_id": f"{language}-{index}", "language": language})
    manifest = tmp_path / "sources.jsonl"
    write_jsonl(manifest, entries)
    return manifest


def test_dynamic_mixtures_are_reproducible_and_sum_to_references(tmp_path):
    manifest = make_sources(tmp_path)
    factory = MixtureFactory(manifest, segment_seconds=.1, seed=42)
    a, refs, recipe = factory.draw(19, "mixed", 5)
    b, second_refs, second_recipe = factory.draw(19, "mixed", 5)
    assert np.array_equal(a, b) and np.array_equal(refs, second_refs)
    assert recipe == second_recipe
    assert np.allclose(a, refs.sum(0))
    assert len({s["identity"] for s in recipe["sources"]}) == 5
    assert {s["language"] for s in recipe["sources"]} == {"en", "hi"}
    recipe["source_manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        factory.render(recipe)


def test_speaker_splits_are_disjoint_even_with_multiple_recordings(tmp_path):
    entries = [{"path": f"/{language}/{speaker}/{clip}.wav", "speaker_id": str(speaker),
                "language": language, "corpus": language}
               for language in ("en", "hi") for speaker in range(100) for clip in range(2)]
    split_sources(entries, tmp_path)
    assert_disjoint(*(tmp_path / f"{name}.jsonl" for name in ("train", "validation", "test")))
    train = read_jsonl(tmp_path / "train.jsonl")
    write_jsonl(tmp_path / "leaked.jsonl", train[:1])
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(tmp_path / "train.jsonl", tmp_path / "leaked.jsonl")

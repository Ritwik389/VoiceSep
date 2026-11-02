from pathlib import Path
from functools import lru_cache
from collections import defaultdict
import hashlib
import json
import os

import numpy as np
from scipy.signal import fftconvolve

from ..audio import load_source


def read_jsonl(path):
    path = Path(path)
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for entry in entries:
        if "path" in entry:
            candidate = Path(entry["path"]).expanduser()
            entry["path"] = str(candidate if candidate.is_absolute() else (path.parent / candidate).resolve())
    return entries


def write_jsonl(path, entries):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries))


def split_sources(entries, output_dir, seed=42):
    """Partition identities before mixing. Each corpus-qualified identity has exactly one split."""
    partitions = {"train": [], "validation": [], "test": []}
    for entry in entries:
        if entry.get("language") not in {"en", "hi"} or not entry.get("speaker_id"):
            raise ValueError("Every source requires language en/hi and a stable speaker_id")
        key = f"{entry.get('corpus', 'custom')}:{entry['speaker_id']}"
        number = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) % 100
        name = "train" if number < 80 else "validation" if number < 90 else "test"
        partitions[name].append(dict(entry, identity=key))
    for name, rows in partitions.items():
        write_jsonl(Path(output_dir) / f"{name}.jsonl", sorted(rows, key=lambda e: (e["identity"], e["path"])))
    return {name: {"sources": len(rows), "speakers": len({r["identity"] for r in rows})}
            for name, rows in partitions.items()}


def assert_disjoint(*manifests):
    seen = set()
    for manifest in manifests:
        rows = read_jsonl(manifest)
        identities = {r.get("identity", f"{r.get('corpus', 'custom')}:{r['speaker_id']}") for r in rows}
        if seen & identities:
            raise ValueError(f"Speaker leakage across splits: {sorted(seen & identities)[:5]}")
        seen |= identities


class MixtureFactory:
    def __init__(self, manifest, sample_rate=8000, segment_seconds=4, slots=5, seed=42,
                 partial_overlap_probability=0, noise_probability=0, reverb_probability=0):
        self.manifest = Path(manifest).resolve()
        self.entries = read_jsonl(manifest)
        self.sample_rate, self.samples, self.slots = sample_rate, int(segment_seconds * sample_rate), slots
        self.seed = seed
        self.partial_probability = partial_overlap_probability
        self.noise_probability, self.reverb_probability = noise_probability, reverb_probability
        self.groups = defaultdict(lambda: defaultdict(list))
        for entry in self.entries:
            if not Path(entry["path"]).is_file():
                raise ValueError(f"Missing source: {entry['path']}")
            identity = entry.get("identity", f"{entry.get('corpus', 'custom')}:{entry['speaker_id']}")
            self.groups[entry["language"]][identity].append(entry)
        for language in ("en", "hi"):
            if len(self.groups[language]) < slots:
                raise ValueError(f"Need at least {slots} distinct {language} speakers in {manifest}")
        self.manifest_sha256 = hashlib.sha256(self.manifest.read_bytes()).hexdigest()

    @lru_cache(maxsize=16)
    def source(self, path):
        path = Path(path)
        return load_source(path if path.is_absolute() else self.manifest.parent / path, self.sample_rate)

    def recipe(self, index, condition=None, count=None):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        condition = condition or str(rng.choice(["en", "hi", "mixed"]))
        count = count or int(rng.integers(2 if condition == "mixed" else 1, self.slots + 1))
        if count > self.slots or count < 1:
            raise ValueError("Source count exceeds this factory's slot capacity")
        if condition == "mixed" and count < 2:
            raise ValueError("Mixed-language mixtures require at least two speakers")
        languages = [condition] * count if condition in {"en", "hi"} else ["en", "hi"] + rng.choice(["en", "hi"], count - 2).tolist()
        selected, used = [], set()
        for language in languages:
            speakers = sorted(set(self.groups[language]) - used)
            identity = str(rng.choice(speakers))
            used.add(identity)
            rows = self.groups[language][identity]
            row = rows[int(rng.integers(len(rows)))]
            source = self.source(row["path"])
            if len(source) == 0 or np.mean(source ** 2) < 1e-10:
                raise ValueError(f"Empty/silent clean source: {row['path']}")
            crop = int(rng.integers(max(1, len(source) - self.samples + 1)))
            offset = int(rng.integers(0, max(1, self.samples // 2))) if rng.random() < self.partial_probability else 0
            portable_path = os.path.relpath(row["path"], self.manifest.parent)
            selected.append({"path": portable_path, "identity": identity, "language": language,
                             "text": row.get("text", ""), "crop": crop, "offset": offset,
                             "gain_db": float(rng.uniform(-5, 0))})
        return {"id": f"mix-{self.seed}-{index}", "sources": selected, "condition": condition,
                "speaker_count": count, "samples": self.samples, "sample_rate": self.sample_rate,
                "noise_seed": int(rng.integers(2**31)) if rng.random() < self.noise_probability else None,
                "rir_seed": int(rng.integers(2**31)) if rng.random() < self.reverb_probability else None,
                "source_manifest_sha256": self.manifest_sha256}

    def render(self, recipe):
        if recipe.get("source_manifest_sha256") != self.manifest_sha256:
            raise ValueError("Frozen recipe source-manifest hash does not match")
        n = recipe["samples"]
        references = np.zeros((self.slots, n), np.float32)
        for index, source in enumerate(recipe["sources"]):
            wave = self.source(source["path"])[source["crop"]:source["crop"] + n - source["offset"]].copy()
            rms = max(float(np.sqrt(np.mean(wave ** 2))), 1e-6)
            wave *= (0.06 * 10 ** (source["gain_db"] / 20)) / rms
            if recipe.get("rir_seed") is not None:
                rng = np.random.default_rng(recipe["rir_seed"] + index)
                t = np.arange(int(0.15 * self.sample_rate)) / self.sample_rate
                impulse = rng.normal(size=len(t)) * np.exp(-t * 30) * 0.02
                impulse[0] = 1
                wave = fftconvolve(wave, impulse)[:len(wave)].astype(np.float32)
            references[index, source["offset"]:source["offset"] + len(wave)] = wave
        mixture = references.sum(0)
        if recipe.get("noise_seed") is not None:
            noise = np.random.default_rng(recipe["noise_seed"]).normal(size=n).astype(np.float32)
            mixture += noise * max(float(np.std(mixture)), 1e-5) * 0.03
        scale = max(1.0, float(np.max(np.abs(mixture))) / 0.95)
        return mixture / scale, references / scale

    def draw(self, index, condition=None, count=None):
        recipe = self.recipe(index, condition, count)
        return (*self.render(recipe), recipe)


def freeze_benchmark(manifest, destination, per_condition=100, counts=(2, 3, 4, 5), seconds=4, seed=12345):
    factory = MixtureFactory(manifest, segment_seconds=seconds, slots=max(counts), seed=seed)
    recipes = []
    for condition in ("en", "hi", "mixed"):
        for count in counts:
            for _ in range(per_condition):
                recipes.append(factory.recipe(len(recipes), condition, count))
    write_jsonl(destination, recipes)
    return {"mixtures": len(recipes), "path": str(destination), "seed": seed}

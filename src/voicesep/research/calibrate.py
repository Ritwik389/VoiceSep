from dataclasses import asdict
from itertools import product
from pathlib import Path
from collections import defaultdict
import hashlib
import json

import numpy as np
import torch

from ..models.separator import load_checkpoint
from ..inference import Calibration, detect_streams
from .data import MixtureFactory, read_jsonl
from .evaluate import score_mixture


@torch.inference_mode()
def calibrate(checkpoint, source_manifest, mixtures, destination, device="cpu", limit=120):
    model, metadata = load_checkpoint(checkpoint, device)
    factory = MixtureFactory(source_manifest, sample_rate=model.config.sample_rate, slots=model.config.slots)
    expected = metadata.get("data_fingerprints", {}).get("validation")
    if expected is None or expected != factory.manifest_sha256:
        raise ValueError("Calibration requires the exact validation source manifest recorded during training")
    recipes = read_jsonl(mixtures)
    # Round-robin groups so a limit never silently drops Hindi or five-speaker validation.
    groups = defaultdict(list)
    for recipe in recipes:
        groups[(recipe["condition"], recipe["speaker_count"])].append(recipe)
    ordered = []
    for index in range(max(map(len, groups.values()), default=0)):
        ordered.extend(items[index] for _, items in sorted(groups.items()) if index < len(items))
    cached = []
    for recipe in ordered[:limit]:
        mix, refs = factory.render(recipe)
        prediction = model(torch.from_numpy(mix).unsqueeze(0).to(device))
        cached.append((recipe, mix, refs, prediction["waveforms"][0].cpu().numpy(),
                       prediction["activity_logits"][0].sigmoid().cpu().numpy()))
    if not cached:
        raise ValueError("Validation mixtures are empty")
    best, best_score = None, -float("inf")
    for activity, fraction, energy, duplicate in product((0.3, 0.5, 0.7), (0.02, 0.05, 0.1),
                                                        (0.0001, 0.001), (0.98, 0.995)):
        candidate = Calibration(activity, fraction, energy, duplicate)
        scores = defaultdict(list)
        for recipe, mix, refs, waves, probs in cached:
            selected, _ = detect_streams(waves, probs, mix, candidate)
            score = score_mixture(waves[selected], refs, mix, compute_stoi=False)
            scores[(recipe["condition"], recipe["speaker_count"])].append(
                score["si_sdri"] - 5 * abs(score["reference_count"] - score["predicted_count"]))
        value = float(np.mean([np.mean(items) for items in scores.values()]))
        if value > best_score:
            best, best_score = candidate, value
    metadata["calibration"] = asdict(best)
    metadata["calibration_provenance"] = {
        "mixtures_sha256": hashlib.sha256(Path(mixtures).read_bytes()).hexdigest(),
        "source_manifest_sha256": factory.manifest_sha256,
        "parent_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "mixtures": len(cached), "balanced_score": best_score, "split": "validation"}
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save(metadata, destination)
    return {"calibration": metadata["calibration"], **metadata["calibration_provenance"]}

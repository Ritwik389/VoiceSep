from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..inference import SeparationEngine, Calibration
from ..losses import si_sdr
from .data import MixtureFactory, read_jsonl


def score_mixture(predictions, references, mixture, sample_rate=8000, compute_stoi=True):
    """Unmatched references score -30 dB. False-positive streams remain explicit penalties."""
    active_refs = references[np.mean(references ** 2, axis=-1) > 1e-9]
    n, m = len(active_refs), len(predictions)
    if not n:
        return {"reference_count": 0, "predicted_count": m, "count_correct": m == 0,
                "si_sdr": 0., "si_sdri": 0., "worst_si_sdri": 0., "false_positives": m,
                "missing": 0, "stoi": None, "leakage": None}
    r = torch.from_numpy(active_refs)
    baseline = si_sdr(torch.from_numpy(np.broadcast_to(mixture, active_refs.shape).copy()), r).numpy()
    matched_scores = np.full(n, -30., dtype=float)
    matched_stoi, leakage = [], []
    if m:
        p = torch.from_numpy(predictions)
        matrix = si_sdr(p[:, None, :], r[None, :, :]).numpy()
        rows, cols = linear_sum_assignment(-matrix)
        for row, col in zip(rows, cols):
            matched_scores[col] = matrix[row, col]
            if compute_stoi:
                try:
                    from pystoi import stoi
                    matched_stoi.append(float(stoi(active_refs[col], predictions[row], sample_rate, extended=False)))
                except ImportError:
                    pass
            dots = np.abs(active_refs @ predictions[row]) / np.maximum(
                np.linalg.norm(active_refs, axis=-1) * np.linalg.norm(predictions[row]), 1e-10)
            leakage.append(float(np.max(np.delete(dots, col), initial=0)))
    false_positives = max(0, m - n)
    # Each additional output contributes a -30 dB score, preventing optimistic over-separation.
    extra = np.full(false_positives, -30.)
    improvements = matched_scores - baseline
    return {"reference_count": n, "predicted_count": m, "count_correct": n == m,
            "si_sdr": float(np.mean(np.r_[matched_scores, extra])),
            "si_sdri": float(np.mean(np.r_[improvements, extra])),
            "worst_si_sdri": float(np.min(improvements)), "false_positives": false_positives,
            "missing": max(0, n - m), "stoi": float(np.mean(matched_stoi)) if matched_stoi else None,
            "leakage": float(np.mean(leakage)) if leakage else None}


def summarize(rows, min_samples=100):
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['condition']}/{row['reference_count']}"] .append(row)
    result = {}
    for name, items in sorted(groups.items()):
        result[name] = {"mixtures": len(items), "mean_si_sdri": float(np.mean([r["si_sdri"] for r in items])),
                        "mean_si_sdr": float(np.mean([r["si_sdr"] for r in items])),
                        "median_si_sdri": float(np.median([r["si_sdri"] for r in items])),
                        "mean_worst_si_sdri": float(np.mean([r["worst_si_sdri"] for r in items])),
                        "exact_count_accuracy": float(np.mean([r["count_correct"] for r in items])),
                        "all_sources_improved_fraction": float(np.mean([r["worst_si_sdri"] > 0 for r in items])),
                        "mean_leakage": float(np.mean([r["leakage"] for r in items if r["leakage"] is not None]))
                        if any(r["leakage"] is not None for r in items) else None,
                        "mean_stoi": float(np.mean([r["stoi"] for r in items if r["stoi"] is not None]))
                        if any(r["stoi"] is not None for r in items) else None}
    required = [f"{lang}/{n}" for lang in ("en", "hi", "mixed") for n in (4, 5)]
    numeric_gate = all(name in result and result[name]["mixtures"] >= min_samples
                       and result[name]["mean_si_sdri"] >= 10
                       and result[name]["exact_count_accuracy"] >= 0.9 for name in required)
    confusion = defaultdict(int)
    for row in rows:
        confusion[f"{row['reference_count']}->{row['predicted_count']}"] += 1
    return {"groups": result, "numeric_gate_passed": numeric_gate,
            "release_ready": False, "listening_review": "pending", "count_confusion": dict(confusion),
            "scoring": "Hungarian assignment; unmatched references and extra outputs incur -30 dB penalties"}


def evaluate_checkpoint(checkpoint, source_manifest, mixtures, output_dir, device="cpu", oracle=False):
    engine = SeparationEngine(checkpoint, device)
    factory = MixtureFactory(source_manifest, sample_rate=engine.sample_rate, slots=engine.model.config.slots)
    fingerprints = engine.metadata.get("data_fingerprints", {})
    if factory.manifest_sha256 in {fingerprints.get("train"), fingerprints.get("validation")}:
        raise ValueError("Final evaluation requires a held-out test source manifest, not training or validation data")
    recipes = read_jsonl(mixtures)
    rows = []
    duration, processing = 0., 0.
    for recipe in recipes:
        mixture, references = factory.render(recipe)
        started = time.perf_counter()
        prediction = engine.separate(mixture, manual_count=recipe["speaker_count"] if oracle else None)
        processing += time.perf_counter() - started
        duration += len(mixture) / engine.sample_rate
        score = score_mixture(prediction["waveforms"], references, mixture, engine.sample_rate)
        rows.append({"id": recipe["id"], "condition": recipe["condition"], **score})
    report = summarize(rows)
    report.update({"checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                   "mixtures_sha256": hashlib.sha256(Path(mixtures).read_bytes()).hexdigest(),
                   "automatic_count": not oracle, "sample_rate": engine.sample_rate,
                   "real_time_factor": processing / max(duration, 1e-8), "results": rows,
                   "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
                   "not_measured": ["WER/CER without reference-aligned ASR run", "human listening review"]})
    if oracle:
        report["numeric_gate_passed"] = False
        report["diagnostic_only"] = True
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    lines = ["# VoiceSep evaluation", "", f"Automatic counting: {not oracle}",
             f"Numeric gates passed: {report['numeric_gate_passed']}", "Release ready: False (listening review required)", "",
             "| Condition | Mixtures | Mean SI-SDRi | Count accuracy |", "|---|---:|---:|---:|"]
    for name, item in report["groups"].items():
        lines.append(f"| {name} | {item['mixtures']} | {item['mean_si_sdri']:.2f} dB | {item['exact_count_accuracy']:.1%} |")
    (output_dir / "evaluation.md").write_text("\n".join(lines) + "\n")
    return report

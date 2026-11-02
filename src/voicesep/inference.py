from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .models.separator import load_checkpoint
from .audio import intervals_from_mask


@dataclass
class Calibration:
    activity_threshold: float = 0.5
    min_active_fraction: float = 0.05
    min_energy_ratio: float = 0.001
    duplicate_threshold: float = 0.98


def detect_streams(waves, probabilities, mixture, calibration: Calibration, manual_count=None, presence_window=None):
    energies = np.mean(waves ** 2, axis=-1)
    mixture_energy = max(float(np.mean(mixture ** 2)), 1e-10)
    fractions = np.mean(probabilities >= calibration.activity_threshold, axis=-1)
    if presence_window and probabilities.shape[-1] > presence_window:
        # Existence is a local-window decision; quiet/brief speakers must not disappear on long files.
        active = (probabilities >= calibration.activity_threshold).astype(np.float64)
        cumulative = np.pad(np.cumsum(active, axis=-1), ((0, 0), (1, 0)))
        fractions = ((cumulative[:, presence_window:] - cumulative[:, :-presence_window]) / presence_window).max(-1)
        squared = np.pad(np.cumsum(waves.astype(np.float64) ** 2, axis=-1), ((0, 0), (1, 0)))
        energies = ((squared[:, presence_window:] - squared[:, :-presence_window]) / presence_window).max(-1)
    scores = fractions * energies / mixture_energy
    ranked = np.argsort(-scores).tolist()
    if manual_count is not None:
        if not 1 <= manual_count <= waves.shape[0]:
            raise ValueError(f"Manual count must be between 1 and {waves.shape[0]}")
        return sorted(ranked[:manual_count]), []
    accepted, duplicates = [], []
    for i in ranked:
        if fractions[i] < calibration.min_active_fraction or energies[i] / mixture_energy < calibration.min_energy_ratio:
            continue
        is_duplicate = False
        for j in accepted:
            correlation = abs(float(np.dot(waves[i], waves[j]))) / max(
                float(np.linalg.norm(waves[i]) * np.linalg.norm(waves[j])), 1e-10)
            if correlation >= calibration.duplicate_threshold:
                duplicates.append(i)
                is_duplicate = True
                break
        if not is_duplicate:
            accepted.append(i)
    return sorted(accepted), duplicates


def align_streams(previous, current, overlap, old_embeddings=None, new_embeddings=None):
    """Map current local slots to persistent slots. No reference audio is used."""
    slots = len(previous)
    costs = np.full((slots, slots), 0.5, np.float64)
    if overlap:
        a, b = previous[:, -overlap:], current[:, :overlap]
        norms = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
        usable = norms > 1e-8
        correlations = np.abs(a @ b.T) / np.maximum(norms, 1e-8)
        costs[usable] = (1 - correlations)[usable]
    if old_embeddings is not None and new_embeddings is not None:
        a = old_embeddings / np.maximum(np.linalg.norm(old_embeddings, axis=-1, keepdims=True), 1e-8)
        b = new_embeddings / np.maximum(np.linalg.norm(new_embeddings, axis=-1, keepdims=True), 1e-8)
        costs = 0.7 * costs + 0.3 * (1 - np.clip(a @ b.T, -1, 1))
    rows, cols = linear_sum_assignment(costs)
    order = np.zeros(slots, dtype=int)
    order[rows] = cols
    return order


class SpeakerEmbedder:
    """Optional frozen ECAPA embeddings; download is enabled only by explicit configuration."""
    def __init__(self, cache_dir):
        from speechbrain.inference.speaker import EncoderClassifier
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(cache_dir), run_opts={"device": "cpu"})

    def __call__(self, waves, sample_rate):
        import torchaudio
        signals = torch.from_numpy(waves)
        signals = torchaudio.functional.resample(signals, sample_rate, 16000)
        with torch.inference_mode():
            return self.encoder.encode_batch(signals).squeeze(1).cpu().numpy()


class SeparationEngine:
    def __init__(self, checkpoint, device="cpu", embedder=None):
        self.model, self.metadata = load_checkpoint(checkpoint, device)
        self.device = device
        self.sample_rate = self.model.config.sample_rate
        self.calibration = Calibration(**self.metadata.get("calibration", {}))
        self.embedder = embedder

    @torch.inference_mode()
    def separate(self, wave, manual_count=None, progress: Callable | None = None,
                 cancelled: Callable | None = None, chunk_seconds=4, overlap_seconds=1):
        progress = progress or (lambda *args: None)
        cancelled = cancelled or (lambda: False)
        sr, slots = self.sample_rate, self.model.config.slots
        if float(np.max(np.abs(wave), initial=0)) < 1e-5:
            return {"waveforms": np.zeros((0, len(wave)), np.float32), "speakers": [],
                    "flags": ["No audible speech signal detected"], "sample_rate": sr}
        window = int(chunk_seconds * sr)
        overlap = int(overlap_seconds * sr)
        if not 0 <= overlap < window:
            raise ValueError("Overlap must be shorter than the chunk")
        hop = window - overlap
        starts = list(range(0, max(1, len(wave) - overlap), hop))
        accumulated = np.zeros((slots, len(wave)), np.float32)
        activity_sum = np.zeros_like(accumulated)
        weights = np.zeros(len(wave), np.float32)
        previous, previous_embeddings = None, None
        for index, start in enumerate(starts):
            if cancelled():
                raise InterruptedError("Separation cancelled")
            segment = wave[start:start + window]
            actual = len(segment)
            padded = np.pad(segment, (0, max(0, window - actual)))
            output = self.model(torch.from_numpy(padded).unsqueeze(0).to(self.device))
            waves = output["waveforms"][0].cpu().numpy()
            probs = torch.nn.functional.interpolate(
                output["activity_logits"].sigmoid(), size=window, mode="linear", align_corners=False)[0].cpu().numpy()
            embeddings = self.embedder(waves, sr) if self.embedder else None
            if embeddings is not None:
                valid = (np.mean(probs >= self.calibration.activity_threshold, axis=-1) >= self.calibration.min_active_fraction)
                valid &= np.mean(waves ** 2, axis=-1) > max(float(np.mean(padded ** 2)) * self.calibration.min_energy_ratio, 1e-10)
                embeddings[~valid] = 0
            if previous is not None:
                order = align_streams(previous, waves, overlap, previous_embeddings, embeddings)
                waves, probs = waves[order], probs[order]
                if embeddings is not None:
                    embeddings = embeddings[order]
            previous = waves
            if embeddings is not None:
                if previous_embeddings is None:
                    previous_embeddings = embeddings
                else:
                    usable = np.linalg.norm(embeddings, axis=-1) > 1e-8
                    previous_embeddings[usable] = .8 * previous_embeddings[usable] + .2 * embeddings[usable]
            blend = np.ones(window, np.float32)
            if index > 0:
                blend[:overlap] = np.linspace(0, 1, overlap, endpoint=False)
            if index < len(starts) - 1:
                blend[-overlap:] = np.linspace(1, 0, overlap, endpoint=False) if overlap else 1
            accumulated[:, start:start + actual] += waves[:, :actual] * blend[:actual]
            activity_sum[:, start:start + actual] += probs[:, :actual] * blend[:actual]
            weights[start:start + actual] += blend[:actual]
            progress("separating", (index + 1) / len(starts))
        accumulated /= weights.clip(1e-6)
        activity_sum /= weights.clip(1e-6)
        accepted, duplicates = detect_streams(accumulated, activity_sum, wave, self.calibration, manual_count,
                                              presence_window=window)
        flags = []
        if duplicates:
            flags.append("Highly similar candidate streams were suppressed; review quiet speakers")
        if len(accepted) == slots:
            flags.append("Model capacity reached: additional speakers cannot be ruled out")
        if not accepted:
            flags.append("No confident speaker streams; try the manual count override")
        if self.embedder is None and len(starts) > 1:
            flags.append("Tracking uses overlap audio only; review identities after long pauses")
        if not self.metadata.get("validation", {}).get("release_ready", False):
            flags.append("Research checkpoint: English/Hindi release gates have not been passed")
        step_samples = max(1, int(sr * 0.04))
        speakers = []
        for number, slot in enumerate(accepted):
            reduced = activity_sum[slot, ::step_samples] >= self.calibration.activity_threshold
            intervals = intervals_from_mask(reduced, step_samples / sr)
            for region in intervals:
                region["end"] = min(region["end"], len(wave) / sr)
            speakers.append({"id": f"speaker-{number + 1}", "name": f"Speaker {chr(65 + number)}",
                             "slot": slot, "activity": intervals,
                             "activity_fraction": round(float(reduced.mean()), 4)})
        return {"waveforms": accumulated[accepted], "speakers": speakers, "flags": flags, "sample_rate": sr}

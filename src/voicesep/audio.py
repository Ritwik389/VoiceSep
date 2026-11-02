from pathlib import Path
from math import gcd
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_audio(path, sample_rate=8000, max_seconds=300):
    """Decode a bounded interval; reject excessive duration instead of silently truncating."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("Audio file does not exist")
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-t", str(max_seconds + 1),
               "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"]
    try:
        result = subprocess.run(command, capture_output=True, timeout=60)
    except FileNotFoundError as exc:
        raise ValueError("FFmpeg is required. Install it before decoding audio.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Audio decoding timed out") from exc
    if result.returncode:
        raise ValueError("Cannot decode this audio file. Try a WAV, FLAC, MP3, or M4A file.")
    wave = np.frombuffer(result.stdout, np.float32).copy()
    if not len(wave):
        raise ValueError("The audio file contains no samples")
    if len(wave) > sample_rate * max_seconds:
        raise ValueError(f"Audio must be at most {max_seconds} seconds")
    if not np.isfinite(wave).all():
        raise ValueError("Audio contains non-finite samples")
    return wave


def load_source(path, sample_rate=8000):
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if rate != sample_rate:
        divisor = gcd(rate, sample_rate)
        wave = resample_poly(wave, sample_rate // divisor, rate // divisor).astype(np.float32)
    if not np.isfinite(wave).all():
        raise ValueError(f"Invalid source: {path}")
    return wave


def write_audio(path, wave, sample_rate=8000):
    # Float WAV preserves relative levels; clipping/normalization belongs to the final remix.
    sf.write(path, np.asarray(wave, np.float32), sample_rate, subtype="FLOAT")


def waveform_peaks(wave, bins=1000):
    if len(wave) == 0:
        return []
    size = max(1, int(np.ceil(len(wave) / bins)))
    padded = np.pad(np.abs(wave), (0, (-len(wave)) % size))
    return padded.reshape(-1, size).max(-1).round(5).tolist()


def intervals_from_mask(mask, step, min_duration=0.08):
    mask = np.asarray(mask, bool)
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return [{"start": round(a * step, 3), "end": round(b * step, 3)}
            for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
            if (b - a) * step >= min_duration]


def energy_activity(wave, sample_rate=8000, step_seconds=0.04):
    frame = max(1, int(sample_rate * step_seconds))
    padded = np.pad(wave, (0, (-len(wave)) % frame))
    powers = np.mean(padded.reshape(-1, frame) ** 2, axis=-1)
    mask = powers > max(1e-8, float(powers.max(initial=0)) * 1e-3)
    regions = intervals_from_mask(mask, step_seconds)
    for region in regions:
        region["end"] = min(region["end"], len(wave) / sample_rate)
    return regions

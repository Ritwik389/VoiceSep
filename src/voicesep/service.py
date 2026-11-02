from pathlib import Path
import gc
import json
import os
import queue
import threading
import zipfile

import numpy as np
import torch

from .audio import read_audio, write_audio, waveform_peaks
from .inference import SeparationEngine
from .storage import TERMINAL
from .transcription import transcribe_tracks, subtitle_text


class Worker:
    def __init__(self, store, checkpoint=None, engine_factory=None):
        self.store = store
        self.checkpoint = Path(checkpoint or os.getenv("VOICESEP_CHECKPOINT", "models/best.pt")).expanduser().resolve()
        self.engine_factory = engine_factory or SeparationEngine
        self.queue = queue.Queue()
        self.cancel_events = {}
        self.engine = None
        self.engine_path = None
        self.model_health_cache = None
        self.thread = threading.Thread(target=self.run, daemon=True, name="voicesep-worker")

    def start(self):
        self.thread.start()

    def close(self):
        for event in self.cancel_events.values():
            event.set()
        self.queue.put(None)
        self.thread.join(timeout=5)

    def health(self):
        available = self.checkpoint.is_file()
        state = {"checkpoint_available": available, "checkpoint_path": str(self.checkpoint),
                 "trained": False, "release_ready": False, "capacity": 5, "sample_rate": 8000,
                 "cuda_available": torch.cuda.is_available(), "error": None}
        if available:
            signature = (str(self.checkpoint), self.checkpoint.stat().st_mtime_ns)
            if self.model_health_cache and self.model_health_cache[0] == signature:
                return self.model_health_cache[1]
            try:
                metadata = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
                state["trained"] = metadata.get("format_version") == 1 and metadata.get("trained") is True
                state["release_ready"] = bool(metadata.get("validation", {}).get("release_ready"))
                state["capacity"] = metadata.get("model_config", {}).get("slots", 5)
                state["sample_rate"] = metadata.get("model_config", {}).get("sample_rate", 8000)
                if state["sample_rate"] != 8000 or state["capacity"] != 5:
                    state["trained"] = False
                state["step"] = metadata.get("step", 0)
                state["calibrated"] = bool(metadata.get("calibration_provenance"))
                if not state["trained"]:
                    state["error"] = "Load a fine-tuned 8 kHz, five-slot VoiceSep checkpoint. Raw two-speaker weights are not a five-speaker model."
            except Exception as error:
                state["error"] = f"Cannot read checkpoint: {type(error).__name__}"
            self.model_health_cache = (signature, state)
        else:
            state["error"] = "Five-speaker checkpoint required. Run the free GPU training notebook, then select best.pt."
        return state

    def submit(self, identifier, process=True):
        event = threading.Event()
        self.cancel_events[identifier] = event
        # Capture the selected checkpoint for this job, not whichever settings a later request chooses.
        self.queue.put((identifier, process, self.checkpoint))

    def cancel(self, identifier):
        event = self.cancel_events.get(identifier)
        if event:
            event.set()
        job = self.store.get(identifier)
        if job and job["status"] not in TERMINAL:
            self.store.update(identifier, status="cancelled", error=None)

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            identifier, process, checkpoint = item
            event = self.cancel_events[identifier]
            try:
                if event.is_set():
                    continue
                self.process(identifier, process, checkpoint, event)
            except InterruptedError:
                self.store.update(identifier, status="cancelled", error=None)
            except Exception as error:
                self.store.update(identifier, status="failed", error=str(error)[:500])
            finally:
                self.cancel_events.pop(identifier, None)
                self.queue.task_done()

    def process(self, identifier, process, checkpoint, event):
        directory = self.store.directory(identifier)
        job = self.store.get(identifier)
        original = directory / "original.wav"
        if not original.is_file():
            self.store.update(identifier, status="decoding", progress=0.02)
            upload = next(directory.glob("input.*"))
            wave = read_audio(upload)
            if event.is_set():
                raise InterruptedError()
            write_audio(original, wave)
            flags = ["Input contains samples near clipping"] if np.max(np.abs(wave)) >= 0.999 else []
            self.store.update(identifier, duration=len(wave) / 8000, original_peaks=waveform_peaks(wave),
                              original_url=f"/api/jobs/{identifier}/audio/original", flags=flags)
        else:
            wave = read_audio(original)
        if not process:
            self.store.update(identifier, status="imported", progress=1.)
            return
        job = self.store.get(identifier)
        start, end = job["options"].get("start", 0), job["options"].get("end")
        end = end if end is not None else len(wave) / 8000
        if not 0 <= start < end <= len(wave) / 8000 + 1e-5:
            raise ValueError("Selected range is outside this recording")
        selection = wave[round(start * 8000):round(end * 8000)]
        # Keep original and returned tracks on the same selected timeline.
        write_audio(directory / "selection.wav", selection)
        self.store.update(identifier, duration=len(selection) / 8000, selection_offset=start,
                          original_peaks=waveform_peaks(selection), original_url=f"/api/jobs/{identifier}/audio/selection")
        if self.engine is None or self.engine_path != checkpoint:
            self.store.update(identifier, status="loading_model", progress=0.05)
            self.engine = self.engine_factory(checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
            self.engine_path = checkpoint
        def progress(stage, value):
            if event.is_set():
                raise InterruptedError()
            self.store.update(identifier, status=stage, progress=0.08 + value * 0.77)
        result = self.engine.separate(selection, manual_count=job["options"].get("speaker_count"),
                                      progress=progress, cancelled=event.is_set)
        paths = []
        speakers = result["speakers"]
        for speaker, stream in zip(speakers, result["waveforms"]):
            if event.is_set():
                raise InterruptedError()
            filename = f"{speaker['id']}.wav"
            write_audio(directory / filename, stream)
            speaker["peaks"] = waveform_peaks(stream)
            speaker["audio_url"] = f"/api/jobs/{identifier}/audio/{speaker['id']}"
            speaker["talk_time"] = round(sum(r["end"] - r["start"] for r in speaker["activity"]), 3)
            paths.append((speaker["id"], directory / filename))
        flags = job.get("flags", []) + result["flags"]
        transcript = []
        if job["options"].get("transcribe", False) and paths:
            self.store.update(identifier, status="transcribing", progress=0.88, speakers=speakers)
            # ASR runs after releasing GPU model memory.
            self.engine, self.engine_path = None, None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                transcript = transcribe_tracks(paths, job["options"].get("language"), cancelled=event.is_set)
            except InterruptedError:
                raise
            except Exception as error:
                flags.append(f"Transcription unavailable ({type(error).__name__}); voice tracks were saved")
        if event.is_set():
            raise InterruptedError()
        exporting = self.store.update(identifier, status="exporting", progress=.97, speakers=speakers,
                                      detected_count=len(speakers), flags=flags, transcript=transcript,
                                      sample_rate=result["sample_rate"])
        final = dict(exporting, status="complete", progress=1.)
        names = {s["id"]: s["name"] for s in speakers}
        for kind in ("srt", "vtt"):
            (directory / f"transcript.{kind}").write_text(subtitle_text(transcript, names, kind))
        (directory / "transcript.txt").write_text("\n".join(f"{names.get(s['speaker_id'], s['speaker_id'])}: {s['text']}" for s in transcript))
        # Do not leak server-local checkpoint or data paths in exported results.
        (directory / "results.json").write_text(json.dumps(final, ensure_ascii=False, indent=2))
        temporary_archive = directory / "results.zip.tmp"
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in [directory / "selection.wav", directory / "results.json",
                         *(path for _, path in paths), *(directory / f"transcript.{kind}" for kind in ("txt", "srt", "vtt"))]:
                archive.write(path, path.name)
        temporary_archive.replace(directory / "results.zip")
        if event.is_set():
            raise InterruptedError()
        self.store.update(identifier, status="complete", progress=1.)

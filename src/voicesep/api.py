from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import json
import math
import os
import shutil

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .storage import JobStore, TERMINAL
from .service import Worker
from .audio import read_audio, write_audio


class ModelSettings(BaseModel):
    checkpoint_path: str = Field(min_length=1, max_length=1024)


class StartOptions(BaseModel):
    speaker_count: int | None = Field(default=None, ge=1, le=5)
    transcribe: bool = False
    language: str | None = None
    start: float = Field(default=0, ge=0, allow_inf_nan=False)
    end: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class RemixOptions(BaseModel):
    gains: dict[str, float]


def create_app(data_dir=None, checkpoint=None, engine_factory=None, analysis_runner=None):
    store = JobStore(data_dir or os.getenv("VOICESEP_DATA_DIR", "data/sessions"))
    worker = Worker(store, checkpoint, engine_factory, analysis_runner)

    @asynccontextmanager
    async def lifespan(app):
        worker.start()
        yield
        worker.close()

    app = FastAPI(title="VoiceSep", version="0.1.0", lifespan=lifespan)
    app.state.store, app.state.worker = store, worker
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET", "POST", "DELETE", "PATCH"], allow_headers=["Content-Type"])

    def get_job(identifier):
        job = store.get(identifier)
        if not job:
            raise HTTPException(404, "Session not found")
        return job

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model": worker.health(), "limits": {"seconds": 300, "bytes": 100 * 1024**2}}

    @app.post("/api/model")
    def settings(options: ModelSettings):
        if any(job["status"] not in TERMINAL for job in store.list()):
            raise HTTPException(409, "Wait for active jobs before changing the checkpoint")
        candidate = Path(options.checkpoint_path).expanduser().resolve()
        if not candidate.is_file():
            raise HTTPException(422, "Checkpoint file does not exist")
        previous = worker.checkpoint
        worker.checkpoint = candidate
        status = worker.health()
        if not status["trained"]:
            worker.checkpoint = previous
            raise HTTPException(422, status["error"])
        worker.engine = None
        return status

    @app.get("/api/jobs")
    def jobs():
        return store.list()

    @app.get("/api/notebooks/training")
    def notebook():
        path = Path(__file__).resolve().parents[2] / "notebooks" / "01_train_colab.ipynb"
        if not path.is_file():
            raise HTTPException(404, "Run from the source checkout to access the training notebook")
        return FileResponse(path, filename="VoiceSep-free-GPU-training.ipynb")

    @app.post("/api/jobs", status_code=201)
    async def create_job(file: UploadFile = File(...), process: bool = Form(True),
                         speaker_count: int | None = Form(None), transcribe: bool = Form(False),
                         language: str | None = Form(None)):
        extension = Path(file.filename or "").suffix.lower()
        if extension not in {".wav", ".flac", ".mp3", ".m4a"}:
            raise HTTPException(415, "Choose a WAV, FLAC, MP3, or M4A audio file")
        if speaker_count is not None and not 1 <= speaker_count <= 5:
            raise HTTPException(422, "Speaker count must be 1–5, or automatic")
        if language not in {None, "", "en", "hi"}:
            raise HTTPException(422, "Language must be en, hi, or automatic")
        if process and not worker.health()["trained"]:
            raise HTTPException(409, worker.health()["error"])
        job = store.create(Path(file.filename).name, {"speaker_count": speaker_count,
                           "transcribe": transcribe, "language": language or None})
        directory = store.directory(job["id"])
        total = 0
        try:
            with (directory / f"input{extension}").open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > 100 * 1024**2:
                        raise HTTPException(413, "Audio must be smaller than 100 MB")
                    stream.write(chunk)
            if total == 0:
                raise HTTPException(422, "This audio file is empty")
        except Exception:
            store.remove(job["id"])
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            await file.close()
        worker.submit(job["id"], process)
        return job

    @app.get("/api/jobs/{identifier}")
    def status(identifier: str):
        return get_job(identifier)

    @app.post('/api/jobs/{identifier}/analysis', status_code=202)
    def analyze_speakers(identifier: str):
        with store.lock:
            job = get_job(identifier)
            if job['status'] != 'complete' or not job['speakers']:
                raise HTTPException(409, 'Complete separation before analyzing speakers')
            if identifier in worker.cancel_events:
                raise HTTPException(409, 'This session already has work queued or running')
            updated = store.update(identifier, speaker_analysis={
                'status': 'queued', 'progress': 0., 'stage': 'Waiting for the local worker', 'error': None})
            worker.submit(identifier, process='analysis')
        return updated

    @app.post("/api/jobs/{identifier}/start")
    def start(identifier: str, options: StartOptions):
        job = get_job(identifier)
        if job["status"] != "imported":
            raise HTTPException(409, "Import an audio recording before starting separation")
        if not worker.health()["trained"]:
            raise HTTPException(409, worker.health()["error"])
        if options.language not in {None, "en", "hi"}:
            raise HTTPException(422, "Language must be en, hi, or automatic")
        end = options.end if options.end is not None else job["duration"]
        if not options.start < end <= job["duration"] + 1e-5:
            raise HTTPException(422, "Choose a valid time range within the recording")
        updated = store.update(identifier, status="queued", progress=0., options=options.model_dump())
        worker.submit(identifier)
        return updated

    @app.get("/api/jobs/{identifier}/events")
    async def events(identifier: str, request: Request):
        get_job(identifier)
        async def stream():
            previous = None
            while not await request.is_disconnected():
                job = get_job(identifier)
                payload = json.dumps(job)
                if payload != previous:
                    yield f"data: {payload}\n\n"
                    previous = payload
                if job["status"] in TERMINAL and job.get('speaker_analysis', {}).get('status') not in {'queued', 'running'}:
                    break
                await asyncio.sleep(0.25)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/api/jobs/{identifier}/cancel")
    def cancel(identifier: str):
        get_job(identifier)
        worker.cancel(identifier)
        return get_job(identifier)

    @app.delete("/api/jobs/{identifier}", status_code=204)
    def delete(identifier: str):
        job = get_job(identifier)
        if job["status"] not in TERMINAL or identifier in worker.cancel_events:
            raise HTTPException(409, "Cancel processing and wait for the worker before deleting this session")
        shutil.rmtree(store.directory(identifier))
        store.remove(identifier)

    @app.get("/api/jobs/{identifier}/audio/{track}")
    def audio(identifier: str, track: str):
        job = get_job(identifier)
        allowed = {"original", "selection", "remix"} | {s["id"] for s in job["speakers"]}
        if track not in allowed:
            raise HTTPException(404, "Track not found")
        path = store.directory(identifier) / f"{track}.wav"
        if not path.is_file():
            raise HTTPException(404, "Track is not ready")
        return FileResponse(path, media_type="audio/wav", filename=f"{track}.wav")

    @app.get("/api/jobs/{identifier}/export/{kind}")
    def export(identifier: str, kind: str):
        job = get_job(identifier)
        if job["status"] != "complete":
            raise HTTPException(409, "Results are not ready")
        filenames = {"zip": "results.zip", "json": "results.json", "txt": "transcript.txt",
                     "srt": "transcript.srt", "vtt": "transcript.vtt", "analysis": "speaker-analysis.json"}
        if kind not in filenames:
            raise HTTPException(404, "Export format not found")
        path = store.directory(identifier) / filenames[kind]
        if not path.is_file():
            raise HTTPException(409, "Export is being finalized; retry shortly")
        return FileResponse(path, filename=path.name)

    @app.post("/api/jobs/{identifier}/remix")
    def remix(identifier: str, options: RemixOptions):
        job = get_job(identifier)
        if job["status"] != "complete" or not job["speakers"]:
            raise HTTPException(409, "Separated tracks are required")
        known = {speaker["id"] for speaker in job["speakers"]}
        if set(options.gains) != known or any(not math.isfinite(v) or not 0 <= v <= 2 for v in options.gains.values()):
            raise HTTPException(422, "Supply one finite gain between 0 and 2 for each speaker")
        directory = store.directory(identifier)
        mixed = sum(read_audio(directory / f"{speaker}.wav") * gain for speaker, gain in options.gains.items())
        peak = float(np.max(np.abs(mixed), initial=0))
        mixed /= max(1., peak / 0.98)
        write_audio(directory / "remix.wav", mixed, job["sample_rate"])
        return FileResponse(directory / "remix.wav", media_type="audio/wav", filename="voicesep-remix.wav")

    frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app

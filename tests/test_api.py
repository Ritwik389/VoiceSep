from dataclasses import asdict
import asyncio
import io
import time
import zipfile

import httpx
import numpy as np
import pytest
import soundfile as sf
import torch

from voicesep.api import create_app
from voicesep.models.separator import ModelConfig


def recording():
    stream = io.BytesIO()
    sf.write(stream, np.sin(np.arange(8000) * 2 * np.pi * 200 / 8000).astype(np.float32) * .1, 8000, format="WAV")
    return stream.getvalue()


class FixtureEngine:
    """API plumbing fixture only; never used by the app or treated as separation evidence."""
    def __init__(self, *args):
        pass

    def separate(self, wave, manual_count=None, progress=None, cancelled=None):
        for i in range(5):
            if cancelled():
                raise InterruptedError()
            progress("separating", (i+1)/5)
            time.sleep(.02)
        return {"waveforms": np.stack([wave*.6, wave*.4]), "sample_rate":8000, "flags":["Software test fixture"],
                "speakers":[{"id":f"speaker-{n}","name":f"Speaker {n}","activity":[{"start":0.,"end":1.}],"activity_fraction":1.,"slot":n-1} for n in (1,2)]}


async def wait_job(client, identifier, states):
    for _ in range(200):
        job = (await client.get(f"/api/jobs/{identifier}")).json()
        if job["status"] in states:
            return job
        await asyncio.sleep(.02)
    raise AssertionError("Worker timed out")


@pytest.mark.asyncio
async def test_import_without_model_then_gate_separation(tmp_path):
    app = create_app(tmp_path / "sessions", tmp_path / "missing.pt")
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/jobs", files={"file":("voice.wav", recording(),"audio/wav")}, data={"process":"false"})
        assert response.status_code == 201
        job = await wait_job(client, response.json()["id"], {"imported", "failed"})
        assert job["status"] == "imported", job
        assert len(job["original_peaks"]) > 0
        assert (await client.get(job["original_url"])).status_code == 200
        assert (await client.post(f"/api/jobs/{job['id']}/start", json={})).status_code == 409
        assert (await client.post("/api/jobs", files={"file":("bad.wav",b"","audio/wav")}, data={"process":"false"})).status_code == 422
        assert (await client.get(f"/api/jobs/{job['id']}/audio/unknown")).status_code == 404


@pytest.mark.asyncio
async def test_completed_job_exports_and_remix(tmp_path):
    checkpoint = tmp_path / "fixture.pt"
    torch.save({"format_version":1,"trained":True,"model_config":asdict(ModelConfig()),"step":1},checkpoint)
    app = create_app(tmp_path / "sessions", checkpoint, FixtureEngine)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/jobs", files={"file":("voice.wav",recording(),"audio/wav")})
        job = await wait_job(client, response.json()["id"], {"complete","failed"})
        assert job["status"] == "complete", job
        assert job["detected_count"] == 2
        for _ in range(100):
            archive = await client.get(f"/api/jobs/{job['id']}/export/zip")
            if archive.status_code == 200:
                break
            await asyncio.sleep(.01)
        assert {"speaker-1.wav","speaker-2.wav","results.json","transcript.srt"}.issubset(zipfile.ZipFile(io.BytesIO(archive.content)).namelist())
        bad = await client.post(f"/api/jobs/{job['id']}/remix",json={"gains":{"speaker-1":1}})
        assert bad.status_code == 422
        remix = await client.post(f"/api/jobs/{job['id']}/remix",json={"gains":{"speaker-1":1,"speaker-2":0}})
        assert remix.status_code == 200 and remix.headers["content-type"] == "audio/wav"
        assert not (await client.get('/api/health')).json()["model"]["release_ready"]


@pytest.mark.asyncio
async def test_cancel_does_not_publish_completed_results(tmp_path):
    checkpoint = tmp_path / "fixture.pt"
    torch.save({"format_version":1,"trained":True,"model_config":asdict(ModelConfig())},checkpoint)
    app = create_app(tmp_path / "sessions", checkpoint, FixtureEngine)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response=await client.post('/api/jobs', files={"file":("voice.wav",recording(),"audio/wav")})
        identifier=response.json()["id"]
        await client.post(f'/api/jobs/{identifier}/cancel')
        await asyncio.sleep(.2)
        assert (await client.get(f'/api/jobs/{identifier}')).json()["status"] == 'cancelled'
        assert (await client.get(f'/api/jobs/{identifier}/export/zip')).status_code == 409

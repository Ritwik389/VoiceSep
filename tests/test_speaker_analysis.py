"""Software fixtures for abstention, lifecycle and exports. No demographic accuracy claims."""
from dataclasses import asdict
import asyncio
import io
import json
import zipfile

import httpx
import numpy as np
import pytest
import soundfile as sf
import torch

from voicesep.api import create_app
from voicesep.models.separator import ModelConfig
from voicesep.speaker_analysis import candidate_segments, analyze_tracks, age_and_presentation, emotion_segment
from voicesep.storage import JobStore
from test_api import FixtureEngine, recording, wait_job


def test_short_silent_invalid_and_clipped_audio_abstain_without_loading_models(tmp_path):
    def forbidden():
        raise AssertionError('A model should not be loaded for unusable audio')
    for name, wave in [('short', np.ones(8000)*.1), ('silent', np.zeros(32000)), ('clipped', np.ones(32000))]:
        path = tmp_path / f'{name}.wav'
        sf.write(path, wave, 8000)
        result = analyze_tracks([('speaker-1', path)], age_factory=forbidden, emotion_factory=forbidden)['speaker-1']
        assert result['age']['status'] == 'uncertain'
        assert result['emotion']['status'] == 'uncertain'
    windows, reason = candidate_segments(np.full(32000, np.nan), 8000)
    assert not windows and reason


def test_age_bands_disagreement_and_uncertain_emotion_are_preserved():
    age, vocal = age_and_presentation([(36., [.01, .94, .05]), (40., [.02, .91, .07])])
    assert age['range'][0] <= 36 <= 40 <= age['range'][1]
    assert age['range'][1] - age['range'][0] >= 20
    assert vocal['label'] == 'feminine_sounding'
    age, vocal = age_and_presentation([(15., [.1, .45, .45]), (80., [.1, .45, .45])])
    assert age['status'] == 'uncertain' and vocal['label'] == 'uncertain'
    assert emotion_segment(np.ones(9)/9, 2, 5)['label'] == 'uncertain'
    scores = np.zeros(9); scores[-1] = 1
    assert emotion_segment(scores, 2, 5)['label'] == 'uncertain'


def test_model_failure_preserves_other_features_and_real_timeline_offsets(tmp_path):
    wave = np.sin(np.arange(9*8000)*2*np.pi*140/8000)*.1
    path = tmp_path / 'voice.wav'; sf.write(path, wave, 8000)
    class MissingAge:
        def __init__(self):
            raise ImportError('fixture optional dependency')
    class EmotionFixture:
        def predict(self, audio):
            assert len(audio) == 4*16000
            scores = np.zeros(9); scores[3] = 1
            return scores
    result = analyze_tracks([('speaker-1', path)], [{'id': 'speaker-1', 'activity': [{'start': 2, 'end': 6}]}],
                            age_factory=MissingAge, emotion_factory=EmotionFixture)['speaker-1']
    assert result['age']['status'] == 'unavailable'
    assert result['emotion']['segments'][0]['start'] == 2
    assert result['emotion']['segments'][0]['end'] == 6
    assert result['emotion']['segments'][0]['label'] == 'happy'
    assert any('8 kHz' in warning for warning in result['warnings'])


@pytest.mark.asyncio
async def test_analysis_on_completed_session_exports_and_failed_retry_keeps_tracks(tmp_path):
    checkpoint = tmp_path / 'fixture.pt'
    torch.save({'format_version': 1, 'trained': True, 'model_config': asdict(ModelConfig()), 'step': 1}, checkpoint)
    should_fail = False
    def analyzer(paths, *args, **kwargs):
        if should_fail:
            raise RuntimeError('fixture analysis failure')
        return {speaker_id: {'age': {'status': 'uncertain'}, 'vocal_presentation': {'status': 'uncertain'},
                             'emotion': {'status': 'uncertain', 'segments': []}}
                for speaker_id, _ in paths}
    app = create_app(tmp_path / 'sessions', checkpoint, FixtureEngine, analyzer)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/jobs', files={'file': ('voice.wav', recording(), 'audio/wav')})
        identifier = response.json()['id']
        await wait_job(client, identifier, {'complete'})
        for _ in range(30):
            response = await client.post(f'/api/jobs/{identifier}/analysis')
            if response.status_code == 202:
                break
            await asyncio.sleep(.02)
        assert response.status_code == 202, response.text
        async def wait_analysis():
            for _ in range(100):
                state = (await client.get(f'/api/jobs/{identifier}')).json()
                if state.get('speaker_analysis', {}).get('status') not in {'queued', 'running'}:
                    return state
                await asyncio.sleep(.02)
            raise AssertionError('Analysis did not finish')
        state = await wait_analysis()
        assert state['status'] == 'complete' and state['speaker_analysis']['status'] == 'complete'
        exported = await client.get(f'/api/jobs/{identifier}/export/analysis')
        assert exported.status_code == 200 and len(exported.json()['speakers']) == 2
        archive = await client.get(f'/api/jobs/{identifier}/export/zip')
        with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
            assert 'speaker-analysis.json' in bundle.namelist()
            assert json.loads(bundle.read('results.json'))['speaker_analysis']['status'] == 'complete'
        should_fail = True
        await client.post(f'/api/jobs/{identifier}/analysis')
        state = await wait_analysis()
        assert state['status'] == 'complete' and state['speaker_analysis']['status'] == 'failed'
        assert (await client.get(f'/api/jobs/{identifier}/audio/speaker-1')).status_code == 200
        assert (await client.get(f'/api/jobs/{identifier}/export/zip')).status_code == 200


def test_restart_marks_analysis_interrupted_and_keeps_completed_separation(tmp_path):
    store = JobStore(tmp_path)
    job = store.create('voice.wav', {})
    store.update(job['id'], status='complete', speaker_analysis={'status': 'running'})
    recovered = JobStore(tmp_path).get(job['id'])
    assert recovered['status'] == 'complete'
    assert recovered['speaker_analysis']['status'] == 'failed'


def test_activity_dropouts_merge_without_removing_real_pauses():
    wave = np.sin(np.arange(8*8000)*2*np.pi*140/8000)*.1
    regions = [{'start': 0., 'end': 1.5}, {'start': 1.65, 'end': 3.8}, {'start': 6., 'end': 8.}]
    windows, reason = candidate_segments(wave, 8000, regions)
    assert reason is None and len(windows) == 2
    assert windows[0]['start'] == 0. and windows[0]['end'] == 3.8
    assert windows[1]['start'] == 6.
    assert candidate_segments(wave, 8000, [])[0] == []


@pytest.mark.asyncio
async def test_duplicate_analysis_request_and_cancel_keep_existing_exports(tmp_path):
    import time
    checkpoint = tmp_path / 'fixture.pt'
    torch.save({'format_version': 1, 'trained': True, 'model_config': asdict(ModelConfig()), 'step': 1}, checkpoint)
    def analyzer(paths, *args, cancelled, **kwargs):
        while not cancelled():
            time.sleep(.01)
        raise InterruptedError('fixture cancelled')
    app = create_app(tmp_path / 'sessions', checkpoint, FixtureEngine, analyzer)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/jobs', files={'file': ('voice.wav', recording(), 'audio/wav')})
        identifier = response.json()['id']
        await wait_job(client, identifier, {'complete'})
        for _ in range(30):
            response = await client.post(f'/api/jobs/{identifier}/analysis')
            if response.status_code == 202: break
            await asyncio.sleep(.02)
        assert response.status_code == 202
        assert (await client.post(f'/api/jobs/{identifier}/analysis')).status_code == 409
        assert (await client.get(f'/api/jobs/{identifier}/export/zip')).status_code == 200
        assert (await client.delete(f'/api/jobs/{identifier}')).status_code == 409
        await client.post(f'/api/jobs/{identifier}/cancel')
        for _ in range(100):
            job = (await client.get(f'/api/jobs/{identifier}')).json()
            if job['speaker_analysis']['status'] == 'cancelled': break
            await asyncio.sleep(.01)
        assert job['status'] == 'complete' and job['speaker_analysis']['status'] == 'cancelled'
        assert (await client.get(f'/api/jobs/{identifier}/export/zip')).status_code == 200

import hashlib
import io
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pytest
import soundfile as sf

from voicesep.research import bootstrap


class Response:
    def __init__(self, body, status=200, headers=None, fail=False):
        self.body=io.BytesIO(body)
        self.status=status
        self.headers=headers or {'Content-Length':str(len(body))}
        self.fail=fail
        self.reads=0

    def __enter__(self): return self
    def __exit__(self,*args): self.body.close()

    def read(self,count):
        if self.fail and self.reads:
            raise TimeoutError('simulated stalled socket')
        self.reads+=1
        return self.body.read(min(count,4) if self.fail else count)


@pytest.mark.parametrize('timeout',[True,False])
def test_interrupted_transfer_resumes_exactly_at_last_saved_byte(tmp_path,monkeypatch,timeout):
    payload=b'0123456789abcdefghijklmnop'
    first=Response(payload if timeout else payload[:4],headers={'Content-Length':str(len(payload))},fail=timeout)
    responses=[first,Response(payload[4:],206,{'Content-Range':f'bytes 4-{len(payload)-1}/{len(payload)}'})]
    requests=[]
    def open_url(request,**kwargs):
        requests.append(request)
        return responses.pop(0)
    monkeypatch.setattr(bootstrap.urllib.request,'urlopen',open_url)
    monkeypatch.setattr(bootstrap.time,'sleep',lambda seconds:None)
    destination=tmp_path/'archive.gz'
    digest=hashlib.md5(payload).hexdigest()
    assert bootstrap.download_archive(('https://first.test/archive','https://second.test/archive'),destination,digest).read_bytes()==payload
    assert requests[1].get_header('Range')=='bytes=4-'
    assert requests[1].full_url=='https://second.test/archive'
    assert not destination.with_suffix('.gz.part').exists()
    assert bootstrap.download_archive(('https://first.test/archive',),destination,digest)==destination
    assert len(requests)==2  # A verified completed cache needs no network.


def test_server_ignoring_range_restarts_without_duplicating_prefix(tmp_path,monkeypatch):
    payload=b'complete-archive'
    destination=tmp_path/'archive.gz'
    destination.with_suffix('.gz.part').write_bytes(payload[:5])
    monkeypatch.setattr(bootstrap.urllib.request,'urlopen',lambda *args,**kwargs:Response(payload))
    assert bootstrap.download_archive(('https://test.test/archive',),destination,hashlib.md5(payload).hexdigest()).read_bytes()==payload


def test_invalid_content_range_never_corrupts_saved_prefix(tmp_path,monkeypatch):
    destination=tmp_path/'archive.gz'
    partial=destination.with_suffix('.gz.part')
    partial.write_bytes(b'prefix')
    monkeypatch.setattr(bootstrap.urllib.request,'urlopen',lambda *args,**kwargs:
        Response(b'wrong',206,{'Content-Range':'bytes 0-4/10'}))
    with pytest.raises(RuntimeError,match='after retries'):
        bootstrap.download_archive(('https://test.test/archive',),destination,'expected',attempts=1)
    assert partial.read_bytes()==b'prefix' and not destination.exists()


def test_checksum_mismatch_is_discarded_before_a_clean_retry(tmp_path,monkeypatch):
    payload=b'correct archive'
    responses=[Response(b'corrupt archive'),Response(payload)]
    requests=[]
    def open_url(request,**kwargs):
        requests.append(request)
        return responses.pop(0)
    monkeypatch.setattr(bootstrap.urllib.request,'urlopen',open_url)
    monkeypatch.setattr(bootstrap.time,'sleep',lambda seconds:None)
    result=bootstrap.download_archive(('https://test.test/archive',),tmp_path/'archive.gz',hashlib.md5(payload).hexdigest())
    assert result.read_bytes()==payload
    assert requests[1].get_header('Range') is None


def test_exhausted_retry_keeps_partial_cache_for_a_later_run(tmp_path,monkeypatch):
    payload=b'0123456789abcdef'
    destination=tmp_path/'archive.gz'
    monkeypatch.setattr(bootstrap.urllib.request,'urlopen',lambda *args,**kwargs:Response(payload,fail=True))
    digest=hashlib.md5(payload).hexdigest()
    with pytest.raises(RuntimeError):
        bootstrap.download_archive(('https://test.test/archive',),destination,digest,attempts=1)
    assert destination.with_suffix('.gz.part').read_bytes()==payload[:4]
    requests=[]
    def resumed(request,**kwargs):
        requests.append(request)
        return Response(payload[4:],206,{'Content-Range':f'bytes 4-{len(payload)-1}/{len(payload)}'})
    monkeypatch.setattr(bootstrap.urllib.request,'urlopen',resumed)
    assert bootstrap.download_archive(('https://test.test/archive',),destination,digest).read_bytes()==payload
    assert requests[0].get_header('Range')=='bytes=4-'


def test_english_extracts_verified_local_archive_and_preserves_later_transcripts(tmp_path,monkeypatch):
    archive=tmp_path/'fixture.tar.gz'
    audio=io.BytesIO()
    t=np.arange(72000)/16000
    sf.write(audio,np.sin(2*np.pi*220*t)*.05,16000,format='FLAC')
    with tarfile.open(archive,'w:gz') as bundle:
        for name,payload in (('LibriSpeech/train-clean-100/123/1/123-1-0001.flac',audio.getvalue()),
                             ('LibriSpeech/train-clean-100/123/1/123-1.trans.txt',b'123-1-0001 KNOWN TRANSCRIPT\n')):
            info=tarfile.TarInfo(name);info.size=len(payload)
            bundle.addfile(info,io.BytesIO(payload))
    def downloaded(urls,destination,expected_md5):
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(archive,destination)
        return destination
    monkeypatch.setattr(bootstrap,'download_archive',downloaded)
    root=tmp_path/'corpus';root.mkdir()
    rows=bootstrap.prepare_english(root,url='https://test.test/fixture')
    assert len(rows)==1 and rows[0]['speaker_id']=='123' and rows[0]['text']=='KNOWN TRANSCRIPT'
    wave,rate=sf.read(root/rows[0]['path'])
    assert rate==8000 and len(wave)==36000
    assert (root/'english-index.jsonl').is_file()
    assert not (root/'downloads/train-clean-100.tar.gz').exists()
    monkeypatch.setattr(bootstrap,'download_archive',lambda *args:pytest.fail('Completed source cache must be reused'))
    assert bootstrap.prepare_english(root)[0]['text']=='KNOWN TRANSCRIPT'

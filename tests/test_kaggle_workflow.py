from datetime import timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location('run_kaggle_workflow',
    Path(__file__).resolve().parents[1]/'scripts/run_kaggle_workflow.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def fixture_corpus(root):
    root.mkdir(parents=True,exist_ok=True)
    (root/'preparation.json').write_text(json.dumps({'stage':'data_prepared','sample_rate':8000}))
    for split in ('train','validation','test'):
        rows=[{'identity':f'{split}-{language}-{i}','language':language,'path':f'audio/{split}-{language}-{i}.flac'}
              for language in ('en','hi') for i in range(5)]
        path=root/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        if split!='train':
            (root/f'{split}-mixtures.jsonl').write_text(json.dumps({
                'source_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})+'\n')


class FakeApi:
    def __init__(self, hours=30):
        self.hours=hours
        self.submissions=[]
        self.fail_benchmark=False

    def kernels_status(self, reference):
        return SimpleNamespace(status=SimpleNamespace(name='ERROR' if self.fail_benchmark and 'benchmark' in reference else 'COMPLETE'))

    def quota_view(self):
        return SimpleNamespace(gpu_quota=SimpleNamespace(total_time_allowed=timedelta(hours=self.hours),time_used=timedelta()))

    def kernels_output(self, reference, destination, **kwargs):
        root=Path(destination)
        if 'preparation' in reference:
            fixture_corpus(root/'voicesep-corpus')
        elif 'benchmark' in reference:
            root.mkdir(parents=True,exist_ok=True)
            (root/'benchmark-selection.json').write_text(json.dumps({'segment_seconds':4,
                'benchmark':{'steps_this_run':100,'device':'cuda','benchmark_only':True,
                             'peak_cuda_memory_bytes':10*1024**3,'seconds_per_step':3}}))
        else:
            root.mkdir(parents=True,exist_ok=True)
            (root/'run-summary.json').write_text(json.dumps({'step':500,'resume_required':True}))
        return [], None

    def kernels_push(self, directory, **kwargs):
        metadata=json.loads((Path(directory)/'kernel-metadata.json').read_text())
        self.submissions.append(metadata)
        return SimpleNamespace(error='',version_number=1)


def make_flow(tmp_path, api, after_preparation_version=0):
    jobs=tmp_path/'jobs'
    for stage,slug in (('benchmark','voicesep-gpu-benchmark'),('train','voicesep-gpu-training-01')):
        root=jobs/stage
        root.mkdir(parents=True)
        (root/'job.ipynb').write_text('{}')
        inputs=['researcher/voicesep-data-preparation']
        if stage=='train': inputs.append('researcher/voicesep-gpu-benchmark')
        (root/'kernel-metadata.json').write_text(json.dumps({'id':f'researcher/{slug}',
            'is_private':True,'enable_gpu':True,'machine_shape':'NvidiaTeslaT4',
            'dataset_sources':['researcher/voicesep-research-source'],'kernel_sources':inputs,
            'code_file':'job.ipynb'}))
    def metadata(api, reference):
        return SimpleNamespace(is_private=True,enable_gpu='preparation' not in reference,
                               enable_tpu=False,current_version_number=4)
    return workflow.Workflow(api,'researcher',jobs,tmp_path/'run',metadata=metadata,poll_seconds=0,
                             after_preparation_version=after_preparation_version)


def test_controller_pins_inputs_and_runs_only_one_bounded_training_session(tmp_path):
    api=FakeApi()
    flow=make_flow(tmp_path,api)
    flow.run()
    assert len(api.submissions)==2
    assert api.submissions[0]['kernel_sources']==['researcher/voicesep-data-preparation/4']
    assert api.submissions[1]['kernel_sources']==['researcher/voicesep-data-preparation/4','researcher/voicesep-gpu-benchmark/1']
    assert flow.state['phase']=='session_complete'
    assert flow.state['resume_required'] is True and flow.state['release_ready'] is False
    flow.run()
    assert len(api.submissions)==2


def test_controller_does_not_submit_training_after_a_failed_benchmark(tmp_path):
    api=FakeApi();api.fail_benchmark=True
    flow=make_flow(tmp_path,api)
    with pytest.raises(RuntimeError,match='ERROR'):
        flow.run()
    assert len(api.submissions)==1 and 'benchmark' in api.submissions[0]['id']


@pytest.mark.parametrize('hours,submitted',[(1,0),(10,1)])
def test_controller_stops_for_insufficient_free_quota_and_stop_file(tmp_path,hours,submitted):
    api=FakeApi(hours=hours)
    flow=make_flow(tmp_path,api)
    with pytest.raises(RuntimeError,match='Insufficient free GPU quota'):
        flow.run()
    assert len(api.submissions)==submitted
    (flow.output/'STOP').touch()
    with pytest.raises(InterruptedError):
        flow.run()
    assert len(api.submissions)==submitted


def test_corpus_gate_rejects_a_modified_manifest_before_any_gpu_submission(tmp_path):
    root=tmp_path/'corpus'
    fixture_corpus(root)
    path=root/'validation.jsonl'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='Frozen benchmark'):
        workflow.validate_corpus(root)


def test_output_download_reaches_metadata_after_pages_of_audio_files(tmp_path):
    api=FakeApi()
    flow=make_flow(tmp_path,api)
    pages=[]
    def output(reference,destination,**kwargs):
        pages.append(kwargs['page_token'])
        return [], 'next-page' if len(pages)==1 else None
    api.kernels_output=output
    flow.download('researcher/voicesep-data-preparation/4',tmp_path/'metadata',r'preparation.json$')
    assert pages==[None,'next-page']


def test_recovery_waits_for_a_new_ui_run_instead_of_reusing_the_failed_version(tmp_path,monkeypatch):
    api=FakeApi()
    flow=make_flow(tmp_path,api,after_preparation_version=5)
    versions=[5,6]
    def metadata(api,reference):
        return SimpleNamespace(is_private=True,enable_gpu='preparation' not in reference,
                               enable_tpu=False,current_version_number=versions[0])
    def ui_restart(seconds):
        assert not api.submissions
        versions.pop(0)
    flow.metadata=metadata
    monkeypatch.setattr(workflow.time,'sleep',ui_restart)
    flow.run()
    assert flow.state['preparation_kernel']=='researcher/voicesep-data-preparation/6'
    assert api.submissions[0]['kernel_sources']==['researcher/voicesep-data-preparation/6']

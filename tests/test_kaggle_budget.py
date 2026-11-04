import ast
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('finish_budget', Path(__file__).resolve().parents[1] / 'scripts/finish_kaggle_budget.py')
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def test_budget_holds_back_final_evaluation_and_user_reserve():
    assert budget.allocation(17.62, 1, 59832, quota_rate=2) == 21600
    assert budget.allocation(10.5, 1, 34000, quota_rate=1) <= 21600
    assert budget.allocation(5.62, 1, 16632, quota_rate=2) == 0
    final = budget.allocation(5.62, 1, 16632, True, quota_rate=2)
    assert 7200 <= final <= 8316
    assert final * 2 + 600 + 3600 <= 5.62 * 3600
    with pytest.raises(RuntimeError, match='reserve intact'):
        budget.allocation(2.9, 1, 8000, True)
    # A refresh cannot expand the original allowance.
    assert budget.allocation(30, 1, 10000, True) == 9400


def fixture_flow(tmp_path, api):
    root = tmp_path / 'template/train'
    root.mkdir(parents=True)
    nb = {'cells': [{'cell_type': 'markdown', 'source': ['original']},
                    {'cell_type': 'code', 'source': ['import subprocess, sys\n']},
                    {'cell_type': 'code', 'source': ['# mount data\n']},
                    {'cell_type': 'code', 'source': ['# train\n']}]}
    (root / 'job.ipynb').write_text(json.dumps(nb))
    (root / 'kernel-metadata.json').write_text(json.dumps({
        'id': 'researcher/voicesep-gpu-training-03', 'is_private': True, 'enable_gpu': True,
        'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': ['researcher/voicesep-research-source'],
        'code_file': 'job.ipynb'}))
    return budget.BudgetWorkflow(api, 'researcher', root.parent, tmp_path / 'run', 17.62)


def test_bounded_notebook_uses_remote_interpreter_and_pinned_parent(tmp_path):
    flow = fixture_flow(tmp_path, None)
    folder = flow.notebook(3, 'researcher/voicesep-gpu-training-02/1', 21600, False)
    nb = json.loads((folder / 'job.ipynb').read_text())
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    last = ''.join(nb['cells'][-1]['source'])
    assert 'sys.executable' in last and '--benchmark-parallel' in last
    assert '--finalize' not in last
    assert 'VOICESEP_JOB_STARTED' in ''.join(nb['cells'][1]['source'])
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    assert meta['is_private'] and meta['kernel_sources'][-1] == 'researcher/voicesep-gpu-training-02/1'
    final = flow.notebook(5, 'researcher/voicesep-gpu-training-04/1', 14400, True)
    assert '--finalize' in (final / 'job.ipynb').read_text()


def test_uncertain_submission_cannot_spend_the_allowance_twice(tmp_path):
    flow = fixture_flow(tmp_path, None)
    flow.save(phase='submitting', unallocated_seconds=10000)
    with pytest.raises(RuntimeError, match='do not submit a duplicate'):
        flow.run()
    assert flow.state['unallocated_seconds'] == 10000


def test_controller_finishes_three_jobs_without_entering_one_hour_buffer(tmp_path):
    class Api:
        def __init__(self):
            self.hours = 17.62
            self.jobs = []
        def authenticate(self):
            pass
        def quota_view(self):
            return SimpleNamespace(gpu_quota=SimpleNamespace(
                total_time_allowed=timedelta(hours=30), time_used=timedelta(hours=30-self.hours)))
        def kernels_status(self, reference):
            for job in self.jobs:
                if job['reference'] == reference and not job['charged']:
                    self.hours -= job['seconds'] / 3600
                    job['charged'] = True
            return SimpleNamespace(status=SimpleNamespace(name='COMPLETE'))
        def kernels_push(self, directory, timeout, acc):
            assert acc == 'NvidiaTeslaT4'
            assert self.hours - timeout / 3600 >= 1
            meta = json.loads((Path(directory) / 'kernel-metadata.json').read_text())
            assert meta['is_private'] is True
            nb = (Path(directory) / 'job.ipynb').read_text()
            self.jobs.append({'reference': meta['id'] + '/1', 'seconds': timeout,
                              'charged': False, 'final': '--finalize' in nb})
            return SimpleNamespace(error='', version_number=1)
        def kernels_output(self, reference, destination, **kwargs):
            job = next(job for job in self.jobs if job['reference'] == reference)
            root = Path(destination) / 'voicesep-checkpoints'
            root.mkdir(parents=True, exist_ok=True)
            (root / 'run-summary.json').write_text(json.dumps({
                'step': 6000 + len(self.jobs)*1000, 'elapsed_seconds': job['seconds'] - 300,
                'training_finalized': job['final'], 'evaluation_complete': job['final'],
                'numeric_gate_passed': False}))
            (root / 'voicesep-research.pt').write_bytes(b'checkpoint software fixture')
            return [], None
    api = Api()
    flow = fixture_flow(tmp_path, api)
    flow.metadata = lambda *args: SimpleNamespace(is_private=True, enable_gpu=True, enable_tpu=False)
    # Keep the inference install confined to this test workspace.
    original_copy = budget.shutil.copyfile
    def copy(source, destination):
        return original_copy(source, tmp_path / 'installed.pt')
    import unittest.mock
    with unittest.mock.patch.object(budget.shutil, 'copyfile', copy):
        flow.run()
    assert len(api.jobs) == 3
    assert [job['seconds'] for job in api.jobs] == [21600, 21600, 14400]
    assert api.hours >= 1 and flow.state['phase'] == 'complete'
    assert flow.state['release_ready'] is False
    flow.run()
    assert len(api.jobs) == 3

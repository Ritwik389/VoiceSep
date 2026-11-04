"""Finish the existing run inside a fixed free-GPU allowance and untouched reserve.

No quota refresh is credited. Each submission is charged its entire hard timeout
against the local allowance before submission. Completed stages may return only
verified unused time; uncertain submissions are never repeated automatically.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('voicesep_previous_workflow', HERE / 'run_kaggle_workflow.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def allocation(remaining_hours, reserve_hours, allowance_seconds, final=False, quota_rate=1):
    """Conservative hard runtime, rounded down, with a separately reserved final job."""
    # Also allow 10 min for platform startup/teardown outside the notebook clock.
    available = min(math.floor((remaining_hours - reserve_hours) * 3600), allowance_seconds) - 600
    if final:
        if available < 7200 * quota_rate:
            raise RuntimeError('Insufficient budget for final calibration and evaluation with reserve intact')
        return min(14400, available // quota_rate)
    # Hold back four hours for the final job. Never start a tiny training continuation.
    final_reserve = 7200 if quota_rate > 1 else 14400
    runtime = min(21600, (available - final_reserve * quota_rate) // quota_rate)
    if runtime < 3600:
        return 0
    return runtime


class BudgetWorkflow(previous.Workflow):
    def __init__(self, api, owner, template, output, initial_gpu_hours, reserve_hours=1, metadata=previous.remote_metadata):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', owner) or reserve_hours < 1:
            raise ValueError('Expected an owned private account and at least a one-hour reserve')
        self.api, self.owner, self.metadata = api, owner, metadata
        self.output, self.template = Path(output).resolve(), Path(template).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.poll_seconds = 45
        self.state_path = self.output / 'status.json'
        signature = hashlib.sha256((self.template / 'train/job.ipynb').read_bytes()
                                   + (self.template / 'train/kernel-metadata.json').read_bytes()).hexdigest()
        self.state = json.loads(self.state_path.read_text()) if self.state_path.is_file() else {}
        budget = max(0, math.floor((initial_gpu_hours - reserve_hours) * 3600))
        policy = {'owner': owner, 'template_sha256': signature, 'initial_gpu_hours': initial_gpu_hours,
                  'reserve_hours': reserve_hours, 'allowance_seconds': budget}
        if self.state:
            if self.state['policy'] != policy:
                raise ValueError('Do not change a saved quota policy or notebook snapshot on restart')
        else:
            self.save(policy=policy, unallocated_seconds=budget, stages=[],
                      parent=f'{owner}/voicesep-gpu-training-02/1', next_run=3,
                      preparation=f'{owner}/voicesep-data-preparation/6',
                      benchmark=f'{owner}/voicesep-gpu-benchmark/2', phase='ready', quota_rate=2,
                      release_ready=False)

    def gpu_hours(self):
        quota = self.read(self.api.quota_view).gpu_quota
        return (quota.total_time_allowed - quota.time_used).total_seconds() / 3600 if quota else 0

    def notebook(self, number, parent, seconds, final):
        folder = self.output / 'jobs' / f'{number:02}'
        folder.mkdir(parents=True, exist_ok=True)
        nb = json.loads((self.template / 'train/job.ipynb').read_text())
        setup = ''.join(nb['cells'][1]['source'])
        # Include installation/data mounting time in the remote wall budget.
        setup = "import time, os\nos.environ['VOICESEP_JOB_STARTED'] = str(time.time())\n" + setup
        nb['cells'][1]['source'] = setup.splitlines(keepends=True)
        # sys.executable must be evaluated on the Kaggle host, not on this Mac.
        options = ['scripts/kaggle_budget_stage.py', '--wall-seconds', str(seconds)]
        if final:
            options.append('--finalize')
        if number == 3:
            options.append('--benchmark-parallel')
        cell = f"subprocess.run([sys.executable, *{options!r}], check=True)\n"
        nb['cells'][-1]['source'] = cell.splitlines(keepends=True)
        nb['cells'][0]['source'] = [f'# VoiceSep quota-bounded {"finalization" if final else "training"} {number:02}\n',
            f'Private research continuation; hard limit {seconds / 3600:.2f}h. No release quality is claimed.\n']
        (folder / 'job.ipynb').write_text(json.dumps(nb, indent=2))
        meta = json.loads((self.template / 'train/kernel-metadata.json').read_text())
        meta.update(id=f'{self.owner}/voicesep-gpu-training-{number:02}',
                    title=f'VoiceSep GPU training {number:02}',
                    kernel_sources=[self.state['preparation'], self.state['benchmark'], parent])
        if (not meta.get('is_private') or not meta.get('enable_gpu') or meta.get('enable_tpu')
                or meta.get('machine_shape') != 'NvidiaTeslaT4'
                or meta.get('dataset_sources') != [f'{self.owner}/voicesep-research-source']):
            raise ValueError('Only private free-T4 continuations are permitted')
        (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
        return folder

    def run(self):
        if self.state['phase'] == 'complete':
            return
        if self.state['phase'] == 'installing':
            self.install()
            return
        if self.state['phase'] == 'submitting':
            raise RuntimeError('Submission response was interrupted. Audit Kaggle; do not submit a duplicate.')
        while True:
            self.check_stop()
            stages = self.state['stages']
            if stages and stages[-1]['status'] == 'running':
                stage = stages[-1]
            else:
                parent = self.state['parent']
                self.wait(parent)
                meta = self.read(self.metadata, self.api, parent)
                if not meta.is_private or not meta.enable_gpu or meta.enable_tpu:
                    raise ValueError('The resume parent must be a private GPU run')
                # Fetch quota after any potentially long network wait, immediately before allocation.
                remaining = self.gpu_hours()
                final = bool(self.state.get('final_next'))
                seconds = allocation(remaining, self.state['policy']['reserve_hours'],
                                     self.state['unallocated_seconds'], final, self.state['quota_rate'])
                if not seconds:
                    final = True
                    seconds = allocation(remaining, self.state['policy']['reserve_hours'],
                                         self.state['unallocated_seconds'], True, self.state['quota_rate'])
                number = self.state['next_run']
                folder = self.notebook(number, parent, seconds, final)
                reserved = seconds * self.state['quota_rate'] + 600
                self.save(phase='submitting', current_submission=number,
                          gpu_hours_before_submission=remaining,
                          unallocated_seconds=self.state['unallocated_seconds'] - reserved)
                # Never automatically retry mutations. The saved phase makes reentry fail closed.
                response = self.api.kernels_push(str(folder), timeout=seconds, acc='NvidiaTeslaT4')
                if response.error or not response.version_number:
                    raise RuntimeError(f'Kaggle rejected the bounded run: {response.error}')
                stage = {'reference': f'{self.owner}/voicesep-gpu-training-{number:02}/{response.version_number}',
                         'run': number, 'hard_seconds': seconds, 'final': final, 'status': 'running',
                         'gpu_hours_before': remaining, 'reserved_quota_seconds': reserved}
                stages.append(stage)
                self.save(stages=stages, phase='waiting', current_kernel=stage['reference'])
            self.wait(stage['reference'])
            destination = self.output / f"session-{stage['run']:02}"
            self.download(stage['reference'], destination,
                r'(run-summary\.json|throughput\.json|last\.pt|best\.pt|best-calibrated\.pt|voicesep-research\.pt|listening-examples\.zip|train\.ya?ml|train\.jsonl|validation\.jsonl|execution-selection\.json|parallel-benchmark.*\.json|evaluation[^/]*/.*\.(json|md))$')
            summaries = list(destination.rglob('run-summary.json'))
            if len(summaries) != 1:
                raise RuntimeError('No final run summary. Preserve checkpoints and inspect the failed job.')
            summary = json.loads(summaries[0].read_text())
            stage.update(status='complete', summary=summary)
            after = self.gpu_hours()
            used = max(0, math.ceil((stage['gpu_hours_before'] - after) * 3600))
            # A quota reset is never credited; the initial allowance is immutable.
            if after > stage['gpu_hours_before'] + .01:
                raise RuntimeError('Quota balance increased unexpectedly; do not use refreshed quota')
            # Initially reserve for two GPU-hours per wall hour. Only lower this after
            # a substantial completed dual-device run confirms wall-time accounting.
            elapsed = summary.get('elapsed_seconds', 0)
            rate = 1 if elapsed > 300 and elapsed * .8 <= used <= elapsed * 1.2 else self.state['quota_rate']
            unused = stage['reserved_quota_seconds'] - used
            self.save(stages=stages, parent=stage['reference'], next_run=stage['run'] + 1,
                      final_next=summary.get('training_finalized', False),
                      latest_step=summary['step'], gpu_hours_after=after, quota_rate=rate,
                      unallocated_seconds=self.state['unallocated_seconds'] + unused, error=None,
                      phase='installing' if stage['final'] else 'ready')
            if stage['final']:
                if not summary.get('evaluation_complete'):
                    raise RuntimeError('Final evaluation did not complete; do not call this a release')
                self.install()
                return

    def install(self):
        stage = self.state['stages'][-1]
        summary = stage['summary']
        if not stage['final'] or not summary.get('evaluation_complete'):
            raise RuntimeError('Final evaluation must complete before local installation')
        destination = self.output / f"session-{stage['run']:02}"
        artifact = next(destination.rglob('voicesep-research.pt'))
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != summary.get('inference_checkpoint_sha256'):
            raise ValueError('Downloaded inference checkpoint does not match its recorded SHA-256')
        evaluation = next(destination.rglob('evaluation/evaluation.json'))
        report = json.loads(evaluation.read_text())
        if (report.get('automatic_count') is not True or
                report.get('numeric_gate_passed') != summary.get('numeric_gate_passed')):
            raise ValueError('Downloaded evaluation is inconsistent with the final automatic-count report')
        root = HERE.parent
        local = root / 'models' / 'voicesep-research.pt'
        temporary = local.with_suffix('.tmp')
        shutil.copyfile(artifact, temporary)
        temporary.replace(local)
        # The local app's documented default, installed only after held-out scoring.
        default = local.with_name('best.pt')
        if not default.exists():
            shutil.copyfile(artifact, temporary)
            temporary.replace(default)
        subprocess.run([sys.executable, str(HERE / 'package_source.py')], check=True)
        package = self.output / 'voicesep-research-bundle.zip'
        partial = package.with_suffix('.tmp')
        with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(root / 'dist/voicesep-source.zip', 'voicesep-source.zip')
            archive.write(artifact, 'voicesep-research.pt')
            archive.write(evaluation, 'evaluation.json')
            archive.write(evaluation.with_suffix('.md'), 'evaluation.md')
            archive.writestr('run-summary.json', json.dumps(summary, indent=2) + '\n')
        partial.replace(package)
        examples = next(destination.rglob('listening-examples.zip'), None)
        self.save(phase='complete', inference_checkpoint=str(local), release_ready=False,
                  numeric_gate_passed=summary.get('numeric_gate_passed', False),
                  evaluation_report=str(evaluation), research_bundle=str(package),
                  listening_examples=str(examples) if examples else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/kaggle-quota-finish'))
    parser.add_argument('--initial-gpu-hours', type=float, required=True)
    parser.add_argument('--reserve-hours', type=float, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        flow = BudgetWorkflow(api, args.owner, args.template, args.output,
                              args.initial_gpu_hours, args.reserve_hours)
        try:
            flow.run()
        except Exception as error:
            flow.save(error=re.sub(r'hf_[A-Za-z0-9]+', '[redacted]', str(error)))
            raise SystemExit(str(error)) from None

"""One bounded run: wait for the UI-prepared corpus, benchmark, then train once.

Run with the isolated Kaggle Python environment. No credential values are saved.
The controller never creates datasets, enables billing, or retries a submission.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import time


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_corpus(folder):
    candidates = list(Path(folder).rglob('preparation.json'))
    if len(candidates) != 1:
        raise ValueError('Expected exactly one completed corpus preparation report')
    root = candidates[0].parent
    report = json.loads(candidates[0].read_text())
    if report.get('stage') != 'data_prepared' or report.get('sample_rate') != 8000:
        raise ValueError('Corpus preparation did not complete at 8 kHz')
    seen = set()
    for name in ('train', 'validation', 'test'):
        rows = read_rows(root / f'{name}.jsonl')
        identities = {row['identity'] for row in rows}
        if identities & seen:
            raise ValueError('Speaker leakage between corpus partitions')
        seen |= identities
        for language in ('en', 'hi'):
            if len({row['identity'] for row in rows if row['language'] == language}) < 5:
                raise ValueError(f'Insufficient {language} speakers in {name}')
        if any(Path(row['path']).is_absolute() or '..' in Path(row['path']).parts for row in rows):
            raise ValueError('Corpus paths must be portable within the input directory')
        if name != 'train':
            digest = hashlib.sha256((root / f'{name}.jsonl').read_bytes()).hexdigest()
            recipes = read_rows(root / f'{name}-mixtures.jsonl')
            if not recipes or any(row.get('source_manifest_sha256') != digest for row in recipes):
                raise ValueError('Frozen benchmark does not match its source manifest')
    return report


def validate_benchmark(folder):
    paths = list(Path(folder).rglob('benchmark-selection.json'))
    if len(paths) != 1:
        raise ValueError('Expected one successful GPU benchmark report')
    choice = json.loads(paths[0].read_text())
    report = choice['benchmark']
    if (choice.get('segment_seconds') not in (2, 4)
            or report.get('steps_this_run') != 100
            or report.get('benchmark_only') is not True
            or not str(report.get('device', '')).startswith('cuda')
            or not 0 < report.get('peak_cuda_memory_bytes', 0) <= 16 * 1024 ** 3
            or report.get('seconds_per_step', 0) <= 0):
        raise ValueError('The 100-step CUDA benchmark did not pass the single-16-GB-device check')
    return choice


def remote_metadata(api, reference):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    owner, slug, *_ = reference.split('/')
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = owner, slug
    with api.build_kaggle_client() as client:
        return client.kernels.kernels_api_client.get_kernel(request).metadata


class Workflow:
    def __init__(self, api, owner, jobs, output, metadata=remote_metadata, poll_seconds=45,
                 after_preparation_version=0):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', owner):
            raise ValueError('Expected a Kaggle username')
        self.api, self.owner, self.metadata = api, owner, metadata
        self.jobs, self.output = Path(jobs).resolve(), Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.poll_seconds = poll_seconds
        if after_preparation_version < 0:
            raise ValueError('The previous preparation version cannot be negative')
        self.after_preparation_version = after_preparation_version
        self.state_path = self.output / 'status.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.is_file() else {}
        signatures = {}
        for stage, slug in (('benchmark', 'voicesep-gpu-benchmark'), ('train', 'voicesep-gpu-training-01')):
            directory = self.jobs / stage
            meta = json.loads((directory / 'kernel-metadata.json').read_text())
            if (meta['id'] != f'{owner}/{slug}' or meta.get('is_private') is not True
                    or meta.get('enable_gpu') is not True or meta.get('enable_tpu', False)
                    or meta.get('machine_shape') != 'NvidiaTeslaT4'
                    or meta.get('dataset_sources') != [f'{owner}/voicesep-research-source']
                    or meta.get('competition_sources') or meta.get('model_sources')):
                raise ValueError('Only the prepared private T4 jobs are allowed')
            expected = [f'{owner}/voicesep-data-preparation']
            if stage == 'train':
                expected.append(f'{owner}/voicesep-gpu-benchmark')
            if meta.get('kernel_sources') != expected or meta['code_file'] != 'job.ipynb':
                raise ValueError('Unexpected notebook inputs or code file')
            signatures[stage] = hashlib.sha256((directory/'job.ipynb').read_bytes()
                + (directory/'kernel-metadata.json').read_bytes()).hexdigest()
        if self.state and (self.state.get('owner') != owner or self.state.get('jobs') != signatures):
            raise ValueError('Keep the same owner and notebook snapshot when restarting this controller')
        if self.state.get('after_preparation_version', after_preparation_version) != after_preparation_version:
            raise ValueError('Keep the same preparation-version boundary when restarting')
        self.state.update(owner=owner, jobs=signatures, release_ready=False,
                          after_preparation_version=after_preparation_version)

    def save(self, **updates):
        self.state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = self.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, indent=2) + '\n')
        temporary.replace(self.state_path)
        print(json.dumps(updates), flush=True)

    def check_stop(self):
        if (self.output/'STOP').exists():
            raise InterruptedError('STOP file present; no further stages will be submitted')

    def wait(self, reference):
        while True:
            self.check_stop()
            status = self.api.kernels_status(reference).status.name
            if self.state.get('remote_status') != status or self.state.get('current_kernel') != reference:
                self.save(current_kernel=reference, remote_status=status)
            if status == 'COMPLETE':
                return
            if status not in ('QUEUED', 'RUNNING'):
                raise RuntimeError(f'{reference} stopped with {status}; inspect its logs before retrying')
            time.sleep(self.poll_seconds)

    def download(self, reference, destination, pattern):
        token = None
        while True:
            self.check_stop()
            _, token = self.api.kernels_output(reference, str(destination), file_pattern=pattern, force=True,
                                              quiet=True, page_size=200, page_token=token)
            if not token:
                break

    def submit(self, stage, inputs, required_hours, timeout):
        self.check_stop()
        if self.state.get(f'{stage}_kernel'):
            return self.state[f'{stage}_kernel']
        if self.state.get('phase') == f'submitting_{stage}':
            raise RuntimeError('Previous submission response was interrupted; inspect Kaggle before retrying')
        quota = self.api.quota_view().gpu_quota
        remaining = ((quota.total_time_allowed-quota.time_used).total_seconds()/3600) if quota else 0
        if remaining < required_hours:
            raise RuntimeError(f'Insufficient free GPU quota: {remaining:.2f} hours remain')
        directory = self.output/'jobs'/stage
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.jobs/stage/'job.ipynb', directory/'job.ipynb')
        meta = json.loads((self.jobs/stage/'kernel-metadata.json').read_text())
        meta['kernel_sources'] = inputs  # Pin the exact completed parent versions.
        (directory/'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
        self.save(phase=f'submitting_{stage}', free_gpu_hours_before_submission=remaining)
        response = self.api.kernels_push(str(directory), timeout=timeout, acc='NvidiaTeslaT4')
        if response.error or not response.version_number:
            raise RuntimeError(f'Kaggle rejected {stage}: {response.error or "missing version number"}')
        reference = f'{meta["id"]}/{response.version_number}'
        self.save(**{f'{stage}_kernel': reference}, phase=f'waiting_{stage}')
        deployed = self.metadata(self.api, meta['id'])
        if not deployed.is_private or not deployed.enable_gpu or deployed.enable_tpu:
            raise RuntimeError('Remote job metadata does not match the private GPU settings')
        return reference

    def run(self):
        if self.state.get('phase') == 'session_complete':
            return
        self.check_stop()
        if self.state.get('error'):
            self.save(error=None)
        if not self.state.get('preparation_kernel'):
            reference = f'{self.owner}/voicesep-data-preparation'
            while True:
                self.check_stop()
                meta = self.metadata(self.api, reference)
                if not meta.is_private or meta.enable_gpu or meta.enable_tpu or not meta.current_version_number:
                    raise ValueError('Preparation must be a private, versioned CPU run')
                if meta.current_version_number > self.after_preparation_version:
                    break
                self.save(phase='waiting_for_ui_restart', observed_preparation_version=meta.current_version_number)
                time.sleep(self.poll_seconds)
            self.save(preparation_kernel=f'{reference}/{meta.current_version_number}', phase='waiting_preparation')
        preparation = self.state['preparation_kernel']
        self.wait(preparation)
        corpus = self.output/'corpus-metadata'
        self.download(preparation, corpus, r'voicesep-corpus/(preparation\.json|(train|validation|test)(-mixtures)?\.jsonl)$')
        report = validate_corpus(corpus)
        self.save(corpus_hours=report.get('hours'), corpus_validated=True)
        benchmark = self.submit('benchmark', [preparation], required_hours=2, timeout=3600)
        self.wait(benchmark)
        results = self.output/'benchmark'
        self.download(benchmark, results, r'(benchmark-selection|throughput)\.json$')
        choice = validate_benchmark(results)
        self.save(benchmark=choice)
        # Reserve conservatively for two allocated T4s, even though this model uses one.
        training = self.submit('train', [preparation, benchmark], required_hours=16, timeout=28800)
        self.wait(training)
        results = self.output/'training'
        self.download(training, results, r'(run-summary\.json|throughput\.json|last\.pt|best\.pt|best-calibrated\.pt|train\.ya?ml|train\.jsonl|evaluation/.*\.(json|md))$')
        summaries = list(results.rglob('run-summary.json'))
        if len(summaries) != 1:
            raise ValueError('Training output is missing its final run summary')
        summary = json.loads(summaries[0].read_text())
        self.save(phase='session_complete', training_step=summary['step'],
                  resume_required=summary['resume_required'], release_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--jobs', type=Path, default=Path('dist/kaggle'))
    parser.add_argument('--output', type=Path, default=Path('runs/kaggle-workflow'))
    parser.add_argument('--after-preparation-version', type=int, default=0,
                        help='Wait for a newer UI-saved preparation version before pinning inputs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        workflow = Workflow(api, args.owner, args.jobs, args.output,
                            after_preparation_version=args.after_preparation_version)
        try:
            workflow.run()
        except Exception as error:
            message = re.sub(r'hf_[A-Za-z0-9]+', '[redacted HF token]', str(error))
            workflow.save(error=message)
            raise SystemExit(message) from None


if __name__ == '__main__':
    main()

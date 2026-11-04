"""Run one quota-bounded continuation on Kaggle; evaluate on the final stage.

The notebook's external Kaggle timeout is the hard limit. Every subprocess also
has a deadline, including package setup time recorded by the first notebook cell.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run_stage(wall_seconds, finalize=False, benchmark_parallel=False):
    import torch
    import yaml

    start = float(os.environ['VOICESEP_JOB_STARTED'])
    deadline = start + wall_seconds
    persist = Path('/kaggle/working/voicesep-checkpoints')
    persist.mkdir(exist_ok=True)
    config = yaml.safe_load(Path('configs/train.yaml').read_text())
    choices = list(Path('/kaggle/input').rglob('benchmark-selection.json'))
    assert len(choices) == 1
    config['data']['segment_seconds'] = json.loads(choices[0].read_text())['segment_seconds']
    text = yaml.safe_dump(config, sort_keys=False)
    parents = [p for p in Path('/kaggle/input').rglob('last.pt')
               if (p.parent.parent / 'train.yaml').is_file()]
    assert len(parents) == 1, 'Exactly one saved training parent is required.'
    parent = parents[0]
    assert (parent.parent.parent / 'train.yaml').read_text() == text
    run = persist / 'tflocoformer'
    shutil.copytree(parent.parent, run, dirs_exist_ok=True)
    (persist / 'train.yaml').write_text(text)
    checkpoint = run / 'last.pt'
    before = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert before['trained'] and before['step'] > 0
    parent_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    selection = {'microbatch_size': 1, 'data_parallel': False, 'effective_batch_size': 8}
    previous_selection = parent.parent.parent / 'execution-selection.json'
    if previous_selection.is_file():
        selection = json.loads(previous_selection.read_text())

    def remaining():
        return deadline - time.time()

    def cli(*args, cap=None):
        allowed = min(remaining() - 120, cap or float('inf'))
        if allowed <= 0:
            raise TimeoutError('Job deadline reached; no further compute is authorized')
        subprocess.run([sys.executable, '-m', 'voicesep.cli', *map(str, args)],
                       check=True, timeout=allowed)

    # Separate process and temporary output: benchmark updates never enter training.
    if benchmark_parallel and torch.cuda.device_count() >= 2 and remaining() > 3600:
        destination = Path('/kaggle/temp/voicesep-dual-t4-benchmark')
        try:
            cli('train', '--config', persist / 'train.yaml', '--output', destination,
                '--resume', checkpoint, '--device', 'cuda', '--benchmark',
                '--microbatch-size', 2, '--data-parallel', '--max-seconds', 1500, cap=1800)
            report = json.loads((destination / 'throughput.json').read_text())
            (persist / 'parallel-benchmark.json').write_text(json.dumps(report, indent=2))
            # Validate a full 100 successful updates, not just successful allocation.
            if (report['steps_this_run'] == 100 and report['seconds_per_step'] < 7.7
                    and report['amp_skipped_updates'] <= 4):
                selection = {'microbatch_size': 2, 'data_parallel': True,
                             'effective_batch_size': config['training']['accumulation']}
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            # The original checkpoint remains intact; release the failed child allocations.
            (persist / 'parallel-benchmark-error.json').write_text(json.dumps({
                'error': type(error).__name__, 'fallback': 'verified single-T4 path'}))
    (persist / 'execution-selection.json').write_text(json.dumps(selection, indent=2))
    print({'execution': selection}, flush=True)
    # Final stage reserves a full hour for calibration/test scoring and 10 min to save.
    # Early stopping in an intermediate stage leads to a separate evaluation stage.
    reserve = 4200 if finalize else 300
    training_seconds = int(remaining() - reserve)
    if training_seconds > 60 and before.get('stale_evaluations', 0) < config['training']['early_stopping_evaluations']:
        options = ['--microbatch-size', selection['microbatch_size']]
        if selection['data_parallel']:
            options.append('--data-parallel')
        cli('train', '--config', persist / 'train.yaml', '--output', run,
            '--resume', checkpoint, '--device', 'cuda', '--max-seconds', training_seconds,
            *options, cap=training_seconds + 180)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    early_stop = state.get('stale_evaluations', 0) >= config['training']['early_stopping_evaluations']
    target_reached = state['step'] >= config['training']['steps']
    finished = finalize or early_stop or target_reached
    reason = 'early_stopping' if early_stop else ('target_steps' if target_reached else
             ('quota_budget' if finalize else 'session_budget'))
    summary = {'step': state['step'], 'starting_step': before['step'],
               'parent_checkpoint_sha256': parent_sha, 'stop_reason': reason,
               'original_step_target': config['training']['steps'],
               'training_finalized': finished, 'resume_required': not finished,
               'release_ready': False, 'evaluation_complete': False,
               'elapsed_seconds': time.time() - start}
    def save_summary():
        (persist / 'run-summary.json').write_text(json.dumps(summary, indent=2))
        print(summary, flush=True)
    save_summary()
    if not finalize:
        return summary
    assert (run / 'best.pt').is_file(), 'No validation-selected checkpoint is available'
    best = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    summary['selected_step'] = best['step']
    summary['best_validation_selection_score'] = best['best_score']
    save_summary()
    # Calibration uses validation only. Test results never select a checkpoint.
    cli('calibrate', run / 'best.pt', 'data/validation.jsonl', 'data/validation-mixtures.jsonl',
        '--destination', persist / 'best-calibrated.pt', '--device', 'cuda', cap=1800)
    cli('evaluate', persist / 'best-calibrated.pt', 'data/test.jsonl', 'data/test-mixtures.jsonl',
        '--output', persist / 'evaluation', '--device', 'cuda', cap=1800)
    report = json.loads((persist / 'evaluation/evaluation.json').read_text())
    summary.update(evaluation_complete=True, numeric_gate_passed=report['numeric_gate_passed'])
    # Compact inference artifact excludes optimizer/RNG; keep provenance and research flags.
    calibrated = torch.load(persist / 'best-calibrated.pt', map_location='cpu', weights_only=True)
    compact = {key: value for key, value in calibrated.items()
               if key not in {'optimizer', 'scheduler', 'scaler', 'rng'}}
    compact['budget_finalization'] = {key: summary[key] for key in
        ('step', 'selected_step', 'stop_reason', 'original_step_target', 'parent_checkpoint_sha256')}
    compact['parent_checkpoint_sha256'] = report['checkpoint_sha256']
    torch.save(compact, persist / 'voicesep-research.pt')
    summary['inference_checkpoint_sha256'] = hashlib.sha256((persist / 'voicesep-research.pt').read_bytes()).hexdigest()
    summary['elapsed_seconds'] = time.time() - start
    save_summary()
    if remaining() > 180:
        # Deterministic examples, never selected by favorable test scores.
        from voicesep.inference import SeparationEngine
        from voicesep.research.data import MixtureFactory, read_jsonl
        import soundfile as sf
        import zipfile
        examples = persist / 'listening-examples'
        examples.mkdir(exist_ok=True)
        engine = SeparationEngine(persist / 'voicesep-research.pt', 'cuda')
        factory = MixtureFactory('data/test.jsonl', slots=engine.model.config.slots)
        selected, index = set(), []
        scores = {row['id']: row for row in report['results']}
        for recipe in read_jsonl('data/test-mixtures.jsonl'):
            group = (recipe['condition'], recipe['speaker_count'])
            if group[1] not in (4, 5) or group in selected:
                continue
            selected.add(group)
            mix, refs = factory.render(recipe)
            prediction = engine.separate(mix)
            folder = examples / f'{group[0]}-{group[1]}'
            folder.mkdir(exist_ok=True)
            sf.write(folder / 'mixture.wav', mix, engine.sample_rate, subtype='PCM_16')
            for number, wave in enumerate(prediction['waveforms'], 1):
                sf.write(folder / f'predicted-{number}.wav', wave, engine.sample_rate, subtype='PCM_16')
            for number, wave in enumerate(refs[:group[1]], 1):
                sf.write(folder / f'reference-{number}.wav', wave, engine.sample_rate, subtype='PCM_16')
            index.append({'id': recipe['id'], 'folder': folder.name, **scores[recipe['id']]})
        (examples / 'manifest.json').write_text(json.dumps(index, indent=2))
        with zipfile.ZipFile(persist / 'listening-examples.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
            for path in examples.rglob('*'):
                if path.is_file():
                    bundle.write(path, path.relative_to(examples))
        del engine
        torch.cuda.empty_cache()
    # Optional known-count diagnostic only if the mandatory automatic-count report is safe.
    if remaining() > 900:
        try:
            cli('evaluate', persist / 'best-calibrated.pt', 'data/test.jsonl', 'data/test-mixtures.jsonl',
                '--output', persist / 'evaluation-known-count', '--device', 'cuda', '--oracle',
                cap=remaining() - 300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            summary['optional_known_count_diagnostic'] = 'incomplete; automatic-count report preserved'
            save_summary()
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-seconds', type=int, required=True)
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--benchmark-parallel', action='store_true')
    args = parser.parse_args()
    run_stage(args.wall_seconds, args.finalize, args.benchmark_parallel)

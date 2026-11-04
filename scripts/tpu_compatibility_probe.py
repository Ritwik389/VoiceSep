"""Bounded TPU viability probe for this exact waveform/PIT pipeline, not training."""
import json
from pathlib import Path
import time
import traceback


def probe():
    report = {'purpose': 'compatibility_only', 'trained_checkpoint_produced': False,
              'usable_for_training': False, 'successful_updates': 0}
    destination = Path('/kaggle/working/tpu-compatibility.json')
    start = time.perf_counter()
    try:
        import numpy as np
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        from voicesep.models.separator import load_checkpoint
        from voicesep.research.data import MixtureFactory
        from voicesep.losses import hungarian_loss
        device = xm.xla_device()
        report.update(torch=torch.__version__, torch_xla=torch_xla.__version__, device=str(device))
        parents = [p for p in Path('/kaggle/input').rglob('best.pt')
                   if (p.parent.parent / 'train.yaml').is_file()]
        assert len(parents) == 1
        prepared = list(Path('/kaggle/input').rglob('preparation.json'))
        assert len(prepared) == 1
        model, _ = load_checkpoint(parents[0], 'cpu')
        factory = MixtureFactory(prepared[0].parent / 'train.jsonl', segment_seconds=4)
        mix, refs, _ = factory.draw(128, condition='en', count=5)
        cpu_input = torch.from_numpy(mix).unsqueeze(0)
        with torch.inference_mode():
            expected = model(cpu_input)['waveforms'].numpy()
        model.to(device)
        x = cpu_input.to(device)
        r = torch.from_numpy(refs).unsqueeze(0).to(device)
        model.eval()
        with torch.inference_mode():
            actual = model(x)['waveforms'].cpu().numpy()
        error = float(np.sqrt(np.mean((actual - expected) ** 2)) / max(np.sqrt(np.mean(expected ** 2)), 1e-7))
        report['relative_rms_error'] = error
        assert error < .01, 'TPU output differs materially from CPU'
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        timings = []
        for _ in range(3):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss, _ = hungarian_loss(output, r, x, model.config.hop, model.config.n_fft)
            loss.backward()
            xm.optimizer_step(optimizer)
            xm.mark_step()
            value = float(loss.cpu())
            assert np.isfinite(value)
            timings.append(time.perf_counter() - step_start)
            report['successful_updates'] += 1
            report['seconds_per_mixture_update'] = timings
            report['elapsed_seconds'] = time.perf_counter() - start
            destination.write_text(json.dumps(report, indent=2))
            print(report, flush=True)
        report['usable_for_training'] = True
        report['next_requirement'] = 'Full effective-batch benchmark and checkpoint-resume equivalence before training'
        report['xla_metrics'] = met.metrics_report()
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(limit=6))
        try:
            report['xla_metrics'] = met.metrics_report()
        except Exception:
            pass
    finally:
        report['elapsed_seconds'] = time.perf_counter() - start
        destination.write_text(json.dumps(report, indent=2))
        print(report, flush=True)


if __name__ == '__main__':
    probe()

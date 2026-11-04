from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import yaml

from voicesep.models.separator import VoiceSeparator
from voicesep.research.data import write_jsonl, freeze_benchmark
from voicesep.research.train import train


def test_optimizer_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    """An interrupted free-GPU run must not silently restart its optimizer or mixture sequence."""
    for split in ('train','validation'):
        rows=[]
        for lang in ('en','hi'):
            for i in range(5):
                audio=tmp_path/f'{split}-{lang}-{i}.wav'
                t=np.arange(1400)/8000
                sf.write(audio,np.sin(t*2*np.pi*(110+i*130))*.05,8000)
                rows.append({'path':str(audio),'speaker_id':f'{split}-{lang}-{i}','language':lang})
        write_jsonl(tmp_path/f'{split}.jsonl',rows)
    freeze_benchmark(tmp_path/'validation.jsonl',tmp_path/'validation-mixtures.jsonl',per_condition=1,seconds=.128)
    config={'seed':42,'model':{'slots':5,'layers':1,'dimension':16,'ffn_dimension':32},
            'data':{'manifest':str(tmp_path/'train.jsonl'),'validation':str(tmp_path/'validation-mixtures.jsonl'),'segment_seconds':.128},
            'training':{'steps':2,'warmup_steps':1,'accumulation':1,'backbone_lr':1e-4,'head_lr':5e-4,
                        'weight_decay':.01,'clip_grad_norm':5,'checkpoint_every':1,'validate_every':1,'amp':False}}
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    # Tiny software fixture: no speech-quality result or checkpoint is published from this test.
    monkeypatch.setattr(VoiceSeparator,'initialize_from_official',lambda self,path:None)
    import voicesep.research.train as trainer
    monkeypatch.setattr(trainer,'validate',lambda *args:1.)
    train(path,tmp_path/'whole',pretrained='fixture',device='cpu')
    train(path,tmp_path/'interrupted',pretrained='fixture',device='cpu',steps_override=1)
    train(path,tmp_path/'resumed',resume=tmp_path/'interrupted/last.pt',device='cpu')
    a=torch.load(tmp_path/'whole/last.pt',weights_only=True)
    b=torch.load(tmp_path/'resumed/last.pt',weights_only=True)
    assert a['draw_index']==b['draw_index']==2
    assert a['stale_evaluations']==b['stale_evaluations']==1
    assert all(torch.equal(a['model'][key],b['model'][key]) for key in a['model'])
    report=train(path,tmp_path/'time-limited',pretrained='fixture',device='cpu',max_seconds=1e-9)
    stopped=torch.load(tmp_path/'time-limited/last.pt',weights_only=True)
    assert report['stopped_for_time_budget'] is True
    assert stopped['step']==0 and stopped['draw_index']==0 and stopped['trained'] is False
    original_step=trainer._optimizer_step
    def skip_first_update():
        calls=0
        def apply(*args):
            nonlocal calls
            calls+=1
            return False if calls==1 else original_step(*args)
        return apply
    monkeypatch.setattr(trainer,'_optimizer_step',skip_first_update())
    train(path,tmp_path/'whole-skipped',pretrained='fixture',device='cpu')
    monkeypatch.setattr(trainer,'_optimizer_step',skip_first_update())
    train(path,tmp_path/'partial-skipped',pretrained='fixture',device='cpu',steps_override=1)
    monkeypatch.setattr(trainer,'_optimizer_step',original_step)
    train(path,tmp_path/'resumed-skipped',resume=tmp_path/'partial-skipped/last.pt',device='cpu')
    a=torch.load(tmp_path/'whole-skipped/last.pt',weights_only=True)
    b=torch.load(tmp_path/'resumed-skipped/last.pt',weights_only=True)
    assert a['draw_index']==b['draw_index']==3
    assert a['step']==b['step']==2
    assert all(torch.equal(a['model'][key],b['model'][key]) for key in a['model'])


def test_microbatch_preserves_effective_batch_and_resume(tmp_path, monkeypatch):
    import pytest
    for split in ('train', 'validation'):
        rows = []
        for lang in ('en', 'hi'):
            for i in range(5):
                audio = tmp_path / f'{split}-{lang}-{i}.wav'
                t = np.arange(1400) / 8000
                sf.write(audio, np.sin(t * 2 * np.pi * (110 + i * 130)) * .05, 8000)
                rows.append({'path': str(audio), 'speaker_id': f'{split}-{lang}-{i}', 'language': lang})
        write_jsonl(tmp_path / f'{split}.jsonl', rows)
    freeze_benchmark(tmp_path / 'validation.jsonl', tmp_path / 'validation-mixtures.jsonl', per_condition=1, seconds=.128)
    config = {'seed': 42, 'model': {'slots': 5, 'layers': 1, 'dimension': 16, 'ffn_dimension': 32},
              'data': {'manifest': str(tmp_path / 'train.jsonl'), 'validation': str(tmp_path / 'validation-mixtures.jsonl'), 'segment_seconds': .128},
              'training': {'steps': 2, 'warmup_steps': 0, 'accumulation': 2, 'backbone_lr': 1e-4, 'head_lr': 5e-4,
                           'weight_decay': .01, 'clip_grad_norm': 5, 'checkpoint_every': 1, 'validate_every': 1, 'amp': False}}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(VoiceSeparator, 'initialize_from_official', lambda self, path: None)
    import voicesep.research.train as trainer
    monkeypatch.setattr(trainer, 'validate', lambda *args: 1.)
    train(path, tmp_path / 'serial', pretrained='fixture', device='cpu')
    train(path, tmp_path / 'batched', pretrained='fixture', device='cpu', microbatch_size=2, steps_override=1)
    report = train(path, tmp_path / 'batched', resume=tmp_path / 'batched/last.pt', device='cpu', microbatch_size=2)
    a = torch.load(tmp_path / 'serial/last.pt', weights_only=True)
    b = torch.load(tmp_path / 'batched/last.pt', weights_only=True)
    assert a['step'] == b['step'] == 2 and a['draw_index'] == b['draw_index'] == 4
    assert a['training_config'] == b['training_config']
    assert report['effective_batch_size'] == 2 and report['microbatch_size'] == 2
    # Floating-point reduction order changes, but each update must use the same mean gradient.
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=2e-5, rtol=2e-4)
    with pytest.raises(ValueError, match='divide'):
        train(path, tmp_path / 'invalid', pretrained='fixture', device='cpu', microbatch_size=3)
    with pytest.raises(ValueError, match='two CUDA'):
        train(path, tmp_path / 'invalid-dp', pretrained='fixture', device='cpu', microbatch_size=2, data_parallel=True)

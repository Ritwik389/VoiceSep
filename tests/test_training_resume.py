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

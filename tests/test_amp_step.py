import pytest
import torch

from voicesep.research.train import _optimizer_step


def test_amp_overflow_skips_update_reduces_scale_then_allows_a_finite_step():
    model=torch.nn.Linear(2,1,bias=False)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    scaler=torch.amp.GradScaler('cpu',init_scale=8)
    original=model.weight.detach().clone()
    scaler.scale(model(torch.ones(1,2)).square().sum()).backward()
    model.weight.grad.fill_(float('inf'))
    assert _optimizer_step(model,optimizer,scaler,1) is False
    assert scaler.get_scale()==4
    assert torch.equal(model.weight,original)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(model(torch.ones(1,2)).square().sum()).backward()
    assert _optimizer_step(model,optimizer,scaler,1) is True
    assert not torch.equal(model.weight,original)


def test_full_precision_nonfinite_gradients_still_fail():
    model=torch.nn.Linear(2,1,bias=False)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    model(torch.ones(1,2)).square().sum().backward()
    model.weight.grad.fill_(float('nan'))
    with pytest.raises(RuntimeError,match='non-finite'):
        _optimizer_step(model,optimizer,scaler,1)

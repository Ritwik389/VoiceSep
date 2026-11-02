from dataclasses import asdict
import numpy as np
import pytest
import torch

from voicesep.models.separator import VoiceSeparator, ModelConfig, load_checkpoint
from voicesep.losses import hungarian_loss
from voicesep.inference import Calibration, detect_streams, align_streams
from voicesep.research.evaluate import score_mixture, summarize
from voicesep.losses import si_sdr


def test_permutation_invariance_and_silent_gradients():
    torch.manual_seed(11)
    references = torch.randn(1, 5, 512) * 0.05
    references[:, 3:] = 0
    predictions = (references + torch.randn_like(references) * 0.001).requires_grad_()
    logits = torch.randn(1, 5, 9, requires_grad=True)
    mixture = references.sum(1)
    first, _ = hungarian_loss({"waveforms": predictions, "activity_logits": logits}, references, mixture)
    permutation = [4, 1, 3, 0, 2]
    second, _ = hungarian_loss({"waveforms": predictions[:, permutation], "activity_logits": logits[:, permutation]}, references, mixture)
    assert torch.allclose(first, second)
    first.backward()
    assert torch.isfinite(predictions.grad).all() and torch.isfinite(logits.grad).all()
    zero = torch.zeros_like(references)
    silence, _ = hungarian_loss({"waveforms": zero.clone().requires_grad_(), "activity_logits": logits}, zero, zero.sum(1))
    assert torch.isfinite(silence)


@pytest.mark.parametrize("length", [1, 64, 1024])
def test_forward_backward_short_audio(length):
    model = VoiceSeparator(ModelConfig(layers=1, dimension=16, ffn_dimension=32))
    wave = torch.randn(1, length) * 0.02
    result = model(wave)
    assert result["waveforms"].shape == (1, 5, length)
    assert torch.isfinite(result["waveforms"]).all()
    result["waveforms"].square().mean().backward()
    assert model.backbone.deconv.weight.grad is not None


def test_untrained_checkpoint_cannot_be_used_for_inference(tmp_path):
    config = ModelConfig(layers=1, dimension=16, ffn_dimension=32)
    model = VoiceSeparator(config)
    checkpoint = tmp_path / "untrained.pt"
    torch.save({"format_version": 1, "trained": False, "model_config": asdict(config), "model": model.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="not been trained"):
        load_checkpoint(checkpoint)


def test_counting_scale_invariance_and_duplicate_suppression():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(2, 1000)).astype(np.float32)
    waves = np.stack([a, b, a.copy(), np.zeros_like(a), np.zeros_like(a)])
    probs = np.ones((5, 20), np.float32)
    selected, duplicates = detect_streams(waves, probs, a + b, Calibration())
    smaller, _ = detect_streams(waves * 0.01, probs, (a + b) * 0.01, Calibration())
    assert len(selected) == 2 and len(duplicates) == 1
    assert selected == smaller
    manual, _ = detect_streams(waves, probs, a + b, Calibration(), manual_count=5)
    assert len(manual) == 5


def test_overlap_tracking_recovers_permuted_streams():
    rng = np.random.default_rng(4)
    previous = rng.normal(size=(5, 800)).astype(np.float32)
    current = np.concatenate([previous[:, -200:], rng.normal(size=(5, 600))], -1)
    permutation = [2, 4, 0, 3, 1]
    order = align_streams(previous, current[permutation], 200)
    assert np.allclose(current[permutation][order, :200], previous[:, -200:])


def test_count_failures_are_not_removed_from_separation_scores():
    rng = np.random.default_rng(9)
    references = rng.normal(size=(4, 1000)).astype(np.float32) * 0.03
    mixture = references.sum(0)
    perfect = score_mixture(references, references, mixture)
    missing = score_mixture(references[:3], references, mixture)
    extra = score_mixture(np.r_[references, references[:1]], references, mixture)
    assert perfect["count_correct"] and not missing["count_correct"] and not extra["count_correct"]
    assert missing["missing"] == 1 and extra["false_positives"] == 1
    assert missing["si_sdri"] < perfect["si_sdri"] and extra["si_sdri"] < perfect["si_sdri"]
    assert not summarize([])["numeric_gate_passed"]


def test_zero_estimate_is_not_rewarded_as_zero_db():
    reference = torch.randn(1000)
    assert si_sdr(torch.zeros_like(reference), reference) <= -30


def test_brief_voice_is_not_dropped_from_long_recording():
    rng=np.random.default_rng(3)
    waves=np.zeros((2,10000),np.float32)
    waves[0]=rng.normal(size=10000)
    waves[1,3000:3400]=rng.normal(size=400)
    probabilities=(waves!=0).astype(np.float32)
    selected,_=detect_streams(waves,probabilities,waves.sum(0),Calibration(),presence_window=4000)
    assert selected==[0,1]

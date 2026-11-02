"""Waveform adapter around MERL's unmodified, vendored TF-Locoformer."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .tflocoformer_upstream import TFLocoformerSeparator

UPSTREAM_SHA256 = "9c691cee4bb0d9664a3fac024b031dfdf8ca3e66fc22a66f2f17119767b49129"


@dataclass
class ModelConfig:
    sample_rate: int = 8000
    slots: int = 5
    n_fft: int = 128
    hop: int = 64
    layers: int = 6
    dimension: int = 128
    ffn_dimension: int = 384
    heads: int = 4
    gradient_checkpointing: bool = False
    architecture: str = "tflocoformer"


class ActivityHead(nn.Module):
    """Shared per-stream temporal head; output slots retain permutation symmetry."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, spectra, mixture):
        power = spectra.abs().square().mean(-1)
        mix_power = mixture.abs().square().mean(-1).unsqueeze(1)
        features = torch.stack((torch.log1p(power), power / (mix_power + 1e-6),
                                torch.log1p(spectra.abs().mean(-1))), -1)
        return self.net(features).squeeze(-1)


class VoiceSeparator(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.architecture != "tflocoformer":
            raise ValueError(f"Unsupported architecture {c.architecture}")
        self.backbone = TFLocoformerSeparator(
            num_spk=c.slots, n_layers=c.layers, emb_dim=c.dimension,
            norm_type="rmsgroupnorm", num_groups=4, tf_order="ft", n_heads=c.heads,
            attention_dim=c.dimension, ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
            ffn_hidden_dim=[c.ffn_dimension, c.ffn_dimension], conv1d_kernel=4,
            conv1d_shift=1, dropout=0.0, eps=1e-5,
        )
        self.activity = ActivityHead()
        self._permanently_frozen = {id(p) for p in self.backbone.parameters() if not p.requires_grad}
        self.register_buffer("window", torch.hann_window(c.n_fft), persistent=False)

    def forward(self, waveform):
        if waveform.ndim != 2:
            raise ValueError("Expected waveform [batch, samples]")
        c = self.config
        length = waveform.shape[-1]
        scale = waveform.std(-1, keepdim=True, correction=0).clamp_min(1e-5)
        x = waveform / scale
        if length < c.n_fft:
            x = F.pad(x, (0, c.n_fft - length))
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = torch.stft(x.float(), c.n_fft, c.hop, window=self.window.float(),
                                  return_complex=True, center=True).transpose(-1, -2)
        if c.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            spectra = checkpoint(self.backbone, spectrum, use_reentrant=False)
        else:
            spectra = self.backbone(spectrum)
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            activity = self.activity(spectra, spectrum)
            b, s, t, f = spectra.shape
            waves = torch.istft(spectra.transpose(-1, -2).reshape(b * s, f, t),
                               c.n_fft, c.hop, window=self.window.float(), length=x.shape[-1])
            waves = waves.reshape(b, s, -1)[..., :length] * scale.unsqueeze(1)
        return {"waveforms": waves, "activity_logits": activity}

    def initialize_from_official(self, path: str | Path):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != UPSTREAM_SHA256:
            raise ValueError("Official checkpoint SHA-256 mismatch; refusing to load")
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {k.removeprefix("separator."): v for k, v in state.items()
                 if k.startswith("separator.")}
        if not state:
            raise ValueError("No separator weights in upstream checkpoint")
        if self.config.slots != 2:
            state = {k: v for k, v in state.items() if k not in {"deconv.weight", "deconv.bias"}}
        result = self.backbone.load_state_dict(state, strict=False)
        expected = {"deconv.weight", "deconv.bias"} if self.config.slots != 2 else set()
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise ValueError(f"Shared backbone incompatibility: {result}")
        return {"loaded_tensors": len(state), "new_output_slots": self.config.slots,
                "five_speaker_trained": False}

    def freeze_backbone(self, frozen: bool):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not frozen and id(parameter) not in self._permanently_frozen)
        for parameter in self.backbone.deconv.parameters():
            parameter.requires_grad_(True)


def load_checkpoint(path: str | Path, device="cpu", require_trained=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Expected a VoiceSep training checkpoint, not raw upstream weights")
    if require_trained and not checkpoint.get("trained", False):
        raise ValueError("This checkpoint has not been trained")
    config = ModelConfig(**checkpoint["model_config"])
    if config.architecture == "convtasnet":
        from .baseline import ConvTasNetSeparator
        model = ConvTasNetSeparator(config)
    else:
        model = VoiceSeparator(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint

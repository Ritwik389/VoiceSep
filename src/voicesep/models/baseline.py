"""Optional lighter comparator, using the same activity/counting and loss pipeline."""
import torch
from torch import nn
from .separator import ModelConfig, ActivityHead


class ConvTasNetSeparator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        from asteroid.models import ConvTasNet
        self.config = config
        self.backbone = ConvTasNet(n_src=config.slots, n_filters=512, kernel_size=16, stride=8,
                                   bn_chan=128, hid_chan=512, skip_chan=128,
                                   n_blocks=8, n_repeats=3, mask_act="relu", sample_rate=8000)
        self.activity = ActivityHead()
        self.register_buffer("window", torch.hann_window(config.n_fft), persistent=False)

    def forward(self, wave):
        waves = self.backbone(wave)
        c = self.config
        b, s, n = waves.shape
        spectra = torch.stft(waves.reshape(b * s, n), c.n_fft, c.hop,
                              window=self.window, return_complex=True).transpose(-1, -2)
        spectra = spectra.reshape(b, s, *spectra.shape[-2:])
        mixture = torch.stft(wave, c.n_fft, c.hop, window=self.window,
                             return_complex=True).transpose(-1, -2)
        return {"waveforms": waves, "activity_logits": self.activity(spectra, mixture)}

    def freeze_backbone(self, frozen):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not frozen)
        for parameter in self.backbone.masker.mask_net.parameters():
            parameter.requires_grad_(True)

    def initialize_from_official(self, path=None):
        from asteroid.models import BaseModel
        pretrained = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri3Mix_sepclean_8k")
        target = self.backbone.state_dict()
        source = pretrained.state_dict()
        incompatible = {k for k in source if k in target and source[k].shape != target[k].shape}
        if any(not k.startswith("masker.mask_net.") for k in incompatible):
            raise ValueError(f"Unexpected shared backbone mismatch: {incompatible}")
        result = self.backbone.load_state_dict({k: v for k, v in source.items() if k not in incompatible}, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != incompatible:
            raise ValueError(f"Unexpected backbone keys: {result}")
        return {"five_speaker_trained": False, "license": "See upstream model's share-alike license notice"}

"""Pinned, local Hugging Face speaker-analysis models. Weights keep publisher terms."""
import os
from pathlib import Path

AGE_MODEL = 'audeering/wav2vec2-large-robust-6-ft-age-gender'
AGE_REVISION = 'a681b720dafd12b9dd7b6d13fb437c7b6b197fd3'
EMOTION_MODEL = 'emotion2vec/emotion2vec_plus_base'
EMOTION_REVISION = 'b318240bfe67db81a8c572ecb37ce9c3759b81c9'
EMOTIONS = ('angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown')
MODEL_INFO = {
    'age_vocal_presentation': {'id': AGE_MODEL, 'revision': AGE_REVISION, 'license': 'CC-BY-NC-SA-4.0'},
    'emotion': {'id': EMOTION_MODEL, 'revision': EMOTION_REVISION,
                'license': 'Publisher model-license; see the model card and FunASR MODEL_LICENSE'},
}


def model_path(kind, download=True):
    # Keep model/cache writes in the project; TLS uses the installed certificate bundle.
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import snapshot_download
    info = MODEL_INFO[kind]
    patterns = (['config.json', 'preprocessor_config.json', 'model.safetensors', 'README.md', 'LICENSE']
                if kind == 'age_vocal_presentation' else
                ['config.yaml', 'configuration.json', 'model.pt', 'tokens.txt', 'README.md'])
    cache = Path(os.getenv('VOICESEP_ANALYSIS_CACHE', 'models/analysis')).resolve()
    return Path(snapshot_download(info['id'], revision=info['revision'], cache_dir=cache,
                                  allow_patterns=patterns, local_files_only=not download))


def download_models():
    return {kind: str(model_path(kind)) for kind in MODEL_INFO}


class AgeVocalModel:
    """Architecture corresponding to audEERING's published six-layer age/gender heads."""
    def __init__(self):
        import torch
        from torch import nn
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class Head(nn.Module):
            def __init__(self, config, outputs):
                super().__init__()
                self.dropout = nn.Dropout(config.final_dropout)
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.out_proj = nn.Linear(config.hidden_size, outputs)

            def forward(self, features):
                hidden = self.dense(self.dropout(features)).tanh()
                return self.out_proj(self.dropout(hidden))

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = Head(config, 1)
                self.gender = Head(config, 3)
                self.post_init()

            def forward(self, input_values):
                pooled = self.wav2vec2(input_values).last_hidden_state.mean(dim=1)
                return self.age(pooled), self.gender(pooled).softmax(dim=-1)

        path = model_path('age_vocal_presentation')
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)
        self.model, loading = AgeGenderModel.from_pretrained(
            path, local_files_only=True, use_safetensors=True, output_loading_info=True)
        if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys')):
            raise RuntimeError('Age model weights do not match the published architecture')
        self.model.eval().to('cpu')

    def predict(self, waveform):
        import torch
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors='pt')['input_values']
        with torch.inference_mode():
            age, probabilities = self.model(inputs)
        return float(age[0, 0]) * 100, probabilities[0].cpu().numpy()


class EmotionModel:
    def __init__(self):
        import torch
        from omegaconf import OmegaConf
        from funasr.models.emotion2vec.model import Emotion2vec
        path = model_path('emotion')
        config = OmegaConf.to_container(OmegaConf.load(path / 'config.yaml'), resolve=True)
        # The published classifier omits the reconstruction decoder used in pretraining.
        config['model_conf']['modalities']['audio']['decoder'] = None
        self.model = Emotion2vec(**config, vocab_size=len(EMOTIONS))
        state = torch.load(path / 'model.pt', weights_only=True, map_location='cpu', mmap=True)['model']
        weights = {name.removeprefix('d2v_model.'): value for name, value in state.items()}
        self.model.load_state_dict(weights, strict=True)
        self.model.eval().to('cpu')

    def predict(self, waveform):
        import torch
        import torch.nn.functional as functional
        signal = torch.from_numpy(waveform).float().reshape(1, -1)
        with torch.inference_mode():
            signal = functional.layer_norm(signal, signal.shape)
            features = self.model.extract_features(signal, padding_mask=None)['x']
            scores = self.model.proj(features.mean(dim=1)).softmax(dim=-1)[0]
        if scores.shape != (len(EMOTIONS),) or not torch.isfinite(scores).all():
            raise RuntimeError('Emotion model returned invalid class scores')
        return scores.cpu().numpy()

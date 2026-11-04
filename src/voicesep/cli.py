from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request

import typer

app = typer.Typer(no_args_is_help=True, help="VoiceSep research and local audio workspace")


def emit(value):
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


@app.command()
def doctor(checkpoint: Path = Path("models/best.pt")):
    """Inspect compute, FFmpeg, and trained-checkpoint availability."""
    import torch
    from .service import Worker
    from .storage import JobStore
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        state = Worker(JobStore(directory), checkpoint).health()
    emit({"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.cuda.is_available(),
          "mps": torch.backends.mps.is_available(), "ffmpeg": shutil.which("ffmpeg"), "model": state})


@app.command("download-backbone")
def download_backbone(destination: Path = Path("models/tflocoformer-libri2mix.pth")):
    """Download and checksum the official TWO-speaker backbone (not a trained five-speaker model)."""
    from .models.separator import UPSTREAM_SHA256
    revision = "7a615460d347ff7334a13dbb831d16280da72cdc"
    url = (f"https://media.githubusercontent.com/media/merlresearch/tf-locoformer/{revision}/"
           "egs2/librimix/enh1/exp/enh_train_enh_tflocoformer_raw/valid.loss.ave_5best.pth")
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == UPSTREAM_SHA256:
        emit({"path": str(destination), "sha256": UPSTREAM_SHA256, "already_verified": True})
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != UPSTREAM_SHA256:
            raise ValueError("Downloaded checkpoint checksum mismatch")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    emit({"path": str(destination), "sha256": UPSTREAM_SHA256, "five_speaker_trained": False})


@app.command("index-sources")
def index_sources(english_root: Path, hindi_manifest: Path, hindi_root: Path,
                  output: Path = Path("data/sources.jsonl"), min_snr: float = 10):
    """Index LibriSpeech and IV-R Hindi metadata; retain real speaker IDs and transcripts."""
    from .research.data import read_jsonl, write_jsonl
    rows = []
    transcripts = {}
    for transcript in english_root.rglob("*.trans.txt"):
        for line in transcript.read_text().splitlines():
            if " " in line:
                key, text = line.split(" ", 1)
                transcripts[key] = text
    for path in sorted(english_root.rglob("*.flac")):
        rows.append({"path": str(path.resolve()), "speaker_id": path.stem.split("-")[0],
                     "language": "en", "corpus": "librispeech", "text": transcripts.get(path.stem, "")})
    for source in read_jsonl(hindi_manifest):
        language = str(source.get("lang", source.get("language", "hi"))).lower()
        if language not in {"hi", "hin", "hindi"}:
            continue
        if source.get("snr") is not None and float(source["snr"]) < min_snr:
            continue
        filename = source.get("filename", source.get("path"))
        speaker = source.get("speaker_id")
        if not filename or speaker is None:
            raise ValueError("IV-R metadata requires filename and speaker_id")
        path = Path(filename)
        path = path if path.is_absolute() else hindi_root / path
        if not path.is_file():
            raise ValueError(f"Hindi audio missing: {path}")
        rows.append({"path": str(path.resolve()), "speaker_id": str(speaker), "language": "hi",
                     "corpus": "indicvoices-r", "text": source.get("text", source.get("normalized", ""))})
    if not rows or not any(r["language"] == "en" for r in rows) or not any(r["language"] == "hi" for r in rows):
        raise ValueError("Both English and Hindi sources are required")
    write_jsonl(output, rows)
    emit({"sources": len(rows), "manifest": str(output)})


@app.command("split")
def split(manifest: Path, output_dir: Path = Path("data"), seed: int = 42):
    """Split identities before making mixtures."""
    from .research.data import read_jsonl, split_sources
    emit(split_sources(read_jsonl(manifest), output_dir, seed))


@app.command("bootstrap-data")
def bootstrap_data(destination: Path = Path("data/corpus"), english_hours: float = 30, hindi_hours: float = 20):
    """Download a bounded real-speech corpus and freeze portable bilingual splits on CPU."""
    if english_hours <= 0 or hindi_hours <= 0:
        raise typer.BadParameter("Both corpus budgets must be positive")
    from .research.bootstrap import bootstrap
    emit(bootstrap(destination, english_hours, hindi_hours))


@app.command("freeze")
def freeze(manifest: Path, destination: Path, per_condition: int = 100, seconds: float = 4, seed: int = 12345):
    """Create a deterministic bilingual 2/3/4/5-speaker benchmark."""
    from .research.data import freeze_benchmark
    emit(freeze_benchmark(manifest, destination, per_condition, seconds=seconds, seed=seed))


@app.command("train")
def training(config: Path = Path("configs/train.yaml"), output: Path = Path("runs/tflocoformer"),
             pretrained: Path | None = None, resume: Path | None = None, device: str | None = None,
             steps: int | None = None, benchmark: bool = False, segment_seconds: float | None = None,
             max_seconds: int | None = None, microbatch_size: int = 1, data_parallel: bool = False):
    """Fine-tune, resume, or benchmark 100 steps. Free GPU notebooks are supplied."""
    from .research.train import train
    emit(train(config, output, pretrained, resume, device, steps, benchmark, segment_seconds, max_seconds,
               microbatch_size, data_parallel))


@app.command("calibrate")
def calibration(checkpoint: Path, sources: Path, mixtures: Path,
                destination: Path = Path("models/best-calibrated.pt"), device: str = "cpu", limit: int = 120):
    """Select activity/counting thresholds on validation mixtures only."""
    from .research.calibrate import calibrate
    emit(calibrate(checkpoint, sources, mixtures, destination, device, limit))


@app.command("evaluate")
def evaluation(checkpoint: Path, sources: Path, mixtures: Path,
               output: Path = Path("reports/evaluation"), device: str = "cpu", oracle: bool = False):
    """Score every output, including missed/extra speakers; emit JSON and Markdown."""
    from .research.evaluate import evaluate_checkpoint
    report = evaluate_checkpoint(checkpoint, sources, mixtures, output, device, oracle)
    emit({k: v for k, v in report.items() if k != "results"})


@app.command("separate")
def separate(audio: Path, checkpoint: Path = Path("models/best.pt"), output: Path = Path("runs/separated"),
             speakers: int | None = None, device: str = "cpu", speaker_embeddings: bool = False):
    """Separate actual audio using a fine-tuned checkpoint. No dummy fallback exists."""
    from .audio import read_audio, write_audio
    from .inference import SeparationEngine, SpeakerEmbedder
    wave = read_audio(audio)
    embedder = SpeakerEmbedder("models/ecapa") if speaker_embeddings else None
    engine = SeparationEngine(checkpoint, device, embedder)
    result = engine.separate(wave, speakers)
    output.mkdir(parents=True, exist_ok=True)
    for speaker, stream in zip(result["speakers"], result.pop("waveforms")):
        write_audio(output / f"{speaker['id']}.wav", stream, result["sample_rate"])
    (output / "results.json").write_text(json.dumps(result, indent=2))
    emit(result)


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the local API and the built frontend."""
    import uvicorn
    uvicorn.run("voicesep.api:create_app", factory=True, host=host, port=port)


@app.command("smoke-test")
def smoke_test(output: Path = Path("reports/software-smoke.json")):
    """Exercise real forward/backward with tones. This is NOT speech-separation evaluation."""
    import torch
    from .models.separator import VoiceSeparator, ModelConfig
    from .losses import hungarian_loss
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = VoiceSeparator(ModelConfig(layers=1, dimension=16, ffn_dimension=32))
    time = torch.arange(1024) / 8000
    refs = torch.stack([torch.sin(time * frequency * 2 * torch.pi) * 0.05
                        for frequency in (180, 310, 440)] + [torch.zeros_like(time)] * 2).unsqueeze(0)
    mixture = refs.sum(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        prediction = model(mixture)
        loss, _ = hungarian_loss(prediction, refs, mixture)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    report = {"forward_backward_passed": True, "finite_losses": losses,
              "five_speaker_speech_evaluated": False, "release_ready": False,
              "note": "Synthetic tones validate software execution only"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    emit(report)


if __name__ == "__main__":
    app()

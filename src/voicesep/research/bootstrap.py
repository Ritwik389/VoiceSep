"""Materialize a bounded, portable bilingual corpus on a CPU notebook before using GPU quota."""
from collections import Counter
from pathlib import Path
from math import gcd
import hashlib
import io
import json
import tarfile
import time
import re
import urllib.error
import urllib.request

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .data import write_jsonl, read_jsonl, split_sources, assert_disjoint, freeze_benchmark

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
LIBRISPEECH_MIRRORS = (LIBRISPEECH_URL,
    "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz",
    "https://openslr.elda.org/resources/12/train-clean-100.tar.gz")
LIBRISPEECH_MD5 = "2a93770f6d5c6c964bc36631d331a522"
HINDI_REPOSITORY = "ai4bharat/indicvoices_r"
RATE = 8000


def _checksum(path):
    digest = hashlib.md5()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(urls, destination, expected_md5, attempts=6):
    """Keep completed chunks on disk and reconnect with HTTP Range after a stalled read."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.part')
    if destination.is_file():
        if expected_md5 is None or _checksum(destination) == expected_md5:
            return destination
        destination.unlink()  # Only this generated download cache, never source recordings.
    for attempt in range(attempts):
        if partial.is_file() and expected_md5 and _checksum(partial) == expected_md5:
            partial.replace(destination)
            return destination
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {'User-Agent': 'VoiceSep-research/0.1', 'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        url = urls[attempt % len(urls)]
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset or not offset <= int(match[2]) < int(match[3]):
                        raise OSError('Invalid HTTP Content-Range; refusing to append mismatched bytes')
                    total = int(match[3])
                    mode = 'ab'
                elif response.status == 200:
                    # Some mirrors ignore Range. Restart instead of duplicating the existing prefix.
                    offset = 0
                    total = int(response.headers['Content-Length']) if response.headers.get('Content-Length') else None
                    mode = 'wb'
                else:
                    raise OSError(f'Unexpected download status {response.status}')
                print(json.dumps({'stage': 'english_download', 'attempt': attempt + 1,
                                  'downloaded_bytes': offset, 'total_bytes': total, 'source': url}), flush=True)
                last_log = time.monotonic()
                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_log >= 15:
                            print(json.dumps({'stage': 'english_download', 'downloaded_bytes': offset,
                                              'total_bytes': total}), flush=True)
                            last_log = time.monotonic()
            if total is not None and partial.stat().st_size != total:
                raise OSError('Archive transfer ended before its declared length')
            if expected_md5 and _checksum(partial) != expected_md5:
                partial.unlink()
                raise OSError('Archive checksum mismatch; retrying a clean download')
            partial.replace(destination)
            return destination
        except (OSError, urllib.error.URLError) as error:
            print(json.dumps({'stage': 'english_download_retry', 'attempt': attempt + 1,
                              'saved_bytes': partial.stat().st_size if partial.exists() else 0,
                              'error': str(error)}), flush=True)
            if attempt + 1 == attempts:
                raise RuntimeError('English archive download failed after retries. Retain the downloads/*.part cache to resume.') from error
            time.sleep(min(5 * (attempt + 1), 30))
    raise RuntimeError('No archive download attempts were made')


def _write_source(root, language, corpus, speaker, identifier, audio, rate, text=""):
    audio = np.asarray(audio, np.float32)
    if audio.ndim == 2:
        audio = audio.mean(-1)
    if rate != RATE:
        divisor = gcd(int(rate), RATE)
        audio = resample_poly(audio, RATE // divisor, int(rate) // divisor).astype(np.float32)
    seconds = len(audio) / RATE
    if not 4 <= seconds <= 30 or not np.isfinite(audio).all() or np.mean(audio ** 2) < 1e-8:
        return None
    speaker = str(speaker)
    key = hashlib.sha256(f"{corpus}:{speaker}:{identifier}".encode()).hexdigest()[:24]
    path = Path("audio") / language / f"{key}.flac"
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        sf.write(destination, audio, RATE, subtype="PCM_16")
    return {"path": path.as_posix(), "speaker_id": speaker, "language": language,
            "corpus": corpus, "text": text, "duration": seconds, "source_id": str(identifier)}


def prepare_english(root, hours=30, clips_per_speaker=40, url=LIBRISPEECH_URL):
    """Extract only after a resumable download has passed the publisher's checksum."""
    root = Path(root)
    cache = root / "english-index.jsonl"
    if cache.is_file():
        return read_jsonl(cache)
    entries, counts, transcripts = [], Counter(), {}
    duration = 0.
    archive_path = download_archive(LIBRISPEECH_MIRRORS if url == LIBRISPEECH_URL else (url,),
        root / 'downloads' / 'train-clean-100.tar.gz', LIBRISPEECH_MD5 if url == LIBRISPEECH_URL else None)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name.endswith(".trans.txt"):
                stream = archive.extractfile(member)
                for line in stream.read().decode("utf-8").splitlines():
                    if " " in line:
                        key, text = line.split(" ", 1)
                        transcripts[key] = text
                continue
            if not name.endswith(".flac") or duration >= hours * 3600:
                continue
            speaker = name.split("-")[0]
            if counts[speaker] >= clips_per_speaker:
                continue
            stream = archive.extractfile(member)
            wave, rate = sf.read(io.BytesIO(stream.read()), dtype="float32")
            entry = _write_source(root, "en", "librispeech", speaker, Path(name).stem, wave, rate)
            if entry:
                entries.append(entry)
                counts[speaker] += 1
                duration += entry["duration"]
                if len(entries) % 500 == 0:
                    print(json.dumps({"stage": "english", "clips": len(entries), "hours": round(duration / 3600, 2),
                                      "speakers": len(counts)}), flush=True)
    for entry in entries:
        entry["text"] = transcripts.get(entry["source_id"], "")
    write_jsonl(cache, entries)
    archive_path.unlink()  # The completed source index and retained FLACs are the durable corpus.
    return entries


def prepare_hindi(root, hours=20, clips_per_speaker=40, token=None, seed=42):
    """Use the publisher's gated dataset through a user-approved HF login; no gate bypass."""
    from datasets import Audio, get_dataset_config_names, get_dataset_split_names, load_dataset
    from huggingface_hub import HfApi, get_token
    root = Path(root)
    cache = root / "hindi-index.jsonl"
    if cache.is_file():
        return read_jsonl(cache)
    token = token or get_token()
    if not token:
        raise ValueError("IndicVoices-R requires approved Hugging Face access. Add HF_TOKEN as a private Kaggle secret; do not paste it into notebook code.")
    revision = HfApi(token=token).dataset_info(HINDI_REPOSITORY).sha
    configs = get_dataset_config_names(HINDI_REPOSITORY, token=token, revision=revision)
    config = next((name for name in configs if name.casefold() in {"hindi", "hi", "hin"}), None)
    if config is None:
        raise ValueError(f"Hindi configuration not found; available configurations: {configs}")
    splits = get_dataset_split_names(HINDI_REPOSITORY, config, token=token, revision=revision)
    split = "train" if "train" in splits else splits[0] if len(splits) == 1 else None
    if split is None:
        raise ValueError("The Hindi dataset has ambiguous splits; select a training split explicitly")
    dataset = load_dataset(HINDI_REPOSITORY, config, split=split, streaming=True, token=token, revision=revision)
    audio_field = next((name for name in ("audio", "audio_filepath", "wav") if name in dataset.features), None)
    if audio_field is None:
        raise ValueError("Hindi dataset must expose an audio column")
    dataset = dataset.cast_column(audio_field, Audio(decode=False)).shuffle(seed=seed, buffer_size=256)
    entries, counts, duration = [], Counter(), 0.
    for index, row in enumerate(dataset):
        details = row.get("metadata") or {}
        if isinstance(details, str):
            details = json.loads(details)
        speaker = row.get("speaker_id", details.get("speaker_id"))
        if speaker is None:
            raise ValueError("Stable original Hindi speaker IDs are required")
        speaker = str(speaker)
        if counts[speaker] >= clips_per_speaker:
            continue
        snr = row.get("snr", details.get("snr"))
        if snr is not None and float(snr) < 10:
            continue
        payload = row[audio_field]
        if payload.get("bytes") is not None:
            audio, rate = sf.read(io.BytesIO(payload["bytes"]), dtype="float32")
        elif payload.get("path") and Path(payload["path"]).is_file():
            audio, rate = sf.read(payload["path"], dtype="float32")
        else:
            raise ValueError("Hindi audio payload is not materialized by the dataset loader")
        identifier = row.get("filename", row.get("id", payload.get("path") or index))
        entry = _write_source(root, "hi", "indicvoices-r", speaker, identifier, audio, rate,
                              row.get("text", details.get("text", "")))
        if entry:
            entries.append(entry)
            counts[speaker] += 1
            duration += entry["duration"]
            if len(entries) % 500 == 0:
                print(json.dumps({"stage": "hindi", "clips": len(entries), "hours": round(duration / 3600, 2),
                                  "speakers": len(counts)}), flush=True)
        if duration >= hours * 3600:
            break
    write_jsonl(cache, entries)
    (root / "hindi-provenance.json").write_text(json.dumps({"dataset": HINDI_REPOSITORY,
        "revision": revision, "configuration": config, "split": split, "license": "CC-BY-4.0"}, indent=2))
    return entries


def bootstrap(destination, english_hours=30, hindi_hours=20, seed=42, token=None):
    from huggingface_hub import HfApi, get_token
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    settings = {"english_hours": english_hours, "hindi_hours": hindi_hours, "seed": seed,
                "sample_rate": RATE, "clips_per_speaker": 40}
    settings_file = root / "bootstrap-settings.json"
    if settings_file.is_file() and json.loads(settings_file.read_text()) != settings:
        raise ValueError("Bootstrap settings changed; use a fresh destination for a different corpus")
    settings_file.write_text(json.dumps(settings, indent=2))
    token = token or get_token()
    if not (root / "hindi-index.jsonl").is_file():
        if not token:
            raise ValueError("Approve IndicVoices-R access and configure HF_TOKEN before downloading the corpus")
        HfApi(token=token).auth_check(repo_id=HINDI_REPOSITORY, repo_type="dataset")
    entries = prepare_english(root, english_hours) + prepare_hindi(root, hindi_hours, token=token, seed=seed)
    # Keep corpus manifests portable between CPU output mounts and GPU notebook inputs.
    for entry in entries:
        if Path(entry["path"]).is_absolute():
            entry["path"] = Path(entry["path"]).relative_to(root).as_posix()
    write_jsonl(root / "sources.jsonl", entries)
    splits = split_sources(entries, root, seed)
    assert_disjoint(*(root / f"{name}.jsonl" for name in ("train", "validation", "test")))
    for name in ("train", "validation", "test"):
        records = read_jsonl(root / f"{name}.jsonl")
        for language in ("en", "hi"):
            if len({entry["identity"] for entry in records if entry["language"] == language}) < 5:
                raise ValueError(f"Not enough {language} identities in {name}; increase the corpus budget")
    freeze_benchmark(root / "validation.jsonl", root / "validation-mixtures.jsonl", per_condition=10, seed=12345)
    freeze_benchmark(root / "test.jsonl", root / "test-mixtures.jsonl", per_condition=100, seed=98765)
    report = {"sample_rate": RATE, "seed": seed, "splits": splits,
              "hours": {lang: sum(e["duration"] for e in entries if e["language"] == lang) / 3600 for lang in ("en", "hi")},
              "english_source": LIBRISPEECH_URL, "hindi_source": HINDI_REPOSITORY,
              "release_ready": False, "stage": "data_prepared"}
    (root / "preparation.json").write_text(json.dumps(report, indent=2))
    return report

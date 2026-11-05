"""Build local 2–5-speaker test WAVs from the downloaded held-out listening references."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import soundfile as sf


def build(archive, destination):
    archive, destination = Path(archive), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cases = []
    with zipfile.ZipFile(archive) as bundle:
        parents = {row['folder']: row for row in json.loads(bundle.read('manifest.json'))}
        for condition in ('en', 'hi', 'mixed'):
            folder = f'{condition}-5'
            sources = [sf.read(io.BytesIO(bundle.read(f'{folder}/reference-{i}.wav')), dtype='float32')
                       for i in range(1, 6)]
            assert all(sr == 8000 and wave.ndim == 1 and np.isfinite(wave).all() for wave, sr in sources)
            assert len({len(wave) for wave, _ in sources}) == 1
            for count in (2, 3, 4, 5):
                name = f'{condition}-{count}-speakers'
                refs = np.stack([wave for wave, _ in sources[:count]])
                mixture = refs.sum(0)
                scale = max(1., float(np.max(np.abs(mixture))) / .95)
                mixture, refs = mixture / scale, refs / scale
                target = destination / f'{count}-speakers' / f'{name}.wav'
                target.parent.mkdir(exist_ok=True)
                sf.write(target, mixture, 8000, subtype='PCM_16')
                reference_folder = destination / 'references' / name
                reference_folder.mkdir(parents=True, exist_ok=True)
                for i, wave in enumerate(refs, 1):
                    sf.write(reference_folder / f'speaker-{i}.wav', wave, 8000, subtype='PCM_16')
                cases.append({'file': str(target.relative_to(destination)), 'speaker_count': count,
                              'condition': condition, 'sample_rate': 8000, 'seconds': len(mixture) / 8000,
                              'reference_folder': str(reference_folder.relative_to(destination)),
                              'parent_test_mixture_id': parents[folder]['id'],
                              'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    (destination / 'manifest.json').write_text(json.dumps({'source_archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'description': 'Synthetic overlapping mixtures derived from held-out recorded voices. Local listening fixtures.',
        'examples': cases}, indent=2) + '\n')
    lines = ['# VoiceSep test audio', '',
             'Upload a WAV from the numbered folders into VoiceSep. Each file is a four-second synthetic mixture of recorded voices, at 8 kHz mono PCM16. Start with automatic speaker counting, then compare using the manual count shown in the folder name.', '',
             '| File | Voices | Language |', '|---|---:|---|']
    for case in sorted(cases, key=lambda row: (row['speaker_count'], row['condition'])):
        lines.append(f"| [{case['file']}]({case['file']}) | {case['speaker_count']} | {case['condition']} |")
    lines.extend(['', 'The `references` folder contains the clean component tracks for listening comparisons. These mixtures reuse the held-out five-speaker example references and serve as local test fixtures. The original benchmark report remains in the research bundle.', '',
                  'English recordings come from LibriSpeech. Hindi recordings come from IndicVoices-R under the access and usage terms accepted for this project. Keep these local fixtures and source audio out of Git. See `docs/DATA.md` for attribution and corpus details.', '',
                  'Rebuild with `python scripts/make_test_examples.py --archive PATH_TO_LISTENING_EXAMPLES_ZIP --output examples/test-audio`.'])
    (destination / 'README.md').write_text('\n'.join(lines) + '\n')
    return {'examples': len(cases), 'directory': str(destination.resolve())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('runs/kaggle-quota-finish/session-05/voicesep-checkpoints/listening-examples.zip'))
    parser.add_argument('--output', type=Path, default=Path('examples/test-audio'))
    args = parser.parse_args()
    print(build(args.archive, args.output))

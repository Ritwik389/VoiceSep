"""Quality-gated experimental speaker attributes, computed locally after separation."""
import gc
from math import gcd

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from .analysis_models import AgeVocalModel, EmotionModel, EMOTIONS, MODEL_INFO


def candidate_segments(wave, sample_rate, activity=None, limit=12):
    """Bounded windows of active audio with original timeline offsets."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not np.isfinite(wave).all():
        return [], 'Audio contains invalid samples'
    if len(wave) < 2 * sample_rate:
        return [], 'At least two seconds of speech is needed'
    if np.std(wave) < 1e-5:
        return [], 'No usable speech signal'
    regions = activity if activity is not None else [{'start': 0., 'end': len(wave) / sample_rate}]
    merged = []
    for region in sorted(regions, key=lambda item: item['start']):
        if merged and region['start'] - merged[-1]['end'] <= .3:
            merged[-1]['end'] = max(merged[-1]['end'], region['end'])
        else:
            merged.append(dict(region))
    windows = []
    for region in merged:
        start = max(0, round(region['start'] * sample_rate))
        end = min(len(wave), round(region['end'] * sample_rate))
        while end - start >= 2 * sample_rate:
            stop = min(end, start + 6 * sample_rate)
            segment = wave[start:stop]
            rms = float(np.sqrt(np.mean(segment ** 2)))
            clipped = float(np.mean(np.abs(segment) >= .999))
            if rms >= .0003 and clipped <= .01:
                windows.append({'start': start / sample_rate, 'end': stop / sample_rate,
                                'audio': segment, 'rms': rms})
            start = stop
    if len(windows) > limit:
        indices = np.linspace(0, len(windows) - 1, limit).round().astype(int)
        windows = [windows[index] for index in indices]
    return windows, None if windows else 'Speech is too short, quiet, or clipped for analysis'


def age_and_presentation(predictions):
    if not predictions:
        reason = 'At least three contiguous seconds of usable speech is needed'
        return {'status': 'uncertain', 'reason': reason}, {'status': 'uncertain', 'reason': reason}
    ages = np.asarray([item[0] for item in predictions], dtype=float)
    probabilities = np.asarray([item[1] for item in predictions], dtype=float)
    if (not np.isfinite(ages).all() or not np.isfinite(probabilities).all()
            or probabilities.shape != (len(predictions), 3)):
        raise ValueError('Invalid age/presentation predictions')
    median = float(np.median(ages))
    deviation = float(np.median(np.abs(ages - median)))
    if not 0 <= median <= 100 or deviation > 15:
        age = {'status': 'uncertain', 'reason': 'Age estimates vary too much across segments'}
    else:
        spread = max(10., 2 * 1.4826 * deviation)
        lower = max(0, int(np.floor((median - spread) / 10)) * 10)
        upper = min(100, int(np.ceil((median + spread) / 10)) * 10)
        age = {'status': 'estimated', 'range': [lower, upper], 'segments': len(predictions),
               'range_method': 'Rounded segment median with a minimum ten-year margin; uncalibrated',
               'reliability': 'experimental'}
    mean = probabilities.mean(0)
    ranked = np.argsort(-mean)
    winner = int(ranked[0])
    agreement = float(np.mean(np.argmax(probabilities, axis=1) == winner))
    scores = {label: float(value) for label, value in zip(('child_like', 'feminine_sounding', 'masculine_sounding'), mean)}
    presentation = {'status': 'uncertain', 'label': 'uncertain', 'scores': scores,
                    'score_kind': 'uncalibrated model probabilities'}
    if mean[winner] >= .65 and mean[winner] - mean[ranked[1]] >= .15 and agreement >= 2 / 3:
        presentation.update(status='estimated', label=tuple(scores)[winner])
    return age, presentation


def emotion_segment(scores, start, end):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(EMOTIONS),) or not np.isfinite(scores).all() or np.any(scores < 0) or scores.sum() <= 0:
        raise ValueError('Invalid emotion class scores')
    scores = scores / scores.sum()
    ranked = np.argsort(-scores)
    top = int(ranked[0])
    confident = scores[top] >= .5 and scores[top] - scores[ranked[1]] >= .15
    label = EMOTIONS[top] if confident and EMOTIONS[top] not in {'other', 'unknown'} else 'uncertain'
    return {'start': start, 'end': end, 'label': label, 'model_label': EMOTIONS[top],
            'model_score': float(scores[top]), 'scores': dict(zip(EMOTIONS, map(float, scores)))}


def _audio16(segment, sample_rate):
    common = gcd(sample_rate, 16000)
    return resample_poly(segment, 16000 // common, sample_rate // common).astype(np.float32)


def _failure(error):
    if isinstance(error, ImportError):
        return 'Install the optional analysis dependencies with pip install -e ".[analysis]"'
    return f'{type(error).__name__} while loading or running the model. Retry voicesep download-analysis-models.'


def analyze_tracks(paths, speakers=None, language=None, cancelled=lambda: False, progress=lambda *args: None,
                   age_factory=None, emotion_factory=None):
    """Return per-speaker results. No reference audio or demographic labels are assumed."""
    age_factory, emotion_factory = age_factory or AgeVocalModel, emotion_factory or EmotionModel
    speakers = {speaker['id']: speaker for speaker in (speakers or [])}
    results, windows_by_id = {}, {}
    for identifier, path in paths:
        if cancelled():
            raise InterruptedError('Speaker analysis cancelled')
        wave, sample_rate = sf.read(path, dtype='float32', always_2d=True)
        wave = wave.mean(axis=1)
        windows, reason = candidate_segments(wave, sample_rate, speakers.get(identifier, {}).get('activity'))
        warnings = ['Experimental estimates; English and Hindi accuracy on separated speech is unvalidated']
        if sample_rate < 16000:
            warnings.append('8 kHz separation output is resampled to 16 kHz; high-frequency information is absent')
        results[identifier] = {
            'version': 1, 'source_sample_rate': sample_rate, 'model_sample_rate': 16000,
            'models': MODEL_INFO, 'language': language or 'undetermined', 'warnings': warnings,
            'quality': {'usable_segments': len(windows), 'usable_seconds': sum(s['end'] - s['start'] for s in windows)},
            'age': {'status': 'uncertain', 'reason': reason or 'Insufficient speech for age estimation'},
            'vocal_presentation': {'status': 'uncertain', 'label': 'uncertain', 'reason': reason or 'Insufficient speech'},
            'emotion': {'status': 'uncertain', 'reason': reason or 'No emotion result', 'segments': []},
        }
        windows_by_id[identifier] = (sample_rate, windows)
    age_inputs = {}
    for identifier, (sr, windows) in windows_by_id.items():
        eligible = [segment for segment in windows if segment['end'] - segment['start'] >= 3]
        if len(eligible) > 3:
            indices = np.linspace(0, len(eligible) - 1, 3).round().astype(int)
            eligible = [eligible[index] for index in indices]
        age_inputs[identifier] = (sr, eligible)
    if any(windows for _, windows in age_inputs.values()):
        model = None
        try:
            progress('Loading age and vocal presentation model', .05)
            model = age_factory()
            for index, (identifier, (sr, windows)) in enumerate(age_inputs.items()):
                predictions = []
                for segment in windows:
                    if cancelled():
                        raise InterruptedError('Speaker analysis cancelled')
                    predictions.append(model.predict(_audio16(segment['audio'], sr)))
                age, vocal = age_and_presentation(predictions)
                results[identifier].update(age=age, vocal_presentation=vocal)
                progress('Estimating age and vocal presentation', .1 + .35 * (index + 1) / max(len(paths), 1))
        except InterruptedError:
            raise
        except Exception as error:
            for identifier, (_, windows) in age_inputs.items():
                if windows:
                    for name in ('age', 'vocal_presentation'):
                        results[identifier][name] = {'status': 'unavailable', 'reason': _failure(error)}
        finally:
            del model
            gc.collect()
    if any(windows for _, windows in windows_by_id.values()):
        model = None
        try:
            progress('Loading emotion model', .5)
            model = emotion_factory()
            for index, (identifier, (sr, windows)) in enumerate(windows_by_id.items()):
                segments = []
                for segment in windows:
                    if cancelled():
                        raise InterruptedError('Speaker analysis cancelled')
                    scores = model.predict(_audio16(segment['audio'], sr))
                    segments.append(emotion_segment(scores, segment['start'], segment['end']))
                if segments:
                    results[identifier]['emotion'] = {'status': 'estimated', 'segments': segments,
                        'score_kind': 'Uncalibrated model probabilities; uncertain labels are retained'}
                progress('Estimating vocal emotion over time', .55 + .4 * (index + 1) / max(len(paths), 1))
        except InterruptedError:
            raise
        except Exception as error:
            for identifier, (_, windows) in windows_by_id.items():
                if windows:
                    results[identifier]['emotion'] = {'status': 'unavailable', 'reason': _failure(error), 'segments': []}
        finally:
            del model
            gc.collect()
    return results

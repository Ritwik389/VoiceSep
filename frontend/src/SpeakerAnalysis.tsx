import {LoaderCircle, ScanLine} from 'lucide-react';
import {clock} from './types';
import type {Job, Speaker} from './types';

type Props = {
  job: Job;
  speaker: Speaker;
  busy: boolean;
  onAnalyze: () => void;
  onCancel: () => void;
  onSeek: (seconds: number) => void;
};

const presentationLabels: Record<string, string> = {
  feminine_sounding: 'Feminine-sounding',
  masculine_sounding: 'Masculine-sounding',
  child_like: 'Child-like',
  uncertain: 'Uncertain',
};

export function SpeakerAnalysis({job, speaker, busy, onAnalyze, onCancel, onSeek}: Props) {
  const analysis = speaker.analysis;
  const state = job.speaker_analysis;
  const analyzing = state?.status === 'queued' || state?.status === 'running';
  const age = analysis?.age;
  const vocal = analysis?.vocal_presentation;
  const emotion = analysis?.emotion;

  return <section className="inspector-section" aria-label="Speaker attributes">
    <div className="flex items-center justify-between gap-2">
      <h3>Voice attributes</h3><span className="badge badge-ghost badge-sm">Experimental</span>
    </div>
    <p className="text-sm text-base-content/70 mb-3">Age range, vocal presentation and emotion over time.</p>
    <button className="btn btn-sm btn-block" disabled={busy || analyzing || job.status !== 'complete'} onClick={onAnalyze}>
      {analyzing ? <LoaderCircle size={15} className="spin"/> : <ScanLine size={15}/>}
      {analyzing ? 'Analyzing voices…' : analysis ? 'Analyze all speakers again' : 'Analyze all speakers'}
    </button>
    {analyzing && <div className="mt-3 text-sm" aria-live="polite">
      <p>{state.stage} · {Math.round(state.progress * 100)}%</p>
      <button className="btn btn-ghost btn-xs mt-1" onClick={onCancel}>Cancel analysis</button>
    </div>}
    {state?.status === 'failed' && <p role="alert" className="text-sm text-error mt-3">{state.error}</p>}
    {state?.status === 'cancelled' && <p className="text-sm text-base-content/70 mt-3">Analysis cancelled. You can retry.</p>}
    {!analysis && !analyzing && <p className="text-xs text-base-content/60 mt-3">First use downloads the Hugging Face models. Inference runs locally. Age model weights are for noncommercial research.</p>}
    {analysis && <>
      <dl>
        <div><dt>Estimated age range</dt><dd>{age?.status === 'estimated' && age.range ? `${age.range[0]}–${age.range[1]} years` : age?.status === 'unavailable' ? 'Unavailable' : 'Uncertain'}</dd></div>
        <div><dt>Vocal presentation</dt><dd>{vocal?.status === 'unavailable' ? 'Unavailable' : presentationLabels[vocal?.label ?? 'uncertain'] ?? 'Uncertain'}</dd></div>
        <div><dt>Analyzed audio</dt><dd>{analysis.quality.usable_seconds.toFixed(1)} s</dd></div>
      </dl>
      {age?.reason && <p className="text-xs text-base-content/70 mt-2">Age · {age.reason}</p>}
      {vocal?.status === 'uncertain' && <p className="text-xs text-base-content/70 mt-2">The vocal presentation estimate has limited agreement.</p>}
      {vocal?.status === 'unavailable' && <p className="text-xs text-base-content/70 mt-2">Vocal presentation · {vocal.reason}</p>}
      <h4 className="text-sm font-medium mt-4 mb-2">Vocal emotion</h4>
      {emotion?.segments.length ? <div className="flex flex-col gap-1" aria-label="Emotion segments">
        {emotion.segments.map((segment, index) => <button key={index} className="btn btn-ghost btn-sm justify-between w-full"
          title="Seek to this speech segment. Scores are uncalibrated model outputs."
          onClick={() => onSeek(segment.start)}>
          <span className="text-xs tabular-nums">{clock(segment.start)}–{clock(segment.end)}</span>
          <span className="capitalize">{segment.label}</span>
        </button>)}
      </div> : <p className="text-xs text-base-content/70">{emotion?.reason ?? 'No usable emotion segment'}</p>}
      <p className="text-xs text-base-content/60 mt-3">Age bands and model scores are uncalibrated. Vocal presentation describes how the voice sounds.</p>
      {analysis.warnings.map(warning => <p key={warning} className="text-xs text-base-content/60 mt-2">{warning}</p>)}
      <a className="btn btn-ghost btn-sm btn-block mt-3" href={`/api/jobs/${job.id}/export/analysis`} download>Download analysis JSON</a>
    </>}
  </section>;
}

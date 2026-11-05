export type Region = {start:number; end:number};
export type AttributeEstimate = {status:'estimated'|'uncertain'|'unavailable'; reason?:string; range?:number[]; label?:string};
export type VoiceAnalysis = {
  version:number; source_sample_rate:number; model_sample_rate:number; warnings:string[];
  quality:{usable_segments:number; usable_seconds:number};
  age:AttributeEstimate; vocal_presentation:AttributeEstimate;
  emotion:{status:string; reason?:string; segments:(Region & {label:string; model_label:string; model_score:number})[]};
};
export type AnalysisState = {status:'queued'|'running'|'complete'|'partial'|'failed'|'cancelled'; progress:number; stage?:string; error:string|null};
export type Speaker = {id:string; name:string; slot:number; activity:Region[]; activity_fraction:number; talk_time:number; peaks:number[]; audio_url:string; analysis?:VoiceAnalysis};
export type Segment = Region & {speaker_id:string; text:string; language:string};
export type Job = {id:string; filename:string; created_at:number; status:string; progress:number; duration:number|null; sample_rate:number; speakers:Speaker[]; detected_count:number|null; flags:string[]; transcript:Segment[]; error:string|null; original_peaks?:number[]; original_url?:string; selection_offset?:number; speaker_analysis?:AnalysisState; options:{speaker_count?:number|null; transcribe?:boolean; language?:string|null}};
export type Health = {status:string; model:{checkpoint_available:boolean; checkpoint_path:string; trained:boolean; release_ready:boolean; capacity:number; sample_rate:number; cuda_available:boolean; calibrated?:boolean; step?:number; error:string|null}};
export const terminal = new Set(['imported','complete','failed','cancelled']);
export const colors = ['#88b9f8','#efc778','#e5a0b5','#83d3ad','#c4b899'];
export function clock(seconds:number) {const total=Math.max(0,Math.floor(seconds)); return `${Math.floor(total/60)}:${String(total%60).padStart(2,'0')}`;}
export async function request<T>(url:string, options?:RequestInit):Promise<T> {
  const response=await fetch(url,options);
  if(!response.ok){const error=await response.json().catch(()=>({detail:'The local server could not complete this request.'})); throw new Error(typeof error.detail==='string'?error.detail:'Please check the selected options.');}
  return response.status===204?undefined as T:response.json();
}

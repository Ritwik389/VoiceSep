export type AudioTrack = {id:string; url:string};
export type PlaybackState = {playing:boolean; position:number; duration:number; loading:boolean};

/** Every source starts on one AudioContext clock; mix changes never restart playback. */
export class SharedPlayer {
  private context:AudioContext|null=null;
  private buffers=new Map<string,AudioBuffer>();
  private sources:AudioBufferSourceNode[]=[];
  private gains=new Map<string,GainNode>();
  private gainValues:Record<string,number>={};
  private master:GainNode|null=null;
  private startedAt=0;
  private offset=0;
  private generation=0;
  private operation=0;
  private animation=0;
  private loop: [number,number]|null=null;
  state:PlaybackState={playing:false,position:0,duration:0,loading:false};
  onChange:(state:PlaybackState)=>void=()=>{};

  private emit(){this.onChange({...this.state});}
  private audioContext(){if(!this.context){this.context=new AudioContext();this.master=this.context.createGain();this.master.gain.value=0.8;this.master.connect(this.context.destination);}return this.context;}
  async load(tracks:AudioTrack[]){
    const generation=++this.generation;
    this.pause();this.buffers.clear();this.offset=0;this.state={playing:false,position:0,duration:0,loading:true};this.emit();
    try{
      if(!tracks.length)return;
      const context=this.audioContext();
      const buffers=await Promise.all(tracks.map(async track=>{const response=await fetch(track.url);if(!response.ok)throw new Error('This audio track is not available.');return [track.id,await context.decodeAudioData(await response.arrayBuffer())] as const;}));
      if(generation!==this.generation)return;
      for(const [id,buffer]of buffers)this.buffers.set(id,buffer);
      this.state.duration=Math.max(0,...buffers.map(([,buffer])=>buffer.duration));
    }finally{if(generation===this.generation){this.state.loading=false;this.emit();}}
  }
  setMix(values:Record<string,number>){this.gainValues={...values};const now=this.context?.currentTime??0;for(const[id,gain]of this.gains)gain.gain.setTargetAtTime(values[id]??0,now,0.012);}
  setVolume(value:number){if(this.master)this.master.gain.setTargetAtTime(value,this.context!.currentTime,0.01);}
  setLoop(range:[number,number]|null){this.loop=range;}
  async play(){
    if(this.state.loading||!this.buffers.size||this.state.playing)return;
    const operation=++this.operation;
    const context=this.audioContext();await context.resume();
    if(operation!==this.operation||this.state.playing)return;
    if(this.offset>=this.state.duration)this.offset=0;
    const start=context.currentTime+0.025;this.startedAt=start;this.state.playing=true;
    for(const[id,buffer]of this.buffers){
      if(this.offset>=buffer.duration)continue;
      const source=context.createBufferSource(),gain=context.createGain();source.buffer=buffer;gain.gain.value=this.gainValues[id]??0;
      source.connect(gain);gain.connect(this.master!);source.start(start,this.offset);this.sources.push(source);this.gains.set(id,gain);
    }
    this.emit();this.tick();
  }
  private tick=()=>{
    if(!this.state.playing)return;
    this.state.position=Math.min(this.state.duration,this.offset+Math.max(0,this.context!.currentTime-this.startedAt));
    if(this.loop&&this.state.position>=this.loop[1]){this.seek(this.loop[0]);return;}
    if(this.state.position>=this.state.duration){this.pause();this.offset=0;this.state.position=0;this.emit();return;}
    this.emit();this.animation=requestAnimationFrame(this.tick);
  };
  pause(){
    this.operation++;
    if(this.state.playing&&this.context)this.offset=Math.min(this.state.duration,this.offset+Math.max(0,this.context.currentTime-this.startedAt));
    this.state.playing=false;this.state.position=this.offset;cancelAnimationFrame(this.animation);
    for(const source of this.sources){try{source.stop();}catch{/* Already ended. */}source.disconnect();}
    for(const gain of this.gains.values())gain.disconnect();this.sources=[];this.gains.clear();this.emit();
  }
  seek(seconds:number){const wasPlaying=this.state.playing;this.pause();this.offset=Math.min(this.state.duration,Math.max(0,seconds));this.state.position=this.offset;this.emit();if(wasPlaying)void this.play();}
  dispose(){this.generation++;this.pause();void this.context?.close();this.context=null;}
}

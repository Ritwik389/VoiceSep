import {beforeEach,describe,expect,it,vi} from 'vitest';
import {SharedPlayer} from './player';

class FakeGain {gain={value:0,setTargetAtTime:vi.fn()};connect=vi.fn();disconnect=vi.fn();}
class FakeSource {buffer:unknown;start=vi.fn();stop=vi.fn();connect=vi.fn();disconnect=vi.fn();}
class FakeContext {
  static latest:FakeContext;currentTime=10;destination={};sources:FakeSource[]=[];gains:FakeGain[]=[];
  constructor(){FakeContext.latest=this;}
  resume=vi.fn(async()=>{});close=vi.fn(async()=>{});
  createGain(){const gain=new FakeGain();this.gains.push(gain);return gain;}
  createBufferSource(){const source=new FakeSource();this.sources.push(source);return source;}
  decodeAudioData=vi.fn(async()=>({duration:24}));
}
beforeEach(()=>{vi.stubGlobal('AudioContext',FakeContext);vi.stubGlobal('requestAnimationFrame',vi.fn(()=>1));vi.stubGlobal('cancelAnimationFrame',vi.fn());vi.stubGlobal('fetch',vi.fn(async()=>({ok:true,arrayBuffer:async()=>new ArrayBuffer(8)})));});
describe('shared audio clock',()=>{
  it('schedules all voices together, and solos without restarting',async()=>{
    const player=new SharedPlayer();await player.load([{id:'a',url:'/a.wav'},{id:'b',url:'/b.wav'}]);player.setMix({a:1,b:1});await player.play();
    const context=FakeContext.latest;
    expect(context.sources).toHaveLength(2);
    expect(context.sources[0].start.mock.calls[0]).toEqual(context.sources[1].start.mock.calls[0]);
    context.currentTime=12;player.setMix({a:1,b:0});
    expect(context.sources).toHaveLength(2);expect(context.sources[0].stop).not.toHaveBeenCalled();
    expect(context.gains[2].gain.setTargetAtTime).toHaveBeenCalledWith(0,12,.012);
    player.pause();expect(player.state.position).toBeCloseTo(1.975);player.dispose();
  });
  it('seeks each source to the same bounded offset',async()=>{
    const player=new SharedPlayer();await player.load([{id:'a',url:'/a.wav'},{id:'b',url:'/b.wav'}]);await player.play();player.seek(7);await Promise.resolve();
    const sources=FakeContext.latest.sources.slice(-2);
    expect(sources[0].start.mock.calls[0][1]).toBe(7);expect(sources[1].start.mock.calls[0][1]).toBe(7);
    player.pause();player.seek(-4);expect(player.state.position).toBe(0);player.seek(100);expect(player.state.position).toBe(24);player.dispose();
  });
  it('cancels an in-flight play on pause and avoids duplicate starts',async()=>{
    const player=new SharedPlayer();await player.load([{id:'a',url:'/a.wav'}]);const pending=player.play();player.pause();await pending;expect(FakeContext.latest.sources).toHaveLength(0);
    await Promise.all([player.play(),player.play()]);expect(FakeContext.latest.sources).toHaveLength(1);player.dispose();
  });
});

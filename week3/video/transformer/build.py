"""Build: audio, sync, preview, render, assemble, check. Use --scene for one scene."""
import argparse
import asyncio
import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path

from lesson import SCENES, SOURCES
from facts import FACTS, verify

ROOT = Path(__file__).resolve().parent
VOICE = 'ko-KR-SunHiNeural'
FINAL = ROOT / 'transformer-ops.ko.mp4'
PREVIEW = ROOT / 'transformer-ops-preview.ko.mp4'
FPS, SECONDS, RATE = 30, 600, 48000


def run(args):
    subprocess.run([str(a) for a in args], check=True)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format',
        '-show_streams', '-show_chapters', '-of', 'json', str(path)]))


def stamp(t):
    ms = round(t*1000)
    return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'


def digest(s):
    return hashlib.sha256((VOICE+'|WordBoundary|'+s['narration']).encode()).hexdigest()


async def audio():
    import edge_tts
    folder = ROOT/'audio'
    folder.mkdir(exist_ok=True)
    for i, s in enumerate(SCENES):
        path, meta = folder/f'{i:02}.mp3', folder/f'{i:02}.json'
        if path.exists() and meta.exists() and json.loads(meta.read_text()).get('digest')==digest(s):
            print(f'Audio cached {i+1}/{len(SCENES)}', flush=True)
            continue
        for attempt in range(3):
            try:
                boundaries = []
                c = edge_tts.Communicate(s['narration'], voice=VOICE, boundary='WordBoundary')
                with path.with_suffix('.part').open('wb') as f:
                    async for chunk in c.stream():
                        if chunk['type']=='audio':
                            f.write(chunk['data'])
                        elif chunk['type']=='WordBoundary':
                            boundaries.append({k: chunk[k] for k in ('offset','duration','text')})
                if not boundaries:
                    raise RuntimeError('Missing speech boundaries')
                path.with_suffix('.part').replace(path)
                save(meta, dict(digest=digest(s), voice=VOICE, boundaries=boundaries))
                print(f'Audio {i+1}/{len(SCENES)}: {probe(path)["format"]["duration"]}s', flush=True)
                break
            except Exception:
                if attempt==2:
                    raise
                await asyncio.sleep(2)
    sync()


def sync():
    lengths = [float(probe(ROOT/'audio'/f'{i:02}.mp3')['format']['duration']) for i in range(len(SCENES))]
    tempo = max(1., sum(lengths)/(SECONDS-32))
    if tempo>1.15:
        raise RuntimeError(f'Narration needs {sum(lengths):.1f}s, tempo={tempo:.3f}; shorten text to <=653.2s.')
    spare = SECONDS-sum(lengths)/tempo
    scenes, cues, cursor = [], [], 0
    for i,(s,length) in enumerate(zip(SCENES,lengths,strict=True)):
        meta = json.loads((ROOT/'audio'/f'{i:02}.json').read_text())
        assert meta['digest']==digest(s)
        frames = round((length/tempo+spare/len(SCENES))*FPS)
        if i==len(SCENES)-1:
            frames=SECONDS*FPS-cursor
        start=cursor/FPS
        # Spoken-word offsets set visual reveal moments and subtitle times.
        boundaries=meta['boundaries']
        reveal=[.35+boundaries[min(len(boundaries)-1,round(len(boundaries)*q))]['offset']/1e7/tempo
                for q in (0,.30,.60)]
        scenes.append(dict(s,index=i,start=start,frames=frames,duration=frames/FPS,
                           audio_duration=length,tempo=tempo,reveal=reveal))
        words, begin, source_cursor = [], None, 0
        for j,w in enumerate(boundaries):
            if begin is None:
                begin=start+.35+w['offset']/1e7/tempo
            text=w['text']
            pos=s['narration'].find(text,source_cursor)
            if pos>=0:
                source_cursor=pos+len(text)
                if source_cursor<len(s['narration']) and s['narration'][source_cursor] in '.?!,':
                    text+=s['narration'][source_cursor]
                    source_cursor+=1
            words.append(text)
            end=start+.35+(w['offset']+w['duration'])/1e7/tempo
            if len(' '.join(words))>=34 or text.endswith(('.','?','!')) or j==len(boundaries)-1:
                cues.append([begin,min(end+.12,start+frames/FPS-.08),' '.join(words)])
                words,begin=[],None
        cursor+=frames
        run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',ROOT/'audio'/f'{i:02}.mp3',
             '-af',f'aresample={RATE},atempo={tempo},asetpts=N/SR/TB,adelay=350:all=1,apad,atrim=end_sample={frames*1600},asetpts=N/SR/TB',
             '-ar',RATE,'-ac','1','-c:a','pcm_s16le',ROOT/'audio'/f'{i:02}-timed.wav'])
    # Avoid flashing a short sentence ending as its own subtitle.
    merged=[]
    for cue in cues:
        if (merged and len(cue[2])<9 and cue[0]-merged[-1][1]<.3
                and len(merged[-1][2])+1+len(cue[2])<=44):
            merged[-1][1]=cue[1]
            merged[-1][2]+=' '+cue[2]
        else:
            merged.append(cue)
    cues=merged
    for j,c in enumerate(cues[:-1]):
        c[1]=min(c[1],cues[j+1][0]-.001)
    save(ROOT/'timeline.json',scenes)
    save(ROOT/'facts.json',FACTS)
    save(ROOT/'lesson.json',SCENES)
    (ROOT/'transformer.ko.srt').write_text('\n\n'.join(f'{i+1}\n{stamp(a)} --> {stamp(b)}\n{txt}' for i,(a,b,txt) in enumerate(cues))+'\n')
    def ass_time(t):
        cs=round(t*100)
        return f'{cs//360000}:{cs//6000%60:02}:{cs//100%60:02}.{cs%100:02}'
    ass_header='''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,UnDotum,42,&H00FAF2ED,&H00FAF2ED,&H00150C07,&H00150C07,0,0,0,0,100,100,0,0,1,1,0,2,80,80,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    (ROOT/'transformer.ko.ass').write_text(ass_header+'\n'.join(
        f'Dialogue: 0,{ass_time(a)},{ass_time(b)},Default,,0,0,0,,{txt}' for a,b,txt in cues)+'\n')
    script=['# Transformer 연산 지도: Params와 FLOPs',
            '한국어 SunHi · 10분 · decoder-only / MHA / LayerNorm / RoPE / SwiGLU · 추론 순전파']
    for s in scenes:
        script.append(f'## {stamp(s["start"])[:-4]} · {s["title"]}\n\n{s["narration"]}\n\n화면: {s["takeaway"]}')
    script.append('## 참고 자료\n\n'+'\n'.join(f'- [{name}]({url})' for name,url in SOURCES))
    (ROOT/'script.ko.md').write_text('\n\n'.join(script)+'\n')
    groups=[s for i,s in enumerate(scenes) if i==0 or s['chapter']!=scenes[i-1]['chapter']]
    lines=[';FFMETADATA1','title=Transformer 연산 지도 — 파라미터와 FLOPs','language=kor']
    for i,g in enumerate(groups):
        end=round(groups[i+1]['start']*1000) if i+1<len(groups) else 600000
        lines+=['[CHAPTER]','TIMEBASE=1/1000',f'START={round(g["start"]*1000)}',f'END={end}',f'title={g["chapter"]}']
    (ROOT/'chapters.ffmetadata').write_text('\n'.join(lines)+'\n')
    listing=ROOT/'audio'/'concat.txt'
    listing.write_text('\n'.join(f"file '{i:02}-timed.wav'" for i in range(len(scenes))))
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',listing,'-c','copy',ROOT/'audio'/'narration.wav'])
    print(f'Synced {len(scenes)} scenes; raw {sum(lengths):.2f}s; tempo {tempo:.4f}; final 600s',flush=True)


def assemble():
    import visuals
    scenes=visuals.scenes()
    for s in scenes:
        assert json.loads((ROOT/'clips'/f'{s["index"]:02}.json').read_text())['signature']==visuals.signature(s['index'])
    listing=ROOT/'clips'/'concat.txt'
    listing.write_text('\n'.join(f"file '{s['index']:02}.mp4'" for s in scenes))
    part=FINAL.with_suffix('.part.mp4')
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',listing,
         '-i',ROOT/'audio'/'narration.wav','-i',ROOT/'chapters.ffmetadata',
         '-map','0:v','-map','1:a','-map_metadata','2','-map_chapters','2',
         '-vf',f"ass='{ROOT/'transformer.ko.ass'}'",
         '-af','loudnorm=I=-16:TP=-1.5:LRA=11','-c:v','libx264','-threads','4',
         '-preset','fast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k',
         '-ar',RATE,'-metadata:s:a:0','language=kor','-t',SECONDS,'-movflags','+faststart',part])
    part.replace(FINAL)
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',FINAL,'-t','60','-map_chapters','-1',
         '-c:v','libx264','-threads','4','-preset','fast','-crf','19','-c:a','aac','-movflags','+faststart',PREVIEW])
    print(FINAL,flush=True)


def check():
    import visuals
    result=verify()
    assert json.loads((ROOT/'facts.json').read_text())==FACTS
    scenes=visuals.scenes()
    assert len(scenes)==len(SCENES)
    cursor=0
    for s,original in zip(scenes,SCENES,strict=True):
        i=s['index']
        assert s['narration']==original['narration']
        assert abs(s['start']-cursor/FPS)<1e-8
        assert s['audio_duration']/s['tempo']+.35<s['duration']
        assert 1<=s['tempo']<=1.15
        meta=json.loads((ROOT/'audio'/f'{i:02}.json').read_text())
        assert meta['digest']==digest(s) and meta['voice']==VOICE and meta['boundaries']
        clip=json.loads((ROOT/'clips'/f'{i:02}.json').read_text())
        assert clip['signature']==visuals.signature(i) and clip['frames']==s['frames']
        cursor+=s['frames']
    assert cursor==18000
    with wave.open(str(ROOT/'audio'/'narration.wav')) as w:
        assert w.getnframes()==SECONDS*RATE and w.getframerate()==RATE
    def seconds(value):
        h,m,s=value.replace(',','.').split(':')
        return 3600*int(h)+60*int(m)+float(s)
    previous=0
    for block in (ROOT/'transformer.ko.srt').read_text().strip().split('\n\n'):
        lines=block.splitlines()
        a,b=map(seconds,lines[1].split(' --> '))
        assert previous<=a<b<=600 and lines[2].strip()
        previous=b
    data=probe(FINAL)
    video=next(s for s in data['streams'] if s['codec_type']=='video')
    assert (video['width'],video['height'],video['r_frame_rate'],int(video['nb_frames']))==(1920,1080,'30/1',18000)
    assert video['codec_name']=='h264'
    assert any(s['codec_type']=='audio' and s['codec_name']=='aac' for s in data['streams'])
    assert abs(float(data['format']['duration'])-600)<.1
    groups=[s for i,s in enumerate(scenes) if i==0 or s['chapter']!=scenes[i-1]['chapter']]
    assert len(data['chapters'])==len(groups)==9
    for ch,s in zip(data['chapters'],groups,strict=True):
        assert abs(float(ch['start_time'])-s['start'])<.002
    preview=probe(PREVIEW)
    assert abs(float(preview['format']['duration'])-60)<.1
    run(['ffmpeg','-hide_banner','-loglevel','error','-xerror','-i',FINAL,'-f','null','-'])
    result.update(voice=VOICE,scenes=len(scenes),duration=600,resolution=[1920,1080],fps=30,frames=18000,
                  chapters=9,full_decode='passed',tempo=scenes[0]['tempo'],
                  sha256=hashlib.sha256(FINAL.read_bytes()).hexdigest(),
                  sources=[dict(title=n,url=u) for n,u in SOURCES])
    save(ROOT/'validation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['audio','sync','preview','render','assemble','check'])
    p.add_argument('--scene',type=int)
    args=p.parse_args()
    if args.command=='audio':
        asyncio.run(audio())
    elif args.command in ('preview','render'):
        import visuals
        getattr(visuals,args.command)(args.scene)
    else:
        globals()[args.command]()

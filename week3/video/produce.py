"""Shared narration and timeline validation. Commands: audio, check."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess

from lesson import SCENES

ROOT = Path(__file__).resolve().parent

def run(args):
    subprocess.run([str(a) for a in args], check=True)

def duration(path):
    return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'default=nw=1:nk=1', str(path)], text=True))

def stamp(t):
    ms = round(t * 1000)
    return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'

async def audio():
    import edge_tts
    from edge_tts import VoicesManager
    directory = ROOT / 'audio'
    directory.mkdir(exist_ok=True)
    voice = 'ko-KR-SunHiNeural'
    voices = await VoicesManager.create()
    if not any(v['ShortName'] == voice for v in voices.voices):
        raise RuntimeError(f'Required Korean voice unavailable: {voice}')
    for i, scene in enumerate(SCENES):
        path = directory / f'{i:02}.mp3'
        meta = directory / f'{i:02}.json'
        digest = hashlib.sha256((voice + scene['narration']).encode()).hexdigest()
        if path.exists() and meta.exists() and json.loads(meta.read_text())['digest'] == digest:
            continue
        for attempt in range(3):
            try:
                boundaries = []
                communicate = edge_tts.Communicate(scene['narration'], voice=voice, boundary='WordBoundary')
                with path.with_suffix('.part').open('wb') as stream:
                    async for chunk in communicate.stream():
                        if chunk['type'] == 'audio':
                            stream.write(chunk['data'])
                        elif chunk['type'] == 'WordBoundary':
                            boundaries.append({k: chunk[k] for k in ('offset', 'duration', 'text')})
                path.with_suffix('.part').replace(path)
                meta.write_text(json.dumps(dict(digest=digest, voice=voice, boundaries=boundaries), ensure_ascii=False, indent=2))
                print(f'Audio {i+1}/{len(SCENES)}: {duration(path):.2f}s', flush=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(2)
    lengths = [duration(directory / f'{i:02}.mp3') for i in range(len(SCENES))]
    # Leave about 1.2 seconds per scene for comfortable transitions.
    tempo = max(1.0, sum(lengths) / (600 - 36))
    if tempo > 1.25:
        raise RuntimeError(f'Narration too long for comfortable delivery: tempo {tempo:.3f}; revise script.')
    spare = 600 - sum(lengths) / tempo
    timeline = []
    cursor = 0
    cues = []
    for i, (scene, length) in enumerate(zip(SCENES, lengths)):
        frames = round((length / tempo + spare / len(SCENES)) * 30)
        if i == len(SCENES)-1:
            frames = 18000 - cursor
        start = cursor / 30
        item = dict(scene, index=i, start=start, frames=frames, duration=frames/30,
                    audio_duration=length, tempo=tempo)
        timeline.append(item)
        raw = json.loads((directory / f'{i:02}.json').read_text())['boundaries']
        words = []
        cue_start = None
        source_cursor=0
        for j, word in enumerate(raw):
            if cue_start is None:
                cue_start = start + .35 + word['offset'] / 1e7 / tempo
            spoken=word['text']
            position=scene['narration'].find(spoken,source_cursor)
            if position>=0:
                source_cursor=position+len(spoken)
                if source_cursor<len(scene['narration']) and scene['narration'][source_cursor] in '.?!,':
                    spoken+=scene['narration'][source_cursor]
                    source_cursor+=1
            words.append(spoken)
            end = start + .35 + (word['offset'] + word['duration']) / 1e7 / tempo
            if len(' '.join(words)) >= 27 or spoken.endswith(('.', '?', '!')) or j == len(raw)-1:
                cues.append((cue_start, min(end+.08, start+frames/30-.1), ' '.join(words)))
                words, cue_start = [], None
        cursor += frames
    cues=[(begin,min(end,cues[j+1][0]-.001) if j+1<len(cues) else end,body)
          for j,(begin,end,body) in enumerate(cues)]
    (ROOT/'timeline.json').write_text(json.dumps(timeline, ensure_ascii=False, indent=2))
    (ROOT/'week3.ko.srt').write_text('\n\n'.join(f'{i+1}\n{stamp(s)} --> {stamp(e)}\n{t}' for i,(s,e,t) in enumerate(cues))+'\n')
    (ROOT/'script.ko.md').write_text('# 3주차: 커널의 시간은 어디에서 소비되는가\n\n' + '\n\n'.join(
        f'## {stamp(s["start"])[:-4]} · {s["title"]}\n\n{s["narration"]}\n\n화면: `{s["equation"]}`\n\n{s["note"]}' for s in timeline))
    print(f'Timeline: 600 seconds; original speech {sum(lengths):.1f}s; tempo {tempo:.4f}', flush=True)

def check():
    timeline=json.loads((ROOT/'timeline.json').read_text())
    assert sum(s['frames'] for s in timeline)==18000
    cursor=0
    for scene in timeline:
        assert abs(scene['start']-cursor/30)<1e-8
        assert scene['audio_duration']/scene['tempo']+.35<scene['duration']
        cursor+=scene['frames']
    def seconds(value):
        h,m,s=value.replace(',','.').split(':')
        return 3600*int(h)+60*int(m)+float(s)
    previous_end=0
    for block in (ROOT/'week3.ko.srt').read_text().strip().split('\n\n'):
        lines=block.splitlines()
        begin,end=map(seconds,lines[1].split(' --> '))
        assert previous_end<=begin<end<=600
        assert len(lines)>=3 and lines[2].strip()
        previous_end=end
    assert 10**8/(20*10**12)==5e-6
    assert 12*10**8/10**12==.0012
    print('PASS: timeline, narration fit, non-overlapping subtitles and lesson arithmetic')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audio', 'check'])
    args = parser.parse_args()
    if args.command == 'audio':
        asyncio.run(audio())
    else:
        check()
